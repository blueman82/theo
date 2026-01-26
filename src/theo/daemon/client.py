"""DaemonClient - Lightweight client for Theo daemon IPC.

This module provides a synchronous client for communicating with the
Theo daemon via Unix socket. It's designed for fast embedding operations
without subprocess overhead.

Features:
    - Lazy connection (only connect when first request is made)
    - Automatic fallback to direct embedding when daemon unavailable
    - Timeout handling at connection and request levels
    - Context manager support for automatic cleanup

Protocol:
    The client sends newline-delimited JSON to the daemon:
    {"cmd": "embed", "args": {"texts": [...]}, "request_id": "abc"}

    And receives JSON responses:
    {"success": true, "data": {...}, "request_id": "abc"}

Usage:
    # Context manager (recommended)
    with DaemonClient() as client:
        result = client.embed(texts=["hello", "world"])

    # Manual management
    client = DaemonClient()
    try:
        result = client.send("embed", texts=["hello", "world"])
    finally:
        client.close()

    # Quick check
    if DaemonClient.is_daemon_running():
        print("Daemon available")
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from theo.constants import (
    PID_FILE,
    RECV_BUFFER,
    SOCKET_PATH,
    get_connect_timeout,
    get_idle_timeout,
    get_request_timeout,
)
from theo.daemon.protocol import DaemonResponse

__all__ = [
    "DaemonClient",
    "DaemonClientError",
    "DaemonConnectionError",
    "DaemonTimeoutError",
    "DaemonProtocolError",
    "get_client",
    "get_shared_client",
    "close_shared_client",
    "quick_embed",
]


# =============================================================================
# Exceptions
# =============================================================================


class DaemonClientError(Exception):
    """Base exception for daemon client errors."""


class DaemonConnectionError(DaemonClientError):
    """Raised when connection to daemon fails."""


class DaemonTimeoutError(DaemonClientError):
    """Raised when a request times out."""


class DaemonProtocolError(DaemonClientError):
    """Raised when response parsing fails."""


# =============================================================================
# DaemonClient
# =============================================================================


class DaemonClient:
    """Synchronous client for communicating with Theo daemon.

    This client provides a simple interface for embedding operations
    via the Theo daemon. It handles connection management, timeouts,
    and automatic fallback when the daemon is unavailable.

    Attributes:
        socket_path: Path to the Unix socket.
        connect_timeout: Timeout for initial connection in seconds.
        request_timeout: Timeout for request/response in seconds.
        auto_fallback: Whether to automatically fall back to direct embedding.

    Example:
        >>> with DaemonClient() as client:
        ...     result = client.embed(["What is Python?"])
        ...     if result["success"]:
        ...         embeddings = result["data"]["embeddings"]
    """

    def __init__(
        self,
        socket_path: Path | str | None = None,
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        auto_fallback: bool = True,
        idle_timeout: float | None = None,
    ) -> None:
        """Initialize the daemon client.

        Args:
            socket_path: Path to Unix socket (default: /tmp/theo.sock).
            connect_timeout: Connection timeout in seconds.
            request_timeout: Request/response timeout in seconds.
            auto_fallback: If True, fall back to direct embedding when daemon unavailable.
            idle_timeout: Connection idle timeout in seconds (default from env or 300s).
        """
        self.socket_path = Path(socket_path) if socket_path else SOCKET_PATH
        self.auto_fallback = auto_fallback

        # All timeouts from param or constants module
        if connect_timeout is not None:
            self.connect_timeout = connect_timeout
        else:
            self.connect_timeout = get_connect_timeout()

        if request_timeout is not None:
            self.request_timeout = request_timeout
        else:
            self.request_timeout = get_request_timeout()

        if idle_timeout is not None:
            self.idle_timeout = idle_timeout
        else:
            self.idle_timeout = get_idle_timeout()

        self._socket: socket.socket | None = None
        self._connected = False
        self._last_error: str | None = None
        self._last_used: float = 0.0  # Timestamp of last successful operation

    def __enter__(self) -> DaemonClient:
        """Context manager entry."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Context manager exit - close connection."""
        self.close()

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """Connect to the daemon's Unix socket.

        This is called automatically on first send() if not already connected.
        Can be called explicitly to eagerly establish connection.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            DaemonConnectionError: If auto_fallback is False and connection fails.
        """
        if self._connected and self._socket:
            return True

        # Check if socket exists
        if not self.socket_path.exists():
            self._last_error = f"Socket not found: {self.socket_path}"
            if self.auto_fallback:
                return False
            raise DaemonConnectionError(self._last_error)

        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect(str(self.socket_path))
            self._connected = True
            self._last_error = None
            self._last_used = time.time()
            return True

        except TimeoutError:
            self._cleanup_socket()
            self._last_error = "Connection timed out"
            if self.auto_fallback:
                return False
            raise DaemonTimeoutError(self._last_error)

        except ConnectionRefusedError:
            self._cleanup_socket()
            self._last_error = "Connection refused"
            if self.auto_fallback:
                return False
            raise DaemonConnectionError(self._last_error)

        except OSError as e:
            self._cleanup_socket()
            self._last_error = f"OSError: {e}"
            if self.auto_fallback:
                return False
            raise DaemonConnectionError(f"Connection failed: {e}")

    def close(self) -> None:
        """Close the connection to the daemon.

        Safe to call multiple times. Does nothing if not connected.
        """
        self._cleanup_socket()
        self._connected = False

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    # =========================================================================
    # Request/Response
    # =========================================================================

    def send(
        self,
        cmd: str,
        request_id: str | None = None,
        timeout: float | None = None,
        **args: Any,
    ) -> dict[str, Any]:
        """Send a command to the daemon and receive the response.

        This is the core method for daemon communication. It handles:
        - Lazy connection (connects if not already connected)
        - JSON serialization/deserialization
        - Timeout handling
        - Automatic fallback if configured

        Args:
            cmd: Command name (ping, embed, embed_async, get_job_status, status).
            request_id: Optional request identifier for correlation.
            timeout: Override request timeout for this call.
            **args: Command arguments passed to the daemon.

        Returns:
            Response dictionary with at minimum {"success": bool}.
            On success: {"success": True, "data": {...}, "request_id": "..."}
            On failure: {"success": False, "error": "...", "request_id": "..."}

        Example:
            >>> result = client.send("embed", texts=["hello", "world"])
            >>> if result["success"]:
            ...     embeddings = result["data"]["embeddings"]
        """
        request_id = request_id or str(uuid.uuid4())[:8]
        effective_timeout = timeout if timeout is not None else self.request_timeout

        # Check if connection is stale (idle too long)
        if self._connected and self._last_used > 0:
            idle_time = time.time() - self._last_used
            if idle_time > self.idle_timeout:
                # Connection is stale, close and reconnect
                self.close()

        # Try daemon connection
        if not self._connected:
            if not self.connect():
                # Connection failed, fall back if configured
                if self.auto_fallback:
                    return self._fallback(cmd, args, request_id)
                return {
                    "success": False,
                    "error": "Daemon not available",
                    "request_id": request_id,
                    "fallback": False,
                }

        # Build request message
        message = {
            "cmd": cmd,
            "args": args,
            "request_id": request_id,
        }

        try:
            return self._send_receive(message, effective_timeout)

        except (DaemonConnectionError, DaemonTimeoutError, DaemonProtocolError) as e:
            # Connection error during send/receive
            self._last_error = f"send_receive failed: {type(e).__name__}: {e}"
            self.close()

            if self.auto_fallback:
                return self._fallback(cmd, args, request_id)

            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
            }

    def _send_receive(
        self,
        message: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Send message and receive response over socket.

        Args:
            message: Request message dictionary.
            timeout: Timeout in seconds.

        Returns:
            Parsed response dictionary.

        Raises:
            DaemonConnectionError: If socket is not connected.
            DaemonTimeoutError: If operation times out.
            DaemonProtocolError: If response parsing fails.
        """
        if not self._socket:
            raise DaemonConnectionError("Not connected")

        # Serialize and send (newline-delimited)
        data = json.dumps(message) + "\n"

        try:
            self._socket.settimeout(timeout)
            self._socket.sendall(data.encode("utf-8"))

        except TimeoutError:
            raise DaemonTimeoutError("Send timed out")
        except BrokenPipeError:
            raise DaemonConnectionError("Connection lost (broken pipe)")
        except OSError as e:
            raise DaemonConnectionError(f"Send failed: {e}")

        # Receive response (newline-delimited)
        try:
            response_data = self._recv_until_newline(timeout)

        except TimeoutError:
            raise DaemonTimeoutError("Receive timed out")
        except OSError as e:
            raise DaemonConnectionError(f"Receive failed: {e}")

        # Parse response
        response = DaemonResponse.from_json(response_data)
        if response is None:
            raise DaemonProtocolError("Invalid JSON response")

        # Update last_used timestamp on successful response
        self._last_used = time.time()

        result: dict[str, Any] = {"success": response.success}
        if response.data:
            result["data"] = response.data
        if response.error:
            result["error"] = response.error
        if response.request_id:
            result["request_id"] = response.request_id

        return result

    def _recv_until_newline(self, timeout: float) -> str:
        """Receive data until newline character.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Received data as string (without trailing newline).

        Raises:
            DaemonConnectionError: If connection is closed.
            socket.timeout: If operation times out.
        """
        if not self._socket:
            raise DaemonConnectionError("Not connected")

        self._socket.settimeout(timeout)
        chunks: list[bytes] = []
        total_size = 0

        while True:
            chunk = self._socket.recv(RECV_BUFFER)

            if not chunk:
                raise DaemonConnectionError("Connection closed by server")

            chunks.append(chunk)
            total_size += len(chunk)

            # Check for newline
            if b"\n" in chunk:
                break

            # Safety limit (100MB)
            if total_size > 100 * 1024 * 1024:
                raise DaemonProtocolError("Response too large")

        data = b"".join(chunks).decode("utf-8")

        # Return everything up to the first newline
        return data.split("\n", 1)[0]

    # =========================================================================
    # Fallback (Direct Embedding)
    # =========================================================================

    def _fallback(
        self,
        cmd: str,
        args: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Fall back to direct embedding when daemon unavailable.

        Args:
            cmd: Command name.
            args: Command arguments.
            request_id: Request ID for correlation.

        Returns:
            Response dictionary with success/failure status.
        """
        # Commands that work without daemon
        if cmd == "ping":
            return {
                "success": True,
                "data": {"pong": False, "fallback": True},
                "request_id": request_id,
                "fallback": True,
            }

        if cmd == "status":
            return {
                "success": True,
                "data": {"running": False, "fallback": True},
                "request_id": request_id,
                "fallback": True,
            }

        # For embed command, use direct embedding
        if cmd == "embed":
            texts = args.get("texts")
            if not texts or not isinstance(texts, list):
                return {
                    "success": False,
                    "error": "'texts' argument required",
                    "request_id": request_id,
                    "fallback": True,
                }

            try:
                from theo.embedding import create_embedding_provider

                # Create provider (will use MLX on Apple Silicon, Ollama otherwise)
                provider = create_embedding_provider()
                embeddings = provider.embed_texts(texts)
                provider.close()

                return {
                    "success": True,
                    "data": {
                        "embeddings": embeddings,
                        "count": len(embeddings),
                        "dimensions": len(embeddings[0]) if embeddings else 0,
                    },
                    "request_id": request_id,
                    "fallback": True,
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Fallback embedding failed: {e}",
                    "request_id": request_id,
                    "fallback": True,
                }

        # For search command, generate query embedding
        if cmd == "search":
            query = args.get("query")
            if not query or not isinstance(query, str):
                return {
                    "success": False,
                    "error": "'query' argument required",
                    "request_id": request_id,
                    "fallback": True,
                }

            try:
                from theo.embedding import create_embedding_provider

                # Create provider (will use MLX on Apple Silicon, Ollama otherwise)
                provider = create_embedding_provider()
                embedding = provider.embed_query(query)
                provider.close()

                return {
                    "success": True,
                    "data": {
                        "embedding": embedding,
                        "dimensions": len(embedding) if embedding else 0,
                    },
                    "request_id": request_id,
                    "fallback": True,
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Fallback search embedding failed: {e}",
                    "request_id": request_id,
                    "fallback": True,
                }

        # Unsupported commands in fallback
        return {
            "success": False,
            "error": f"Command '{cmd}' not supported in fallback mode",
            "request_id": request_id,
            "fallback": True,
        }

    # =========================================================================
    # High-Level Commands
    # =========================================================================

    def ping(self) -> dict[str, Any]:
        """Ping the daemon to check if it's running.

        Returns:
            Response with pong=True if daemon responded, pong=False if fallback.

        Example:
            >>> result = client.ping()
            >>> if result["success"] and result.get("data", {}).get("pong"):
            ...     print("Daemon is running")
        """
        return self.send("ping")

    def embed(
        self,
        texts: list[str],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings for texts (synchronous).

        Blocks until embeddings are generated.

        Args:
            texts: List of texts to embed.
            timeout: Optional timeout override.

        Returns:
            Response with embeddings on success.

        Example:
            >>> result = client.embed(["What is Python?", "What is Java?"])
            >>> if result["success"]:
            ...     embeddings = result["data"]["embeddings"]
            ...     print(f"Generated {len(embeddings)} embeddings")
        """
        return self.send("embed", texts=texts, timeout=timeout)

    def embed_async(self, texts: list[str]) -> dict[str, Any]:
        """Queue embedding job for async processing.

        Returns immediately with job_id.

        Args:
            texts: List of texts to embed.

        Returns:
            Response with job_id for status tracking.

        Example:
            >>> result = client.embed_async(["hello", "world"])
            >>> if result["success"]:
            ...     job_id = result["data"]["job_id"]
            ...     # Later check status
            ...     status = client.get_job_status(job_id)
        """
        return self.send("embed_async", texts=texts)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of an async embedding job.

        Args:
            job_id: The job identifier returned by embed_async.

        Returns:
            Response with job status and embeddings if completed.

        Example:
            >>> result = client.get_job_status("abc123")
            >>> if result["success"]:
            ...     status = result["data"]["status"]
            ...     if status == "completed":
            ...         embeddings = result["data"]["embeddings"]
        """
        return self.send("get_job_status", job_id=job_id)

    def status(self) -> dict[str, Any]:
        """Get daemon status including worker and queue stats.

        Returns:
            Response with status information.

        Example:
            >>> result = client.status()
            >>> if result["success"]:
            ...     print(f"Jobs processed: {result['data']['worker']['jobs_processed']}")
        """
        return self.send("status")

    # =========================================================================
    # Static Methods
    # =========================================================================

    @staticmethod
    def is_daemon_running() -> bool:
        """Check if the daemon is running.

        This performs two checks:
        1. PID file exists and process is alive
        2. Socket exists and responds to ping

        Returns:
            True if daemon is running and responsive.

        Example:
            >>> if DaemonClient.is_daemon_running():
            ...     client = DaemonClient()
            ...     # Use daemon
            ... else:
            ...     # Use direct embedding
        """
        # Check PID file
        if not PID_FILE.exists():
            return False

        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            # Check if process is alive
            os.kill(pid, 0)
        except (ValueError, OSError):
            return False

        # Check socket exists
        if not SOCKET_PATH.exists():
            return False

        # Try a quick ping
        try:
            client = DaemonClient(
                connect_timeout=0.5,
                request_timeout=1.0,
                auto_fallback=False,
            )
            result = client.ping()
            client.close()
            return result.get("success", False) and result.get("data", {}).get("pong", False)
        except Exception:
            return False


# =============================================================================
# Module-Level Connection Pool (Singleton)
# =============================================================================

_shared_client: DaemonClient | None = None


def get_shared_client() -> DaemonClient:
    """Get the shared module-level DaemonClient singleton.

    This client is reused across all calls within the same process,
    with automatic staleness detection and reconnection based on
    THEO_CONNECTION_IDLE_TIMEOUT_SECONDS.

    Returns:
        Shared DaemonClient instance.

    Example:
        >>> client = get_shared_client()
        >>> result = client.embed(["hello"])
        >>> # Later calls reuse the same connection
        >>> result2 = client.embed(["world"])
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = DaemonClient()
    return _shared_client


def close_shared_client() -> None:
    """Close the shared client connection.

    Call this at process exit to cleanly close the shared connection.
    """
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def get_client(
    auto_fallback: bool = True,
    connect_timeout: float | None = None,
    request_timeout: float | None = None,
) -> DaemonClient:
    """Create a new DaemonClient instance.

    Convenience function for creating a client with common settings.
    For connection reuse, prefer get_shared_client() instead.

    Args:
        auto_fallback: Whether to fall back to direct embedding.
        connect_timeout: Connection timeout (defaults to env setting).
        request_timeout: Request timeout (defaults to env setting).

    Returns:
        Configured DaemonClient instance.
    """
    return DaemonClient(
        auto_fallback=auto_fallback,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )


def quick_embed(texts: list[str]) -> list[list[float]]:
    """Quick embedding with automatic resource management.

    Convenience function for one-off embedding operations.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors, or empty list on error.

    Example:
        >>> embeddings = quick_embed(["hello", "world"])
        >>> print(f"Generated {len(embeddings)} embeddings")
    """
    with DaemonClient() as client:
        result = client.embed(texts=texts)
        if result.get("success"):
            return result.get("data", {}).get("embeddings", [])
        return []


# =============================================================================
# CLI Testing
# =============================================================================


if __name__ == "__main__":
    import sys

    print("DaemonClient Test")
    print("=" * 50)

    # Check daemon status
    print(f"\nDaemon running: {DaemonClient.is_daemon_running()}")
    print(f"Socket path: {SOCKET_PATH}")
    print(f"PID file: {PID_FILE}")

    # Test client
    print("\nTesting client...")
    with DaemonClient() as client:
        # Ping
        result = client.ping()
        print(f"\nPing result: {result}")

        if result.get("fallback"):
            print("  (Using fallback mode - daemon not running)")

        # Status (if daemon running)
        if result.get("data", {}).get("pong"):
            status = client.status()
            print(f"\nStatus: {json.dumps(status, indent=2)}")

        # Test embed (will use fallback if daemon not running)
        if len(sys.argv) > 1 and sys.argv[1] == "--test-embed":
            print("\nTesting embed...")
            result = client.embed(texts=["Hello world", "Goodbye world"])
            if result.get("success"):
                print(
                    f"Embed result: count={result['data']['count']}, "
                    f"dim={result['data']['dimensions']}"
                )
            else:
                print(f"Embed failed: {result.get('error')}")

    print("\nTest complete!")
