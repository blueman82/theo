"""DaemonClient - Lightweight client for theo-daemon IPC.

This module provides a synchronous client for communicating with the
theo-daemon via Unix socket. It's designed for use in hook scripts
that need fast memory operations without subprocess overhead.

Features:
    - Lazy connection (only connect when first request is made)
    - Automatic fallback to subprocess when daemon unavailable
    - Timeout handling at connection and request levels
    - Context manager support for automatic cleanup

Protocol:
    The client sends newline-delimited JSON to the daemon:
    {"cmd": "fetch", "args": {"namespace": "project:foo"}, "request_id": "abc"}

    And receives JSON responses:
    {"success": true, "data": {...}, "request_id": "abc"}

Usage:
    # Context manager (recommended)
    with DaemonClient() as client:
        result = client.fetch(namespace="project:foo", query="context")

    # Manual management
    client = DaemonClient()
    try:
        result = client.send("fetch", namespace="project:foo")
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
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

# =============================================================================
# HARDCODED CONSTANTS (implementation details)
# =============================================================================

SOCKET_PATH = Path("/tmp/theo.sock")
PID_FILE = Path("/tmp/theo.pid")
RECV_BUFFER = 65536  # bytes


# =============================================================================
# CONFIGURABLE SETTINGS (from .env)
# =============================================================================

def _load_env() -> None:
    """Load .env file from theo project root.

    Searches standard locations for the theo .env file and loads
    any environment variables not already set.
    """
    env_paths = [
        Path.home() / "Documents" / "Github" / "theo" / ".env",
        Path.home() / "Github" / "theo" / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key not in os.environ:
                            os.environ[key] = value
            break

_load_env()


def _require(key: str) -> str:
    """Get required environment variable or raise error.

    Args:
        key: Environment variable name to retrieve.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    value = os.environ.get(key)
    if value is None:
        raise RuntimeError(f"Required env var {key} not set")
    return value


def get_connect_timeout() -> float:
    """Get socket connection timeout in seconds.

    Returns:
        Connection timeout from THEO_CONNECT_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_CONNECT_TIMEOUT_SECONDS"))


def get_request_timeout() -> float:
    """Get request/response timeout in seconds.

    Returns:
        Request timeout from THEO_REQUEST_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_REQUEST_TIMEOUT_SECONDS"))


def get_idle_timeout() -> float:
    """Get connection idle timeout in seconds.

    Returns:
        Idle timeout from THEO_CONNECTION_IDLE_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_CONNECTION_IDLE_TIMEOUT_SECONDS"))

THEO_PATHS = (
    Path.home() / "Github" / "theo",
    Path(__file__).parent.parent,
    Path.home() / ".local" / "share" / "theo",
    Path("/opt/theo"),
)

UV_PATHS = (
    Path.home() / ".local" / "bin" / "uv",
    Path("/opt/homebrew/bin/uv"),
    Path("/usr/local/bin/uv"),
)


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
# Helper Functions
# =============================================================================


def _find_theo_dir() -> Path | None:
    """Find theo installation directory.

    Returns:
        Path to theo directory or None if not found.
    """
    for path in THEO_PATHS:
        if (path / "src" / "theo" / "__main__.py").exists():
            return path
    return None


def _find_uv_executable() -> str:
    """Find uv executable path.

    Returns:
        Path to uv executable or 'uv' for PATH lookup.
    """
    for uv_path in UV_PATHS:
        if uv_path.exists():
            return str(uv_path)
    return "uv"


# =============================================================================
# DaemonClient
# =============================================================================


class DaemonClient:
    """Synchronous client for communicating with theo-daemon.

    This client provides a simple interface for hook scripts to
    communicate with the theo-daemon. It handles connection
    management, timeouts, and automatic fallback to subprocess
    when the daemon is unavailable.

    Attributes:
        socket_path: Path to the Unix socket.
        connect_timeout: Timeout for initial connection in seconds.
        request_timeout: Timeout for request/response in seconds.
        auto_fallback: Whether to automatically fall back to subprocess.

    Example:
        >>> with DaemonClient() as client:
        ...     result = client.fetch(namespace="project:foo")
        ...     if result["success"]:
        ...         memories = result["data"]["memories"]
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
            connect_timeout: Connection timeout in seconds (from env if None).
            request_timeout: Request/response timeout in seconds (from env if None).
            auto_fallback: If True, fall back to subprocess when daemon unavailable.
            idle_timeout: Connection idle timeout in seconds (from env if None).
        """
        self.socket_path = Path(socket_path) if socket_path else SOCKET_PATH
        self.auto_fallback = auto_fallback

        # All timeouts from param or env
        self.connect_timeout = connect_timeout if connect_timeout is not None else get_connect_timeout()
        self.request_timeout = request_timeout if request_timeout is not None else get_request_timeout()
        self.idle_timeout = idle_timeout if idle_timeout is not None else get_idle_timeout()

        self._socket: socket.socket | None = None
        self._connected = False
        self._last_used: float = 0.0  # Timestamp of last successful operation

    def __enter__(self) -> DaemonClient:
        """Context manager entry."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Context manager exit - close the connection.

        Properly closes the socket to prevent resource leaks.
        Each hook subprocess should clean up its own connection.
        """
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
            if self.auto_fallback:
                return False
            raise DaemonConnectionError(f"Socket not found: {self.socket_path}")

        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect(str(self.socket_path))
            self._connected = True
            self._last_used = time.time()
            return True

        except TimeoutError:
            self._cleanup_socket()
            if self.auto_fallback:
                return False
            raise DaemonTimeoutError("Connection timed out")

        except ConnectionRefusedError:
            self._cleanup_socket()
            if self.auto_fallback:
                return False
            raise DaemonConnectionError("Connection refused")

        except OSError as e:
            self._cleanup_socket()
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
        - Automatic fallback to subprocess if configured

        Args:
            cmd: Command name (ping, fetch, store, curate, invalidate, status).
            request_id: Optional request identifier for correlation.
            timeout: Override request timeout for this call.
            **args: Command arguments passed to the daemon.

        Returns:
            Response dictionary with at minimum {"success": bool}.
            On success: {"success": True, "data": {...}, "request_id": "..."}
            On failure: {"success": False, "error": "...", "request_id": "..."}

        Example:
            >>> result = client.send("fetch", namespace="project:foo", query="context")
            >>> if result["success"]:
            ...     memories = result["data"]["memories"]
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
                # Connection failed, fall back to subprocess
                return self._fallback_subprocess(cmd, args)

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
            self.close()

            if self.auto_fallback:
                return self._fallback_subprocess(cmd, args)

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
        try:
            return json.loads(response_data)
        except json.JSONDecodeError as e:
            raise DaemonProtocolError(f"Invalid JSON response: {e}")

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

            # Safety limit
            if total_size > 10 * 1024 * 1024:  # 10MB
                raise DaemonProtocolError("Response too large")

        data = b"".join(chunks).decode("utf-8")

        # Return everything up to the first newline
        return data.split("\n", 1)[0]

    # =========================================================================
    # Subprocess Fallback
    # =========================================================================

    def _fallback_subprocess(
        self,
        cmd: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Fall back to subprocess call when daemon unavailable.

        Maps daemon commands to theo CLI calls:
        - fetch -> memory_recall
        - store -> memory_store
        - ping -> returns mock success
        - status -> returns mock status
        - invalidate -> returns success (no-op without daemon)
        - curate -> not supported (requires Ollama)

        Args:
            cmd: Command name.
            args: Command arguments.

        Returns:
            Response dictionary with success/failure status.
        """
        # Commands that don't need subprocess
        if cmd == "ping":
            return {
                "success": True,
                "data": {"pong": False, "fallback": True},
                "fallback": True,
            }

        if cmd == "status":
            return {
                "success": True,
                "data": {"running": False, "fallback": True},
                "fallback": True,
            }

        if cmd == "invalidate":
            return {
                "success": True,
                "data": {"invalidated": 0, "fallback": True},
                "fallback": True,
            }

        if cmd == "warmup":
            return {
                "success": True,
                "data": {"warmed_up": False, "fallback": True},
                "fallback": True,
            }

        # Map daemon commands to theo tool names
        tool_map = {
            "fetch": "memory_recall",
            "store": "memory_store",
        }

        tool_name = tool_map.get(cmd)
        if not tool_name:
            return {
                "success": False,
                "error": f"Command '{cmd}' not supported in fallback mode",
                "fallback": True,
            }

        # Build subprocess command
        theo_dir = _find_theo_dir()
        uv_exe = _find_uv_executable()

        if theo_dir is None:
            cmd_list = [
                uv_exe,
                "run",
                "python",
                "-m",
                "theo",
                "--call",
                tool_name,
                "--args",
                json.dumps(args),
            ]
            working_dir = str(Path.cwd())
        else:
            cmd_list = [
                uv_exe,
                "run",
                "--directory",
                str(theo_dir),
                "python",
                "-m",
                "theo",
                "--call",
                tool_name,
                "--args",
                json.dumps(args),
            ]
            working_dir = str(theo_dir)

        try:
            result = subprocess.run(
                cmd_list,
                check=False,
                capture_output=True,
                timeout=self.request_timeout,
                cwd=working_dir,
                text=True,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Subprocess failed: {result.stderr[:200]}",
                    "fallback": True,
                }

            data = json.loads(result.stdout)
            return {
                "success": True,
                "data": data,
                "fallback": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Subprocess timed out",
                "fallback": True,
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid JSON from subprocess",
                "fallback": True,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "uv or python not found",
                "fallback": True,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
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
            >>> if result["success"] and result["data"].get("pong"):
            ...     print("Daemon is running")
        """
        return self.send("ping")

    def fetch(
        self,
        namespace: str = "global",
        query: str = "",
        n_results: int = 10,
        force_refresh: bool = False,
        include_related: bool = True,
        max_depth: int = 1,
    ) -> dict[str, Any]:
        """Fetch memories from the daemon (with caching).

        Args:
            namespace: Memory namespace (e.g., "project:myproject").
            query: Search query for memory retrieval.
            n_results: Maximum number of results to return.
            force_refresh: If True, bypass cache and fetch fresh.
            include_related: Include related memories.
            max_depth: Maximum depth for related memory traversal.

        Returns:
            Response with data.memories list on success.

        Example:
            >>> result = client.fetch(namespace="project:foo", query="preferences")
            >>> if result["success"]:
            ...     for mem in result["data"]["memories"]:
            ...         print(mem["content"])
        """
        return self.send(
            "fetch",
            namespace=namespace,
            query=query,
            n_results=n_results,
            force_refresh=force_refresh,
            include_related=include_related,
            max_depth=max_depth,
        )

    def store(
        self,
        content: str,
        namespace: str = "global",
        memory_type: str = "session",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new memory.

        Args:
            content: Memory content text.
            namespace: Memory namespace.
            memory_type: Type of memory (preference, decision, pattern, session).
            importance: Importance score (0.0 to 1.0).
            metadata: Optional additional metadata.

        Returns:
            Response with store result.

        Example:
            >>> result = client.store(
            ...     content="User prefers dark mode",
            ...     namespace="project:foo",
            ...     memory_type="preference",
            ...     importance=0.7,
            ... )
        """
        return self.send(
            "store",
            content=content,
            namespace=namespace,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )

    def curate(
        self,
        memories: list[dict[str, Any]],
        project_name: str = "unknown",
        project_root: str = "",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Curate memories using Ollama LLM.

        Note: This command requires the daemon to be running (no fallback).

        Args:
            memories: List of memory dictionaries to curate.
            project_name: Name of the current project.
            project_root: Root path of the project.
            model: Ollama model to use (default: daemon's default).

        Returns:
            Response with curated context string.

        Example:
            >>> result = client.curate(
            ...     memories=memories_list,
            ...     project_name="myproject",
            ... )
            >>> if result["success"]:
            ...     curated_text = result["data"]["curated"]
        """
        args: dict[str, Any] = {
            "memories": memories,
            "project_name": project_name,
            "project_root": project_root,
        }
        if model:
            args["model"] = model

        return self.send("curate", **args)

    def invalidate(self, namespace: str | None = None) -> dict[str, Any]:
        """Invalidate cached memories.

        Args:
            namespace: Specific namespace to invalidate, or None for all.

        Returns:
            Response with number of entries invalidated.

        Example:
            >>> client.invalidate(namespace="project:foo")  # One namespace
            >>> client.invalidate()  # All namespaces
        """
        args: dict[str, Any] = {}
        if namespace:
            args["namespace"] = namespace
        return self.send("invalidate", **args)

    def warmup(self, model: str | None = None) -> dict[str, Any]:
        """Trigger Ollama model warmup.

        Args:
            model: Model to warm up (default: daemon's default).

        Returns:
            Response with warmup status.
        """
        args: dict[str, Any] = {}
        if model:
            args["model"] = model
        return self.send("warmup", **args)

    def status(self) -> dict[str, Any]:
        """Get daemon status including cache and Ollama keeper stats.

        Returns:
            Response with status information.

        Example:
            >>> result = client.status()
            >>> if result["success"]:
            ...     print(f"Cache entries: {result['data']['cache']['namespace_count']}")
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
            ...     # Use direct subprocess calls
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
            return result.get("success", False) and result.get("data", {}).get(
                "pong", False
            )
        except Exception:
            return False


# =============================================================================
# Module-Level Singleton for Connection Reuse
# =============================================================================

_shared_client: DaemonClient | None = None


def get_shared_client() -> DaemonClient:
    """Get shared DaemonClient instance for connection reuse.

    Returns the same client across multiple calls within a process,
    reducing connection overhead for hooks that make multiple requests.

    Returns:
        Shared DaemonClient instance.
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = DaemonClient()
    return _shared_client


def _cleanup_shared_client() -> None:
    """Clean up the shared client on process exit.

    Registered with atexit to ensure socket resources are properly
    released when the process exits, preventing resource leaks.
    """
    global _shared_client
    if _shared_client is not None:
        try:
            _shared_client.close()
        except Exception:
            pass  # Ignore errors during cleanup
        _shared_client = None


# Register cleanup handler for process exit
import atexit
atexit.register(_cleanup_shared_client)


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

    Args:
        auto_fallback: Whether to fall back to subprocess.
        connect_timeout: Connection timeout (from env if None).
        request_timeout: Request timeout (from env if None).

    Returns:
        Configured DaemonClient instance.
    """
    return DaemonClient(
        auto_fallback=auto_fallback,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )


def quick_fetch(
    namespace: str,
    query: str = "",
    n_results: int = 10,
) -> list[dict[str, Any]]:
    """Quick fetch memories with automatic resource management.

    Convenience function for one-off memory fetches.

    Args:
        namespace: Memory namespace.
        query: Search query.
        n_results: Maximum results.

    Returns:
        List of memory dictionaries, or empty list on error.
    """
    with DaemonClient() as client:
        result = client.fetch(namespace=namespace, query=query, n_results=n_results)
        if result.get("success"):
            return result.get("data", {}).get("memories", [])
        return []


def quick_store(
    content: str,
    namespace: str = "global",
    memory_type: str = "observation",
) -> bool:
    """Quick store a memory with automatic resource management.

    Convenience function for one-off memory stores.

    Args:
        content: Memory content.
        namespace: Memory namespace.
        memory_type: Type of memory.

    Returns:
        True if store succeeded, False otherwise.
    """
    with DaemonClient() as client:
        result = client.store(
            content=content, namespace=namespace, memory_type=memory_type
        )
        return result.get("success", False)


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

        # Test fetch (will use fallback if daemon not running)
        if len(sys.argv) > 1 and sys.argv[1] == "--test-fetch":
            print("\nTesting fetch...")
            result = client.fetch(namespace="global", query="test", n_results=3)
            print(f"Fetch result: {json.dumps(result, indent=2)}")

    print("\nTest complete!")
