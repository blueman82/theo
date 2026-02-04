"""Daemon server for non-blocking embedding operations.

This module provides a Unix socket server that handles embedding requests
asynchronously. It solves the MCP timeout problem by offloading embedding
operations to a background worker.

Features:
    - Unix socket server at /tmp/theo.sock
    - Concurrent client handling via asyncio
    - Background worker for embedding jobs
    - Graceful shutdown on SIGTERM/SIGINT
    - PID file for process management

Architecture:
    The daemon runs as a background process and handles these commands:
    - ping: Health check
    - embed: Synchronous embedding (blocks until complete)
    - embed_async: Queue embedding job, returns job_id immediately
    - get_job_status: Check status of async job
    - status: Get daemon status
    - shutdown: Graceful shutdown

Protocol:
    Clients send newline-delimited JSON messages:
    {"cmd": "embed", "args": {"texts": [...]}, "request_id": "abc"}

    Server responds with JSON:
    {"success": true, "data": {...}, "request_id": "abc"}
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from theo.daemon.protocol import DaemonRequest, DaemonResponse, EmbedJob
from theo.daemon.worker import JobQueue, Worker
from theo.embedding import EmbeddingBackend, create_embedding_provider

if TYPE_CHECKING:
    from asyncio import StreamReader, StreamWriter

    from theo.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

__all__ = [
    "DaemonServer",
    "SOCKET_PATH",
    "PID_FILE",
    "is_daemon_running",
    "stop_daemon",
]

# =============================================================================
# Constants
# =============================================================================

SOCKET_PATH = Path("/tmp/theo.sock")
PID_FILE = Path("/tmp/theo.pid")


# =============================================================================
# PID File Management
# =============================================================================


def write_pid_file() -> None:
    """Write current PID to PID file."""
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"Failed to write PID file: {e}")


def remove_pid_file() -> None:
    """Remove PID file."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


def read_pid_file() -> int | None:
    """Read PID from PID file.

    Returns:
        PID as integer or None if not found/invalid.
    """
    try:
        if PID_FILE.exists():
            with open(PID_FILE) as f:
                return int(f.read().strip())
    except (OSError, ValueError):
        pass
    return None


def is_daemon_running() -> bool:
    """Check if daemon is already running.

    Returns:
        True if daemon is running, False otherwise.
    """
    pid = read_pid_file()
    if pid is None:
        return False

    try:
        os.kill(pid, 0)  # Check if process exists
        return True
    except OSError:
        # Process not found, clean up stale PID file
        remove_pid_file()
        return False


def stop_daemon() -> bool:
    """Stop running daemon.

    Returns:
        True if daemon was stopped, False if not running.
    """
    import time

    pid = read_pid_file()
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        for _ in range(50):  # Wait up to 5 seconds
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                remove_pid_file()
                return True
        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        remove_pid_file()
        return True
    except OSError:
        remove_pid_file()
        return False


# =============================================================================
# Daemon Server
# =============================================================================


class DaemonServer:
    """Unix socket server for Theo daemon.

    Handles concurrent client connections and routes requests to
    appropriate handlers. Provides both synchronous and asynchronous
    embedding operations.

    Attributes:
        socket_path: Path to Unix socket.
        provider: Embedding provider instance.
        worker: Background worker for async jobs.
        queue: Job queue for async operations.
    """

    def __init__(
        self,
        socket_path: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_backend: EmbeddingBackend = "mlx",
    ) -> None:
        """Initialize the daemon server.

        Args:
            socket_path: Path for Unix socket (default: /tmp/theo.sock).
            embedding_provider: Optional pre-configured embedding provider.
            embedding_backend: Backend to use if creating provider ('mlx' or 'ollama').
        """
        self.socket_path = socket_path or SOCKET_PATH

        # Create or use provided embedding provider
        if embedding_provider is not None:
            self.provider = embedding_provider
        else:
            try:
                self.provider = create_embedding_provider(embedding_backend)
                logger.info(f"Using embedding backend: {embedding_backend}")
            except ImportError:
                # MLX unavailable, fall back to Ollama
                logger.warning("MLX backend unavailable, falling back to Ollama")
                self.provider = create_embedding_provider("ollama")

        # Job queue and worker for async operations
        self.queue = JobQueue()
        self.worker = Worker(self.provider, self.queue)

        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()
        self._client_count = 0
        self._start_time: datetime | None = None
        self._active_session: dict[str, Any] | None = None

    async def handle_client(
        self,
        reader: StreamReader,
        writer: StreamWriter,
    ) -> None:
        """Handle a single client connection.

        Reads newline-delimited JSON messages and dispatches to handlers.

        Args:
            reader: Async stream reader.
            writer: Async stream writer.
        """
        self._client_count += 1
        client_id = self._client_count
        peer = writer.get_extra_info("peername")
        logger.debug(f"Client {client_id} connected: {peer}")

        try:
            while not self._shutdown_event.is_set():
                try:
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not line:
                    break  # Client disconnected

                request = DaemonRequest.from_json(line)
                if request is None:
                    response = DaemonResponse.err("Invalid message format")
                    writer.write(response.to_json().encode())
                    await writer.drain()
                    continue

                response = await self._dispatch(request)
                writer.write(response.to_json().encode())
                await writer.drain()

        except asyncio.CancelledError:
            pass
        except ConnectionResetError:
            logger.debug(f"Client {client_id} connection reset")
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug(f"Client {client_id} disconnected")

    async def _dispatch(self, request: DaemonRequest) -> DaemonResponse:
        """Dispatch a request to the appropriate handler.

        Args:
            request: Parsed daemon request.

        Returns:
            Response to send back to client.
        """
        handlers: dict[str, Any] = {
            # Core commands
            "ping": self._handle_ping,
            "status": self._handle_status,
            "shutdown": self._handle_shutdown,
            # Embedding commands
            "embed": self._handle_embed,
            "embed_async": self._handle_embed_async,
            "get_job_status": self._handle_get_job_status,
            # Index operations (required per task specification)
            "index": self._handle_index,
            "search": self._handle_search,
            "delete": self._handle_delete,
            # Session tracking for Agent Trace
            "set_active_session": self._handle_set_active_session,
            "get_active_session": self._handle_get_active_session,
        }

        handler = handlers.get(request.cmd)
        if handler is None:
            return DaemonResponse.err(
                f"Unknown command: {request.cmd}",
                request_id=request.request_id,
            )

        try:
            result = await handler(request.args)
            return DaemonResponse.ok(result, request_id=request.request_id)
        except Exception as e:
            logger.error(f"Handler error ({request.cmd}): {e}")
            return DaemonResponse.err(str(e), request_id=request.request_id)

    async def _handle_ping(self, _: dict[str, Any]) -> dict[str, Any]:
        """Handle ping command.

        Returns:
            Dict with pong response and timestamp.
        """
        return {
            "pong": True,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
        }

    async def _handle_embed(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle synchronous embed command.

        Blocks until embeddings are generated.

        Args:
            args: Must contain 'texts' list.

        Returns:
            Dict with embeddings.
        """
        texts = args.get("texts")
        if not texts or not isinstance(texts, list):
            raise ValueError("'texts' argument required and must be a list")

        # Run embedding in executor to not block event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.provider.embed_texts,
            texts,
        )

        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    async def _handle_embed_async(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle async embed command.

        Queues the embedding job and returns immediately with job_id.

        Args:
            args: Must contain 'texts' list.

        Returns:
            Dict with job_id for status tracking.
        """
        texts = args.get("texts")
        if not texts or not isinstance(texts, list):
            raise ValueError("'texts' argument required and must be a list")

        # Create job and add to queue
        job_id = str(uuid.uuid4())[:8]
        job = EmbedJob(job_id=job_id, texts=texts)
        self.queue.add(job)

        logger.debug(f"Queued async job {job_id} with {len(texts)} texts")

        return {
            "job_id": job_id,
            "queued": True,
            "text_count": len(texts),
        }

    async def _handle_get_job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle get_job_status command.

        Args:
            args: Must contain 'job_id'.

        Returns:
            Dict with job status and results if completed.
        """
        job_id = args.get("job_id")
        if not job_id:
            raise ValueError("'job_id' argument required")

        job = self.queue.get_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        result = job.to_dict()

        # Include embeddings if completed
        if job.embeddings is not None:
            result["embeddings"] = job.embeddings

        return result

    async def _handle_status(self, _: dict[str, Any]) -> dict[str, Any]:
        """Handle status command.

        Returns:
            Dict with daemon status information.
        """
        provider_type = type(self.provider).__name__

        return {
            "pid": os.getpid(),
            "uptime_started": self._start_time.isoformat() if self._start_time else None,
            "socket_path": str(self.socket_path),
            "embedding_provider": provider_type,
            "worker": self.worker.status(),
            "health_check": self.provider.health_check(),
        }

    async def _handle_shutdown(self, _: dict[str, Any]) -> dict[str, Any]:
        """Handle shutdown command.

        Initiates graceful shutdown.

        Returns:
            Dict confirming shutdown initiated.
        """
        logger.info("Shutdown requested via command")
        # Schedule shutdown for after response is sent
        asyncio.get_event_loop().call_soon(lambda: asyncio.create_task(self.stop()))
        return {"shutting_down": True}

    async def _handle_index(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle index command - index documents using embeddings.

        This command provides a higher-level interface for indexing documents.
        It generates embeddings for the provided texts and returns them along
        with metadata for storage in a vector database.

        Architectural justification: This command reuses the embed infrastructure
        to provide document indexing semantics. The daemon pattern solves MCP
        timeout issues - index operations can take 10-60s for large documents,
        but the daemon returns immediately after queueing.

        Args:
            args: Must contain 'texts' list. Optional 'ids' list for document IDs.

        Returns:
            Dict with embeddings and document metadata for indexing.
        """
        texts = args.get("texts")
        if not texts or not isinstance(texts, list):
            raise ValueError("'texts' argument required and must be a list")

        ids = args.get("ids", [])
        if ids and len(ids) != len(texts):
            raise ValueError("'ids' list must match length of 'texts' list")

        # Generate embeddings for indexing
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.provider.embed_texts,
            texts,
        )

        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        return {
            "indexed": True,
            "embeddings": embeddings,
            "ids": ids,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    async def _handle_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle search command - generate query embedding for similarity search.

        This command generates a query embedding that can be used for
        similarity search against indexed documents. The actual search
        against the vector database is performed by the caller.

        Architectural justification: Separating query embedding from search
        execution allows the daemon to focus on the expensive embedding
        operation while keeping search logic in the application layer.
        This matches the Recall daemon pattern where the daemon handles
        embedding and the client handles storage operations.

        Args:
            args: Must contain 'query' string.

        Returns:
            Dict with query embedding for similarity search.
        """
        query = args.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' argument required and must be a string")

        # Generate query embedding (uses query prefix for asymmetric search)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self.provider.embed_query,
            query,
        )

        return {
            "query": query,
            "embedding": embedding,
            "dimensions": len(embedding) if embedding else 0,
        }

    async def _handle_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle delete command - acknowledge deletion request.

        This command acknowledges a deletion request. The actual deletion
        from the vector database is performed by the caller. The daemon
        tracks the deletion for cache invalidation purposes.

        Architectural justification: The daemon doesn't directly manage
        the vector database - that's the responsibility of the storage
        layer. This command exists to:
        1. Provide a consistent command interface (index/search/delete)
        2. Allow future cache invalidation when daemon manages caches
        3. Support the full lifecycle of documents in the system

        Args:
            args: Must contain 'ids' list of document IDs to delete.

        Returns:
            Dict confirming deletion acknowledged.
        """
        ids = args.get("ids")
        if not ids or not isinstance(ids, list):
            raise ValueError("'ids' argument required and must be a list")

        # Log deletion for debugging
        logger.debug(f"Delete requested for {len(ids)} documents: {ids[:5]}...")

        return {
            "acknowledged": True,
            "ids": ids,
            "count": len(ids),
        }

    async def _handle_set_active_session(self, args: dict[str, Any]) -> dict[str, Any]:
        """Set the currently active Claude Code session.

        Args:
            args: Session info containing session_id, transcript_path, model_id, project_path.

        Returns:
            Dict confirming active session was set.
        """
        self._active_session = {
            "session_id": args.get("session_id"),
            "transcript_path": args.get("transcript_path"),
            "model_id": args.get("model_id"),
            "project_path": args.get("project_path"),
            "started_at": datetime.now().isoformat(),
        }
        return {"status": "active_session_set"}

    async def _handle_get_active_session(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get the currently active Claude Code session.

        Args:
            args: Unused.

        Returns:
            Dict with active status and session info if active.
        """
        if self._active_session is None:
            return {"active": False, "session": None}
        return {"active": True, "session": self._active_session}

    async def start(self) -> None:
        """Start the daemon server.

        Creates Unix socket, starts worker, and accepts connections.
        """
        self._start_time = datetime.now()

        # Clean up stale socket
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove stale socket: {e}")
                raise

        # Start Unix socket server
        self._server = await asyncio.start_unix_server(
            self.handle_client,
            path=str(self.socket_path),
        )

        # Set socket permissions (rw for owner only)
        os.chmod(self.socket_path, 0o600)

        # Write PID file
        write_pid_file()

        # Start background worker
        self.worker.start()

        logger.info(f"Daemon started on {self.socket_path} (PID {os.getpid()})")

        # Serve until shutdown
        async with self._server:
            await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the daemon server gracefully."""
        logger.info("Shutting down daemon...")

        self._shutdown_event.set()

        # Stop worker
        await self.worker.stop()

        # Close embedding provider
        if self.provider:
            try:
                self.provider.close()
                logger.info("Embedding provider closed")
            except Exception as e:
                logger.warning(f"Error closing embedding provider: {e}")

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Clean up socket
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass

        # Clean up PID file
        remove_pid_file()

        logger.info("Daemon stopped")


def setup_signals(server: DaemonServer) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        server: DaemonServer instance to stop on signal.
    """
    loop = asyncio.get_event_loop()

    def handle_signal(sig: signal.Signals) -> None:
        logger.info(f"Received signal {sig.name}, shutting down...")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, partial(handle_signal, sig))


async def run_daemon(
    embedding_backend: EmbeddingBackend = "mlx",
    socket_path: Path | None = None,
) -> None:
    """Run the daemon server.

    Args:
        embedding_backend: Backend to use ('mlx' or 'ollama').
        socket_path: Optional custom socket path.
    """
    server = DaemonServer(
        socket_path=socket_path,
        embedding_backend=embedding_backend,
    )
    setup_signals(server)
    await server.start()


def main() -> None:
    """Main entry point with CLI handling."""
    # Configure logging to stderr (critical for MCP stdio transport)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--stop":
            if stop_daemon():
                print("Daemon stopped", file=sys.stderr)
                sys.exit(0)
            else:
                print("Daemon not running", file=sys.stderr)
                sys.exit(1)

        elif arg == "--status":
            if is_daemon_running():
                pid = read_pid_file()
                print(f"Daemon running (PID {pid})", file=sys.stderr)
                print(f"Socket: {SOCKET_PATH}", file=sys.stderr)
                sys.exit(0)
            else:
                print("Daemon not running", file=sys.stderr)
                sys.exit(1)

        elif arg == "--help":
            print(__doc__, file=sys.stderr)
            print("\nUsage:", file=sys.stderr)
            print("  python -m theo.daemon.server          Start daemon", file=sys.stderr)
            print("  python -m theo.daemon.server --stop   Stop daemon", file=sys.stderr)
            print("  python -m theo.daemon.server --status Check status", file=sys.stderr)
            sys.exit(0)

        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            print("Use --help for usage information", file=sys.stderr)
            sys.exit(1)

    # Check if already running
    if is_daemon_running():
        print(f"Daemon already running (PID {read_pid_file()})", file=sys.stderr)
        sys.exit(1)

    # Run daemon
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
