"""Daemon server for non-blocking embedding operations.

This module provides a Unix socket daemon that handles embedding requests
asynchronously, solving the MCP timeout problem that occurs when blocking
on expensive embedding operations.

Architecture:
    - DaemonServer: Unix socket server at /tmp/theo.sock
    - DaemonClient: Client with subprocess fallback
    - Worker: Background embedding job processor
    - Protocol: JSON message definitions for IPC

Key insight: MCP servers use stdio transport, blocking operations cause timeouts.
Daemon provides non-blocking embedding via Unix socket IPC (<10ms overhead).

Usage:
    # Server (run in background)
    >>> from theo.daemon import DaemonServer
    >>> server = DaemonServer()
    >>> await server.start()

    # Client
    >>> from theo.daemon import DaemonClient
    >>> with DaemonClient() as client:
    ...     result = client.embed(["hello", "world"])
    ...     embeddings = result["data"]["embeddings"]

    # Quick embedding (auto-connects or falls back)
    >>> from theo.daemon import quick_embed
    >>> embeddings = quick_embed(["hello", "world"])
"""

from theo.daemon.client import (
    DaemonClient,
    DaemonClientError,
    DaemonConnectionError,
    DaemonProtocolError,
    DaemonTimeoutError,
    get_client,
    quick_embed,
)
from theo.daemon.protocol import (
    DaemonCommand,
    DaemonRequest,
    DaemonResponse,
    EmbedJob,
    JobStatus,
)
from theo.daemon.server import (
    PID_FILE,
    SOCKET_PATH,
    DaemonServer,
    is_daemon_running,
    run_daemon,
    stop_daemon,
)
from theo.daemon.worker import JobQueue, Worker

__all__ = [
    # Server
    "DaemonServer",
    "run_daemon",
    "is_daemon_running",
    "stop_daemon",
    "SOCKET_PATH",
    "PID_FILE",
    # Client
    "DaemonClient",
    "DaemonClientError",
    "DaemonConnectionError",
    "DaemonTimeoutError",
    "DaemonProtocolError",
    "get_client",
    "quick_embed",
    # Protocol
    "DaemonCommand",
    "DaemonRequest",
    "DaemonResponse",
    "JobStatus",
    "EmbedJob",
    # Worker
    "Worker",
    "JobQueue",
]
