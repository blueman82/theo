#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "pydantic",
#     "pydantic-settings",
#     "python-dotenv",
#     "mlx-embeddings",
# ]
# ///
"""Recall Daemon - Fast IPC server for Claude Code hooks.

This daemon provides a persistent Unix socket server that hook scripts
can connect to for fast memory operations. It eliminates the subprocess
overhead of calling theo CLI for each operation.

Features:
    - Unix socket server at /tmp/theo.sock
    - Ollama model warmup every 60 seconds (keep_alive=-1)
    - Memory cache with 5-minute TTL per namespace
    - Concurrent client handling via asyncio
    - Graceful shutdown on SIGTERM/SIGINT
    - PID file for process management

Architecture:
    The daemon runs as a background process and handles these commands:
    - ping: Health check
    - warmup: Explicit Ollama warmup
    - fetch: Fetch memories (with caching)
    - curate: Curate memories with Ollama
    - store: Store a new memory
    - invalidate: Invalidate cache for namespace

Protocol:
    Clients send newline-delimited JSON messages:
    {"cmd": "fetch", "args": {"namespace": "project:foo", "query": "..."}, "request_id": "abc"}

    Server responds with JSON:
    {"success": true, "data": {...}, "request_id": "abc"}

Usage:
    Start daemon:
        python theo-daemon.py

    Stop daemon:
        python theo-daemon.py --stop

    Check status:
        python theo-daemon.py --status
"""

from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing
import os
import resource
import signal
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Set multiprocessing start method to 'spawn' before any multiprocessing usage
# This is required on macOS with Python 3.13+ to prevent semaphore leaks
# and crashes during MLX model loading (which uses multiprocessing internally)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add theo source to path FIRST (before any theo imports)
_theo_dir = Path.home() / "Documents" / "Github" / "theo"
if _theo_dir.exists() and str(_theo_dir / "src") not in sys.path:
    sys.path.insert(0, str(_theo_dir / "src"))

# Load .env file from theo project root before any config access
from dotenv import load_dotenv

_theo_env = _theo_dir / ".env"
if _theo_env.exists():
    load_dotenv(_theo_env)

# Import store queue components for async embedding
from theo_queue import StoreQueue, QueuedStore, IndexQueue, QueuedIndex
from theo_batcher import EmbeddingBatcher
from theo_worker import embed_worker, index_worker

# Import HybridStore for direct memory writes (bypasses subprocess overhead)

from theo.storage.hybrid import HybridStore
from theo.config import RecallSettings
from theo.embedding.factory import create_embedding_provider
from theo.embedding.provider import EmbeddingProvider

if TYPE_CHECKING:
    from asyncio import StreamReader, StreamWriter

# =============================================================================
# Constants
# =============================================================================

SOCKET_PATH = Path("/tmp/theo.sock")
PID_FILE = Path("/tmp/theo.pid")
LOG_DIR = Path.home() / ".claude" / "hooks" / "logs"
LOG_FILE = LOG_DIR / "theo-daemon.log"

DEFAULT_LLM_MODEL = os.environ.get("THEO_OLLAMA_LLM_MODEL", "")
DEFAULT_EMBED_MODEL = os.environ.get("THEO_OLLAMA_MODEL", "")

if not DEFAULT_LLM_MODEL:
    raise RuntimeError("THEO_OLLAMA_LLM_MODEL environment variable must be set")
if not DEFAULT_EMBED_MODEL:
    raise RuntimeError("THEO_OLLAMA_MODEL environment variable must be set")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# All settings from .env (no hardcoded fallbacks)
CACHE_TTL_SECONDS = int(os.environ["THEO_CACHE_TTL_SECONDS"])
WARMUP_INTERVAL_SECONDS = int(os.environ["THEO_WARMUP_INTERVAL_SECONDS"])
OLLAMA_TIMEOUT_SECONDS = int(os.environ["THEO_OLLAMA_TIMEOUT"])
THEO_TIMEOUT_SECONDS = int(os.environ["THEO_TIMEOUT_SECONDS"])

# LLM Classification Queue Settings
CLASSIFICATION_BATCH_SIZE = int(os.environ["THEO_CLASSIFICATION_BATCH_SIZE"])
CLASSIFICATION_INTERVAL_SECONDS = int(os.environ["THEO_CLASSIFICATION_INTERVAL_SECONDS"])
CLASSIFICATION_MAX_QUEUE_SIZE = int(os.environ["THEO_CLASSIFICATION_MAX_QUEUE_SIZE"])

# Memory Management Settings
GC_INTERVAL_SECONDS = int(os.environ["THEO_GC_INTERVAL_SECONDS"])
MEMORY_WATCHDOG_INTERVAL_SECONDS = int(os.environ["THEO_MEMORY_WATCHDOG_INTERVAL_SECONDS"])
MEMORY_WATCHDOG_MAX_RSS_MB = int(os.environ["THEO_MAX_MEMORY_MB"])
QUEUE_CLEANUP_INTERVAL_SECONDS = int(os.environ["THEO_QUEUE_CLEANUP_INTERVAL_SECONDS"])
QUEUE_CLEANUP_AGE_HOURS = int(os.environ["THEO_QUEUE_CLEANUP_AGE_HOURS"])
MAX_CACHE_ENTRIES = int(os.environ["THEO_MAX_CACHE_ENTRIES"])

THEO_PATHS = (
    Path.home() / "Documents" / "Github" / "theo",
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
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class DaemonProtocol:
    """Protocol message for daemon IPC.

    All messages to/from the daemon follow this format.

    Attributes:
        cmd: Command name (ping, warmup, fetch, curate, store, invalidate).
        args: Command arguments as dictionary.
        request_id: Optional request identifier for correlation.
    """

    cmd: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    @classmethod
    def from_json(cls, data: str | bytes) -> DaemonProtocol | None:
        """Parse a DaemonProtocol from JSON string.

        Args:
            data: JSON string or bytes to parse.

        Returns:
            Parsed DaemonProtocol or None if invalid.
        """
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                return None

            cmd = parsed.get("cmd")
            if not cmd or not isinstance(cmd, str):
                return None

            return cls(
                cmd=cmd,
                args=parsed.get("args") or {},
                request_id=parsed.get("request_id"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def to_response(
        self,
        success: bool,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        """Create a JSON response string.

        Args:
            success: Whether the operation succeeded.
            data: Response data if successful.
            error: Error message if failed.

        Returns:
            JSON-encoded response string with newline.
        """
        response = {
            "success": success,
            "request_id": self.request_id,
        }

        if data is not None:
            response["data"] = data

        if error is not None:
            response["error"] = error

        return json.dumps(response) + "\n"


@dataclass(slots=True)
class ClassificationQueueItem:
    """Item in the LLM classification queue.

    Attributes:
        memory_id: ID of the memory to classify.
        content: Memory content for classification.
        namespace: Memory namespace.
        queued_at: When this item was queued.
    """

    memory_id: str
    content: str
    namespace: str
    queued_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class CacheEntry:
    """A cached memory result with timestamp.

    Attributes:
        memories: List of memory dictionaries.
        fetched_at: When this data was fetched.
        query: The query used to fetch (for debugging).
    """

    memories: list[dict[str, Any]]
    fetched_at: datetime
    query: str = ""

    def is_expired(self, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
        """Check if this cache entry has expired.

        Args:
            ttl_seconds: TTL in seconds (default 300 = 5 minutes).

        Returns:
            True if expired, False otherwise.
        """
        age = datetime.now() - self.fetched_at
        return age > timedelta(seconds=ttl_seconds)


# =============================================================================
# Logger
# =============================================================================


class Logger:
    """Simple file logger for daemon.

    Thread-safe logging to a file with timestamps.

    Attributes:
        path: Path to the log file.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        """Initialize the logger.

        Args:
            log_path: Path to log file. Creates parent dirs if needed.
        """
        self.path = log_path or LOG_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO") -> None:
        """Write a timestamped message to the log file.

        Args:
            message: The message to log.
            level: Log level (INFO, WARN, ERROR).
        """
        try:
            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | {level} | {message}\n")
        except Exception:
            pass  # Logging should never crash the daemon

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, "INFO")

    def warn(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, "WARN")

    def warning(self, message: str) -> None:
        """Alias for warn() for stdlib logging compatibility."""
        self.warn(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "ERROR")


# =============================================================================
# Memory Cache
# =============================================================================


class MemoryCache:
    """TTL-based cache for project memories.

    Thread-safe cache keyed by namespace with automatic expiration.

    Attributes:
        _cache: Internal cache dictionary.
        _ttl: TTL in seconds.
    """

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries.
        """
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds

    def get(self, namespace: str) -> list[dict[str, Any]] | None:
        """Get memories from cache if not expired.

        Args:
            namespace: The memory namespace.

        Returns:
            List of memories or None if not cached/expired.
        """
        entry = self._cache.get(namespace)
        if entry is None:
            return None

        if entry.is_expired(self._ttl):
            del self._cache[namespace]
            return None

        return entry.memories

    def set(
        self,
        namespace: str,
        memories: list[dict[str, Any]],
        query: str = "",
    ) -> None:
        """Store memories in cache.

        Evicts oldest entry if cache is at capacity (LRU eviction).

        Args:
            namespace: The memory namespace.
            memories: List of memory dictionaries.
            query: The query used (for debugging).
        """
        # Evict oldest entry if at capacity (and not updating existing)
        if namespace not in self._cache and len(self._cache) >= MAX_CACHE_ENTRIES:
            oldest_ns = min(self._cache, key=lambda k: self._cache[k].fetched_at)
            del self._cache[oldest_ns]

        self._cache[namespace] = CacheEntry(
            memories=memories,
            fetched_at=datetime.now(),
            query=query,
        )

    def invalidate(self, namespace: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            namespace: Specific namespace to invalidate, or None for all.

        Returns:
            Number of entries invalidated.
        """
        if namespace is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        if namespace in self._cache:
            del self._cache[namespace]
            return 1

        return 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats (namespaces, total_memories, etc.).
        """
        total_memories = sum(len(e.memories) for e in self._cache.values())
        oldest = None
        if self._cache:
            oldest = min(e.fetched_at for e in self._cache.values()).isoformat()

        return {
            "namespaces": list(self._cache.keys()),
            "namespace_count": len(self._cache),
            "total_memories": total_memories,
            "oldest_entry": oldest,
            "ttl_seconds": self._ttl,
        }


# =============================================================================
# LLM Classification Queue
# =============================================================================


class ClassificationQueue:
    """Queue for async LLM classification of memory relationships.

    Memories are stored immediately with relates_to edges (fast, embedding-only).
    This queue processes them in batches to detect supersedes/contradicts
    relationships via LLM, then upgrades the edges.

    Attributes:
        _queue: List of items awaiting classification.
        _processed_count: Total items processed.
        _upgraded_count: Edges upgraded to supersedes/contradicts.
        logger: Logger instance.
    """

    def __init__(self, logger: Logger | None = None) -> None:
        """Initialize the classification queue.

        Args:
            logger: Logger instance for debugging.
        """
        self._queue: list[ClassificationQueueItem] = []
        self._processed_count = 0
        self._upgraded_count = 0
        self.logger = logger or Logger()

    def enqueue(self, memory_id: str, content: str, namespace: str) -> bool:
        """Add a memory to the classification queue.

        Args:
            memory_id: ID of the stored memory.
            content: Memory content for classification.
            namespace: Memory namespace.

        Returns:
            True if enqueued, False if queue is full.
        """
        if len(self._queue) >= CLASSIFICATION_MAX_QUEUE_SIZE:
            # Drop oldest item to make room
            self._queue.pop(0)
            self.logger.warning("Classification queue full, dropped oldest item")

        self._queue.append(ClassificationQueueItem(
            memory_id=memory_id,
            content=content,
            namespace=namespace,
        ))
        return True

    def dequeue_batch(self, batch_size: int = CLASSIFICATION_BATCH_SIZE) -> list[ClassificationQueueItem]:
        """Get a batch of items for processing.

        Args:
            batch_size: Maximum items to return.

        Returns:
            List of queue items (removed from queue).
        """
        batch = self._queue[:batch_size]
        self._queue = self._queue[batch_size:]
        return batch

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": len(self._queue),
            "processed_count": self._processed_count,
            "upgraded_count": self._upgraded_count,
            "max_size": CLASSIFICATION_MAX_QUEUE_SIZE,
        }

    def increment_processed(self, count: int = 1) -> None:
        """Increment processed counter."""
        self._processed_count += count

    def increment_upgraded(self, count: int = 1) -> None:
        """Increment upgraded counter."""
        self._upgraded_count += count


class ClassificationWorker:
    """Background worker for LLM classification of memory relationships.

    Processes the classification queue in batches, calling LLM to detect
    supersedes/contradicts relationships and upgrading edges.

    Attributes:
        queue: ClassificationQueue instance.
        logger: Logger instance.
        interval: Processing interval in seconds.
    """

    def __init__(
        self,
        queue: ClassificationQueue,
        logger: Logger | None = None,
        interval: int = CLASSIFICATION_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the classification worker.

        Args:
            queue: ClassificationQueue to process.
            logger: Logger instance.
            interval: Seconds between processing runs.
        """
        self.queue = queue
        self.logger = logger or Logger()
        self.interval = interval
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def _process_batch(self, batch: list[ClassificationQueueItem]) -> int:
        """Process a batch of memories for LLM classification.

        For each memory, calls both classification tools:
        1. memory_detect_contradictions - finds contradicting memories
        2. memory_check_supersedes - checks if memory supersedes older ones

        Args:
            batch: List of queue items to process.

        Returns:
            Number of edges upgraded.
        """
        upgraded = 0

        for item in batch:
            contradictions_count = 0
            supersedes_count = 0

            # 1. Check for contradictions
            try:
                result = await call_theo_async(
                    "memory_detect_contradictions",
                    {
                        "memory_id": item.memory_id,
                    },
                    self.logger,
                )

                if result.get("success"):
                    contradictions = result.get("contradictions", [])
                    contradictions_count = len(contradictions)
                    upgraded += contradictions_count

            except Exception as e:
                self.logger.warning(
                    f"Contradiction detection failed for {item.memory_id}: {e}"
                )

            # 2. Check for supersedes relationships
            try:
                result = await call_theo_async(
                    "memory_check_supersedes",
                    {
                        "memory_id": item.memory_id,
                    },
                    self.logger,
                )

                if result.get("success") and result.get("superseded_id"):
                    supersedes_count = 1
                    upgraded += 1

            except Exception as e:
                self.logger.warning(
                    f"Supersedes check failed for {item.memory_id}: {e}"
                )

            # Log if any classification occurred
            if contradictions_count or supersedes_count:
                self.logger.info(
                    f"Classified {item.memory_id}: "
                    f"{contradictions_count} contradicts, {supersedes_count} supersedes"
                )

        return upgraded

    async def _worker_loop(self) -> None:
        """Background worker loop."""
        self._running = True
        self.logger.info(f"Classification worker started (interval={self.interval}s)")

        while self._running:
            try:
                await asyncio.sleep(self.interval)

                if not self._running:
                    break

                # Get batch from queue
                batch = self.queue.dequeue_batch()
                if not batch:
                    continue

                self.logger.info(f"Processing {len(batch)} memories for classification")

                # Process batch
                upgraded = await self._process_batch(batch)

                # Update stats
                self.queue.increment_processed(len(batch))
                self.queue.increment_upgraded(upgraded)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Classification worker error: {e}")
                await asyncio.sleep(5)

        self.logger.info("Classification worker stopped")

    def start(self) -> None:
        """Start the background worker."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> dict[str, Any]:
        """Get worker status."""
        return {
            "running": self._running,
            "interval_seconds": self.interval,
            "queue": self.queue.stats(),
        }


# =============================================================================
# Ollama Integration
# =============================================================================


class OllamaKeeper:
    """Background task for keeping Ollama model warm.

    Sends periodic empty prompts with keep_alive=-1 to prevent
    model unloading from VRAM.

    Attributes:
        model: The Ollama model name.
        interval: Warmup interval in seconds.
        logger: Logger instance.
        _task: Background asyncio task.
        _running: Whether the keeper is running.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        interval: int = WARMUP_INTERVAL_SECONDS,
        logger: Logger | None = None,
    ) -> None:
        """Initialize the Ollama keeper.

        Args:
            model: Model name to keep warm.
            interval: Interval between warmup requests.
            logger: Logger instance.
        """
        self.model = model
        self.interval = interval
        self.logger = logger or Logger()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_warmup: datetime | None = None

    def warmup_sync(self) -> bool:
        """Synchronously warm up the Ollama model.

        Sends empty prompt with keep_alive=-1 to load model into VRAM.

        Returns:
            True if successful, False otherwise.
        """
        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": "",
                "keep_alive": -1,
            }).encode()

            req = urllib.request.Request(
                OLLAMA_GENERATE_ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
                success = resp.status == 200
                if success:
                    self._last_warmup = datetime.now()
                return success

        except urllib.error.URLError as e:
            self.logger.warning(f"Ollama warmup failed (URLError): {e.reason}")
            return False
        except Exception as e:
            self.logger.warning(f"Ollama warmup failed: {e}")
            return False

    async def warmup_async(self) -> bool:
        """Asynchronously warm up the Ollama model.

        Returns:
            True if successful, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.warmup_sync)

    async def _warmup_loop(self) -> None:
        """Background warmup loop."""
        self._running = True
        self.logger.info(f"Ollama keeper started (model={self.model}, interval={self.interval}s)")

        # Initial warmup
        success = await self.warmup_async()
        self.logger.info(f"Initial warmup: {'success' if success else 'failed'}")

        while self._running:
            try:
                await asyncio.sleep(self.interval)

                if not self._running:
                    break

                success = await self.warmup_async()
                if not success:
                    self.logger.warning("Periodic warmup failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Warmup loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

        self.logger.info("Ollama keeper stopped")

    def start(self) -> None:
        """Start the background warmup task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._warmup_loop())

    async def stop(self) -> None:
        """Stop the background warmup task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> dict[str, Any]:
        """Get keeper status.

        Returns:
            Dict with running status and last warmup time.
        """
        return {
            "running": self._running,
            "model": self.model,
            "interval_seconds": self.interval,
            "last_warmup": self._last_warmup.isoformat() if self._last_warmup else None,
        }


# =============================================================================
# Recall Integration
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


async def call_theo_async(
    tool_name: str,
    args: dict[str, Any],
    logger: Logger | None = None,
) -> dict[str, Any]:
    """Call theo MCP tool asynchronously via --call mode.

    Args:
        tool_name: Name of the tool (memory_recall, memory_store, etc.).
        args: Dictionary of tool arguments.
        logger: Optional logger for debugging.

    Returns:
        Tool result as dictionary, or error dict on failure.

    Note:
        Uses process groups (start_new_session=True) to ensure all child
        processes are killed on timeout, preventing zombie processes.
    """
    import os
    import signal

    theo_dir = _find_theo_dir()
    uv_exe = _find_uv_executable()

    # Get embedding backend from environment (daemon plist sets this)
    embedding_backend = os.environ.get("THEO_EMBEDDING_BACKEND", "mlx")

    if theo_dir is None:
        cmd = [
            uv_exe, "run", "python", "-m", "theo",
            "--embedding-backend", embedding_backend,
            "--call", tool_name,
            "--args", json.dumps(args),
        ]
        working_dir = str(Path.cwd())
    else:
        cmd = [
            uv_exe, "run",
            "--directory", str(theo_dir),
            "python", "-m", "theo",
            "--embedding-backend", embedding_backend,
            "--call", tool_name,
            "--args", json.dumps(args),
        ]
        working_dir = str(theo_dir)

    proc = None
    try:
        # Run subprocess asynchronously with new session for process group
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            start_new_session=True,  # Creates new process group
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=THEO_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # Kill the entire process group to prevent zombie children
            if proc.pid:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # Process already dead or no permission
            await proc.wait()
            return {"success": False, "error": "theo timed out"}

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            if logger:
                logger.warning(f"theo {tool_name} failed: {error_msg[:100]}")
            return {"success": False, "error": f"theo failed: {error_msg}"}

        result = json.loads(stdout.decode("utf-8"))
        if result is None:
            return {"success": False, "error": "theo returned null"}
        return result

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON response: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": "uv or python not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Ensure cleanup if proc exists and hasn't been waited on
        if proc is not None and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            await proc.wait()


# =============================================================================
# Daemon Server
# =============================================================================


class DaemonServer:
    """Unix socket server for theo daemon.

    Handles concurrent client connections and routes requests to
    appropriate handlers.

    Attributes:
        socket_path: Path to Unix socket.
        cache: Memory cache instance.
        keeper: Ollama keeper instance.
        logger: Logger instance.
    """

    def __init__(
        self,
        socket_path: Path = SOCKET_PATH,
        cache: MemoryCache | None = None,
        keeper: OllamaKeeper | None = None,
        logger: Logger | None = None,
        classification_queue: ClassificationQueue | None = None,
        classification_worker: ClassificationWorker | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the daemon server.

        Args:
            socket_path: Path for Unix socket.
            cache: Memory cache (created if not provided).
            keeper: Ollama keeper (created if not provided).
            logger: Logger (created if not provided).
            classification_queue: Queue for async LLM classification.
            classification_worker: Worker for processing classification queue.
            embedding_provider: EmbeddingProvider for generating embeddings.
                               If not provided, creates one based on RecallSettings.
        """
        self.socket_path = socket_path
        self.cache = cache or MemoryCache()
        self.logger = logger or Logger()
        self.keeper = keeper or OllamaKeeper(logger=self.logger)

        # LLM classification queue for async supersedes/contradicts detection
        self.classification_queue = classification_queue or ClassificationQueue(logger=self.logger)
        self.classification_worker = classification_worker or ClassificationWorker(
            queue=self.classification_queue,
            logger=self.logger,
        )

        # Create embedding provider based on settings if not provided
        if embedding_provider is None:
            settings = RecallSettings()  # type: ignore[call-arg]  # Loads from env vars
            try:
                self.embedding_provider = create_embedding_provider(
                    backend=settings.embedding_backend,
                    host=settings.ollama_host,
                    model=settings.ollama_model,
                    mlx_model=settings.mlx_model,
                )
                self.logger.info(f"Using embedding backend: {settings.embedding_backend}")
            except ImportError as e:
                # MLX unavailable, fall back to Ollama
                self.logger.warning(f"MLX backend unavailable ({e}), falling back to Ollama")
                self.embedding_provider = create_embedding_provider(
                    backend="ollama",
                    host=settings.ollama_host,
                    model=settings.ollama_model,
                )
                self.logger.info("Using embedding backend: ollama (fallback)")
        else:
            self.embedding_provider = embedding_provider

        # Store queue for immediate SQLite writes with async embedding
        self.store_queue = StoreQueue()
        self.batcher = EmbeddingBatcher(provider=self.embedding_provider)
        self._embed_worker_task: asyncio.Task[None] | None = None

        # Index queue for async document indexing
        self.index_queue = IndexQueue()
        self.index_batcher = EmbeddingBatcher(provider=self.embedding_provider)
        self._index_worker_task: asyncio.Task[None] | None = None

        # HybridStore for direct memory writes (initialized in start())
        self.hybrid_store: HybridStore | None = None

        # SQLiteStore for document storage (initialized in start())
        self.sqlite_store = None

        # Memory management tasks (initialized in start())
        self._gc_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()
        self._client_count = 0

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
        self.logger.info(f"Client {client_id} connected: {peer}")

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

                message = DaemonProtocol.from_json(line)
                if message is None:
                    response = json.dumps({
                        "success": False,
                        "error": "Invalid message format",
                    }) + "\n"
                    writer.write(response.encode())
                    await writer.drain()
                    continue

                response = await self._dispatch(message)
                writer.write(response.encode())
                await writer.drain()

        except asyncio.CancelledError:
            pass
        except ConnectionResetError:
            # Normal for single-request clients that close after response
            pass
        except Exception as e:
            self.logger.error(f"Client {client_id} error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.info(f"Client {client_id} disconnected")

    async def _dispatch(self, message: DaemonProtocol) -> str:
        """Dispatch a message to the appropriate handler.

        Args:
            message: Parsed daemon protocol message.

        Returns:
            JSON response string.
        """
        handlers = {
            "ping": self._handle_ping,
            "warmup": self._handle_warmup,
            "fetch": self._handle_fetch,
            "curate": self._handle_curate,
            "store": self._handle_store,
            "invalidate": self._handle_invalidate,
            "status": self._handle_status,
            "search": self._handle_search,
            "embed": self._handle_embed,
            "index": self._handle_index,
        }

        handler = handlers.get(message.cmd)
        if handler is None:
            return message.to_response(
                success=False,
                error=f"Unknown command: {message.cmd}",
            )

        try:
            result = await handler(message.args)
            return message.to_response(success=True, data=result)
        except Exception as e:
            self.logger.error(f"Handler error ({message.cmd}): {e}")
            return message.to_response(success=False, error=str(e))

    async def _handle_ping(self, _: dict[str, Any]) -> dict[str, Any]:
        """Handle ping command.

        Args:
            _: Command arguments (ignored).

        Returns:
            Dict with pong response and timestamp.
        """
        return {
            "pong": True,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
        }

    async def _handle_warmup(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle warmup command.

        Args:
            args: Command arguments (model optional).

        Returns:
            Dict with warmup status.
        """
        model = args.get("model", DEFAULT_LLM_MODEL)

        # Use the keeper's warmup
        success = await self.keeper.warmup_async()
        return {
            "warmed_up": success,
            "model": model,
        }

    async def _handle_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle fetch command - fetch memories with caching.

        Args:
            args: Command arguments including namespace, query, etc.

        Returns:
            Dict with memories and cache status.
        """
        namespace = args.get("namespace", "global")
        query = args.get("query", "")
        n_results = args.get("n_results", 10)
        force_refresh = args.get("force_refresh", False)

        # Check cache first (if not forcing refresh)
        if not force_refresh:
            cached = self.cache.get(namespace)
            if cached is not None:
                return {
                    "memories": cached,
                    "from_cache": True,
                    "namespace": namespace,
                }

        # Fetch from theo
        result = await call_theo_async(
            "memory_recall",
            {
                "query": query or f"{namespace} context preferences patterns decisions",
                "namespace": namespace,
                "n_results": n_results,
                "include_related": args.get("include_related", True),
                "max_depth": args.get("max_depth", 1),
            },
            self.logger,
        )

        if not result.get("success"):
            return {
                "memories": [],
                "from_cache": False,
                "error": result.get("error"),
            }

        # Extract from data wrapper (theo MCP returns {"success": true, "data": {...}})
        data = result.get("data", {})
        memories = data.get("memories", [])

        # Include expanded memories
        expanded = data.get("expanded", [])
        if expanded:
            memories.extend(expanded)

        # Cache the result
        self.cache.set(namespace, memories, query)

        return {
            "memories": memories,
            "from_cache": False,
            "namespace": namespace,
        }

    async def _handle_curate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle curate command - curate memories with Ollama.

        Args:
            args: Command arguments including memories, project_name, etc.

        Returns:
            Dict with curated context.
        """
        memories = args.get("memories", [])
        project_name = args.get("project_name", "unknown")
        project_root = args.get("project_root", "")

        if not memories:
            return {"curated": "", "memory_count": 0}

        # Format memories for Ollama
        memory_lines: list[str] = []
        for mem in memories:
            source = mem.get("source", "unknown")
            mem_type = mem.get("type", "unknown")
            importance = mem.get("importance", 0.5)
            confidence = mem.get("confidence", 0.3)
            content = mem.get("content", "")

            line = f"[{source}|{mem_type}|imp:{importance:.1f}|conf:{confidence:.1f}] {content}"
            memory_lines.append(line)

        memory_text = "\n".join(memory_lines)

        # Build curation prompt
        rfc_preamble = (
            'The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", '
            '"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in these '
            "memories are to be interpreted as described in RFC 2119."
        )

        prompt = f"""You are curating memories for a Claude Code session.
Current project: {project_name}
Current path: {project_root}

Raw memories (format: [source|type|importance|confidence] content):
{memory_text}

CRITICAL INSTRUCTIONS:
1. PRESERVE exact RFC 2119 keywords from source (MUST, MUST NOT, SHOULD, etc.)
2. PRESERVE important adverbs like "PROACTIVELY" - copy them exactly
3. Synthesize only TRUE duplicates (same meaning)
4. DO NOT remove rules just because of confidence differences
5. DO NOT add notes, explanations, or commentary
6. EXCLUDE memories clearly for different projects

OUTPUT FORMAT:
# Memory Context

{rfc_preamble}

---

## Golden Rules
- [highest priority rules, if any]

## Preferences
- [user preferences]

## Patterns
- [coding patterns]

## Recent Decisions
- [decisions, if any]

OUTPUT:"""

        # Call Ollama for curation
        try:
            payload = json.dumps({
                "model": args.get("model", DEFAULT_LLM_MODEL),
                "prompt": prompt,
                "stream": False,
            }).encode()

            req = urllib.request.Request(
                OLLAMA_GENERATE_ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            loop = asyncio.get_event_loop()

            def do_ollama() -> str | None:
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read())
                        return result.get("response", "")
                except Exception:
                    return None

            curated = await loop.run_in_executor(None, do_ollama)

            if curated and "Memory Context" in curated:
                return {"curated": curated, "memory_count": len(memories)}
            return {
                "curated": "",
                "memory_count": len(memories),
                "error": "Ollama curation failed or returned invalid output",
            }

        except Exception as e:
            return {
                "curated": "",
                "memory_count": len(memories),
                "error": str(e),
            }

    async def _handle_store(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle store command - enqueue a memory for async embedding.

        Enqueues immediately to SQLite queue (<10ms), then returns.
        Background embed_worker handles embedding and storage to theo.
        This ensures store calls never block on Ollama/embedding operations.

        Args:
            args: Command arguments including content, type, namespace, etc.

        Returns:
            Dict with queue result (queued=True, queue_id).
        """
        # Extract required fields from args
        content = args.get("content", "")
        namespace = args.get("namespace", "global")
        memory_type = args.get("type", "session")
        importance = args.get("importance", 0.5)
        metadata = args.get("metadata")

        if not content:
            return {"success": False, "error": "content is required"}

        # Create QueuedStore entry
        entry = QueuedStore(
            content=content,
            namespace=namespace,
            memory_type=memory_type,
            importance=float(importance),
            metadata=metadata,
        )

        # Enqueue for async embedding (immediate SQLite INSERT, <10ms)
        queue_id = self.store_queue.enqueue(entry)

        # Invalidate cache for this namespace immediately
        # (memory will appear after embed_worker processes it)
        self.cache.invalidate(namespace)

        self.logger.info(
            f"Enqueued store for namespace={namespace} (queue_id={queue_id})\n"
            f"         Content: {content}"
        )

        return {
            "success": True,
            "queued": True,
            "queue_id": queue_id,
            "namespace": namespace,
        }

    async def _handle_invalidate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle invalidate command - invalidate cache.

        Args:
            args: Command arguments (namespace optional).

        Returns:
            Dict with invalidation result.
        """
        namespace = args.get("namespace")
        count = self.cache.invalidate(namespace)
        return {
            "invalidated": count,
            "namespace": namespace or "all",
        }

    async def _handle_status(self, _: dict[str, Any]) -> dict[str, Any]:
        """Handle status command - get daemon status.

        Args:
            _: Command arguments (ignored).

        Returns:
            Dict with daemon status including cache, keeper, queue, and classification stats.
        """
        # Get store queue stats
        queue_stats = self.store_queue.stats()

        # Get embedding provider type
        provider_type = type(self.embedding_provider).__name__

        # Get index queue stats
        index_queue_stats = self.index_queue.stats()

        return {
            "pid": os.getpid(),
            "uptime_started": getattr(self, "_start_time", datetime.now()).isoformat(),
            "cache": self.cache.stats(),
            "ollama_keeper": self.keeper.status(),
            "embedding_provider": provider_type,
            "store_queue": {
                "pending_count": queue_stats.get("pending", 0),
                "completed_count": queue_stats.get("completed", 0),
                "total_count": queue_stats.get("total", 0),
                "queue_path": queue_stats.get("db_path", ""),
                "embed_worker_running": self._embed_worker_task is not None and not self._embed_worker_task.done(),
            },
            "index_queue": {
                "pending_count": index_queue_stats.get("pending", 0),
                "completed_count": index_queue_stats.get("completed", 0),
                "total_count": index_queue_stats.get("total", 0),
                "unique_files": index_queue_stats.get("unique_files", 0),
                "index_worker_running": self._index_worker_task is not None and not self._index_worker_task.done(),
            },
            "classification_worker": self.classification_worker.status(),
            "socket_path": str(self.socket_path),
        }

    async def _handle_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle search command - generate query embedding for similarity search.

        Args:
            args: Must contain 'query' string.

        Returns:
            Dict with query embedding for similarity search.
        """
        query = args.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("'query' argument required and must be a string")

        # Generate query embedding using the embedding provider
        # MLX Metal requires main thread execution - run sync, not in executor
        embedding = self.embedding_provider.embed_query(query)

        return {
            "query": query,
            "embedding": embedding,
            "dimensions": len(embedding) if embedding else 0,
        }

    async def _handle_embed(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle embed command - generate embeddings for texts.

        Args:
            args: Must contain 'texts' list.

        Returns:
            Dict with embeddings.
        """
        texts = args.get("texts")
        if not texts or not isinstance(texts, list):
            raise ValueError("'texts' argument required and must be a list")

        # Generate embeddings using the embedding provider
        # MLX Metal requires main thread execution - run sync, not in executor
        embeddings = self.embedding_provider.embed_texts(texts)

        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    async def _handle_index(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle index command - queue document chunks for async indexing.

        This command accepts document chunks and queues them for background
        embedding and storage. Returns immediately with queue statistics.

        Args:
            args: Must contain 'chunks' list, each with:
                - content: Chunk text
                - source_file: Source file path
                - chunk_index: Position within file
                - namespace: Document namespace
                - doc_hash: SHA-256 hash of content
                - metadata: Optional metadata dict

        Returns:
            Dict with queued count and queue_ids.
        """
        import hashlib

        chunks = args.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValueError("'chunks' argument required and must be a list")

        namespace = args.get("namespace", "default")
        queued_entries: list[QueuedIndex] = []

        for chunk in chunks:
            content = chunk.get("content")
            source_file = chunk.get("source_file")
            chunk_index = chunk.get("chunk_index", 0)

            if not content or not source_file:
                continue

            # Compute doc_hash if not provided
            doc_hash = chunk.get("doc_hash")
            if not doc_hash:
                doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            entry = QueuedIndex(
                content=content,
                source_file=source_file,
                chunk_index=chunk_index,
                namespace=chunk.get("namespace", namespace),
                doc_hash=doc_hash,
                metadata=chunk.get("metadata"),
            )
            queued_entries.append(entry)

        if not queued_entries:
            return {
                "success": True,
                "queued": 0,
                "queue_ids": [],
                "message": "No valid chunks to queue",
            }

        # Batch enqueue for efficiency
        queue_ids = self.index_queue.enqueue_batch(queued_entries)

        self.logger.info(f"Queued {len(queue_ids)} chunks for indexing from {len(set(e.source_file for e in queued_entries))} files")

        return {
            "success": True,
            "queued": len(queue_ids),
            "queue_ids": queue_ids,
            "pending_total": self.index_queue.pending_count(),
        }

    # =========================================================================
    # Memory Management Loops
    # =========================================================================

    async def _gc_loop(self) -> None:
        """Periodically force garbage collection to prevent memory accumulation."""
        while True:
            try:
                await asyncio.sleep(GC_INTERVAL_SECONDS)
                gc.collect()
                # Also try to clear MLX metal cache if available
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except ImportError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"GC loop error: {e}")

    async def _memory_watchdog_loop(self) -> None:
        """Monitor memory usage and request restart if threshold exceeded."""
        while True:
            try:
                await asyncio.sleep(MEMORY_WATCHDOG_INTERVAL_SECONDS)
                # Get RSS in MB (macOS returns bytes, Linux returns KB)
                rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == "darwin":
                    rss_mb = rss_bytes / (1024 * 1024)
                else:
                    rss_mb = rss_bytes / 1024

                if rss_mb > MEMORY_WATCHDOG_MAX_RSS_MB:
                    self.logger.warning(
                        f"Memory usage {rss_mb:.0f}MB exceeds {MEMORY_WATCHDOG_MAX_RSS_MB}MB threshold, "
                        "initiating graceful restart"
                    )
                    # Signal graceful shutdown - launchd will restart the daemon
                    # Don't call sys.exit() here as it causes event loop errors
                    await self.stop()
                    return  # Exit watchdog loop, let main loop exit cleanly
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory watchdog error: {e}")

    async def _queue_cleanup_loop(self) -> None:
        """Periodically clean up completed queue entries to prevent disk bloat."""
        while True:
            try:
                await asyncio.sleep(QUEUE_CLEANUP_INTERVAL_SECONDS)
                deleted = self.store_queue.cleanup_embedded(older_than_hours=QUEUE_CLEANUP_AGE_HOURS)
                if deleted > 0:
                    self.logger.info(f"Cleaned up {deleted} old queue entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Queue cleanup error: {e}")

    async def start(self) -> None:
        """Start the daemon server.

        Creates Unix socket, starts Ollama keeper, and accepts connections.
        """
        self._start_time = datetime.now()

        # Clean up stale socket
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception as e:
                self.logger.error(f"Failed to remove stale socket: {e}")
                raise

        # Start Unix socket server
        # Increase buffer limit to 10MB to handle large index payloads (default is 64KB)
        self._server = await asyncio.start_unix_server(
            self.handle_client,
            path=str(self.socket_path),
            limit=10 * 1024 * 1024,  # 10MB buffer for large chunk batches
        )

        # Set socket permissions (rw for owner)
        os.chmod(self.socket_path, 0o600)

        # Write PID file
        write_pid_file()

        # Start Ollama keeper
        self.keeper.start()

        # Create HybridStore for direct memory writes (bypasses subprocess overhead)
        try:
            settings = RecallSettings()  # type: ignore[call-arg]  # Loads from env vars
            self.hybrid_store = await HybridStore.create(
                sqlite_path=settings.get_sqlite_path(),
                chroma_path=settings.get_chroma_path(),
                collection_name=settings.collection_name,
                embedding_client=self.embedding_provider,  # Reuse existing provider (avoid duplicate Metal contexts)
            )
            self.logger.info("HybridStore created for direct writes")
        except Exception as e:
            self.logger.error(f"Failed to create HybridStore: {e}")
            raise

        # Start embed worker for async embedding of queued stores
        # NOTE: MLX embed_batch now runs synchronously (not via asyncio.to_thread)
        # to avoid Metal command buffer race conditions
        self._embed_worker_task = asyncio.create_task(
            embed_worker(
                self.store_queue,
                self.batcher,
                self.logger,  # type: ignore[arg-type]  # Logger matches LoggerProtocol
                self.hybrid_store,
                classification_queue=self.classification_queue,  # type: ignore[arg-type]  # ClassificationQueue matches protocol
            )
        )
        self.logger.info("Embed worker started")

        # Create ChromaStore for document indexing
        try:
            from theo.storage.chroma_store import ChromaStore
            self.chroma_store = ChromaStore(
                db_path=settings.get_chroma_path(),
                collection_name=settings.collection_name,
            )
            self.logger.info("ChromaStore created for document indexing")
        except Exception as e:
            self.logger.error(f"Failed to create ChromaStore: {e}")
            # Non-fatal - document indexing won't work but memory operations will

        # Start index worker for async document indexing
        if self.chroma_store:
            self._index_worker_task = asyncio.create_task(
                index_worker(
                    self.index_queue,
                    self.index_batcher,
                    self.logger,  # type: ignore[arg-type]
                    self.chroma_store,
                )
            )

        # Start classification worker
        self.classification_worker.start()

        # Start memory management tasks
        self._gc_task = asyncio.create_task(self._gc_loop())
        self._watchdog_task = asyncio.create_task(self._memory_watchdog_loop())
        self._cleanup_task = asyncio.create_task(self._queue_cleanup_loop())

        self.logger.info(f"Daemon started on {self.socket_path} (PID {os.getpid()})")

        # Serve until shutdown
        async with self._server:
            await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the daemon server gracefully."""
        self.logger.info("Shutting down daemon...")

        self._shutdown_event.set()

        # Stop embed worker task
        if self._embed_worker_task and not self._embed_worker_task.done():
            self._embed_worker_task.cancel()
            try:
                await self._embed_worker_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Embed worker stopped")

        # Stop index worker task
        if self._index_worker_task and not self._index_worker_task.done():
            self._index_worker_task.cancel()
            try:
                await self._index_worker_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Index worker stopped")

        # Stop memory management tasks
        for task in [self._gc_task, self._watchdog_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.logger.info("Memory management tasks stopped")

        # Stop classification worker
        await self.classification_worker.stop()

        # Stop Ollama keeper
        await self.keeper.stop()

        # Close embedding provider (sync method, not async)
        if self.embedding_provider:
            try:
                self.embedding_provider.close()
                self.logger.info("Embedding provider closed")
            except Exception as e:
                self.logger.warning(f"Error closing embedding provider: {e}")

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

        self.logger.info("Daemon stopped")


# =============================================================================
# Signal Handling
# =============================================================================


def setup_signals(server: DaemonServer) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        server: DaemonServer instance to stop on signal.
    """
    loop = asyncio.get_event_loop()

    def handle_signal(sig: signal.Signals) -> None:
        server.logger.info(f"Received signal {sig.name}, shutting down...")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))


# =============================================================================
# PID File Management
# =============================================================================


def write_pid_file() -> None:
    """Write current PID to PID file."""
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


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
    pid = read_pid_file()
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        import time
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
# Main Entry Point
# =============================================================================


async def run_daemon() -> None:
    """Run the daemon server."""
    logger = Logger()
    server = DaemonServer(logger=logger)
    setup_signals(server)
    await server.start()


def main() -> None:
    """Main entry point with CLI handling."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--stop":
            if stop_daemon():
                print("Daemon stopped")
                sys.exit(0)
            else:
                print("Daemon not running")
                sys.exit(1)

        elif arg == "--status":
            if is_daemon_running():
                pid = read_pid_file()
                print(f"Daemon running (PID {pid})")
                print(f"Socket: {SOCKET_PATH}")
                print(f"PID file: {PID_FILE}")
                sys.exit(0)
            else:
                print("Daemon not running")
                sys.exit(1)

        elif arg == "--help":
            print(__doc__)
            print("\nUsage:")
            print("  python theo-daemon.py          Start daemon")
            print("  python theo-daemon.py --stop   Stop daemon")
            print("  python theo-daemon.py --status Check daemon status")
            print("  python theo-daemon.py --test-mode  Run validation tests")
            sys.exit(0)

        elif arg == "--test-mode":
            # Validation mode - test that all components can be instantiated
            print("Running validation tests...")

            # Test DaemonProtocol
            proto = DaemonProtocol(cmd="ping", args={}, request_id="test")
            assert proto.cmd == "ping"
            parsed = DaemonProtocol.from_json('{"cmd": "test", "args": {"key": "value"}}')
            assert parsed is not None
            assert parsed.cmd == "test"
            print("✓ DaemonProtocol OK")

            # Test MemoryCache
            cache = MemoryCache(ttl_seconds=10)
            cache.set("test-ns", [{"content": "test"}], "query")
            assert cache.get("test-ns") is not None
            assert cache.invalidate("test-ns") == 1
            assert cache.get("test-ns") is None
            print("✓ MemoryCache OK")

            # Test Logger
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tf:
                test_logger = Logger(Path(tf.name))
                test_logger.info("test message")
            print("✓ Logger OK")

            # Test OllamaKeeper instantiation
            keeper = OllamaKeeper(model="test", interval=60)
            assert keeper.model == "test"
            print("✓ OllamaKeeper OK")

            # Test DaemonServer instantiation
            server = DaemonServer()
            assert server.socket_path == SOCKET_PATH
            assert server.cache is not None
            assert server.keeper is not None
            print("✓ DaemonServer OK")

            print("\nAll validation tests passed!")
            sys.exit(0)

        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)

    # Check if already running
    if is_daemon_running():
        print(f"Daemon already running (PID {read_pid_file()})")
        sys.exit(1)

    # Run daemon
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
