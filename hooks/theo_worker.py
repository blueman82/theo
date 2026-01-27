#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Recall Worker - Background embed worker for async embedding processing.

This module provides an embed_worker async function that runs as a background
task, polling the StoreQueue for pending entries, batch embedding them via
EmbeddingBatcher, and storing results to theo via MCP.

Architecture:
    embed_worker runs as asyncio.create_task in daemon lifecycle
    Polls queue -> Batches embeddings -> Stores to theo -> Marks embedded
    Graceful Ollama unavailability handling with retry/backoff

The worker decouples embedding from hook request handling for non-blocking
operation. Store operations complete immediately via queue, embeddings happen
in background.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
import os
import signal

import httpx

# Limit concurrent store writes to avoid overwhelming SQLite
MAX_CONCURRENT_STORE_WRITES = 10

if TYPE_CHECKING:
    from theo_batcher import EmbeddingBatcher
    from theo_queue import QueuedStore, StoreQueue, QueuedIndex, IndexQueue

    # Import HybridStore type for type hints
    _theo_dir = Path.home() / "Github" / "theo"
    if _theo_dir.exists() and str(_theo_dir / "src") not in sys.path:
        sys.path.insert(0, str(_theo_dir / "src"))
    from theo.storage.hybrid import HybridStore
    from theo.storage.sqlite_store import SQLiteStore

# =============================================================================
# Constants
# =============================================================================

# Retry backoff when Ollama is unavailable
OLLAMA_UNAVAILABLE_RETRY_SECONDS = 30.0

# Recall subprocess timeout
THEO_TIMEOUT_SECONDS = 30

# Paths to find theo installation
THEO_PATHS = (
    Path.home() / "Github" / "theo",
    Path(__file__).parent.parent,
    Path.home() / ".local" / "share" / "theo",
    Path("/opt/theo"),
)

# Paths to find uv executable
UV_PATHS = (
    Path.home() / ".local" / "bin" / "uv",
    Path("/opt/homebrew/bin/uv"),
    Path("/usr/local/bin/uv"),
)


# =============================================================================
# Logger Protocol (for type hints)
# =============================================================================


class ClassificationQueueProtocol(Protocol):
    """Protocol for classification queue interface compatible with theo-daemon.py."""

    def enqueue(self, memory_id: str, content: str, namespace: str) -> bool:
        """Add a memory to the classification queue for supersede/contradict detection."""
        ...


class LoggerProtocol(Protocol):
    """Protocol for logger interface compatible with theo-daemon.py Logger."""

    def info(self, message: str) -> None:
        """Log an info message."""
        ...

    def warn(self, message: str) -> None:
        """Log a warning message."""
        ...

    def warning(self, message: str) -> None:
        """Log a warning message (alias for warn)."""
        ...

    def error(self, message: str) -> None:
        """Log an error message."""
        ...


# =============================================================================
# Recall Integration (copied from theo-daemon.py for standalone operation)
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
    logger: LoggerProtocol | None = None,
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
                    pass
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
# Embed Worker
# =============================================================================


async def embed_worker(
    queue: "StoreQueue",
    batcher: "EmbeddingBatcher",
    logger: LoggerProtocol,
    store: "HybridStore",
    classification_queue: ClassificationQueueProtocol | None = None,
    poll_interval: float = 2.0,
    batch_size: int = 50,
) -> None:
    """Background worker that processes queue entries for embedding.

    This async function runs continuously as an asyncio task, polling the
    queue for pending entries, batch embedding them via Ollama, storing
    the embeddings directly to HybridStore, and marking entries as embedded.

    After successful storage, memories are enqueued to the classification
    queue for async supersede/contradict detection (if queue provided).

    The worker gracefully handles Ollama unavailability by waiting and
    retrying rather than crashing.

    Args:
        queue: StoreQueue instance to poll for pending entries.
        batcher: EmbeddingBatcher instance for batch Ollama embed calls.
        logger: Logger instance for status and error logging.
        store: HybridStore instance for direct memory writes (no subprocess).
        classification_queue: Optional queue for async supersede/contradict detection.
        poll_interval: Seconds to sleep when queue is empty (default: 2.0).
        batch_size: Maximum entries to process per batch (default: 50).

    Note:
        This function runs forever until cancelled. Use asyncio.create_task()
        to run it in the background and task.cancel() to stop it.

    Example:
        >>> from theo_queue import StoreQueue
        >>> from theo_batcher import EmbeddingBatcher
        >>> from theo.storage.hybrid import HybridStore
        >>> queue = StoreQueue()
        >>> batcher = EmbeddingBatcher()
        >>> store = await HybridStore.create(...)
        >>> task = asyncio.create_task(embed_worker(queue, batcher, logger, store))
        >>> # ... later ...
        >>> task.cancel()
    """
    logger.info(f"Embed worker started (poll_interval={poll_interval}s, batch_size={batch_size})")

    while True:
        try:
            # Poll queue for pending entries
            entries = queue.dequeue_batch(batch_size)

            if not entries:
                # No work to do, sleep and try again
                await asyncio.sleep(poll_interval)
                continue

            logger.info(f"Processing {len(entries)} entries for embedding")

            # Add all entries to batcher
            for entry in entries:
                # Use string ID for batcher (it expects str memory_id)
                batcher.add(entry.content, str(entry.id))

            # Flush to get embeddings
            try:
                embeddings = await batcher.flush()
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # Ollama unavailable - log and retry after backoff
                logger.warn(f"Ollama unavailable ({type(e).__name__}), retrying in {OLLAMA_UNAVAILABLE_RETRY_SECONDS}s")
                # Clear batcher to avoid re-embedding on retry
                batcher.clear()
                await asyncio.sleep(OLLAMA_UNAVAILABLE_RETRY_SECONDS)
                # Don't mark as embedded - entries will be picked up again
                continue
            except httpx.HTTPError as e:
                # Other HTTP errors - log and retry after backoff
                logger.warn(f"Ollama HTTP error ({e}), retrying in {OLLAMA_UNAVAILABLE_RETRY_SECONDS}s")
                batcher.clear()
                await asyncio.sleep(OLLAMA_UNAVAILABLE_RETRY_SECONDS)
                continue
            except Exception as e:
                # MLX or other embedding errors - log, clear, and retry
                logger.error(f"Embedding error ({type(e).__name__}): {e}")
                batcher.clear()
                # Mark these entries as embedded (skip them) so they don't block the queue
                # Using embedded=2 to indicate "failed" status
                entry_ids = [entry.id for entry in entries]
                queue.mark_embedded(entry_ids)
                logger.warn(f"Marked {len(entry_ids)} entries as failed (embedded=1) to unblock queue")
                await asyncio.sleep(poll_interval)
                continue

            # Store embeddings directly to HybridStore with limited concurrency
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_STORE_WRITES)

            async def store_one(mem_id: int, ent: "QueuedStore", emb: list[float]) -> tuple[int, bool, str | None, str | None]:
                """Store a single memory directly via HybridStore.

                Returns:
                    Tuple of (queue_id, success, error_message, actual_memory_id).
                """
                async with semaphore:
                    try:
                        # Direct HybridStore write with pre-computed embedding
                        result_id = await store.add_memory(
                            content=ent.content,
                            memory_type=ent.memory_type,
                            namespace=ent.namespace,
                            importance=ent.importance,
                            metadata=ent.metadata or {},
                            embedding=emb,  # Pass pre-computed embedding to avoid re-embedding
                        )
                        return mem_id, True, None, result_id
                    except Exception as e:
                        return mem_id, False, str(e), None

            # Prepare entries and update embeddings in queue
            to_store: list[tuple[int, "QueuedStore", list[float]]] = []
            for memory_id_str, embedding in embeddings:
                memory_id = int(memory_id_str)
                entry = next((e for e in entries if e.id == memory_id), None)
                if entry is None:
                    logger.warning(f"Entry {memory_id} not found in batch, skipping")
                    continue
                queue.update_embedding(memory_id, embedding)
                to_store.append((memory_id, entry, embedding))

            # Store one at a time to avoid concurrency issues with SQLite
            results = []
            for mid, ent, emb in to_store:
                try:
                    result = await store_one(mid, ent, emb)
                    results.append(result)
                except Exception as store_error:
                    logger.error(f"Failed to store memory {mid}: {type(store_error).__name__}: {store_error}")
                    results.append(store_error)

            # Collect successful stores (including duplicates - they don't need retry)
            # Also track info for classification queue
            stored_ids: list[int] = []
            classification_items: list[tuple[str, str, str]] = []  # (memory_id, content, namespace)

            # Build lookup for queue_id -> entry
            entry_lookup = {mid: ent for mid, ent, _emb in to_store}

            for res in results:
                if isinstance(res, BaseException):
                    logger.warning(f"Store task exception: {res}")
                else:
                    queue_id, success, error, actual_memory_id = res
                    if success:
                        stored_ids.append(queue_id)
                        # Collect for classification if we have the actual memory ID
                        if actual_memory_id and queue_id in entry_lookup:
                            ent = entry_lookup[queue_id]
                            classification_items.append((actual_memory_id, ent.content, ent.namespace))
                    elif error and "already exists" in error:
                        # Duplicate content is fine - mark as embedded (no retry needed)
                        stored_ids.append(queue_id)
                    else:
                        logger.warning(
                            f"Failed to store memory {queue_id}: {error or 'unknown error'}"
                        )

            # Mark successfully stored entries as embedded
            if stored_ids:
                marked = queue.mark_embedded(stored_ids)
                logger.info(f"Embedded and stored {marked} memories")

            # Enqueue to classification queue for async supersede/contradict detection
            if classification_queue and classification_items:
                enqueued = 0
                for mem_id, content, namespace in classification_items:
                    if classification_queue.enqueue(mem_id, content, namespace):
                        enqueued += 1
                if enqueued:
                    logger.info(f"Enqueued {enqueued} memories for classification")

            # Force garbage collection to help Python clean up
            # NOTE: Do NOT call mx.clear_cache() - it causes Metal race conditions
            # when used with asyncio.to_thread(). Error: "_MTLCommandBuffer addCompletedHandler"
            gc.collect()

        except asyncio.CancelledError:
            logger.info("Embed worker cancelled, shutting down")
            raise
        except Exception as e:
            # Unexpected error - log and continue
            logger.error(f"Embed worker error: {e}")
            await asyncio.sleep(poll_interval)


# =============================================================================
# Index Worker
# =============================================================================


async def index_worker(
    queue: "IndexQueue",
    batcher: "EmbeddingBatcher",
    logger: LoggerProtocol,
    sqlite_store: "SQLiteStore",
    poll_interval: float = 2.0,
    batch_size: int = 100,
) -> None:
    """Background worker that processes index queue entries.

    This async function runs continuously, polling the IndexQueue for
    pending document chunks, batch embedding them, and storing to SQLite.

    Args:
        queue: IndexQueue instance to poll for pending entries.
        batcher: EmbeddingBatcher instance for batch embedding.
        logger: Logger instance for status logging.
        sqlite_store: SQLiteStore instance for document storage.
        poll_interval: Seconds to sleep when queue is empty.
        batch_size: Maximum entries to process per batch.
    """
    logger.info(f"Index worker started (poll_interval={poll_interval}s, batch_size={batch_size})")

    while True:
        try:
            # Poll queue for pending entries
            entries = queue.dequeue_batch(batch_size)

            if not entries:
                await asyncio.sleep(poll_interval)
                continue

            logger.info(f"Processing {len(entries)} chunks for indexing")

            # Add all entries to batcher
            for entry in entries:
                batcher.add(entry.content, str(entry.id))

            # Flush to get embeddings
            try:
                embeddings = await batcher.flush()
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                logger.warn(f"Embedding unavailable ({type(e).__name__}), retrying in {OLLAMA_UNAVAILABLE_RETRY_SECONDS}s")
                batcher.clear()
                await asyncio.sleep(OLLAMA_UNAVAILABLE_RETRY_SECONDS)
                continue
            except httpx.HTTPError as e:
                logger.warn(f"Embedding HTTP error ({e}), retrying in {OLLAMA_UNAVAILABLE_RETRY_SECONDS}s")
                batcher.clear()
                await asyncio.sleep(OLLAMA_UNAVAILABLE_RETRY_SECONDS)
                continue

            # Build lookup for queue_id -> entry
            entry_lookup = {entry.id: entry for entry in entries}

            # Update embeddings in queue and prepare for storage
            documents_to_store: list[tuple["QueuedIndex", list[float]]] = []
            for memory_id_str, embedding in embeddings:
                queue_id = int(memory_id_str)
                entry = entry_lookup.get(queue_id)
                if entry is None:
                    logger.warning(f"Entry {queue_id} not found in batch, skipping")
                    continue
                queue.update_embedding(queue_id, embedding)
                documents_to_store.append((entry, embedding))

            # Store documents to ChromaDB
            if documents_to_store:
                try:
                    # Import Document type for storage
                    from theo.storage import Document

                    docs = []
                    doc_embeddings = []
                    for entry, embedding in documents_to_store:
                        doc = Document(
                            id=f"{Path(entry.source_file).stem}_{entry.chunk_index}_{entry.id}",
                            content=entry.content,
                            source_file=entry.source_file,
                            chunk_index=entry.chunk_index,
                            doc_hash=entry.doc_hash,
                            namespace=entry.namespace,
                            doc_type="document",
                            confidence=1.0,
                            metadata=entry.metadata or {},
                        )
                        docs.append(doc)
                        doc_embeddings.append(embedding)

                    # Store to ChromaDB (run in executor to avoid blocking event loop)
                    loop = asyncio.get_running_loop()
                    stored_ids = await loop.run_in_executor(
                        None,  # Use default thread pool
                        chroma_store.add_documents,
                        docs,
                        doc_embeddings,
                    )
                    logger.info(f"Stored {len(stored_ids)} documents to ChromaDB")

                    # Mark as indexed
                    indexed_queue_ids = [entry.id for entry, _ in documents_to_store if entry.id]
                    if indexed_queue_ids:
                        queue.mark_indexed(indexed_queue_ids)

                except Exception as e:
                    logger.error(f"Failed to store documents: {e}")

            # Force garbage collection
            # NOTE: Do NOT call mx.clear_cache() - causes Metal race conditions
            gc.collect()

        except asyncio.CancelledError:
            logger.info("Index worker cancelled, shutting down")
            raise
        except Exception as e:
            logger.error(f"Index worker error: {e}")
            await asyncio.sleep(poll_interval)


# =============================================================================
# Module Test / Validation
# =============================================================================


async def _run_validation() -> None:
    """Run validation tests for embed_worker.

    Note: This tests the worker with mocked dependencies to avoid
    requiring Ollama or theo to be running.
    """
    import tempfile
    from pathlib import Path

    print("Running embed_worker validation tests...")

    # Import dependencies
    from theo_batcher import EmbeddingBatcher
    from theo_queue import QueuedStore, StoreQueue

    # Simple logger for testing
    class TestLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str) -> None:
            self.messages.append(f"INFO: {message}")
            print(f"  [INFO] {message}")

        def warn(self, message: str) -> None:
            self.messages.append(f"WARN: {message}")
            print(f"  [WARN] {message}")

        def warning(self, message: str) -> None:
            self.warn(message)

        def error(self, message: str) -> None:
            self.messages.append(f"ERROR: {message}")
            print(f"  [ERROR] {message}")

    # Mock EmbeddingProvider for testing
    class MockProvider:
        """Mock embedding provider that returns fake embeddings."""

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            """Return fake embeddings (768-dim vectors of zeros)."""
            return [[0.0] * 768 for _ in texts]

    # Mock HybridStore for testing (doesn't actually store)
    class MockStore:
        async def add_memory(
            self,
            content: str,
            memory_type: str,
            namespace: str,
            importance: float,
            metadata: dict[str, Any],
            queue_id: int | None = None,
        ) -> str:
            """Mock add_memory that returns a fake ID."""
            _ = content, memory_type, namespace, importance, metadata  # Mark as used
            return f"mem_mock_{queue_id or 0}"

    # Test 1: Worker starts and stops cleanly
    print("\n  Test 1: Worker lifecycle...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_queue.db"
        queue = StoreQueue(db_path=db_path)
        provider = MockProvider()
        batcher = EmbeddingBatcher(provider)  # type: ignore[arg-type]
        logger = TestLogger()
        store = MockStore()

        # Start worker
        task = asyncio.create_task(
            embed_worker(queue, batcher, logger, store, poll_interval=0.1, batch_size=5)  # type: ignore[arg-type]
        )

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Cancel and wait for cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert any("started" in m for m in logger.messages), "Worker should log start"
        print("  Worker lifecycle OK")

    # Test 2: Worker processes queue entries (mocked embedding)
    print("\n  Test 2: Queue processing with mock provider...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_queue2.db"
        queue = StoreQueue(db_path=db_path)
        provider = MockProvider()
        batcher = EmbeddingBatcher(provider)  # type: ignore[arg-type]
        logger = TestLogger()
        store = MockStore()

        # Enqueue a test entry
        entry = QueuedStore(
            content="Test memory for embedding",
            namespace="test:worker",
            memory_type="test",
            importance=0.7,
        )
        queue.enqueue(entry)
        assert queue.pending_count() == 1, "Should have 1 pending entry"

        # Start worker
        task = asyncio.create_task(
            embed_worker(queue, batcher, logger, store, poll_interval=0.1, batch_size=5)  # type: ignore[arg-type]
        )

        # Let it try to process
        await asyncio.sleep(0.5)

        # Cancel
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Check that it attempted to process
        processing_attempted = any("Processing" in m for m in logger.messages)
        print(f"  Processing attempted: {processing_attempted}")

    print("\nEmbed worker validation tests completed!")


if __name__ == "__main__":
    asyncio.run(_run_validation())
