#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Recall Queue - SQLite-backed immediate storage for memory operations.

This module provides a persistent queue for memory store operations that need
embedding. Store operations are written immediately to SQLite (fast, <10ms),
then processed asynchronously by a background worker that handles embedding.

Architecture:
    Hook calls enqueue() -> Immediate SQLite INSERT (<10ms)
    Worker polls dequeue_batch() -> Gets pending entries
    Worker embeds and stores -> mark_embedded()

This separation ensures:
    1. Hooks never block on Ollama/embedding calls
    2. Store operations are durably persisted immediately
    3. Embedding can be batched for efficiency
    4. System remains responsive even when Ollama is slow/down

Thread Safety:
    SQLite WAL mode allows concurrent reads during writes.
    Each connection should be used from a single thread.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# Constants
# =============================================================================

DEFAULT_QUEUE_DB_PATH = Path.home() / ".claude" / "hooks" / "data" / "queue.db"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class QueuedStore:
    """A queued memory store operation awaiting embedding.

    This dataclass represents a memory that has been queued for storage
    but not yet embedded. The embedding happens asynchronously via the
    worker process.

    Attributes:
        content: The memory content text to embed and store.
        namespace: Memory namespace (e.g., "project:myapp", "global").
        memory_type: Type of memory (preference, pattern, decision, etc.).
        importance: Importance score from 0.0 to 1.0.
        metadata: Optional additional metadata dictionary.
        created_at: Unix timestamp when this was queued.
        id: Database row ID (None until persisted, populated on dequeue).
        embedding: Optional pre-computed 1024-dim embedding vector (mxbai-embed-large).
    """

    content: str
    namespace: str
    memory_type: str
    importance: float
    metadata: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    id: int | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "id": self.id,
            "content": self.content,
            "namespace": self.namespace,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "embedding": self.embedding,
        }


# =============================================================================
# StoreQueue Class
# =============================================================================


class StoreQueue:
    """SQLite-backed queue for immediate memory store persistence.

    This class provides fast (<10ms target) synchronous storage of memory
    store operations. It uses SQLite with WAL mode for concurrent access
    and durability.

    The queue operates in two phases:
    1. enqueue(): Immediate INSERT, no embedding (fast)
    2. Worker: dequeue_batch() -> embed -> mark_embedded() (async)

    Attributes:
        db_path: Path to the SQLite database file.
        _conn: SQLite connection (lazy initialized).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the store queue.

        Args:
            db_path: Path to SQLite database. Defaults to
                     ~/.claude/hooks/data/queue.db
        """
        self.db_path = db_path or DEFAULT_QUEUE_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database directory exists and schema is created.

        This method handles both fresh database creation and graceful migration
        of existing databases. SQLite doesn't support ADD COLUMN IF NOT EXISTS,
        so we check PRAGMA table_info first before attempting ALTER TABLE.
        """
        # Create data directory if not exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create queue table with embedding column for pre-computed embeddings
        # The embedding column stores 1024-dim float arrays as BLOB (mxbai-embed-large)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                namespace TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                embedded INTEGER DEFAULT 0,
                embedding BLOB
            )
        """)

        # Migration: Add embedding column if it doesn't exist (for existing databases)
        # SQLite doesn't support ADD COLUMN IF NOT EXISTS, so check PRAGMA table_info
        cursor.execute("PRAGMA table_info(store_queue)")
        columns = {row[1] for row in cursor.fetchall()}

        if "embedding" not in columns:
            cursor.execute("""
                ALTER TABLE store_queue ADD COLUMN embedding BLOB
            """)

        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_store_queue_embedded
            ON store_queue(embedded, created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_store_queue_namespace
            ON store_queue(namespace)
        """)

        # Partial index for efficient lookup of entries with pre-computed embeddings
        # This allows fast queries like "SELECT ... WHERE embedding IS NOT NULL"
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_store_queue_embedding_exists
            ON store_queue(id) WHERE embedding IS NOT NULL
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection with WAL mode.

        Returns:
            SQLite connection configured for WAL mode.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                isolation_level=None,  # Autocommit mode for speed
                check_same_thread=False,
            )
            # Enable WAL mode for concurrent reads during writes
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Synchronous NORMAL for balance of speed and durability
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # Use memory for temp tables
            self._conn.execute("PRAGMA temp_store=MEMORY")
            # Row factory for easier access
            self._conn.row_factory = sqlite3.Row

        return self._conn

    def enqueue(self, entry: QueuedStore) -> int:
        """Enqueue a store operation immediately.

        This method performs a synchronous INSERT and returns immediately.
        Target latency is <10ms. No embedding or Ollama calls happen here.

        Args:
            entry: The QueuedStore entry to persist.

        Returns:
            The row ID of the inserted entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Serialize metadata to JSON if present
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None

        # Serialize embedding to BLOB if present using struct.pack for efficiency
        embedding_blob: bytes | None = None
        if entry.embedding is not None:
            embedding_blob = struct.pack(f"{len(entry.embedding)}f", *entry.embedding)

        cursor.execute(
            """
            INSERT INTO store_queue (
                content, namespace, memory_type, importance, metadata, created_at, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.content,
                entry.namespace,
                entry.memory_type,
                entry.importance,
                metadata_json,
                entry.created_at,
                embedding_blob,
            ),
        )

        return cursor.lastrowid or 0

    def dequeue_batch(self, batch_size: int = 10) -> list[QueuedStore]:
        """Get a batch of unprocessed entries for embedding.

        Returns the oldest unembedded entries up to batch_size.
        Does NOT mark them as embedded - call mark_embedded() after
        successful processing.

        Args:
            batch_size: Maximum number of entries to return.

        Returns:
            List of QueuedStore entries with id and embedding populated.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, content, namespace, memory_type, importance, metadata, created_at, embedding
            FROM store_queue
            WHERE embedded = 0
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (batch_size,),
        )

        entries: list[QueuedStore] = []
        for row in cursor.fetchall():
            # Parse metadata JSON if present
            metadata = None
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    metadata = None

            # Deserialize embedding BLOB if present using struct.unpack
            embedding: list[float] | None = None
            if row["embedding"]:
                embedding_blob: bytes = row["embedding"]
                # Calculate number of floats (4 bytes per float)
                num_floats = len(embedding_blob) // 4
                embedding = list(struct.unpack(f"{num_floats}f", embedding_blob))

            entries.append(
                QueuedStore(
                    id=row["id"],
                    content=row["content"],
                    namespace=row["namespace"],
                    memory_type=row["memory_type"],
                    importance=row["importance"],
                    metadata=metadata,
                    created_at=row["created_at"],
                    embedding=embedding,
                )
            )

        return entries

    def mark_embedded(self, ids: list[int]) -> int:
        """Mark entries as embedded after successful processing.

        Args:
            ids: List of row IDs to mark as embedded.

        Returns:
            Number of rows updated.
        """
        if not ids:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Use parameterized query with placeholders
        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"""
            UPDATE store_queue
            SET embedded = 1
            WHERE id IN ({placeholders})
            """,
            ids,
        )

        return cursor.rowcount

    def pending_count(self) -> int:
        """Get count of entries awaiting embedding.

        Returns:
            Number of unembedded entries in the queue.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM store_queue WHERE embedded = 0
            """
        )

        row = cursor.fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics including counts and oldest entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get counts
        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN embedded = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN embedded = 1 THEN 1 ELSE 0 END) as completed,
                MIN(CASE WHEN embedded = 0 THEN created_at ELSE NULL END) as oldest_pending
            FROM store_queue
            """
        )
        row = cursor.fetchone()

        return {
            "total": row["total"] or 0,
            "pending": row["pending"] or 0,
            "completed": row["completed"] or 0,
            "oldest_pending_at": row["oldest_pending"],
            "db_path": str(self.db_path),
        }

    def cleanup_embedded(self, older_than_hours: int = 24) -> int:
        """Remove old embedded entries to prevent database bloat.

        Args:
            older_than_hours: Remove entries embedded more than this many hours ago.

        Returns:
            Number of rows deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = time.time() - (older_than_hours * 3600)

        cursor.execute(
            """
            DELETE FROM store_queue
            WHERE embedded = 1 AND created_at < ?
            """,
            (cutoff,),
        )

        return cursor.rowcount

    def update_embedding(self, queue_id: int, embedding: list[float]) -> bool:
        """Update the embedding for a queued entry.

        This method is used by the EmbedWorker to store computed embeddings
        back into the queue entry. The embedding is serialized as a BLOB
        using struct.pack for efficient storage.

        Args:
            queue_id: The row ID of the queue entry to update.
            embedding: The 1024-dim embedding vector to store.

        Returns:
            True if the update was successful, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Serialize embedding to BLOB using struct.pack for efficiency
        embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)

        cursor.execute(
            """
            UPDATE store_queue
            SET embedding = ?
            WHERE id = ?
            """,
            (embedding_blob, queue_id),
        )

        return cursor.rowcount > 0

    def get_embedding(self, queue_id: int) -> list[float] | None:
        """Get the embedding for a queued entry.

        Retrieves the pre-computed embedding for a queue entry if it exists.
        Returns None if the entry doesn't exist or has no embedding.

        Args:
            queue_id: The row ID of the queue entry.

        Returns:
            The embedding vector as a list of floats, or None if not available.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT embedding
            FROM store_queue
            WHERE id = ? AND embedding IS NOT NULL
            """,
            (queue_id,),
        )

        row = cursor.fetchone()
        if row is None or row["embedding"] is None:
            return None

        # Deserialize embedding BLOB using struct.unpack
        embedding_blob: bytes = row["embedding"]
        num_floats = len(embedding_blob) // 4
        return list(struct.unpack(f"{num_floats}f", embedding_blob))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# QueuedIndex Data Structure
# =============================================================================


@dataclass(frozen=True, slots=True)
class QueuedIndex:
    """A queued document chunk awaiting embedding and storage.

    This dataclass represents a document chunk that has been queued for indexing
    but not yet embedded. The embedding happens asynchronously via the worker.

    Attributes:
        content: The chunk text content to embed and store.
        source_file: Absolute path to the source file.
        chunk_index: Position of this chunk within the source file.
        namespace: Document namespace (e.g., "default", "project:myapp").
        doc_hash: SHA-256 hash of the chunk content for deduplication.
        metadata: Optional additional metadata dictionary.
        created_at: Unix timestamp when this was queued.
        id: Database row ID (None until persisted, populated on dequeue).
        embedding: Optional pre-computed embedding vector.
    """

    content: str
    source_file: str
    chunk_index: int
    namespace: str
    doc_hash: str
    metadata: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    id: int | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "namespace": self.namespace,
            "doc_hash": self.doc_hash,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "embedding": self.embedding,
        }


# =============================================================================
# IndexQueue Class
# =============================================================================


class IndexQueue:
    """SQLite-backed queue for immediate document chunk persistence.

    Similar to StoreQueue but for document indexing. Provides fast (<10ms)
    synchronous storage of document chunks for async embedding.

    The queue operates in two phases:
    1. enqueue(): Immediate INSERT, no embedding (fast)
    2. Worker: dequeue_batch() -> embed -> store to ChromaDB -> mark_indexed()
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the index queue.

        Args:
            db_path: Path to SQLite database. Defaults to queue.db in hooks/data/
        """
        self.db_path = db_path or DEFAULT_QUEUE_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database and schema exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                namespace TEXT NOT NULL,
                doc_hash TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                indexed INTEGER DEFAULT 0,
                embedding BLOB
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_index_queue_indexed
            ON index_queue(indexed, created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_index_queue_source
            ON index_queue(source_file)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                isolation_level=None,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.row_factory = sqlite3.Row

        return self._conn

    def enqueue(self, entry: QueuedIndex) -> int:
        """Enqueue a document chunk immediately.

        Args:
            entry: The QueuedIndex entry to persist.

        Returns:
            The row ID of the inserted entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
        embedding_blob: bytes | None = None
        if entry.embedding is not None:
            embedding_blob = struct.pack(f"{len(entry.embedding)}f", *entry.embedding)

        cursor.execute(
            """
            INSERT INTO index_queue (
                content, source_file, chunk_index, namespace, doc_hash,
                metadata, created_at, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.content,
                entry.source_file,
                entry.chunk_index,
                entry.namespace,
                entry.doc_hash,
                metadata_json,
                entry.created_at,
                embedding_blob,
            ),
        )

        return cursor.lastrowid or 0

    def enqueue_batch(self, entries: list[QueuedIndex]) -> list[int]:
        """Enqueue multiple document chunks in a single transaction.

        Args:
            entries: List of QueuedIndex entries to persist.

        Returns:
            List of row IDs for inserted entries.
        """
        if not entries:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()
        ids: list[int] = []

        cursor.execute("BEGIN TRANSACTION")
        try:
            for entry in entries:
                metadata_json = json.dumps(entry.metadata) if entry.metadata else None
                embedding_blob: bytes | None = None
                if entry.embedding is not None:
                    embedding_blob = struct.pack(f"{len(entry.embedding)}f", *entry.embedding)

                cursor.execute(
                    """
                    INSERT INTO index_queue (
                        content, source_file, chunk_index, namespace, doc_hash,
                        metadata, created_at, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.content,
                        entry.source_file,
                        entry.chunk_index,
                        entry.namespace,
                        entry.doc_hash,
                        metadata_json,
                        entry.created_at,
                        embedding_blob,
                    ),
                )
                ids.append(cursor.lastrowid or 0)

            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

        return ids

    def dequeue_batch(self, batch_size: int = 50) -> list[QueuedIndex]:
        """Dequeue entries awaiting indexing.

        Args:
            batch_size: Maximum entries to return.

        Returns:
            List of QueuedIndex entries (oldest first).
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, content, source_file, chunk_index, namespace, doc_hash,
                   metadata, created_at, embedding
            FROM index_queue
            WHERE indexed = 0
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (batch_size,),
        )

        entries: list[QueuedIndex] = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            embedding: list[float] | None = None
            if row["embedding"]:
                num_floats = len(row["embedding"]) // 4
                embedding = list(struct.unpack(f"{num_floats}f", row["embedding"]))

            entries.append(
                QueuedIndex(
                    id=row["id"],
                    content=row["content"],
                    source_file=row["source_file"],
                    chunk_index=row["chunk_index"],
                    namespace=row["namespace"],
                    doc_hash=row["doc_hash"],
                    metadata=metadata,
                    created_at=row["created_at"],
                    embedding=embedding,
                )
            )

        return entries

    def mark_indexed(self, ids: list[int]) -> int:
        """Mark entries as indexed.

        Args:
            ids: List of row IDs to mark as indexed.

        Returns:
            Number of rows updated.
        """
        if not ids:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"""
            UPDATE index_queue
            SET indexed = 1
            WHERE id IN ({placeholders})
            """,
            ids,
        )

        return cursor.rowcount

    def update_embedding(self, queue_id: int, embedding: list[float]) -> bool:
        """Update the embedding for a queued entry."""
        conn = self._get_connection()
        cursor = conn.cursor()

        embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
        cursor.execute(
            "UPDATE index_queue SET embedding = ? WHERE id = ?",
            (embedding_blob, queue_id),
        )

        return cursor.rowcount > 0

    def pending_count(self) -> int:
        """Get count of entries awaiting indexing."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM index_queue WHERE indexed = 0")
        row = cursor.fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN indexed = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN indexed = 1 THEN 1 ELSE 0 END) as completed,
                COUNT(DISTINCT source_file) as unique_files
            FROM index_queue
            """
        )
        row = cursor.fetchone()

        return {
            "total": row["total"] or 0,
            "pending": row["pending"] or 0,
            "completed": row["completed"] or 0,
            "unique_files": row["unique_files"] or 0,
            "db_path": str(self.db_path),
        }

    def cleanup_indexed(self, older_than_hours: int = 24) -> int:
        """Remove old indexed entries."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = time.time() - (older_than_hours * 3600)
        cursor.execute(
            "DELETE FROM index_queue WHERE indexed = 1 AND created_at < ?",
            (cutoff,),
        )

        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# Module Test / Validation
# =============================================================================


def _run_validation() -> None:
    """Run validation tests for StoreQueue."""
    import tempfile

    print("Running StoreQueue validation tests...")

    # Use temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_queue.db"
        queue = StoreQueue(db_path=db_path)

        # Test 1: Enqueue
        entry = QueuedStore(
            content="Test memory content",
            namespace="test:project",
            memory_type="preference",
            importance=0.8,
            metadata={"source": "test"},
        )
        row_id = queue.enqueue(entry)
        assert row_id > 0, f"Expected positive row_id, got {row_id}"
        print(f"  Enqueue OK (id={row_id})")

        # Test 2: Pending count
        count = queue.pending_count()
        assert count == 1, f"Expected 1 pending, got {count}"
        print(f"  Pending count OK ({count})")

        # Test 3: Dequeue batch
        batch = queue.dequeue_batch(batch_size=5)
        assert len(batch) == 1, f"Expected 1 entry, got {len(batch)}"
        assert batch[0].id == row_id
        assert batch[0].content == "Test memory content"
        assert batch[0].namespace == "test:project"
        assert batch[0].metadata == {"source": "test"}
        print(f"  Dequeue batch OK (got {len(batch)} entries)")

        # Test 4: Mark embedded
        updated = queue.mark_embedded([row_id])
        assert updated == 1, f"Expected 1 updated, got {updated}"
        print(f"  Mark embedded OK ({updated} updated)")

        # Test 5: Pending count after marking
        count = queue.pending_count()
        assert count == 0, f"Expected 0 pending after mark, got {count}"
        print(f"  Pending count after mark OK ({count})")

        # Test 6: Stats
        stats = queue.stats()
        assert stats["total"] == 1
        assert stats["pending"] == 0
        assert stats["completed"] == 1
        print(f"  Stats OK: {stats}")

        # Test 7: Multiple enqueues for performance
        import time as time_module

        start = time_module.perf_counter()
        for i in range(100):
            queue.enqueue(
                QueuedStore(
                    content=f"Memory {i}",
                    namespace="perf:test",
                    memory_type="test",
                    importance=0.5,
                )
            )
        elapsed = time_module.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        print(f"  Performance: 100 enqueues in {elapsed*1000:.2f}ms (avg {avg_ms:.2f}ms each)")
        assert avg_ms < 10, f"Average enqueue time {avg_ms:.2f}ms exceeds 10ms target"

        # Test 8: Cleanup
        queue.mark_embedded(list(range(2, 102)))  # Mark all new entries
        deleted = queue.cleanup_embedded(older_than_hours=0)
        assert deleted >= 1, f"Expected at least 1 deleted, got {deleted}"
        print(f"  Cleanup OK ({deleted} deleted)")

        # Test 9: Enqueue with pre-computed embedding
        test_embedding = [0.1 * i for i in range(1024)]  # 1024-dim vector
        entry_with_embedding = QueuedStore(
            content="Memory with embedding",
            namespace="test:embedding",
            memory_type="pattern",
            importance=0.9,
            embedding=test_embedding,
        )
        embed_row_id = queue.enqueue(entry_with_embedding)
        assert embed_row_id > 0, f"Expected positive row_id, got {embed_row_id}"
        print(f"  Enqueue with embedding OK (id={embed_row_id})")

        # Test 10: Dequeue includes embedding
        batch_with_embedding = queue.dequeue_batch(batch_size=5)
        assert len(batch_with_embedding) == 1
        assert batch_with_embedding[0].embedding is not None
        assert len(batch_with_embedding[0].embedding) == 1024
        # Check first few values match (floating point comparison)
        for i in range(10):
            assert abs(batch_with_embedding[0].embedding[i] - test_embedding[i]) < 1e-6
        print(f"  Dequeue with embedding OK (embedding len={len(batch_with_embedding[0].embedding)})")

        # Test 11: update_embedding
        entry_no_embed = QueuedStore(
            content="Memory without embedding",
            namespace="test:update",
            memory_type="decision",
            importance=0.7,
        )
        update_row_id = queue.enqueue(entry_no_embed)
        # Verify no embedding initially
        initial_embedding = queue.get_embedding(update_row_id)
        assert initial_embedding is None, "Expected no embedding initially"

        # Update with embedding
        new_embedding = [0.5] * 1024
        updated = queue.update_embedding(update_row_id, new_embedding)
        assert updated, "Expected update_embedding to return True"
        print(f"  update_embedding OK (id={update_row_id})")

        # Test 12: get_embedding
        retrieved_embedding = queue.get_embedding(update_row_id)
        assert retrieved_embedding is not None, "Expected embedding to be retrieved"
        assert len(retrieved_embedding) == 1024
        assert abs(retrieved_embedding[0] - 0.5) < 1e-6
        print(f"  get_embedding OK (len={len(retrieved_embedding)})")

        # Test 13: get_embedding returns None for non-existent entry
        none_embedding = queue.get_embedding(99999)
        assert none_embedding is None, "Expected None for non-existent entry"
        print("  get_embedding None for missing entry OK")

        queue.close()

    print("\nAll StoreQueue validation tests passed!")


if __name__ == "__main__":
    _run_validation()
