"""SQLite storage layer for Theo memory system.

This module provides persistent storage for:
- Memory content and metadata
- Vector embeddings (sqlite-vec) for semantic search
- Full-text search (FTS5) for keyword search
- Relationship graphs (edges) for memory connections
- Embedding cache for performance optimization

Edge types:
- relates_to: General relationship between memories
- supersedes: One memory replaces another (newer info)
- caused_by: Causation relationship
- contradicts: Conflicting information

Example:
    >>> store = SQLiteStore(Path("~/.theo/theo.db"))
    >>> mem_id = store.add_memory("Python is great", embedding=[0.1]*1024)
    >>> results = store.search_vector(embedding=[0.1]*1024, n_results=5)
"""

import hashlib
import json
import logging
import math
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import sqlite_vec

from theo.constants import RECENCY_DECAY_BASE, RECENCY_HALF_LIFE_DAYS


@dataclass
class SearchResult:
    """Result from a search operation."""

    id: str
    content: str
    score: float
    memory_type: str
    namespace: str
    confidence: float
    importance: float
    source_file: str | None
    chunk_index: int
    metadata: dict | None = None
    last_accessed: float | None = None
    created_at: float | None = None
    access_count: int = 0


@dataclass
class TraceRecord:
    """Record of AI attribution for a git commit."""

    commit_sha: str
    conversation_url: str
    model_id: str | None
    session_id: str | None
    files: list[str]
    created_at: float


# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)


def compute_recency_score(
    importance: float,
    last_accessed: float | None,
    created_at: float,
    access_count: int,
) -> float:
    """Compute recency-weighted score for a memory.

    Formula (from Recall): score = importance * recency_factor * access_factor
    - recency_factor = 0.5^(age_days / 7)
    - access_factor = log(access_count + 1) + 1
    """
    reference_time = last_accessed if last_accessed is not None else created_at
    age_days = max(0.0, time.time() - reference_time) / 86400

    recency_factor = RECENCY_DECAY_BASE ** (age_days / RECENCY_HALF_LIFE_DAYS)
    access_factor = math.log(access_count + 1) + 1

    return importance * recency_factor * access_factor


class SQLiteStoreError(Exception):
    """Custom exception for SQLite storage errors."""

    pass


class SQLiteStore:
    """SQLite storage for memory relationships and metadata.

    Provides persistent storage for the memory graph that enables
    relationship-based retrieval during recall operations.

    Args:
        db_path: Path to SQLite database file.
                 Defaults to ~/.theo/theo.db

    Attributes:
        db_path: Path to database file
        _conn: SQLite connection
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize SQLite storage.

        Args:
            db_path: Path to database file. Defaults to ~/.theo/theo.db

        Raises:
            SQLiteStoreError: If database initialization fails
        """
        self.db_path = db_path or Path.home() / ".theo" / "theo.db"

        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect with optimized settings for concurrent access
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA foreign_keys = ON")
            # WAL mode for better concurrent read performance (reduces "database locked" errors)
            self._conn.execute("PRAGMA journal_mode = WAL")
            # Higher busy timeout (10s) to handle contention gracefully
            self._conn.execute("PRAGMA busy_timeout = 10000")
            self._conn.row_factory = sqlite3.Row

            # Enable extension loading and load sqlite-vec for vector storage
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)  # Disable for security

            # Initialize schema
            self._init_schema()

        except Exception as e:
            raise SQLiteStoreError(f"Failed to initialize SQLite storage: {e}") from e

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()

        # Create edges table for relationship graph
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'relates_to',
                weight REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                metadata TEXT,
                UNIQUE(source_id, target_id, edge_type)
            )
        """
        )

        # Create indexes for efficient graph traversal
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_source
            ON edges(source_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges(target_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_type
            ON edges(edge_type)
        """
        )

        # Create schema version table for future migrations
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
        """
        )

        # Create validation_events table for TRY/LEARN cycle tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                context TEXT,
                session_id TEXT,
                created_at REAL NOT NULL
            )
        """
        )

        # Create index for validation event queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_memory
            ON validation_events(memory_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_type
            ON validation_events(event_type)
        """
        )

        # =====================================================================
        # Memory Storage Tables (sqlite-vec migration - schema v2)
        # =====================================================================

        # Memory content and metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT,

                -- Type and scope
                memory_type TEXT NOT NULL DEFAULT 'document',
                namespace TEXT NOT NULL DEFAULT 'default',

                -- Confidence (CRITICAL: >= 0.9 means golden rule)
                confidence REAL NOT NULL DEFAULT 0.3,
                importance REAL NOT NULL DEFAULT 0.5,

                -- Document provenance
                source_file TEXT,
                chunk_index INTEGER DEFAULT 0,
                start_line INTEGER,
                end_line INTEGER,

                -- Timestamps
                created_at REAL NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,

                -- Additional metadata as JSON
                tags TEXT
            )
        """
        )

        # Vector embeddings (sqlite-vec) - MLX uses 1024 dimensions
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """
        )

        # Full-text search (FTS5)
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                id UNINDEXED,
                memory_type UNINDEXED,
                namespace UNINDEXED,
                source_file UNINDEXED
            )
        """
        )

        # Embedding cache (avoid re-computation)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dims INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (content_hash, provider, model)
            )
        """
        )

        # Indexes for memories
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_namespace
            ON memories(namespace)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_confidence
            ON memories(confidence DESC)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_source_file
            ON memories(source_file)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_hash
            ON memories(content_hash)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash
            ON embedding_cache(content_hash)
        """
        )

        # =====================================================================
        # Transcription Storage Tables (schema v3)
        # =====================================================================

        # Transcription sessions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                audio_path TEXT,
                full_text TEXT NOT NULL,
                duration_seconds REAL,
                model_used TEXT,
                language TEXT,
                namespace TEXT DEFAULT 'default',
                memory_id TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
        """
        )

        # Transcription segments with timing
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcription_segments (
                id TEXT PRIMARY KEY,
                transcription_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                confidence REAL DEFAULT 0.8,
                FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
            )
        """
        )

        # Indexes for transcriptions
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transcription_segments_parent
            ON transcription_segments(transcription_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transcriptions_namespace
            ON transcriptions(namespace)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transcriptions_created
            ON transcriptions(created_at DESC)
        """
        )

        # =====================================================================
        # Agent Trace Tables (schema v4)
        # =====================================================================

        # Traces table for commit attribution
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                commit_sha TEXT PRIMARY KEY,
                conversation_url TEXT NOT NULL,
                model_id TEXT,
                session_id TEXT,
                files TEXT,
                created_at REAL NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_session
            ON traces(session_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_conversation
            ON traces(conversation_url)
        """
        )

        # Record schema versions
        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (1, ?)
        """,
            (time.time(),),
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (2, ?)
        """,
            (time.time(),),
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (3, ?)
        """,
            (time.time(),),
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (4, ?)
        """,
            (time.time(),),
        )

        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "relates_to",
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add a relationship edge between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: Type of relationship (relates_to, supersedes, caused_by, contradicts)
            weight: Relationship strength 0.0-1.0 (default: 1.0)
            metadata: Optional additional metadata

        Returns:
            Edge ID

        Raises:
            SQLiteStoreError: If edge creation fails
            ValueError: If weight is out of range
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO edges
                (source_id, target_id, edge_type, weight, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    target_id,
                    edge_type,
                    weight,
                    time.time(),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid or 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to add edge: {e}") from e

    def add_edges_batch(self, edges: list[dict[str, Any]]) -> int:
        """Add multiple edges in a single transaction.

        Args:
            edges: List of edge dicts with keys:
                   source_id, target_id, edge_type, weight, metadata (optional)

        Returns:
            Number of edges added

        Raises:
            SQLiteStoreError: If batch insert fails
        """
        if not edges:
            return 0

        try:
            cursor = self._conn.cursor()
            now = time.time()

            for edge in edges:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO edges
                    (source_id, target_id, edge_type, weight, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge["source_id"],
                        edge["target_id"],
                        edge.get("edge_type", "relates_to"),
                        edge.get("weight", 1.0),
                        edge.get("created_at", now),
                        json.dumps(edge.get("metadata")) if edge.get("metadata") else None,
                    ),
                )

            self._conn.commit()
            return len(edges)

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to add edges batch: {e}") from e

    def _query_edges(
        self,
        field: str,
        value: str,
        edge_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query edges by a specific field (source_id or target_id).

        Args:
            field: Field to filter by ('source_id' or 'target_id')
            value: Value to match
            edge_type: Optional filter by edge type

        Returns:
            List of edge dicts
        """
        cursor = self._conn.cursor()
        base_query = """
            SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
            FROM edges WHERE {} = ?"""
        if edge_type:
            cursor.execute(base_query.format(field) + " AND edge_type = ?", (value, edge_type))
        else:
            cursor.execute(base_query.format(field), (value,))
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_edges_from(
        self,
        source_id: str,
        edge_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all edges originating from a memory.

        Args:
            source_id: Source memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of edge dicts

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            return self._query_edges("source_id", source_id, edge_type)
        except Exception as e:
            raise SQLiteStoreError(f"Failed to get edges: {e}") from e

    def get_edges_to(
        self,
        target_id: str,
        edge_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all edges pointing to a memory.

        Args:
            target_id: Target memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of edge dicts

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            return self._query_edges("target_id", target_id, edge_type)
        except Exception as e:
            raise SQLiteStoreError(f"Failed to get edges: {e}") from e

    def get_related(
        self,
        memory_id: str,
        max_depth: int = 1,
        edge_types: Optional[list[str]] = None,
        min_weight: float = 0.0,
    ) -> list[str]:
        """Get related memory IDs via graph traversal.

        Performs breadth-first traversal of the relationship graph
        to find memories connected to the given memory.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum traversal depth (default: 1 = direct connections only)
            edge_types: Optional filter by edge types
            min_weight: Minimum edge weight to follow (default: 0.0)

        Returns:
            List of related memory IDs (excluding the starting memory)

        Raises:
            SQLiteStoreError: If traversal fails
        """
        try:
            visited: set[str] = {memory_id}
            current_level: set[str] = {memory_id}
            result: list[str] = []

            for _ in range(max_depth):
                next_level: set[str] = set()

                for node_id in current_level:
                    # Get outgoing edges
                    edges = self.get_edges_from(node_id)

                    for edge in edges:
                        # Apply filters
                        if edge_types and edge["edge_type"] not in edge_types:
                            continue
                        if edge["weight"] < min_weight:
                            continue

                        target = edge["target_id"]
                        if target not in visited:
                            visited.add(target)
                            next_level.add(target)
                            result.append(target)

                    # Get incoming edges (for bidirectional traversal)
                    edges = self.get_edges_to(node_id)

                    for edge in edges:
                        if edge_types and edge["edge_type"] not in edge_types:
                            continue
                        if edge["weight"] < min_weight:
                            continue

                        source = edge["source_id"]
                        if source not in visited:
                            visited.add(source)
                            next_level.add(source)
                            result.append(source)

                current_level = next_level

                if not current_level:
                    break

            return result

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get related memories: {e}") from e

    def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Delete a specific edge.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: Edge type

        Returns:
            True if edge was deleted, False if not found

        Raises:
            SQLiteStoreError: If delete fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                DELETE FROM edges
                WHERE source_id = ? AND target_id = ? AND edge_type = ?
                """,
                (source_id, target_id, edge_type),
            )
            self._conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to delete edge: {e}") from e

    def delete_edges_for_memory(self, memory_id: str) -> int:
        """Delete all edges involving a memory (both directions).

        Args:
            memory_id: Memory ID

        Returns:
            Number of edges deleted

        Raises:
            SQLiteStoreError: If delete fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                DELETE FROM edges
                WHERE source_id = ? OR target_id = ?
                """,
                (memory_id, memory_id),
            )
            self._conn.commit()
            return cursor.rowcount

        except Exception as e:
            raise SQLiteStoreError(f"Failed to delete edges for memory: {e}") from e

    def count_edges(self) -> int:
        """Get total number of edges.

        Returns:
            Edge count

        Raises:
            SQLiteStoreError: If count fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM edges")
            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to count edges: {e}") from e

    def _orphan_where_clause(
        self, namespace: str | None
    ) -> tuple[str, tuple[str, ...] | tuple[()]]:
        """Build WHERE clause for orphan memory queries (DRY helper)."""
        base = """m.id NOT IN (SELECT source_id FROM edges)
            AND m.id NOT IN (SELECT target_id FROM edges)"""
        if namespace:
            return f"m.namespace = ? AND {base}", (namespace,)
        return base, ()

    def find_orphan_memories(
        self,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Find memories with no edges (neither source nor target)."""
        try:
            cursor = self._conn.cursor()
            where, params = self._orphan_where_clause(namespace)
            cursor.execute(
                f"""SELECT m.* FROM memories m
                    WHERE {where}
                    ORDER BY m.created_at DESC
                    LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            )
            return [self._row_to_memory(row) for row in cursor.fetchall()]
        except Exception as e:
            raise SQLiteStoreError(f"Failed to find orphan memories: {e}") from e

    def count_orphan_memories(self, namespace: str | None = None) -> int:
        """Count memories with no edges."""
        try:
            cursor = self._conn.cursor()
            where, params = self._orphan_where_clause(namespace)
            cursor.execute(f"SELECT COUNT(*) FROM memories m WHERE {where}", params)
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            raise SQLiteStoreError(f"Failed to count orphan memories: {e}") from e

    def _row_to_edge(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert SQLite row to edge dict."""
        edge = {
            "id": row["id"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "edge_type": row["edge_type"],
            "weight": row["weight"],
            "created_at": row["created_at"],
        }

        if row["metadata"]:
            try:
                edge["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                edge["metadata"] = None
        else:
            edge["metadata"] = None

        return edge

    # =========================================================================
    # Validation Event Operations (for TRY/LEARN cycle)
    # =========================================================================

    def add_validation_event(
        self,
        memory_id: str,
        event_type: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a validation event for a memory.

        Records events in the TRY/LEARN validation cycle:
        - 'applied': Memory was used/applied
        - 'succeeded': Application was successful
        - 'failed': Application failed

        Args:
            memory_id: ID of the memory
            event_type: Type of event ('applied', 'succeeded', 'failed')
            context: Optional description of the context
            session_id: Optional session identifier

        Returns:
            Event ID

        Raises:
            SQLiteStoreError: If insert fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO validation_events
                (memory_id, event_type, context, session_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, event_type, context, session_id, time.time()),
            )
            self._conn.commit()
            return cursor.lastrowid or 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to add validation event: {e}") from e

    def get_validation_events(
        self,
        memory_id: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get validation events for a memory.

        Args:
            memory_id: Memory ID to get events for
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of event dicts with id, memory_id, event_type, context, session_id, created_at

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            cursor = self._conn.cursor()

            if event_type:
                cursor.execute(
                    """
                    SELECT id, memory_id, event_type, context, session_id, created_at
                    FROM validation_events
                    WHERE memory_id = ? AND event_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (memory_id, event_type, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, memory_id, event_type, context, session_id, created_at
                    FROM validation_events
                    WHERE memory_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (memory_id, limit),
                )

            return [
                {
                    "id": row["id"],
                    "memory_id": row["memory_id"],
                    "event_type": row["event_type"],
                    "context": row["context"],
                    "session_id": row["session_id"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get validation events: {e}") from e

    def delete_edge_by_id(self, edge_id: int) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: The edge ID to delete

        Returns:
            True if edge was deleted, False if not found

        Raises:
            SQLiteStoreError: If delete fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self._conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to delete edge: {e}") from e

    def get_edges(
        self,
        memory_id: str,
        direction: str = "both",
        edge_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get edges connected to a memory.

        Args:
            memory_id: Memory ID
            direction: 'outgoing', 'incoming', or 'both'
            edge_type: Optional filter by edge type

        Returns:
            List of edge dicts

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            edges = []

            if direction in ("outgoing", "both"):
                edges.extend(self.get_edges_from(memory_id, edge_type))

            if direction in ("incoming", "both"):
                edges.extend(self.get_edges_to(memory_id, edge_type))

            return edges

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get edges: {e}") from e

    # =========================================================================
    # Memory CRUD Operations
    # =========================================================================

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for sqlite-vec storage."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize embedding from bytes."""
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"{count}f", data))

    def add_memory(
        self,
        content: str,
        embedding: list[float],
        memory_type: str = "document",
        namespace: str = "default",
        confidence: float = 0.3,
        importance: float = 0.5,
        source_file: str | None = None,
        chunk_index: int = 0,
        content_hash: str | None = None,
        tags: dict | None = None,
    ) -> str:
        """Add memory to all three tables in a single transaction.

        Args:
            content: Memory content text
            embedding: Vector embedding (1024 dimensions for MLX)
            memory_type: Type of memory (document, fact, procedure, etc.)
            namespace: Namespace for organization
            confidence: Initial confidence score (0.0-1.0)
            importance: Importance score (0.0-1.0)
            source_file: Optional source file path
            chunk_index: Index of chunk within source file
            content_hash: Optional pre-computed content hash
            tags: Optional metadata dictionary

        Returns:
            Memory ID (uuid4 hex)

        Raises:
            SQLiteStoreError: If memory creation fails
        """
        memory_id = uuid4().hex
        now = time.time()
        content_hash = content_hash or hashlib.sha256(content.encode()).hexdigest()

        try:
            cursor = self._conn.cursor()

            # Insert into memories table
            cursor.execute(
                """
                INSERT INTO memories (
                    id, content, content_hash, memory_type, namespace,
                    confidence, importance, source_file, chunk_index,
                    created_at, last_accessed, access_count, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    memory_type = excluded.memory_type,
                    namespace = excluded.namespace,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    source_file = excluded.source_file,
                    chunk_index = excluded.chunk_index,
                    tags = excluded.tags
                """,
                (
                    memory_id,
                    content,
                    content_hash,
                    memory_type,
                    namespace,
                    confidence,
                    importance,
                    source_file,
                    chunk_index,
                    now,
                    now,
                    0,
                    json.dumps(tags) if tags else None,
                ),
            )

            # Insert into memories_vec (sqlite-vec)
            cursor.execute(
                "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                (memory_id, self._serialize_embedding(embedding)),
            )

            # Insert into memories_fts (FTS5)
            cursor.execute(
                """
                INSERT INTO memories_fts (id, content, memory_type, namespace, source_file)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, content, memory_type, namespace, source_file),
            )

            self._conn.commit()
            return memory_id

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to add memory: {e}") from e

    def get_memory(self, memory_id: str) -> dict | None:
        """Get memory by ID with all metadata.

        Args:
            memory_id: Memory ID

        Returns:
            Memory dict or None if not found
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, content, content_hash, memory_type, namespace,
                       confidence, importance, source_file, chunk_index,
                       start_line, end_line, created_at, last_accessed,
                       access_count, tags
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_memory(row)

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get memory: {e}") from e

    def update_memory(self, memory_id: str, **fields) -> bool:
        """Update memory fields.

        Args:
            memory_id: Memory ID
            **fields: Fields to update (confidence, importance, content, etc.)

        Returns:
            True if updated, False if not found

        Raises:
            SQLiteStoreError: If update fails
        """
        if not fields:
            return False

        # Allowed fields to update
        allowed_fields = {
            "content",
            "confidence",
            "importance",
            "memory_type",
            "namespace",
            "tags",
            "last_accessed",
            "access_count",
        }

        update_fields = {k: v for k, v in fields.items() if k in allowed_fields}
        if not update_fields:
            return False

        # Handle special cases
        if "tags" in update_fields and update_fields["tags"] is not None:
            update_fields["tags"] = json.dumps(update_fields["tags"])

        try:
            cursor = self._conn.cursor()

            # Build dynamic UPDATE query
            set_clause = ", ".join(f"{k} = ?" for k in update_fields)
            values = list(update_fields.values()) + [memory_id]

            cursor.execute(
                f"UPDATE memories SET {set_clause} WHERE id = ?",
                values,
            )

            # If content was updated, also update FTS
            if "content" in update_fields:
                cursor.execute(
                    """
                    UPDATE memories_fts
                    SET content = ?
                    WHERE id = ?
                    """,
                    (update_fields["content"], memory_id),
                )

            self._conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to update memory: {e}") from e

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from all three tables.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found

        Raises:
            SQLiteStoreError: If delete fails
        """
        try:
            cursor = self._conn.cursor()

            # Check if memory exists
            cursor.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,))
            if not cursor.fetchone():
                return False

            # Delete from all three tables atomically
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            cursor.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            cursor.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))

            # Also delete any edges involving this memory
            cursor.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )

            self._conn.commit()
            return True

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to delete memory: {e}") from e

    def list_memories(
        self,
        namespace: str | None = None,
        memory_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[dict]:
        """List memories with optional filters and pagination.

        Args:
            namespace: Optional namespace filter
            memory_type: Optional memory type filter
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Column to sort by (default: created_at)
            descending: Sort descending (default: True)

        Returns:
            List of memory dicts
        """
        try:
            cursor = self._conn.cursor()

            # Build query with optional filters
            conditions = []
            params: list[Any] = []

            if namespace is not None:
                conditions.append("namespace = ?")
                params.append(namespace)

            if memory_type is not None:
                conditions.append("memory_type = ?")
                params.append(memory_type)

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            # Whitelist allowed columns to prevent SQL injection
            allowed_columns = {"created_at", "importance", "confidence", "last_accessed"}
            sort_col = order_by if order_by in allowed_columns else "created_at"
            sort_dir = "DESC" if descending else "ASC"

            cursor.execute(
                f"""
                SELECT id, content, content_hash, memory_type, namespace,
                       confidence, importance, source_file, chunk_index,
                       start_line, end_line, created_at, last_accessed,
                       access_count, tags
                FROM memories
                {where_clause}
                ORDER BY {sort_col} {sort_dir}
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            )

            return [self._row_to_memory(row) for row in cursor.fetchall()]

        except Exception as e:
            raise SQLiteStoreError(f"Failed to list memories: {e}") from e

    def count_memories(
        self,
        namespace: str | None = None,
        memory_type: str | None = None,
    ) -> int:
        """Count memories with optional filters.

        Args:
            namespace: Optional namespace filter
            memory_type: Optional memory type filter

        Returns:
            Count of memories
        """
        try:
            cursor = self._conn.cursor()

            conditions = []
            params: list[Any] = []

            if namespace is not None:
                conditions.append("namespace = ?")
                params.append(namespace)

            if memory_type is not None:
                conditions.append("memory_type = ?")
                params.append(memory_type)

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            cursor.execute(f"SELECT COUNT(*) FROM memories{where_clause}", params)
            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            raise SQLiteStoreError(f"Failed to count memories: {e}") from e

    def list_namespaces(self) -> list[dict[str, Any]]:
        """List all namespaces with their memory counts.

        Returns:
            List of dicts with namespace name and count, sorted by count descending
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT namespace, COUNT(*) as count
                FROM memories
                GROUP BY namespace
                ORDER BY count DESC
                """
            )
            return [{"namespace": row[0], "count": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            raise SQLiteStoreError(f"Failed to list namespaces: {e}") from e

    def _row_to_memory(self, row: sqlite3.Row) -> dict:
        """Convert SQLite row to memory dict."""
        memory = {
            "id": row["id"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "memory_type": row["memory_type"],
            "namespace": row["namespace"],
            "confidence": row["confidence"],
            "importance": row["importance"],
            "source_file": row["source_file"],
            "chunk_index": row["chunk_index"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
        }

        if row["tags"]:
            try:
                memory["tags"] = json.loads(row["tags"])
            except json.JSONDecodeError:
                memory["tags"] = None
        else:
            memory["tags"] = None

        return memory

    # =========================================================================
    # Search Operations
    # =========================================================================

    # sqlite-vec KNN search has an undocumented LIMIT ceiling that varies by database size
    # and exhibits transient "unknown error" failures. Use conservative limit with retry.
    _VEC_KNN_MAX_LIMIT = 100
    _VEC_KNN_MAX_RETRIES = 3

    def search_vector(
        self,
        embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """KNN search using sqlite-vec with retry for transient errors.

        Args:
            embedding: Query embedding vector
            n_results: Number of results to return (capped at _VEC_KNN_MAX_LIMIT)
            where: Optional filter dict (namespace, memory_type)

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        # Cap n_results to avoid sqlite-vec errors
        n_results = min(n_results, self._VEC_KNN_MAX_LIMIT)

        # Build pre-filter for KNN (sqlite-vec supports AND id IN (...) with MATCH)
        knn_filter: str = ""
        knn_params: list[Any] = []
        if where:
            conditions: list[str] = []
            if "namespace" in where:
                conditions.append("namespace = ?")
                knn_params.append(where["namespace"])
            if "memory_type" in where:
                conditions.append("memory_type = ?")
                knn_params.append(where["memory_type"])
            if conditions:
                knn_filter = (
                    f" AND id IN (SELECT id FROM memories WHERE {' AND '.join(conditions)})"
                )

        # sqlite-vec KNN search with cosine distance and pre-filtering
        query = f"""
            SELECT m.*, v.distance
            FROM memories m
            JOIN (
                SELECT id, distance
                FROM memories_vec
                WHERE embedding MATCH ?{knn_filter}
                ORDER BY distance
                LIMIT ?
            ) v ON m.id = v.id
            ORDER BY v.distance
        """

        vec_param = self._serialize_embedding(embedding)
        last_error: Exception | None = None

        # Retry with progressively smaller limits on transient errors
        limits_to_try = [n_results, n_results // 2, 10, 5]
        limits_to_try = [lim for lim in limits_to_try if lim > 0]

        for attempt, limit in enumerate(limits_to_try):
            try:
                cursor = self._conn.cursor()
                cursor.execute(query, [vec_param] + knn_params + [limit])

                results: list[SearchResult] = []
                for row in cursor.fetchall():
                    distance = row["distance"]
                    similarity = 1.0 - (distance / 2.0)

                    tags = None
                    if row["tags"]:
                        try:
                            tags = json.loads(row["tags"])
                        except json.JSONDecodeError:
                            pass

                    results.append(
                        SearchResult(
                            id=row["id"],
                            content=row["content"],
                            score=similarity,
                            memory_type=row["memory_type"],
                            namespace=row["namespace"],
                            confidence=row["confidence"],
                            importance=row["importance"],
                            source_file=row["source_file"],
                            chunk_index=row["chunk_index"],
                            metadata=tags,
                            last_accessed=row["last_accessed"],
                            created_at=row["created_at"],
                            access_count=row["access_count"] or 0,
                        )
                    )

                return results[:n_results]

            except sqlite3.OperationalError as e:
                last_error = e
                logger.warning(
                    f"sqlite-vec search failed (attempt {attempt + 1}, limit={limit}): {e}"
                )
                continue

        raise SQLiteStoreError(
            f"Failed to search vectors after retries: {last_error}"
        ) from last_error

    def search_fts(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """FTS5 search with BM25 ranking.

        Args:
            query: Search query string
            n_results: Number of results to return
            where: Optional filter dict (namespace, memory_type)

        Returns:
            List of SearchResult sorted by relevance (highest first)
        """
        try:
            cursor = self._conn.cursor()

            # Escape FTS5 special characters
            escaped_query = self._escape_fts_query(query)

            # Build filter conditions
            conditions = []
            params: list[Any] = []

            if where:
                if "namespace" in where:
                    conditions.append("m.namespace = ?")
                    params.append(where["namespace"])
                if "memory_type" in where:
                    conditions.append("m.memory_type = ?")
                    params.append(where["memory_type"])

            filter_clause = " AND " + " AND ".join(conditions) if conditions else ""

            # FTS5 with BM25 ranking (lower is better, so negate for consistency)
            sql_query = f"""
                SELECT m.*, -bm25(memories_fts) as score
                FROM memories_fts f
                JOIN memories m ON f.id = m.id
                WHERE memories_fts MATCH ? {filter_clause}
                ORDER BY score DESC
                LIMIT ?
            """

            cursor.execute(sql_query, [escaped_query] + params + [n_results])

            results = []
            for row in cursor.fetchall():
                tags = None
                if row["tags"]:
                    try:
                        tags = json.loads(row["tags"])
                    except json.JSONDecodeError:
                        pass

                results.append(
                    SearchResult(
                        id=row["id"],
                        content=row["content"],
                        score=row["score"],
                        memory_type=row["memory_type"],
                        namespace=row["namespace"],
                        confidence=row["confidence"],
                        importance=row["importance"],
                        source_file=row["source_file"],
                        chunk_index=row["chunk_index"],
                        metadata=tags,
                        last_accessed=row["last_accessed"],
                        created_at=row["created_at"],
                        access_count=row["access_count"] or 0,
                    )
                )

            return results

        except Exception as e:
            raise SQLiteStoreError(f"Failed to search FTS: {e}") from e

    def _escape_fts_query(self, query: str) -> str:
        """Escape FTS5 special characters in query.

        FTS5 has many special characters: AND, OR, NOT, |, ., :, ^, -, +, *, ", (, )
        Always quote the query to treat it as a literal phrase search.
        """
        cleaned = query.strip()
        if not cleaned:
            return '""'

        # Escape internal quotes and wrap in quotes for literal matching
        escaped = cleaned.replace('"', '""')
        return f'"{escaped}"'

    def search_hybrid(
        self,
        embedding: list[float],
        query: str,
        n_results: int = 5,
        vector_weight: float = 0.7,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Combine vector and FTS results with weighted scoring.

        Uses Reciprocal Rank Fusion (RRF) to combine results from
        vector search and FTS search.

        Args:
            embedding: Query embedding vector
            query: Search query string
            n_results: Number of results to return
            vector_weight: Weight for vector scores (0.0-1.0)
            where: Optional filter dict (namespace, memory_type)

        Returns:
            List of SearchResult sorted by combined score
        """
        fts_weight = 1.0 - vector_weight

        try:
            # Get more results from each search to have overlap for fusion
            # Cap to MAX_LIMIT to avoid sqlite-vec errors
            fetch_count = min(n_results * 3, self._VEC_KNN_MAX_LIMIT)

            # Perform both searches with filters
            # search_vector handles retries internally for transient errors
            vector_results = self.search_vector(embedding, n_results=fetch_count, where=where)
            fts_results = self.search_fts(query, n_results=fetch_count, where=where)

            # Build combined scores using RRF
            # RRF score = sum(1 / (k + rank)) where k=60 is typical
            k = 60
            scores: dict[str, float] = {}
            result_map: dict[str, SearchResult] = {}

            # Add vector results
            for rank, result in enumerate(vector_results, 1):
                rrf_score = vector_weight * (1.0 / (k + rank))
                scores[result.id] = scores.get(result.id, 0) + rrf_score
                result_map[result.id] = result

            # Add FTS results
            for rank, result in enumerate(fts_results, 1):
                rrf_score = fts_weight * (1.0 / (k + rank))
                scores[result.id] = scores.get(result.id, 0) + rrf_score
                if result.id not in result_map:
                    result_map[result.id] = result

            # Apply recency weighting to combined scores
            for memory_id in scores:
                result = result_map[memory_id]
                recency = compute_recency_score(
                    result.importance,
                    result.last_accessed,
                    result.created_at or time.time(),
                    result.access_count,
                )
                scores[memory_id] *= recency

            # Sort by combined score
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

            # Build final results with combined scores
            results = []
            for memory_id in sorted_ids[:n_results]:
                result = result_map[memory_id]
                # Replace score with combined RRF score
                results.append(
                    SearchResult(
                        id=result.id,
                        content=result.content,
                        score=scores[memory_id],
                        memory_type=result.memory_type,
                        namespace=result.namespace,
                        confidence=result.confidence,
                        importance=result.importance,
                        source_file=result.source_file,
                        chunk_index=result.chunk_index,
                        metadata=result.metadata,
                        last_accessed=result.last_accessed,
                        created_at=result.created_at,
                        access_count=result.access_count,
                    )
                )

            return results

        except Exception as e:
            raise SQLiteStoreError(f"Failed to perform hybrid search: {e}") from e

    # =========================================================================
    # Embedding Cache Operations
    # =========================================================================

    def get_cached_embedding(
        self, content_hash: str, provider: str, model: str
    ) -> list[float] | None:
        """Retrieve cached embedding.

        Args:
            content_hash: SHA256 hash of content
            provider: Embedding provider (e.g., 'mlx', 'ollama')
            model: Model name

        Returns:
            Cached embedding or None if not found
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT embedding
                FROM embedding_cache
                WHERE content_hash = ? AND provider = ? AND model = ?
                """,
                (content_hash, provider, model),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return self._deserialize_embedding(row["embedding"])

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get cached embedding: {e}") from e

    def cache_embedding(
        self, content_hash: str, provider: str, model: str, embedding: list[float]
    ) -> None:
        """Store embedding in cache.

        Args:
            content_hash: SHA256 hash of content
            provider: Embedding provider (e.g., 'mlx', 'ollama')
            model: Model name
            embedding: Embedding vector

        Raises:
            SQLiteStoreError: If caching fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (content_hash, provider, model, embedding, dims, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    provider,
                    model,
                    self._serialize_embedding(embedding),
                    len(embedding),
                    time.time(),
                ),
            )
            self._conn.commit()

        except Exception as e:
            raise SQLiteStoreError(f"Failed to cache embedding: {e}") from e

    # =========================================================================
    # Transcription Operations
    # =========================================================================

    def save_transcription(
        self,
        transcription_id: str,
        full_text: str,
        duration_seconds: float | None = None,
        model_used: str | None = None,
        language: str | None = None,
        namespace: str = "default",
        segments: list[dict] | None = None,
        audio_path: str | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Save a transcription with optional segments.

        Args:
            transcription_id: Unique ID for the transcription
            full_text: Complete transcription text
            duration_seconds: Recording duration in seconds
            model_used: Whisper model identifier
            language: Detected or specified language code
            namespace: Storage namespace
            segments: List of segment dicts with text, start_time, end_time, confidence
            audio_path: Path to saved audio file
            memory_id: Optional linked memory ID

        Returns:
            The transcription ID

        Raises:
            SQLiteStoreError: If save fails
        """
        try:
            cursor = self._conn.cursor()
            now = time.time()

            # Insert transcription record
            cursor.execute(
                """
                INSERT INTO transcriptions (
                    id, audio_path, full_text, duration_seconds, model_used,
                    language, namespace, memory_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transcription_id,
                    audio_path,
                    full_text,
                    duration_seconds,
                    model_used,
                    language,
                    namespace,
                    memory_id,
                    now,
                ),
            )

            # Insert segments if provided
            if segments:
                for segment in segments:
                    segment_id = uuid4().hex
                    cursor.execute(
                        """
                        INSERT INTO transcription_segments (
                            id, transcription_id, text, start_time, end_time, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            segment_id,
                            transcription_id,
                            segment["text"],
                            segment["start_time"],
                            segment["end_time"],
                            segment.get("confidence", 0.8),
                        ),
                    )

            self._conn.commit()
            return transcription_id

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to save transcription: {e}") from e

    def get_transcription(self, transcription_id: str) -> dict | None:
        """Get transcription by ID with all segments.

        Args:
            transcription_id: Transcription ID

        Returns:
            Transcription dict with segments, or None if not found
        """
        try:
            cursor = self._conn.cursor()

            # Get transcription record
            cursor.execute(
                """
                SELECT id, audio_path, full_text, duration_seconds, model_used,
                       language, namespace, memory_id, created_at
                FROM transcriptions
                WHERE id = ?
                """,
                (transcription_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            transcription = {
                "id": row["id"],
                "audio_path": row["audio_path"],
                "full_text": row["full_text"],
                "duration_seconds": row["duration_seconds"],
                "model_used": row["model_used"],
                "language": row["language"],
                "namespace": row["namespace"],
                "memory_id": row["memory_id"],
                "created_at": row["created_at"],
            }

            # Get segments
            cursor.execute(
                """
                SELECT id, text, start_time, end_time, confidence
                FROM transcription_segments
                WHERE transcription_id = ?
                ORDER BY start_time
                """,
                (transcription_id,),
            )

            transcription["segments"] = [
                {
                    "id": seg["id"],
                    "text": seg["text"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "confidence": seg["confidence"],
                }
                for seg in cursor.fetchall()
            ]

            return transcription

        except Exception as e:
            raise SQLiteStoreError(f"Failed to get transcription: {e}") from e

    def list_transcriptions(
        self,
        namespace: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List transcriptions with optional namespace filter.

        Args:
            namespace: Optional namespace filter
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of transcription dicts (without segments)
        """
        try:
            cursor = self._conn.cursor()

            if namespace:
                cursor.execute(
                    """
                    SELECT id, audio_path, full_text, duration_seconds, model_used,
                           language, namespace, memory_id, created_at
                    FROM transcriptions
                    WHERE namespace = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (namespace, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, audio_path, full_text, duration_seconds, model_used,
                           language, namespace, memory_id, created_at
                    FROM transcriptions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )

            return [
                {
                    "id": row["id"],
                    "audio_path": row["audio_path"],
                    "full_text": row["full_text"],
                    "duration_seconds": row["duration_seconds"],
                    "model_used": row["model_used"],
                    "language": row["language"],
                    "namespace": row["namespace"],
                    "memory_id": row["memory_id"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

        except Exception as e:
            raise SQLiteStoreError(f"Failed to list transcriptions: {e}") from e

    def delete_transcription(self, transcription_id: str) -> bool:
        """Delete transcription and its segments.

        Note: Does NOT delete the audio file. Caller must handle that.

        Args:
            transcription_id: Transcription ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            SQLiteStoreError: If delete fails
        """
        try:
            cursor = self._conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT 1 FROM transcriptions WHERE id = ?",
                (transcription_id,),
            )
            if not cursor.fetchone():
                return False

            # Segments deleted via ON DELETE CASCADE
            cursor.execute(
                "DELETE FROM transcriptions WHERE id = ?",
                (transcription_id,),
            )

            self._conn.commit()
            return True

        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to delete transcription: {e}") from e

    # =========================================================================
    # Trace Operations (Agent Trace)
    # =========================================================================

    def add_trace(
        self,
        commit_sha: str,
        conversation_url: str,
        model_id: str | None = None,
        session_id: str | None = None,
        files: list[str] | None = None,
    ) -> None:
        """Record AI attribution for a git commit.

        Args:
            commit_sha: Git commit SHA
            conversation_url: Path to conversation transcript
            model_id: AI model identifier (e.g., "anthropic/claude-opus-4-5-20251101")
            session_id: Claude Code session ID
            files: List of files changed in commit

        Raises:
            SQLiteStoreError: If insert fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO traces
                (commit_sha, conversation_url, model_id, session_id, files, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    commit_sha,
                    conversation_url,
                    model_id,
                    session_id,
                    json.dumps(files or []),
                    time.time(),
                ),
            )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to add trace: {e}") from e

    def get_trace(self, commit_sha: str) -> TraceRecord | None:
        """Get trace record for a commit.

        Args:
            commit_sha: Git commit SHA

        Returns:
            TraceRecord or None if not found

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM traces WHERE commit_sha = ?",
                (commit_sha,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return TraceRecord(
                commit_sha=row["commit_sha"],
                conversation_url=row["conversation_url"],
                model_id=row["model_id"],
                session_id=row["session_id"],
                files=json.loads(row["files"]) if row["files"] else [],
                created_at=row["created_at"],
            )
        except Exception as e:
            raise SQLiteStoreError(f"Failed to get trace: {e}") from e

    def list_traces_for_conversation(self, conversation_url: str) -> list[TraceRecord]:
        """List all traces for a conversation.

        Args:
            conversation_url: Conversation transcript path

        Returns:
            List of TraceRecord objects

        Raises:
            SQLiteStoreError: If query fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM traces WHERE conversation_url = ? ORDER BY created_at",
                (conversation_url,),
            )
            return [
                TraceRecord(
                    commit_sha=row["commit_sha"],
                    conversation_url=row["conversation_url"],
                    model_id=row["model_id"],
                    session_id=row["session_id"],
                    files=json.loads(row["files"]) if row["files"] else [],
                    created_at=row["created_at"],
                )
                for row in cursor.fetchall()
            ]
        except Exception as e:
            raise SQLiteStoreError(f"Failed to list traces: {e}") from e
