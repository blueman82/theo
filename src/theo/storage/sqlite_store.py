"""SQLite storage layer for Theo relationship graphs.

This module provides persistent storage for memory relationships (edges)
that cannot be efficiently stored in ChromaDB. It enables graph expansion
during memory recall - when searching for a memory, related memories can
also be returned based on relationship type and weight.

Edge types:
- relates_to: General relationship between memories
- supersedes: One memory replaces another (newer info)
- caused_by: Causation relationship
- contradicts: Conflicting information

Example:
    >>> store = SQLiteStore(Path("~/.theo/theo.db"))
    >>> store.add_edge("mem1", "mem2", "supersedes", weight=1.0)
    >>> related = store.get_related("mem1", max_depth=2)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import sqlite_vec

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)


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

            # Connect with foreign keys enabled
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.row_factory = sqlite3.Row

            # Load sqlite-vec extension for vector storage
            sqlite_vec.load(self._conn)

            # Initialize schema
            self._init_schema()

        except Exception as e:
            raise SQLiteStoreError(f"Failed to initialize SQLite storage: {e}") from e

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()

        # Create edges table for relationship graph
        cursor.execute("""
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
        """)

        # Create indexes for efficient graph traversal
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_source
            ON edges(source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges(target_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_type
            ON edges(edge_type)
        """)

        # Create schema version table for future migrations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
        """)

        # Create validation_events table for TRY/LEARN cycle tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                context TEXT,
                session_id TEXT,
                created_at REAL NOT NULL
            )
        """)

        # Create index for validation event queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_memory
            ON validation_events(memory_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_type
            ON validation_events(event_type)
        """)

        # =====================================================================
        # Memory Storage Tables (sqlite-vec migration - schema v2)
        # =====================================================================

        # Memory content and metadata (replaces ChromaDB)
        cursor.execute("""
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
        """)

        # Vector embeddings (sqlite-vec) - MLX uses 1024 dimensions
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        # Full-text search (FTS5)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                id UNINDEXED,
                memory_type UNINDEXED,
                namespace UNINDEXED,
                source_file UNINDEXED
            )
        """)

        # Embedding cache (avoid re-computation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dims INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (content_hash, provider, model)
            )
        """)

        # Indexes for memories
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace
            ON memories(namespace)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_confidence
            ON memories(confidence DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_source_file
            ON memories(source_file)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_hash
            ON memories(content_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash
            ON embedding_cache(content_hash)
        """)

        # Record schema version 2 (sqlite-vec migration)
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (1, ?)
        """, (time.time(),))
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version, applied_at)
            VALUES (2, ?)
        """, (time.time(),))

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
            cursor = self._conn.cursor()

            if edge_type:
                cursor.execute(
                    """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges
                    WHERE source_id = ? AND edge_type = ?
                    """,
                    (source_id, edge_type),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges
                    WHERE source_id = ?
                    """,
                    (source_id,),
                )

            return [self._row_to_edge(row) for row in cursor.fetchall()]

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
            cursor = self._conn.cursor()

            if edge_type:
                cursor.execute(
                    """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges
                    WHERE target_id = ? AND edge_type = ?
                    """,
                    (target_id, edge_type),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges
                    WHERE target_id = ?
                    """,
                    (target_id,),
                )

            return [self._row_to_edge(row) for row in cursor.fetchall()]

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
