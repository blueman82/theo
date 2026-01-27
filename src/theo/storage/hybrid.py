"""Hybrid storage layer coordinating SQLite operations.

This module provides a HybridStore that wraps SQLiteStore, providing
a high-level API for memory operations with automatic embedding generation.

Key architecture:
- SQLiteStore is the source of truth for documents/memories (content + embeddings + metadata)
- Embeddings are generated via EmbeddingProvider and stored in sqlite-vec
- Full-text search via FTS5, vector search via sqlite-vec

Usage:
    >>> store = await HybridStore.create()
    >>> mem_id = await store.add_memory("Important fact", memory_type="fact")
    >>> results = await store.search("relevant query", n_results=5)
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from theo.config import EmbeddingBackend, TheoSettings
from theo.embedding import EmbeddingProvider, create_embedding_provider
from theo.storage.sqlite_store import SQLiteStore, SQLiteStoreError

logger = logging.getLogger(__name__)


class HybridStoreError(Exception):
    """Custom exception for hybrid storage operations."""
    pass


class HybridStore:
    """Storage layer wrapping SQLiteStore with automatic embedding generation.

    SQLiteStore is the source of truth for all document/memory content,
    embeddings, and relationship edges.

    Args:
        sqlite_store: SQLiteStore instance for all storage operations
        embedding_client: EmbeddingProvider for generating embeddings

    Example:
        >>> async with HybridStore.create() as store:
        ...     mem_id = await store.add_memory("Important fact", memory_type="fact")
        ...     results = await store.search("relevant query", n_results=5)
    """

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        embedding_client: EmbeddingProvider,
    ):
        """Initialize HybridStore with SQLiteStore and embedding provider.

        Args:
            sqlite_store: SQLiteStore instance
            embedding_client: EmbeddingProvider instance
        """
        self._sqlite = sqlite_store
        self._embedding_client = embedding_client

    @classmethod
    async def create(
        cls,
        sqlite_path: Optional[Path] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "mxbai-embed-large",
        embedding_backend: EmbeddingBackend = "mlx",
        mlx_model: str = "mlx-community/mxbai-embed-large-v1",
        embedding_client: Optional[EmbeddingProvider] = None,
        # Legacy parameters (ignored, kept for backward compatibility)
        chroma_path: Optional[Path] = None,
        collection_name: str = "documents",
        ephemeral: bool = False,
    ) -> "HybridStore":
        """Create a HybridStore with new component instances.

        Factory method for convenient HybridStore creation with
        default configurations.

        Args:
            sqlite_path: Path to SQLite database (default: ~/.theo/theo.db)
            ollama_host: Ollama server host (default: http://localhost:11434)
            ollama_model: Embedding model name (default: mxbai-embed-large)
            embedding_backend: Embedding backend to use ('ollama' or 'mlx')
            mlx_model: MLX model identifier (used when embedding_backend='mlx')
            embedding_client: Optional existing EmbeddingProvider to reuse
                (avoids creating duplicate Metal contexts on Apple Silicon)
            chroma_path: DEPRECATED - ignored (ChromaDB removed)
            collection_name: DEPRECATED - ignored (ChromaDB removed)
            ephemeral: DEPRECATED - ignored (ChromaDB removed)

        Returns:
            Configured HybridStore instance

        Raises:
            HybridStoreError: If store initialization fails
        """
        try:
            # Use settings defaults if paths not provided
            settings = TheoSettings()

            sqlite_store = SQLiteStore(
                db_path=sqlite_path or settings.get_sqlite_path()
            )

            # Use provided embedding client or create new one
            if embedding_client is None:
                embedding_client = create_embedding_provider(
                    backend=embedding_backend,
                    host=ollama_host,
                    model=ollama_model,
                    mlx_model=mlx_model,
                )

            return cls(
                sqlite_store=sqlite_store,
                embedding_client=embedding_client,
            )

        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to create HybridStore: {e}") from e

    def close(self) -> None:
        """Close all underlying stores and clients."""
        self._embedding_client.close()
        self._sqlite.close()

    async def __aenter__(self) -> "HybridStore":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - close all resources."""
        self.close()

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        namespace: str = "global",
        importance: float = 0.5,
        confidence: float = 0.3,
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        queue_id: Optional[int] = None,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Add a new memory with embedding to SQLite.

        Args:
            content: The memory content text
            memory_type: Type of memory (e.g., 'fact', 'decision', 'context')
            namespace: Namespace for organizing memories
            importance: Importance score from 0.0 to 1.0
            confidence: Confidence score from 0.0 to 1.0 (default: 0.3)
            metadata: Optional additional metadata as dict
            memory_id: Optional custom ID (ignored - SQLiteStore generates UUIDs)
            queue_id: Optional queue entry ID (unused, for compatibility)
            embedding: Optional pre-computed embedding

        Returns:
            The ID of the created memory

        Raises:
            HybridStoreError: If add operation fails
        """
        try:
            # Generate embedding if not provided
            if embedding is None:
                embeddings = self._embedding_client.embed_texts([content])
                embedding = embeddings[0] if embeddings else []

            # Compute content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Store in SQLite with all three tables (memories, memories_vec, memories_fts)
            memory_id = self._sqlite.add_memory(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                namespace=namespace,
                confidence=confidence,
                importance=importance,
                content_hash=content_hash,
                tags=metadata,
            )

            return memory_id

        except Exception as e:
            raise HybridStoreError(f"Failed to add memory: {e}") from e

    async def get_memory(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Get a memory by ID from SQLite.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            Memory dict or None if not found
        """
        try:
            result = self._sqlite.get_memory(memory_id)
            if result:
                return {
                    "id": result["id"],
                    "content": result["content"],
                    "type": result["memory_type"],
                    "namespace": result["namespace"],
                    "importance": result["importance"],
                    "confidence": result["confidence"],
                    "metadata": result.get("tags"),
                }
            return None
        except Exception as e:
            raise HybridStoreError(f"Failed to get memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from SQLite.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if memory was deleted, False if not found
        """
        try:
            return self._sqlite.delete_memory(memory_id)
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
            return False

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        n_results: int = 5,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining vector and FTS using SQLite.

        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)
            vector_weight: Weight for vector scores vs FTS (default: 0.7)

        Returns:
            List of memory dicts with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._embedding_client.embed_query(query)

            # Search using hybrid (vector + FTS) via SQLiteStore
            # Note: search_hybrid doesn't support namespace/memory_type filters directly,
            # so filtering is done post-search if needed
            results = self._sqlite.search_hybrid(
                embedding=query_embedding,
                query=query,
                n_results=n_results * 2 if namespace or memory_type else n_results,
                vector_weight=vector_weight,
            )

            # Convert SearchResult dataclass to dict format and apply filters
            memories = []
            for r in results:
                # Apply namespace and memory_type filters
                if namespace and r.namespace != namespace:
                    continue
                if memory_type and r.memory_type != memory_type:
                    continue

                memories.append({
                    "id": r.id,
                    "content": r.content,
                    "type": r.memory_type,
                    "namespace": r.namespace,
                    "importance": r.importance,
                    "confidence": r.confidence,
                    "similarity": r.score,
                    "metadata": r.metadata,
                })

                # Stop once we have enough results after filtering
                if len(memories) >= n_results:
                    break

            return memories

        except Exception as e:
            raise HybridStoreError(f"Search failed: {e}") from e

    # =========================================================================
    # Edge Operations (delegated to SQLite)
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "relates_to",
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add an edge between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: Type of relationship (default: 'relates_to')
            weight: Edge weight (default: 1.0)
            metadata: Optional edge metadata

        Returns:
            The ID of the created edge
        """
        try:
            return self._sqlite.add_edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata,
            )
        except Exception as e:
            raise HybridStoreError(f"Failed to add edge: {e}") from e

    def get_related(
        self,
        memory_id: str,
        max_depth: int = 1,
        edge_types: Optional[list[str]] = None,
        min_weight: float = 0.0,
    ) -> list[str]:
        """Get related memory IDs via graph traversal.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum traversal depth (default: 1)
            edge_types: Filter by edge types (optional)
            min_weight: Minimum edge weight (default: 0.0)

        Returns:
            List of related memory IDs
        """
        try:
            return self._sqlite.get_related(
                memory_id=memory_id,
                max_depth=max_depth,
                edge_types=edge_types,
                min_weight=min_weight,
            )
        except Exception as e:
            raise HybridStoreError(f"Failed to get related memories: {e}") from e

    # =========================================================================
    # Stats and Utility
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with document count, edge count, etc.
        """
        memory_count = self._sqlite.count_memories()
        edge_count = self._sqlite.count_edges()

        # Get namespace breakdown
        namespaces: list[str] = []
        # Note: SQLiteStore doesn't have a distinct namespaces query,
        # so we provide just the counts for now
        return {
            "document_count": memory_count,
            "sources": [],  # Not tracked in same way as ChromaDB
            "namespaces": namespaces,
            "edge_count": edge_count,
        }

    # =========================================================================
    # Validation Event Operations (delegated to SQLite)
    # =========================================================================

    def add_validation_event(
        self,
        memory_id: str,
        event_type: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a validation event for a memory.

        Args:
            memory_id: ID of the memory
            event_type: Type of event ('applied', 'succeeded', 'failed')
            context: Optional description of the context
            session_id: Optional session identifier

        Returns:
            Event ID
        """
        return self._sqlite.add_validation_event(
            memory_id=memory_id,
            event_type=event_type,
            context=context,
            session_id=session_id,
        )

    def get_validation_events(
        self,
        memory_id: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get validation events for a memory.

        Args:
            memory_id: Memory ID
            event_type: Optional filter by event type
            limit: Maximum number of events

        Returns:
            List of event dicts
        """
        return self._sqlite.get_validation_events(
            memory_id=memory_id,
            event_type=event_type,
            limit=limit,
        )

    # =========================================================================
    # Memory Update and Count Operations
    # =========================================================================

    async def update_memory(
        self,
        memory_id: str,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update a memory's fields.

        Args:
            memory_id: ID of the memory to update
            confidence: New confidence score (optional)
            importance: New importance score (optional)
            memory_type: New memory type (optional)
            metadata: New metadata (optional - replaces existing tags)

        Returns:
            True if memory was updated, False if not found
        """
        try:
            # Build update fields
            fields: dict[str, Any] = {}
            if confidence is not None:
                fields["confidence"] = confidence
            if importance is not None:
                fields["importance"] = importance
            if memory_type is not None:
                fields["memory_type"] = memory_type
            if metadata is not None:
                fields["tags"] = metadata

            if not fields:
                return False

            return self._sqlite.update_memory(memory_id, **fields)

        except Exception as e:
            raise HybridStoreError(f"Failed to update memory: {e}") from e

    def count_memories(
        self,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> int:
        """Count memories with optional filters.

        Args:
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)

        Returns:
            Number of matching memories
        """
        return self._sqlite.count_memories(
            namespace=namespace,
            memory_type=memory_type,
        )

    def list_memories(
        self,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        """List memories with filtering and pagination.

        Args:
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to sort by (unused - SQLiteStore returns by created_at DESC)
            descending: Sort in descending order (unused)

        Returns:
            List of memory dicts
        """
        results = self._sqlite.list_memories(
            namespace=namespace,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        )

        # Convert to HybridStore format
        memories = []
        for r in results:
            memories.append({
                "id": r["id"],
                "content": r["content"],
                "type": r["memory_type"],
                "namespace": r["namespace"],
                "importance": r["importance"],
                "confidence": r["confidence"],
                "created_at": r.get("created_at"),
                "accessed_at": r.get("last_accessed"),
            })

        return memories

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
        """
        return self._sqlite.get_edges(
            memory_id=memory_id,
            direction=direction,
            edge_type=edge_type,
        )

    def delete_edge_by_id(self, edge_id: int) -> bool:
        """Delete an edge by its ID.

        Args:
            edge_id: The edge ID to delete

        Returns:
            True if edge was deleted
        """
        return self._sqlite.delete_edge_by_id(edge_id)
