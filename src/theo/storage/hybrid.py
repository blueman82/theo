"""Hybrid storage layer coordinating ChromaDB and SQLite operations.

This module provides a HybridStore that coordinates ChromaDB (vector storage,
document content) and SQLite (relationship graph) operations.

Key architecture:
- ChromaDB is the source of truth for documents/memories (content + embeddings)
- SQLite is only used for relationship edges between memories

Usage:
    >>> store = await HybridStore.create()
    >>> mem_id = await store.add_memory("Important fact", memory_type="fact")
    >>> results = await store.search("relevant query", n_results=5)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from theo.config import EmbeddingBackend, TheoSettings
from theo.embedding import EmbeddingProvider, create_embedding_provider
from theo.storage.chroma_store import ChromaStore
from theo.storage.chroma_store import StorageError as ChromaStorageError
from theo.storage.sqlite_store import SQLiteStore, SQLiteStoreError
from theo.storage.types import Document

logger = logging.getLogger(__name__)


class HybridStoreError(Exception):
    """Custom exception for hybrid storage operations."""
    pass


class HybridStore:
    """Coordinated storage layer combining ChromaDB and SQLite.

    ChromaDB is the source of truth for all document/memory content.
    SQLite handles relationship edges for graph traversal.

    Args:
        chroma_store: ChromaStore instance for vector operations
        sqlite_store: SQLiteStore instance for edge operations
        embedding_client: EmbeddingProvider for generating embeddings

    Example:
        >>> async with HybridStore.create() as store:
        ...     mem_id = await store.add_memory("Important fact", memory_type="fact")
        ...     results = await store.search("relevant query", n_results=5)
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        sqlite_store: SQLiteStore,
        embedding_client: EmbeddingProvider,
    ):
        """Initialize HybridStore with component stores.

        Args:
            chroma_store: ChromaStore instance
            sqlite_store: SQLiteStore instance
            embedding_client: EmbeddingProvider instance
        """
        self._chroma = chroma_store
        self._sqlite = sqlite_store
        self._embedding_client = embedding_client

    @classmethod
    async def create(
        cls,
        sqlite_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        collection_name: str = "documents",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "mxbai-embed-large",
        ephemeral: bool = False,
        embedding_backend: EmbeddingBackend = "mlx",
        mlx_model: str = "mlx-community/mxbai-embed-large-v1",
        embedding_client: Optional[EmbeddingProvider] = None,
    ) -> "HybridStore":
        """Create a HybridStore with new component instances.

        Factory method for convenient HybridStore creation with
        default configurations.

        Args:
            sqlite_path: Path to SQLite database (default: ~/.theo/theo.db)
            chroma_path: Path to ChromaDB storage (default: ~/.theo/chroma_db)
            collection_name: ChromaDB collection name (default: "documents")
            ollama_host: Ollama server host (default: http://localhost:11434)
            ollama_model: Embedding model name (default: mxbai-embed-large)
            ephemeral: Use in-memory storage for testing (default: False)
            embedding_backend: Embedding backend to use ('ollama' or 'mlx')
            mlx_model: MLX model identifier (used when embedding_backend='mlx')
            embedding_client: Optional existing EmbeddingProvider to reuse
                (avoids creating duplicate Metal contexts on Apple Silicon)

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
            chroma_store = ChromaStore(
                db_path=chroma_path or settings.get_chroma_path(),
                collection_name=collection_name,
                ephemeral=ephemeral,
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
                chroma_store=chroma_store,
                sqlite_store=sqlite_store,
                embedding_client=embedding_client,
            )

        except (SQLiteStoreError, ChromaStorageError) as e:
            raise HybridStoreError(f"Failed to create HybridStore: {e}") from e

    def close(self) -> None:
        """Close all underlying stores and clients."""
        self._embedding_client.close()
        self._sqlite.close()
        # ChromaDB doesn't require explicit close

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
        """Add a new memory with embedding to ChromaDB.

        Args:
            content: The memory content text
            memory_type: Type of memory (e.g., 'fact', 'decision', 'context')
            namespace: Namespace for organizing memories
            importance: Importance score from 0.0 to 1.0
            confidence: Confidence score from 0.0 to 1.0 (default: 0.3)
            metadata: Optional additional metadata as dict
            memory_id: Optional custom ID (auto-generated if not provided)
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

            # Build document - store namespace in metadata for consistency with migrated data
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["namespace"] = namespace
            doc_metadata["importance"] = importance

            doc = Document(
                id=memory_id or "",  # ChromaStore will generate if empty
                content=content,
                source_file=None,  # Not used for memories
                doc_type=memory_type,
                confidence=confidence,
                metadata=doc_metadata,
            )

            # Add to ChromaDB
            ids = self._chroma.add_documents([doc], [embedding])
            return ids[0] if ids else ""

        except Exception as e:
            raise HybridStoreError(f"Failed to add memory: {e}") from e

    async def get_memory(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Get a memory by ID from ChromaDB.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            Memory dict or None if not found
        """
        try:
            result = self._chroma.get(ids=[memory_id])
            if result and result["ids"]:
                doc = Document.from_chroma_result(
                    doc_id=result["ids"][0],
                    content=result["documents"][0],
                    metadata=result["metadatas"][0] if result["metadatas"] else {},
                )
                return {
                    "id": doc.id,
                    "content": doc.content,
                    "type": doc.doc_type,
                    "namespace": doc.metadata.get("namespace", "global") if doc.metadata else "global",
                    "importance": doc.metadata.get("importance", 0.5) if doc.metadata else 0.5,
                    "confidence": doc.confidence,
                    "metadata": doc.metadata,
                }
            return None
        except Exception as e:
            raise HybridStoreError(f"Failed to get memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from ChromaDB.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if memory was deleted, False if not found
        """
        try:
            self._chroma.delete([memory_id])
            return True
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
    ) -> list[dict[str, Any]]:
        """Perform semantic search using ChromaDB.

        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)

        Returns:
            List of memory dicts with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._embedding_client.embed_query(query)

            # Build metadata filter
            where: Optional[dict[str, Any]] = None
            if namespace or memory_type:
                where = {}
                if namespace:
                    where["namespace"] = namespace
                if memory_type:
                    where["doc_type"] = memory_type

            # Search ChromaDB
            results = self._chroma.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
            )

            # Convert to memory format
            memories = []
            for r in results:
                doc = r.document
                memories.append({
                    "id": doc.id,
                    "content": doc.content,
                    "type": doc.doc_type,
                    "namespace": doc.metadata.get("namespace", "global") if doc.metadata else "global",
                    "importance": doc.metadata.get("importance", 0.5) if doc.metadata else 0.5,
                    "confidence": doc.confidence,
                    "similarity": r.similarity,
                    "metadata": doc.metadata,
                })

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
        chroma_stats = self._chroma.get_stats()
        edge_count = self._sqlite.count_edges()

        return {
            "document_count": chroma_stats.total_documents,
            "sources": chroma_stats.source_files,
            "namespaces": chroma_stats.namespaces,
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
            metadata: New metadata to merge (optional)

        Returns:
            True if memory was updated, False if not found
        """
        try:
            # Get existing memory
            result = self._chroma.get(ids=[memory_id])
            if not result or not result["ids"]:
                return False

            # Get existing metadata
            existing_meta = result["metadatas"][0] if result["metadatas"] else {}

            # Update fields
            if confidence is not None:
                existing_meta["confidence"] = confidence
            if importance is not None:
                existing_meta["importance"] = importance
            if memory_type is not None:
                existing_meta["doc_type"] = memory_type
            if metadata is not None:
                existing_meta.update(metadata)

            # Update accessed_at timestamp
            import time
            existing_meta["accessed_at"] = time.time()

            # Update in ChromaDB
            self._chroma.update(
                ids=[memory_id],
                metadatas=[existing_meta],
            )
            return True

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
        where: Optional[dict[str, Any]] = None
        if namespace or memory_type:
            where = {}
            if namespace:
                where["namespace"] = namespace
            if memory_type:
                where["doc_type"] = memory_type

        return self._chroma.count(where=where)

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
            order_by: Field to sort by
            descending: Sort in descending order

        Returns:
            List of memory dicts
        """
        where: Optional[dict[str, Any]] = None
        if namespace or memory_type:
            where = {}
            if namespace:
                where["namespace"] = namespace
            if memory_type:
                where["doc_type"] = memory_type

        # Get all matching memories
        result = self._chroma.get(where=where, limit=limit + offset)

        if not result or not result["ids"]:
            return []

        # Convert to memory dicts
        memories = []
        for i, doc_id in enumerate(result["ids"]):
            metadata = result["metadatas"][i] if result["metadatas"] else {}
            content = result["documents"][i] if result["documents"] else ""
            memories.append({
                "id": doc_id,
                "content": content,
                "type": metadata.get("doc_type", "document"),
                "namespace": metadata.get("namespace", "global"),
                "importance": metadata.get("importance", 0.5),
                "confidence": metadata.get("confidence", 0.3),
                "created_at": metadata.get("created_at"),
                "accessed_at": metadata.get("accessed_at"),
            })

        # Sort (ChromaDB doesn't support ordering)
        if order_by in ("importance", "confidence"):
            memories.sort(key=lambda x: x.get(order_by, 0), reverse=descending)
        elif order_by in ("created_at", "accessed_at"):
            memories.sort(
                key=lambda x: x.get(order_by) or 0,
                reverse=descending,
            )

        # Apply pagination
        return memories[offset:offset + limit]

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
