"""ChromaDB storage layer with hybrid FTS search capability.

This module provides vector storage using ChromaDB with support for:
- Semantic search with metadata filtering
- Hash-based deduplication tracking
- Hybrid search combining vector and text matching
- Confidence field for memory validation
- Both persistent and ephemeral storage modes

Merges DocVec's clean CRUD interface with Recall's hybrid search approach.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from theo.storage.types import Document, HybridSearchResult, SearchResult, StoreStats

# MCP servers must never write to stdout (corrupts JSON-RPC)
# All logging goes to stderr
logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Custom exception for storage-related errors."""

    pass


class ChromaStore:
    """Vector storage layer using ChromaDB.

    Provides storage for embeddings with metadata, semantic search,
    hybrid search, and support for both persistent and ephemeral modes.

    Args:
        db_path: Path to ChromaDB persistent storage directory.
                 Defaults to ~/.theo/chroma_db for persistent storage.
        collection_name: Name of the collection (default: "documents")
        ephemeral: If True, use in-memory storage for testing (default: False)

    Attributes:
        db_path: Path to database storage (None if ephemeral)
        collection_name: Name of the active collection
        ephemeral: Whether using ephemeral storage
        _client: ChromaDB client instance
        _collection: ChromaDB collection instance
        _id_counter: Counter for generating unique IDs

    Example:
        >>> # Persistent storage
        >>> store = ChromaStore(Path("~/.theo/chroma_db"))
        >>> ids = store.add_documents(docs, embeddings)
        >>> results = store.search(query_embedding, n_results=5)

        >>> # Ephemeral for testing
        >>> store = ChromaStore(ephemeral=True)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: str = "documents",
        ephemeral: bool = False,
    ):
        """Initialize ChromaStore with persistent or ephemeral storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory.
                     Defaults to ~/.theo/chroma_db if not ephemeral.
            collection_name: Name of the collection to use
            ephemeral: If True, use in-memory EphemeralClient for testing

        Raises:
            StorageError: If database initialization fails
        """
        self.collection_name = collection_name
        self.ephemeral = ephemeral
        self._id_counter = 0

        # Set default path if not provided and not ephemeral
        if ephemeral:
            self.db_path = None
        else:
            self.db_path = db_path or Path.home() / ".theo" / "chroma_db"

        try:
            if ephemeral:
                # Use in-memory client for testing
                self._client = chromadb.EphemeralClient()
            else:
                # Ensure database directory exists
                if self.db_path is not None:
                    self.db_path.mkdir(parents=True, exist_ok=True)
                # Initialize persistent ChromaDB client with telemetry disabled
                self._client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(anonymized_telemetry=False),
                )

            # Get or create collection with cosine distance
            self._collection = self._get_or_create_collection()

        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB storage: {e}") from e

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one with cosine distance.

        Uses cosine similarity which is critical for mxbai-embed-large
        compatibility (NOT L2 which performs poorly with this model).

        Returns:
            ChromaDB Collection instance

        Raises:
            StorageError: If collection operations fail
        """
        try:
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return collection

        except Exception as e:
            raise StorageError(f"Failed to get or create collection: {e}") from e

    def _generate_id(self) -> str:
        """Generate unique, sortable ID using timestamp and sequence.

        Returns:
            Unique ID string in format: timestamp_sequence
        """
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{self._id_counter}"
        self._id_counter += 1
        return unique_id

    # =========================================================================
    # Document Operations (DocVec-style CRUD)
    # =========================================================================

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add documents with embeddings to storage.

        Args:
            documents: List of Document objects to store
            embeddings: List of embedding vectors (must match documents length)

        Returns:
            List of generated IDs for the added documents

        Raises:
            StorageError: If add operation fails
            ValueError: If input lists have different lengths
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Length mismatch: documents={len(documents)}, embeddings={len(embeddings)}"
            )

        if not documents:
            return []

        try:
            # Generate IDs if not provided
            ids = []
            contents = []
            metadatas = []

            for doc in documents:
                doc_id = doc.id if doc.id else self._generate_id()
                ids.append(doc_id)
                contents.append(doc.content)
                metadatas.append(doc.to_chroma_metadata())

            # Upsert to ChromaDB collection (handles duplicates gracefully)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=contents,
                metadatas=metadatas,  # type: ignore[arg-type]
            )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add documents to storage: {e}") from e

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """Add embeddings with documents and metadata to storage (raw API).

        This is a lower-level API for direct ChromaDB-style operations.
        For type-safe operations, prefer add_documents().

        Args:
            embeddings: List of embedding vectors
            documents: List of document content strings
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of generated IDs for the added documents

        Raises:
            StorageError: If add operation fails
            ValueError: If input lists have different lengths
        """
        if metadatas is not None and len(embeddings) != len(metadatas):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if len(embeddings) != len(documents):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, documents={len(documents)}"
            )

        if not embeddings:
            return []

        try:
            # Generate unique IDs for each document
            ids = [self._generate_id() for _ in range(len(embeddings))]

            # Add to ChromaDB collection
            if metadatas is not None:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=documents,
                    metadatas=metadatas,  # type: ignore[arg-type]
                )
            else:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=documents,
                )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add documents to storage: {e}") from e

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Perform semantic search with optional metadata filtering.

        Args:
            query_embedding: Query vector to search for
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter dict
                   (e.g., {"namespace": "project1"}, {"doc_type": "memory"})
            where_document: Optional document content filter
                   (e.g., {"$contains": "keyword"})

        Returns:
            List of SearchResult objects with documents and similarity scores

        Raises:
            StorageError: If search operation fails
        """
        try:
            # Build query kwargs
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }

            if where is not None:
                query_kwargs["where"] = where

            if where_document is not None:
                query_kwargs["where_document"] = where_document

            results = self._collection.query(**query_kwargs)

            # Convert to SearchResult objects
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    content = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    search_results.append(
                        SearchResult.from_chroma_result(
                            doc_id=doc_id,
                            content=content,
                            metadata=metadata,
                            distance=distance,
                            rank=i,
                        )
                    )

            return search_results

        except Exception as e:
            raise StorageError(f"Failed to search documents: {e}") from e

    def search_raw(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform semantic search returning raw ChromaDB results.

        This is a lower-level API returning the raw result dict.
        For type-safe operations, prefer search().

        Args:
            query_embedding: Query vector to search for
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter dict

        Returns:
            Dictionary with search results containing:
                - ids: List of document IDs
                - documents: List of document content
                - metadatas: List of metadata dictionaries
                - distances: List of distance scores (lower is better)

        Raises:
            StorageError: If search operation fails
        """
        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }

            if where is not None:
                query_kwargs["where"] = where

            results = self._collection.query(**query_kwargs)

            # ChromaDB returns results wrapped in lists (for batch queries)
            # Extract the first (and only) result set
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }

        except Exception as e:
            raise StorageError(f"Failed to search documents: {e}") from e

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
        fts_weight: float = 0.3,
        min_vector_similarity: float = 0.5,
    ) -> HybridSearchResult:
        """Perform hybrid search combining vector similarity and text matching.

        Implements Recall's hybrid search approach:
        1. Perform vector search with query embedding
        2. If results have low similarity, augment with FTS results
        3. Merge and re-rank results

        Args:
            query_embedding: Query vector for semantic search
            query_text: Query text for FTS matching
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter dict
            fts_weight: Weight for FTS results in final ranking (0.0-1.0)
            min_vector_similarity: Minimum similarity to skip FTS augmentation

        Returns:
            HybridSearchResult with combined results

        Raises:
            StorageError: If search operation fails
        """
        try:
            # Step 1: Vector search
            vector_results = self.search(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # Get more for merging
                where=where,
            )

            # Check if vector results are good enough
            avg_similarity = (
                sum(r.similarity for r in vector_results) / len(vector_results)
                if vector_results
                else 0.0
            )

            fts_used = False
            fts_count = 0

            # Step 2: FTS augmentation if vector results are weak
            if avg_similarity < min_vector_similarity and query_text.strip():
                fts_used = True

                # Use ChromaDB's where_document for text matching
                fts_results = self.search(
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where=where,
                    where_document={"$contains": query_text.lower()},
                )

                fts_count = len(fts_results)

                # Merge results (FTS results boost existing vector results)
                result_map: dict[str, SearchResult] = {}

                for result in vector_results:
                    result_map[result.document.id] = result

                for fts_result in fts_results:
                    doc_id = fts_result.document.id
                    if doc_id in result_map:
                        # Boost similarity for FTS matches
                        existing = result_map[doc_id]
                        boosted_similarity = existing.similarity + (fts_weight * 0.2)
                        result_map[doc_id] = SearchResult(
                            document=existing.document,
                            distance=existing.distance,
                            similarity=min(1.0, boosted_similarity),
                            rank=existing.rank,
                        )
                    else:
                        # Add FTS-only results with adjusted similarity
                        adjusted = SearchResult(
                            document=fts_result.document,
                            distance=fts_result.distance,
                            similarity=fts_result.similarity * fts_weight,
                            rank=fts_result.rank,
                        )
                        result_map[doc_id] = adjusted

                vector_results = list(result_map.values())

            # Step 3: Sort by similarity and limit
            vector_results.sort(key=lambda r: r.similarity, reverse=True)
            final_results = vector_results[:n_results]

            # Update ranks
            for i, result in enumerate(final_results):
                final_results[i] = SearchResult(
                    document=result.document,
                    distance=result.distance,
                    similarity=result.similarity,
                    rank=i,
                )

            return HybridSearchResult(
                results=final_results,
                vector_count=len(vector_results),
                fts_count=fts_count,
                fts_used=fts_used,
            )

        except Exception as e:
            raise StorageError(f"Failed to perform hybrid search: {e}") from e

    # =========================================================================
    # Delete Operations
    # =========================================================================

    def delete_by_id(self, ids: list[str]) -> None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            StorageError: If delete operation fails
        """
        if not ids:
            return

        try:
            self._collection.delete(ids=ids)

        except Exception as e:
            raise StorageError(f"Failed to delete documents: {e}") from e

    def delete_by_source(self, source_file: str) -> int:
        """Delete all documents from a specific source file.

        Args:
            source_file: Source file path to delete documents for

        Returns:
            Number of documents deleted

        Raises:
            StorageError: If delete operation fails
        """
        try:
            result = self.get_by_source_file(source_file)
            if result is None:
                return 0

            ids_to_delete = result["ids"]
            self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete documents by source: {e}") from e

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs (alias for delete_by_id).

        Args:
            ids: List of document IDs to delete

        Raises:
            StorageError: If delete operation fails
        """
        self.delete_by_id(ids)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_by_hash(self, doc_hash: str) -> Optional[dict[str, Any]]:
        """Get documents by hash for deduplication checking.

        Args:
            doc_hash: Document hash to search for

        Returns:
            Dictionary with document data if found, None otherwise.
            Contains: ids, documents, metadatas

        Raises:
            StorageError: If query operation fails
        """
        try:
            results = self._collection.get(
                where={"doc_hash": doc_hash},
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return None

            return {
                "ids": results["ids"],
                "documents": results["documents"],
                "metadatas": results["metadatas"],
            }

        except Exception as e:
            raise StorageError(f"Failed to get document by hash: {e}") from e

    def get_by_source_file(self, source_file: str) -> Optional[dict[str, Any]]:
        """Get all documents from a specific source file.

        Args:
            source_file: Source file path to query for

        Returns:
            Dictionary with document data if found, None otherwise.
            Contains: ids, documents, metadatas

        Raises:
            StorageError: If query operation fails
        """
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return None

            return {
                "ids": results["ids"],
                "documents": results["documents"],
                "metadatas": results["metadatas"],
            }

        except Exception as e:
            raise StorageError(f"Failed to get documents by source file: {e}") from e

    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get documents by IDs or metadata filter.

        Args:
            ids: Optional list of document IDs to retrieve
            where: Optional metadata filter dict
            limit: Optional maximum number of documents to return

        Returns:
            Dictionary with document data containing:
                - ids: List of document IDs
                - documents: List of document content
                - metadatas: List of metadata dictionaries

        Raises:
            StorageError: If get operation fails
        """
        try:
            get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if ids is not None:
                get_kwargs["ids"] = ids
            if where is not None:
                get_kwargs["where"] = where
            if limit is not None:
                get_kwargs["limit"] = limit

            results = self._collection.get(**get_kwargs)

            return {
                "ids": results["ids"] if results["ids"] else [],
                "documents": results["documents"] if results["documents"] else [],
                "metadatas": results["metadatas"] if results["metadatas"] else [],
            }

        except Exception as e:
            raise StorageError(f"Failed to get documents: {e}") from e

    # =========================================================================
    # Collection Operations
    # =========================================================================

    def count(self, where: Optional[dict[str, Any]] = None) -> int:
        """Get number of documents in collection with optional filter.

        Args:
            where: Optional metadata filter dict

        Returns:
            Number of documents stored matching the filter

        Raises:
            StorageError: If count operation fails
        """
        try:
            if where is None:
                return self._collection.count()
            else:
                # ChromaDB doesn't have a count with filter, so we get and count
                results = self._collection.get(where=where)
                return len(results["ids"]) if results["ids"] else 0

        except Exception as e:
            raise StorageError(f"Failed to count documents: {e}") from e

    def update(
        self,
        ids: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Update documents in the collection.

        Args:
            ids: List of document IDs to update
            metadatas: Optional list of new metadata dicts
            documents: Optional list of new document contents
            embeddings: Optional list of new embedding vectors

        Raises:
            StorageError: If update operation fails
        """
        try:
            update_kwargs: dict[str, Any] = {"ids": ids}
            if metadatas is not None:
                update_kwargs["metadatas"] = metadatas
            if documents is not None:
                update_kwargs["documents"] = documents
            if embeddings is not None:
                update_kwargs["embeddings"] = embeddings

            self._collection.update(**update_kwargs)

        except Exception as e:
            raise StorageError(f"Failed to update documents: {e}") from e

    def clear(self) -> int:
        """Delete all documents by dropping and recreating the collection.

        This method uses atomic collection deletion/recreation instead of
        deleting documents individually, which prevents WAL corruption
        when clearing large collections.

        Returns:
            Number of documents deleted

        Raises:
            StorageError: If clear operation fails
        """
        try:
            current_count = self._collection.count()
            if current_count == 0:
                return 0

            # Delete and recreate collection atomically
            self._client.delete_collection(self.collection_name)
            self._collection = self._get_or_create_collection()
            return current_count

        except Exception as e:
            raise StorageError(f"Failed to clear collection: {e}") from e

    def get_stats(self) -> StoreStats:
        """Get collection statistics.

        Returns:
            StoreStats object with collection statistics

        Raises:
            StorageError: If stats operation fails
        """
        try:
            total_documents = self._collection.count()
            if total_documents == 0:
                return StoreStats()

            all_docs = self._collection.get(include=["metadatas"])
            source_files: set[str] = set()
            namespaces: set[str] = set()
            doc_types: dict[str, int] = {}

            if all_docs["metadatas"]:
                for metadata in all_docs["metadatas"]:
                    if metadata:
                        if "source_file" in metadata:
                            source_files.add(metadata["source_file"])
                        if "namespace" in metadata:
                            namespaces.add(metadata["namespace"])
                        if "doc_type" in metadata:
                            doc_type = metadata["doc_type"]
                            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            return StoreStats(
                total_documents=total_documents,
                unique_sources=len(source_files),
                source_files=sorted(source_files),
                namespaces=sorted(namespaces),
                doc_types=doc_types,
            )

        except Exception as e:
            raise StorageError(f"Failed to get collection stats: {e}") from e

    def update_confidence(self, doc_id: str, confidence: float) -> bool:
        """Update the confidence score for a document.

        Used by the memory validation loop to adjust confidence
        based on practical use outcomes.

        Args:
            doc_id: Document ID to update
            confidence: New confidence score (0.0 to 1.0)

        Returns:
            True if document was updated, False if not found

        Raises:
            StorageError: If update operation fails
            ValueError: If confidence is out of range
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        try:
            # Get existing document
            result = self._collection.get(ids=[doc_id], include=["metadatas", "embeddings"])

            if not result["ids"]:
                return False

            # Update metadata with new confidence
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            metadata["confidence"] = confidence

            # ChromaDB doesn't support partial updates, so we need to update
            self._collection.update(
                ids=[doc_id],
                metadatas=[metadata],  # type: ignore[arg-type]
            )

            return True

        except Exception as e:
            raise StorageError(f"Failed to update confidence: {e}") from e
