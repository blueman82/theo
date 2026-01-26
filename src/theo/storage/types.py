"""Storage layer types for Theo.

This module defines data structures used by the storage layer:
- Document: Dataclass for document/chunk storage with unified metadata
- SearchResult: Structured search result with similarity score
- StoreStats: Collection statistics

These types merge DocVec's document-centric model with Recall's
confidence and relationship tracking fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Document:
    """A document/chunk stored in the vector database.

    Represents a single piece of content with metadata for both
    document indexing (DocVec) and memory (Recall) use cases.

    Attributes:
        id: Unique identifier for the document
        content: The actual text content
        embedding: Vector embedding (optional, may be generated separately)
        source_file: Source file path (for document indexing)
        chunk_index: Index of this chunk within the source file
        doc_hash: Hash of content for deduplication
        confidence: Confidence score from 0.0 to 1.0 (for memory validation)
        namespace: Scope/namespace for organizing documents
        doc_type: Type categorization (e.g., 'document', 'memory', 'fact')
        created_at: When the document was created
        metadata: Additional metadata as dict
    """

    id: str
    content: str
    embedding: Optional[list[float]] = None
    source_file: Optional[str] = None
    chunk_index: int = 0
    doc_hash: Optional[str] = None
    confidence: float = 1.0
    namespace: str = "default"
    doc_type: str = "document"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = None

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Convert to ChromaDB-compatible metadata dict.

        ChromaDB only supports primitive types in metadata:
        str, int, float, bool. This method flattens the document
        fields into a compatible format.

        Returns:
            Dictionary suitable for ChromaDB metadata field
        """
        meta: dict[str, Any] = {
            "namespace": self.namespace,
            "doc_type": self.doc_type,
            "confidence": self.confidence,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at.isoformat(),
        }

        if self.source_file is not None:
            meta["source_file"] = self.source_file

        if self.doc_hash is not None:
            meta["doc_hash"] = self.doc_hash

        # Flatten additional metadata (only primitive types)
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[f"meta_{key}"] = value

        return meta

    @classmethod
    def from_chroma_result(
        cls,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> "Document":
        """Create a Document from ChromaDB query result.

        Args:
            doc_id: The document ID from ChromaDB
            content: The document content
            metadata: The metadata dict from ChromaDB
            embedding: Optional embedding vector

        Returns:
            Document instance
        """
        # Extract standard fields
        source_file = metadata.get("source_file")
        chunk_index = metadata.get("chunk_index", 0)
        doc_hash = metadata.get("doc_hash")
        confidence = metadata.get("confidence", 1.0)
        namespace = metadata.get("namespace", "default")
        doc_type = metadata.get("doc_type", "document")

        # Parse created_at
        created_at_str = metadata.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        # Extract custom metadata (prefixed with meta_)
        custom_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            if key.startswith("meta_"):
                custom_metadata[key[5:]] = value  # Remove meta_ prefix

        return cls(
            id=doc_id,
            content=content,
            embedding=embedding,
            source_file=source_file,
            chunk_index=chunk_index,
            doc_hash=doc_hash,
            confidence=confidence,
            namespace=namespace,
            doc_type=doc_type,
            created_at=created_at,
            metadata=custom_metadata if custom_metadata else None,
        )


@dataclass
class SearchResult:
    """Result from a search operation.

    Contains the matched document along with relevance scoring.

    Attributes:
        document: The matched Document
        distance: Raw distance from query (lower is better for cosine)
        similarity: Similarity score from 0.0 to 1.0 (higher is better)
        rank: Position in result set (0-indexed)
    """

    document: Document
    distance: float
    similarity: float
    rank: int = 0

    @classmethod
    def from_chroma_result(
        cls,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        distance: float,
        rank: int = 0,
    ) -> "SearchResult":
        """Create SearchResult from ChromaDB query result.

        Args:
            doc_id: Document ID
            content: Document content
            metadata: ChromaDB metadata dict
            distance: Cosine distance (0 = identical, 2 = opposite)
            rank: Result position

        Returns:
            SearchResult instance
        """
        document = Document.from_chroma_result(doc_id, content, metadata)

        # Convert cosine distance to similarity
        # Cosine distance: 0 = identical, 2 = opposite
        # Similarity: 1.0 = identical, 0.0 = opposite
        similarity = 1.0 - (distance / 2.0)

        return cls(
            document=document,
            distance=distance,
            similarity=similarity,
            rank=rank,
        )


@dataclass
class StoreStats:
    """Statistics about the storage collection.

    Provides aggregate information about stored documents.

    Attributes:
        total_documents: Total number of documents/chunks
        unique_sources: Number of unique source files
        source_files: List of unique source file paths
        namespaces: List of unique namespaces
        doc_types: Mapping of doc_type to count
    """

    total_documents: int = 0
    unique_sources: int = 0
    source_files: list[str] = field(default_factory=list)
    namespaces: list[str] = field(default_factory=list)
    doc_types: dict[str, int] = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining vector and FTS.

    Attributes:
        results: List of SearchResult from combined search
        vector_count: Number of results from vector search
        fts_count: Number of results from FTS (if used)
        fts_used: Whether FTS was used in this search
    """

    results: list[SearchResult] = field(default_factory=list)
    vector_count: int = 0
    fts_count: int = 0
    fts_used: bool = False
