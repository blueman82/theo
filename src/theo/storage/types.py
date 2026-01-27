"""Storage layer types for Theo.

This module defines data structures used by the storage layer:
- Document: Dataclass for document/chunk storage with unified metadata
- SearchResult: Structured search result with similarity score
- StoreStats: Collection statistics

These types merge DocVec's document-centric model with Recall's
confidence and relationship tracking fields.
"""

import json
import sqlite3
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
        content_hash: Hash of content for deduplication
        confidence: Confidence score from 0.0 to 1.0 (for memory validation)
        importance: Importance score from 0.0 to 1.0
        namespace: Scope/namespace for organizing documents
        memory_type: Type categorization (e.g., 'document', 'memory', 'fact')
        created_at: When the document was created (unix timestamp)
        last_accessed: When the document was last accessed (unix timestamp)
        access_count: Number of times the document has been accessed
        start_line: Starting line number in source file
        end_line: Ending line number in source file
        tags: Additional metadata as dict (stored as JSON)
    """

    id: str
    content: str
    embedding: Optional[list[float]] = None
    source_file: Optional[str] = None
    chunk_index: int = 0
    content_hash: Optional[str] = None
    confidence: float = 1.0
    importance: float = 0.5
    namespace: str = "default"
    memory_type: str = "document"
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_accessed: Optional[float] = None
    access_count: int = 0
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    tags: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict suitable for SQLite INSERT.

        Returns:
            Dictionary with column names matching SQLite schema.
            The 'tags' field is serialized as JSON string.
        """
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type,
            "namespace": self.namespace,
            "importance": self.importance,
            "confidence": self.confidence,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "tags": json.dumps(self.tags) if self.tags else None,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Document":
        """Create a Document from a sqlite3.Row.

        Args:
            row: A sqlite3.Row object from a SELECT query

        Returns:
            Document instance
        """
        # Get row as dict for easier access
        row_dict = dict(row)

        # Parse tags JSON if present
        tags_raw = row_dict.get("tags")
        tags = json.loads(tags_raw) if tags_raw else None

        return cls(
            id=row_dict["id"],
            content=row_dict["content"],
            embedding=None,  # Embeddings stored separately in vec table
            source_file=row_dict.get("source_file"),
            chunk_index=row_dict.get("chunk_index", 0),
            content_hash=row_dict.get("content_hash"),
            confidence=row_dict.get("confidence", 1.0),
            importance=row_dict.get("importance", 0.5),
            namespace=row_dict.get("namespace", "default"),
            memory_type=row_dict.get("memory_type", "document"),
            created_at=row_dict.get("created_at", datetime.now().timestamp()),
            last_accessed=row_dict.get("last_accessed"),
            access_count=row_dict.get("access_count", 0),
            start_line=row_dict.get("start_line"),
            end_line=row_dict.get("end_line"),
            tags=tags,
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
    def from_sqlite(
        cls,
        doc: Document,
        distance: float,
        rank: int = 0,
    ) -> "SearchResult":
        """Create SearchResult from SQLite query result.

        Args:
            doc: Document instance
            distance: Vector distance (0 = identical, higher = less similar)
            rank: Result position (0-indexed)

        Returns:
            SearchResult instance with similarity = 1.0 - distance
        """
        # Convert distance to similarity
        # Distance: 0 = identical, 1 = orthogonal
        # Similarity: 1.0 = identical, 0.0 = orthogonal
        similarity = max(0.0, 1.0 - distance)

        return cls(
            document=doc,
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
        memory_types: Mapping of memory_type to count
    """

    total_documents: int = 0
    unique_sources: int = 0
    source_files: list[str] = field(default_factory=list)
    namespaces: list[str] = field(default_factory=list)
    memory_types: dict[str, int] = field(default_factory=dict)


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
