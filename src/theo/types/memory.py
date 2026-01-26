"""Unified MemoryDocument type for Theo.

This module defines the MemoryDocument type that unifies DocVec's Chunk concept
with Recall's Memory concept. Both document chunks and memories are treated as
"remembered information" with:
- Provenance fields (source file, line numbers) from DocVec Chunk
- Context fields (when stored, importance) from Recall Memory
- Confidence field for the validation loop
- Relationships for the memory graph

The key insight: Document chunks and memories are both "remembered information"
that can be stored, retrieved, and validated through use.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Result Types for Memory Operations
# =============================================================================


@dataclass
class ApplyResult:
    """Result of applying a memory (recording its use).

    Attributes:
        success: Whether the apply was recorded
        memory_id: ID of the applied memory
        event_id: ID of the validation event created
        error: Error message (if failed)
    """

    success: bool
    memory_id: Optional[str] = None
    event_id: Optional[int] = None
    error: Optional[str] = None


@dataclass
class OutcomeResult:
    """Result of recording an outcome for a memory application.

    Attributes:
        success: Whether the outcome was recorded
        memory_id: ID of the memory
        outcome_success: Whether the memory application succeeded
        new_confidence: Updated confidence score
        promoted: Whether memory was promoted to golden rule
        error: Error message (if failed)
    """

    success: bool
    memory_id: Optional[str] = None
    outcome_success: Optional[bool] = None
    new_confidence: Optional[float] = None
    promoted: bool = False
    error: Optional[str] = None


@dataclass
class EdgeForgetResult:
    """Result of deleting edges.

    Attributes:
        success: Whether the operation succeeded
        deleted_ids: List of deleted edge IDs
        deleted_count: Number of edges deleted
        error: Error message (if failed)
    """

    success: bool
    deleted_ids: list[int] = field(default_factory=list)
    deleted_count: int = 0
    error: Optional[str] = None


@dataclass
class ContradictionResult:
    """Result of detecting contradictions.

    Attributes:
        memory_id: ID of the memory checked
        contradictions: List of contradicting memory IDs
        edges_created: Number of CONTRADICTS edges created
        error: Error message (if failed)
    """

    memory_id: str
    contradictions: list[str] = field(default_factory=list)
    edges_created: int = 0
    error: Optional[str] = None


@dataclass
class SupersedesResult:
    """Result of checking if a memory supersedes another.

    Attributes:
        memory_id: ID of the memory checked
        superseded_id: ID of the memory that was superseded (if any)
        edge_created: Whether a SUPERSEDES edge was created
        reason: Explanation of why supersession was determined
        error: Error message (if failed)
    """

    memory_id: str
    superseded_id: Optional[str] = None
    edge_created: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GraphNode:
    """Summarized memory info for graph visualization.

    Attributes:
        id: Memory ID
        content_preview: Truncated content (max 150 chars)
        memory_type: Memory type as string
        confidence: Confidence score
        importance: Importance score
    """

    id: str
    content_preview: str
    memory_type: str
    confidence: float
    importance: float


@dataclass
class GraphEdge:
    """Relationship info for graph visualization.

    Attributes:
        id: Edge identifier
        source_id: Source memory ID
        target_id: Target memory ID
        edge_type: Relationship type as string
        weight: Edge weight
    """

    id: int
    source_id: str
    target_id: str
    edge_type: str
    weight: float


@dataclass
class GraphPath:
    """Path from origin with scoring.

    Attributes:
        node_ids: List of memory IDs in traversal order
        edge_types: List of edge types traversed
        total_weight: Product of all edge weights
        relevance_score: Combined relevance score
    """

    node_ids: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    total_weight: float = 1.0
    relevance_score: float = 1.0


@dataclass
class GraphStats:
    """Summary statistics for graph inspection.

    Attributes:
        node_count: Total number of nodes discovered
        edge_count: Total number of edges discovered
        max_depth_reached: Maximum hop distance traversed
        origin_id: ID of the origin memory node
    """

    node_count: int
    edge_count: int
    max_depth_reached: int
    origin_id: str


@dataclass
class GraphInspectionResult:
    """Combined result for graph inspection tool.

    Attributes:
        success: Whether the inspection succeeded
        origin_id: ID of the origin memory node
        nodes: List of GraphNode objects discovered
        edges: List of GraphEdge objects discovered
        paths: List of GraphPath objects
        stats: GraphStats summary
        error: Error message (if failed)
    """

    success: bool
    origin_id: str = ""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    stats: Optional[GraphStats] = None
    error: Optional[str] = None

    def to_mermaid(self) -> str:
        """Generate Mermaid flowchart syntax for the graph."""
        if not self.nodes:
            return "flowchart TD\n    empty[No nodes found]"

        lines = ["flowchart TD"]

        # Auto-adjust label length based on node count
        node_count = len(self.nodes)
        if node_count <= 3:
            max_label_len = 150
        elif node_count <= 8:
            max_label_len = 100
        elif node_count <= 15:
            max_label_len = 60
        else:
            max_label_len = 40

        for node in self.nodes:
            label = node.content_preview.replace('"', "'").replace("\n", " ")
            if len(label) > max_label_len:
                label = label[: max_label_len - 3] + "..."
            lines.append(f'    {node.id}["{label}"]')

        for edge in self.edges:
            lines.append(f"    {edge.source_id} -->|{edge.edge_type}| {edge.target_id}")

        return "\n".join(lines)


class MemoryType(str, Enum):
    """Types of memories/documents that can be stored.

    Categorizes stored information by its semantic purpose:
    - DOCUMENT: Document chunk from file indexing (DocVec origin)
    - PREFERENCE: User preferences or settings (Recall origin)
    - DECISION: Design or implementation decisions
    - PATTERN: Recognized patterns or recurring behaviors
    - SESSION: Session-related information or conversation context
    - FILE_CONTEXT: File activity tracking
    - GOLDEN_RULE: High-confidence memories that are constitutional principles
    - FACT: Factual information
    """

    DOCUMENT = "document"
    PREFERENCE = "preference"
    DECISION = "decision"
    PATTERN = "pattern"
    SESSION = "session"
    FILE_CONTEXT = "file_context"
    GOLDEN_RULE = "golden_rule"
    FACT = "fact"


class RelationType(str, Enum):
    """Types of relationships between memories.

    Defines how memories can be linked together:
    - RELATES_TO: General relationship between memories
    - SUPERSEDES: One memory replaces another
    - CAUSED_BY: One memory was caused by another
    - CONTRADICTS: Memories contain conflicting information
    """

    RELATES_TO = "relates_to"
    SUPERSEDES = "supersedes"
    CAUSED_BY = "caused_by"
    CONTRADICTS = "contradicts"


class Relationship(BaseModel):
    """A relationship between two memories.

    Defines a directed edge in the memory graph connecting two memories
    with a specific relationship type and weight.

    Attributes:
        target_id: ID of the target memory
        relation: Type of relationship (RelationType enum)
        weight: Strength of the relationship from 0.0 to 1.0
        created_at: When the relationship was created
    """

    model_config = ConfigDict(frozen=True)

    target_id: str
    relation: RelationType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryDocument(BaseModel):
    """Unified type for document chunks and memories.

    MemoryDocument unifies DocVec's Chunk and Recall's Memory into a single
    type that represents any piece of "remembered information" in the system.

    Document chunks have provenance (source file, line numbers).
    Memories have context (when stored, importance).
    Both have:
    - Content and embeddings for semantic search
    - Confidence scoring for validation loop
    - Relationships for graph traversal
    - Metadata for additional context

    Attributes:
        id: Unique identifier
        content: The actual text content
        content_hash: SHA-256 hash for deduplication
        memory_type: Type categorization (MemoryType enum)
        namespace: Scope ('default', 'global', or 'project:{name}')

        # Provenance fields (from DocVec Chunk)
        source_file: Source file path (for document indexing)
        chunk_index: Index of this chunk within the source file
        start_line: Starting line number in source file
        end_line: Ending line number in source file

        # Context fields (from Recall Memory)
        importance: Importance score from 0.0 to 1.0
        confidence: Confidence score from 0.0 to 1.0 (validated through usage)
        created_at: When the document/memory was created
        accessed_at: When last accessed
        access_count: Number of times accessed

        # Relationships for memory graph
        relationships: List of relationships to other memories

        # Additional metadata
        metadata: Optional additional metadata dict
        embedding: Optional pre-computed embedding vector
    """

    model_config = ConfigDict(frozen=False)

    # Core fields
    id: str
    content: str
    content_hash: Optional[str] = None
    memory_type: MemoryType = MemoryType.DOCUMENT
    namespace: str = Field(default="default")

    # Provenance fields (from DocVec Chunk)
    source_file: Optional[str] = None
    chunk_index: int = 0
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    # Context fields (from Recall Memory)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0, ge=0)

    # Relationships for memory graph
    relationships: list[Relationship] = Field(default_factory=list)

    # Additional metadata and embedding
    metadata: Optional[dict[str, Any]] = None
    embedding: Optional[list[float]] = None

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace format.

        Valid formats:
        - 'default' (default namespace)
        - 'global' (global namespace)
        - 'project:{name}' (project-scoped namespace)
        """
        import re

        if v in ("default", "global"):
            return v
        if re.match(r"^project:[a-zA-Z0-9_-]+$", v):
            return v
        raise ValueError(
            f"Invalid namespace '{v}'. "
            "Must be 'default', 'global', or match 'project:{{name}}' pattern."
        )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace-only")
        return v

    def is_golden_rule(self) -> bool:
        """Check if this memory qualifies as a golden rule.

        Returns:
            True if confidence >= 0.9 or type is GOLDEN_RULE
        """
        return self.confidence >= 0.9 or self.memory_type == MemoryType.GOLDEN_RULE

    def can_be_promoted(self) -> bool:
        """Check if this memory can be promoted to golden rule.

        Only PREFERENCE, DECISION, and PATTERN types can be promoted.

        Returns:
            True if eligible for golden rule promotion
        """
        promotable_types = {MemoryType.PREFERENCE, MemoryType.DECISION, MemoryType.PATTERN}
        return self.memory_type in promotable_types and self.confidence >= 0.9

    def is_document_chunk(self) -> bool:
        """Check if this is a document chunk (has source file provenance).

        Returns:
            True if this represents a document chunk
        """
        return self.source_file is not None

    def add_relationship(
        self,
        target_id: str,
        relation: RelationType,
        weight: float = 1.0,
    ) -> "MemoryDocument":
        """Add a relationship to another memory.

        Args:
            target_id: ID of the target memory
            relation: Type of relationship
            weight: Strength of relationship (0.0-1.0)

        Returns:
            Self for method chaining
        """
        self.relationships.append(
            Relationship(target_id=target_id, relation=relation, weight=weight)
        )
        return self

    def record_access(self) -> "MemoryDocument":
        """Record an access to this memory.

        Updates accessed_at timestamp and increments access_count.

        Returns:
            Self for method chaining
        """
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self

    def update_confidence(self, delta: float) -> "MemoryDocument":
        """Update confidence score by a delta amount.

        Clamps the result to [0.0, 1.0] range.

        Args:
            delta: Amount to add (positive) or subtract (negative)

        Returns:
            Self for method chaining
        """
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        return self

    @classmethod
    def from_chunk(
        cls,
        chunk_id: str,
        content: str,
        source_file: str,
        chunk_index: int,
        content_hash: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: str = "default",
    ) -> "MemoryDocument":
        """Create a MemoryDocument from a DocVec-style chunk.

        Factory method for creating MemoryDocument from document indexing.

        Args:
            chunk_id: Unique identifier for the chunk
            content: The chunk content
            source_file: Path to the source file
            chunk_index: Index of this chunk in the document
            content_hash: Optional SHA-256 hash for deduplication
            start_line: Optional starting line number
            end_line: Optional ending line number
            metadata: Optional additional metadata
            namespace: Namespace for organizing (default: "default")

        Returns:
            MemoryDocument instance with DOCUMENT type
        """
        return cls(
            id=chunk_id,
            content=content,
            content_hash=content_hash,
            memory_type=MemoryType.DOCUMENT,
            namespace=namespace,
            source_file=source_file,
            chunk_index=chunk_index,
            start_line=start_line,
            end_line=end_line,
            importance=0.5,  # Document chunks start at neutral importance
            confidence=1.0,  # Documents have full confidence (they exist)
            metadata=metadata,
        )

    @classmethod
    def from_memory(
        cls,
        memory_id: str,
        content: str,
        memory_type: MemoryType,
        content_hash: Optional[str] = None,
        namespace: str = "global",
        importance: float = 0.5,
        confidence: float = 0.3,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "MemoryDocument":
        """Create a MemoryDocument from a Recall-style memory.

        Factory method for creating MemoryDocument from memory storage.

        Args:
            memory_id: Unique identifier for the memory
            content: The memory content
            memory_type: Type of memory (MemoryType enum)
            content_hash: Optional SHA-256 hash for deduplication
            namespace: Namespace for scoping (default: "global")
            importance: Importance score 0.0-1.0 (default: 0.5)
            confidence: Confidence score 0.0-1.0 (default: 0.3)
            metadata: Optional additional metadata

        Returns:
            MemoryDocument instance with specified memory type
        """
        return cls(
            id=memory_id,
            content=content,
            content_hash=content_hash,
            memory_type=memory_type,
            namespace=namespace,
            importance=importance,
            confidence=confidence,
            metadata=metadata,
        )

    def to_storage_format(self) -> dict[str, Any]:
        """Convert to ChromaStore-compatible format.

        Returns a dictionary suitable for use with ChromaStore's add_documents()
        method. Flattens metadata to ChromaDB-compatible primitive types.

        Returns:
            Dictionary with keys: id, content, and metadata dict
        """
        meta: dict[str, Any] = {
            "namespace": self.namespace,
            "doc_type": self.memory_type.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "chunk_index": self.chunk_index,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
        }

        if self.source_file is not None:
            meta["source_file"] = self.source_file

        if self.content_hash is not None:
            meta["doc_hash"] = self.content_hash

        if self.start_line is not None:
            meta["start_line"] = self.start_line

        if self.end_line is not None:
            meta["end_line"] = self.end_line

        # Flatten additional metadata (only primitive types for ChromaDB)
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[f"meta_{key}"] = value

        return {
            "id": self.id,
            "content": self.content,
            "metadata": meta,
        }

    @classmethod
    def from_storage_format(
        cls,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> "MemoryDocument":
        """Create a MemoryDocument from ChromaStore result format.

        Reconstructs a MemoryDocument from the format returned by ChromaStore
        queries. Handles the flattened metadata format.

        Args:
            doc_id: Document ID from storage
            content: Document content
            metadata: Metadata dict from ChromaStore
            embedding: Optional embedding vector

        Returns:
            MemoryDocument instance
        """
        # Parse memory type
        doc_type_str = metadata.get("doc_type", "document")
        try:
            memory_type = MemoryType(doc_type_str)
        except ValueError:
            memory_type = MemoryType.DOCUMENT

        # Parse timestamps
        created_at = datetime.now()
        if created_at_str := metadata.get("created_at"):
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                pass

        accessed_at = datetime.now()
        if accessed_at_str := metadata.get("accessed_at"):
            try:
                accessed_at = datetime.fromisoformat(accessed_at_str)
            except (ValueError, TypeError):
                pass

        # Extract custom metadata (prefixed with meta_)
        custom_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            if key.startswith("meta_"):
                custom_metadata[key[5:]] = value  # Remove meta_ prefix

        return cls(
            id=doc_id,
            content=content,
            content_hash=metadata.get("doc_hash"),
            memory_type=memory_type,
            namespace=metadata.get("namespace", "default"),
            source_file=metadata.get("source_file"),
            chunk_index=metadata.get("chunk_index", 0),
            start_line=metadata.get("start_line"),
            end_line=metadata.get("end_line"),
            importance=metadata.get("importance", 0.5),
            confidence=metadata.get("confidence", 1.0),
            created_at=created_at,
            accessed_at=accessed_at,
            access_count=metadata.get("access_count", 0),
            metadata=custom_metadata if custom_metadata else None,
            embedding=embedding,
        )
