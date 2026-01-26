"""Unified type system for Theo.

This module provides the unified type system that treats document chunks
and memories as the same underlying concept: "remembered information."

The key types are:
- MemoryDocument: Unified type combining DocVec's Chunk and Recall's Memory
- MemoryType: Enum categorizing types of stored information
- RelationType: Enum for relationships between memories
- UnifiedSearchResult: Search result with match type tracking
- MatchType: How a search result was found

Document compatibility:
- Document: Re-exported from storage for backward compatibility

Example:
    >>> from theo.types import MemoryDocument, MemoryType
    >>> # Create from document indexing
    >>> chunk = MemoryDocument.from_chunk(
    ...     chunk_id="doc_001",
    ...     content="Function definition here",
    ...     source_file="main.py",
    ...     chunk_index=0,
    ... )
    >>> # Create from memory storage
    >>> memory = MemoryDocument.from_memory(
    ...     memory_id="mem_001",
    ...     content="User prefers dark mode",
    ...     memory_type=MemoryType.PREFERENCE,
    ... )
    >>> # Convert to storage format
    >>> storage_dict = chunk.to_storage_format()
"""

from theo.types.document import Document
from theo.types.memory import (
    ApplyResult,
    ContradictionResult,
    EdgeForgetResult,
    GraphEdge,
    GraphInspectionResult,
    GraphNode,
    GraphPath,
    GraphStats,
    MemoryDocument,
    MemoryType,
    OutcomeResult,
    Relationship,
    RelationType,
    SupersedesResult,
)
from theo.types.search_result import (
    HybridSearchSummary,
    MatchType,
    UnifiedSearchResult,
)

__all__ = [
    # Core unified type
    "MemoryDocument",
    "MemoryType",
    "RelationType",
    "Relationship",
    # Operation result types
    "ApplyResult",
    "OutcomeResult",
    "EdgeForgetResult",
    "ContradictionResult",
    "SupersedesResult",
    # Graph inspection types
    "GraphNode",
    "GraphEdge",
    "GraphPath",
    "GraphStats",
    "GraphInspectionResult",
    # Search result types
    "UnifiedSearchResult",
    "MatchType",
    "HybridSearchSummary",
    # Backward compatibility
    "Document",
]
