"""Search result types for Theo.

This module defines search result types that track how results were found:
- MatchType: Enum for categorizing how a result was matched
- UnifiedSearchResult: Search result with match type tracking
- HybridSearchSummary: Summary of a hybrid search operation

These types extend the basic SearchResult from storage with additional
information about the search method used.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from theo.types.memory import MemoryDocument


class MatchType(str, Enum):
    """How a search result was matched.

    Categorizes the method used to find a result:
    - VECTOR: Found via vector similarity search
    - FTS: Found via full-text search
    - HYBRID: Found via combined vector + FTS search
    - EXACT: Found via exact match (hash, ID)
    - GRAPH: Found via graph traversal from another result
    """

    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"
    EXACT = "exact"
    GRAPH = "graph"


class UnifiedSearchResult(BaseModel):
    """Result from a search operation with match type tracking.

    Extends the basic search result concept with information about
    how the result was found, enabling better ranking and explanation.

    Attributes:
        document: The matched MemoryDocument
        distance: Raw distance from query (lower is better for cosine)
        similarity: Similarity score from 0.0 to 1.0 (higher is better)
        rank: Position in result set (0-indexed)
        match_type: How this result was found (MatchType enum)
        match_details: Optional details about the match (e.g., matched terms)
        graph_path: For GRAPH matches, the traversal path taken
        boosted: Whether this result was boosted by multiple match types
    """

    model_config = ConfigDict(frozen=False)

    document: MemoryDocument
    distance: float = Field(default=0.0, ge=0.0)
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: int = Field(default=0, ge=0)
    match_type: MatchType = MatchType.VECTOR
    match_details: Optional[dict[str, Any]] = None
    graph_path: Optional[list[str]] = None
    boosted: bool = False

    @classmethod
    def from_vector_search(
        cls,
        document: MemoryDocument,
        distance: float,
        rank: int = 0,
    ) -> "UnifiedSearchResult":
        """Create a search result from vector similarity search.

        Args:
            document: The matched MemoryDocument
            distance: Cosine distance (0 = identical, 2 = opposite)
            rank: Result position in the result set

        Returns:
            UnifiedSearchResult with VECTOR match type
        """
        # Convert cosine distance to similarity
        # Cosine distance: 0 = identical, 2 = opposite
        # Similarity: 1.0 = identical, 0.0 = opposite
        similarity = 1.0 - (distance / 2.0)

        return cls(
            document=document,
            distance=distance,
            similarity=similarity,
            rank=rank,
            match_type=MatchType.VECTOR,
        )

    @classmethod
    def from_fts_search(
        cls,
        document: MemoryDocument,
        distance: float,
        rank: int = 0,
        matched_terms: Optional[list[str]] = None,
    ) -> "UnifiedSearchResult":
        """Create a search result from full-text search.

        Args:
            document: The matched MemoryDocument
            distance: Distance score (may be from underlying vector search)
            rank: Result position in the result set
            matched_terms: Optional list of terms that matched

        Returns:
            UnifiedSearchResult with FTS match type
        """
        similarity = 1.0 - (distance / 2.0)

        match_details = None
        if matched_terms:
            match_details = {"matched_terms": matched_terms}

        return cls(
            document=document,
            distance=distance,
            similarity=similarity,
            rank=rank,
            match_type=MatchType.FTS,
            match_details=match_details,
        )

    @classmethod
    def from_graph_traversal(
        cls,
        document: MemoryDocument,
        relevance_score: float,
        rank: int = 0,
        path: Optional[list[str]] = None,
    ) -> "UnifiedSearchResult":
        """Create a search result from graph traversal.

        Args:
            document: The matched MemoryDocument
            relevance_score: Computed relevance (0.0-1.0)
            rank: Result position in the result set
            path: List of memory IDs in the traversal path

        Returns:
            UnifiedSearchResult with GRAPH match type
        """
        return cls(
            document=document,
            distance=0.0,  # Not applicable for graph
            similarity=relevance_score,
            rank=rank,
            match_type=MatchType.GRAPH,
            graph_path=path,
        )

    def boost(self, additional_similarity: float = 0.1) -> "UnifiedSearchResult":
        """Create a boosted copy of this result.

        Used when a result is found via multiple methods (e.g., both
        vector and FTS).

        Args:
            additional_similarity: Amount to boost similarity (default 0.1)

        Returns:
            New UnifiedSearchResult with boosted similarity
        """
        return UnifiedSearchResult(
            document=self.document,
            distance=self.distance,
            similarity=min(1.0, self.similarity + additional_similarity),
            rank=self.rank,
            match_type=MatchType.HYBRID,
            match_details=self.match_details,
            graph_path=self.graph_path,
            boosted=True,
        )


class HybridSearchSummary(BaseModel):
    """Summary of a hybrid search operation.

    Provides metadata about how results were found in a hybrid search,
    useful for debugging and explaining search behavior.

    Attributes:
        results: List of UnifiedSearchResult objects
        total_vector_matches: Number of results from vector search
        total_fts_matches: Number of results from FTS
        fts_used: Whether FTS was actually used (may be skipped if vector results are good)
        fts_triggered_reason: Why FTS was triggered (if applicable)
        query_text: The original query text (if provided)
        search_duration_ms: Time taken for the search in milliseconds
    """

    model_config = ConfigDict(frozen=False)

    results: list[UnifiedSearchResult] = Field(default_factory=list)
    total_vector_matches: int = 0
    total_fts_matches: int = 0
    fts_used: bool = False
    fts_triggered_reason: Optional[str] = None
    query_text: Optional[str] = None
    search_duration_ms: Optional[float] = None

    @property
    def total_results(self) -> int:
        """Get total number of results."""
        return len(self.results)

    @property
    def boosted_count(self) -> int:
        """Get number of results that were boosted by multiple match types."""
        return sum(1 for r in self.results if r.boosted)

    def get_by_match_type(self, match_type: MatchType) -> list[UnifiedSearchResult]:
        """Get results filtered by match type.

        Args:
            match_type: The match type to filter by

        Returns:
            List of results with the specified match type
        """
        return [r for r in self.results if r.match_type == match_type]
