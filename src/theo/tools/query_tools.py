"""Query tools for the Theo MCP server.

This module provides MCP tools for semantic search operations:
- search: Basic semantic search
- search_with_filters: Search with metadata filtering
- search_with_budget: Search with token budget limiting

Uses DaemonClient for query embedding generation and integrates with
the ValidationLoop for confidence-based result boosting.
"""

import logging
from typing import Any, Optional

from theo.daemon import DaemonClient
from theo.storage.sqlite_store import SQLiteStore, SearchResult as SQLiteSearchResult
from theo.validation import FeedbackCollector, UsageFeedback
from theo.validation.feedback import FeedbackType

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)

# Token estimation: ~4 characters per token
CHARS_PER_TOKEN = 4


class QueryTools:
    """Tool implementations for semantic search operations.

    Provides search, search_with_filters, and search_with_budget operations
    that use the DaemonClient for query embedding generation.

    Optionally integrates with FeedbackCollector to track which results
    are used, enabling the validation loop to adjust confidence scores.

    Architectural justification:
    - DaemonClient handles query embedding asynchronously
    - SQLiteStore provides vector, FTS, and hybrid search via sqlite-vec + FTS5
    - FeedbackCollector aggregates usage signals for validation

    Args:
        daemon_client: Client for daemon embedding operations
        store: SQLiteStore instance for all storage operations
        feedback_collector: Optional collector for validation feedback

    Example:
        >>> tools = QueryTools(daemon_client, store, feedback_collector)
        >>> result = await tools.search("How do I implement authentication?", n_results=5)
        >>> for hit in result["data"]["results"]:
        ...     print(f"{hit['similarity']:.2f}: {hit['content'][:50]}...")
    """

    def __init__(
        self,
        daemon_client: DaemonClient,
        store: SQLiteStore,
        feedback_collector: Optional[FeedbackCollector] = None,
    ) -> None:
        """Initialize QueryTools with dependencies.

        Args:
            daemon_client: Client for daemon embedding operations
            store: SQLiteStore instance for all storage operations
            feedback_collector: Optional collector for validation feedback
        """
        self._daemon = daemon_client
        self._store = store
        self._feedback = feedback_collector

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Uses chars/4 approximation consistent with existing codebase patterns.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // CHARS_PER_TOKEN

    def _result_to_dict(self, result: SQLiteSearchResult, rank: int) -> dict[str, Any]:
        """Convert SQLiteSearchResult to response dictionary.

        Args:
            result: SQLiteSearchResult from SQLiteStore
            rank: Position in result set (0-indexed)

        Returns:
            Dictionary with result data
        """
        # SQLiteStore.SQLiteSearchResult is a dataclass with direct attributes
        return {
            "id": result.id,
            "content": result.content,
            "source_file": result.source_file,
            "chunk_index": result.chunk_index,
            "namespace": result.namespace,
            "doc_type": result.memory_type,
            "confidence": result.confidence,
            "similarity": result.score,
            "distance": 1.0 - result.score,  # Convert similarity to distance
            "rank": rank,
            "token_estimate": self._estimate_tokens(result.content),
        }

    def _record_feedback(
        self,
        results: list[SQLiteSearchResult],
        query: str,
    ) -> None:
        """Record search results for validation feedback.

        Args:
            results: Search results to record
            query: The search query
        """
        if self._feedback is None:
            return

        for result in results:
            # Record implicit usage - documents were retrieved
            # SQLiteStore.SQLiteSearchResult has id directly (not nested in document)
            feedback = UsageFeedback(
                doc_id=result.id,
                was_helpful=True,  # Retrieved = potentially helpful
                feedback_type=FeedbackType.IMPLICIT_USED,
                context=query,
            )
            self._feedback.collect(feedback)

    async def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Perform semantic search for a query.

        Generates query embedding via daemon and searches SQLiteStore
        for similar documents using sqlite-vec.

        Args:
            query: Search query string
            n_results: Maximum number of results to return (default: 5)

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with results, total, query
            - error: Error message if operation failed
        """
        try:
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty",
                }

            # Generate query embedding via daemon
            embed_result = self._daemon.send("search", query=query)

            if not embed_result.get("success"):
                return {
                    "success": False,
                    "error": f"Query embedding failed: {embed_result.get('error', 'Unknown error')}",
                }

            query_embedding = embed_result.get("data", {}).get("embedding", [])

            if not query_embedding:
                return {
                    "success": False,
                    "error": "No query embedding returned",
                }

            # Search using SQLiteStore.search_vector (KNN via sqlite-vec)
            results = self._store.search_vector(
                embedding=query_embedding,
                n_results=n_results,
            )

            # Record feedback for validation loop
            self._record_feedback(results, query)

            # Convert results to response format
            results_data = [
                self._result_to_dict(result, i)
                for i, result in enumerate(results)
            ]

            # Calculate total tokens in results
            total_tokens = sum(r["token_estimate"] for r in results_data)

            return {
                "success": True,
                "data": {
                    "results": results_data,
                    "total": len(results_data),
                    "query": query,
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"search failed for query '{query}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def search_with_filters(
        self,
        query: str,
        filters: dict[str, Any],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Perform semantic search with metadata filtering.

        Args:
            query: Search query string
            filters: Metadata filters (e.g., {"namespace": "project1"}, {"doc_type": "memory"})
            n_results: Maximum number of results to return (default: 5)

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with results, total, query, filters
            - error: Error message if operation failed
        """
        try:
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty",
                }

            # Generate query embedding via daemon
            embed_result = self._daemon.send("search", query=query)

            if not embed_result.get("success"):
                return {
                    "success": False,
                    "error": f"Query embedding failed: {embed_result.get('error', 'Unknown error')}",
                }

            query_embedding = embed_result.get("data", {}).get("embedding", [])

            if not query_embedding:
                return {
                    "success": False,
                    "error": "No query embedding returned",
                }

            # Search using SQLiteStore.search_vector with filters
            # SQLiteStore.search_vector supports where filters
            results = self._store.search_vector(
                embedding=query_embedding,
                n_results=n_results,
                where=filters,
            )

            # Record feedback for validation loop
            self._record_feedback(results, query)

            # Convert results to response format
            results_data = [
                self._result_to_dict(result, i)
                for i, result in enumerate(results)
            ]

            # Calculate total tokens in results
            total_tokens = sum(r["token_estimate"] for r in results_data)

            return {
                "success": True,
                "data": {
                    "results": results_data,
                    "total": len(results_data),
                    "query": query,
                    "filters": filters,
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"search_with_filters failed for query '{query}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def search_with_budget(
        self,
        query: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Search and return results within a token budget.

        Fetches results and truncates to fit within the specified
        token budget.

        Args:
            query: Search query string
            max_tokens: Maximum total tokens allowed in results

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with results, total, query, tokens_used, budget_remaining
            - error: Error message if operation failed
        """
        try:
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty",
                }

            if max_tokens <= 0:
                return {
                    "success": False,
                    "error": "max_tokens must be positive",
                }

            # Generate query embedding via daemon
            embed_result = self._daemon.send("search", query=query)

            if not embed_result.get("success"):
                return {
                    "success": False,
                    "error": f"Query embedding failed: {embed_result.get('error', 'Unknown error')}",
                }

            query_embedding = embed_result.get("data", {}).get("embedding", [])

            if not query_embedding:
                return {
                    "success": False,
                    "error": "No query embedding returned",
                }

            # Fetch more results than needed to ensure we have enough after budget filtering
            results = self._store.search_vector(
                embedding=query_embedding,
                n_results=20,  # Fetch extra for budget filtering
            )

            # Filter results to fit within token budget
            filtered_results: list[dict[str, Any]] = []
            tokens_used = 0

            for i, result in enumerate(results):
                result_dict = self._result_to_dict(result, i)
                result_tokens = result_dict["token_estimate"]

                if tokens_used + result_tokens <= max_tokens:
                    filtered_results.append(result_dict)
                    tokens_used += result_tokens
                else:
                    # Budget exhausted
                    break

            # Record feedback for validation loop (only for included results)
            included_results = results[: len(filtered_results)]
            self._record_feedback(included_results, query)

            return {
                "success": True,
                "data": {
                    "results": filtered_results,
                    "total": len(filtered_results),
                    "query": query,
                    "tokens_used": tokens_used,
                    "budget_remaining": max_tokens - tokens_used,
                    "max_tokens": max_tokens,
                },
            }

        except Exception as e:
            logger.error(f"search_with_budget failed for query '{query}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        fts_weight: float = 0.3,
        min_vector_similarity: float = 0.5,
    ) -> dict[str, Any]:
        """Perform hybrid search combining vector and text matching.

        Uses SQLiteStore's search_hybrid which combines vector similarity
        (sqlite-vec) with full-text search (FTS5) using RRF scoring.

        Args:
            query: Search query string
            n_results: Maximum number of results to return (default: 5)
            fts_weight: Weight for FTS results in final ranking (0.0-1.0)
            min_vector_similarity: Minimum similarity to skip FTS augmentation (unused)

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with results, fts_used, vector_count, fts_count
            - error: Error message if operation failed
        """
        try:
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Query cannot be empty",
                }

            # Generate query embedding via daemon
            embed_result = self._daemon.send("search", query=query)

            if not embed_result.get("success"):
                return {
                    "success": False,
                    "error": f"Query embedding failed: {embed_result.get('error', 'Unknown error')}",
                }

            query_embedding = embed_result.get("data", {}).get("embedding", [])

            if not query_embedding:
                return {
                    "success": False,
                    "error": "No query embedding returned",
                }

            # Perform hybrid search using SQLiteStore.search_hybrid
            # Note: vector_weight = 1.0 - fts_weight
            vector_weight = 1.0 - fts_weight
            results = self._store.search_hybrid(
                embedding=query_embedding,
                query=query,
                n_results=n_results,
                vector_weight=vector_weight,
            )

            # Record feedback for validation loop
            self._record_feedback(results, query)

            # Convert results to response format
            results_data = [
                self._result_to_dict(result, i)
                for i, result in enumerate(results)
            ]

            # Calculate total tokens in results
            total_tokens = sum(r["token_estimate"] for r in results_data)

            return {
                "success": True,
                "data": {
                    "results": results_data,
                    "total": len(results_data),
                    "query": query,
                    "fts_used": True,  # search_hybrid always uses FTS
                    "vector_count": len(results),  # Approximate
                    "fts_count": len(results),  # Approximate
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"hybrid_search failed for query '{query}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
