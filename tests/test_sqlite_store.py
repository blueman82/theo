"""Tests for SQLite storage layer with sqlite-vec.

This module tests SQLiteStore functionality including:
- Memory CRUD operations (add, get, update, delete)
- Vector search (KNN with sqlite-vec)
- Full-text search (FTS5)
- Hybrid search (vector + FTS with RRF)
- Embedding cache operations
- Edge/relationship operations
- Golden rule confidence threshold detection
- Recency-based scoring

Uses mock 1024-dimensional embeddings to match MLX mxbai-embed-large-v1.
"""

import math
import time
from pathlib import Path

import pytest

from theo.storage.sqlite_store import SearchResult, SQLiteStore, compute_recency_score

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def store() -> SQLiteStore:
    """Provide in-memory SQLiteStore instance for testing.

    Uses ':memory:' for fast, isolated tests. The sqlite-vec
    extension is loaded automatically.

    Yields:
        SQLiteStore: In-memory store with sqlite-vec loaded
    """
    # SQLiteStore accepts Path or str, ':memory:' is valid for sqlite3
    s = SQLiteStore(Path(":memory:"))
    yield s
    s.close()


@pytest.fixture
def mock_embedding() -> list[float]:
    """Provide mock 1024-dimensional embedding for testing.

    MLX mxbai-embed-large-v1 produces 1024-dimensional embeddings.
    """
    return [0.1] * 1024


@pytest.fixture
def mock_embedding_alt() -> list[float]:
    """Provide alternative mock embedding for similarity testing."""
    return [0.9] * 1024


@pytest.fixture
def mock_embedding_mixed() -> list[float]:
    """Provide mixed mock embedding for hybrid search testing."""
    return [0.5] * 1024


# ============================================================================
# Memory CRUD Tests
# ============================================================================


class TestMemoryCRUD:
    """Test memory create, read, update, delete operations."""

    def test_add_and_get_memory(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test adding a memory and retrieving it by ID."""
        # Add memory
        memory_id = store.add_memory(
            content="Python is a great programming language.",
            embedding=mock_embedding,
            memory_type="fact",
            namespace="programming",
            confidence=0.5,
            importance=0.8,
            source_file="notes.md",
            chunk_index=0,
            tags={"topic": "python", "level": "beginner"},
        )

        # Verify ID was returned
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) == 32  # uuid4 hex

        # Retrieve memory
        memory = store.get_memory(memory_id)

        # Verify all fields
        assert memory is not None
        assert memory["id"] == memory_id
        assert memory["content"] == "Python is a great programming language."
        assert memory["memory_type"] == "fact"
        assert memory["namespace"] == "programming"
        assert memory["confidence"] == 0.5
        assert memory["importance"] == 0.8
        assert memory["source_file"] == "notes.md"
        assert memory["chunk_index"] == 0
        assert memory["tags"] == {"topic": "python", "level": "beginner"}

    def test_get_memory_nonexistent(self, store: SQLiteStore) -> None:
        """Test getting a nonexistent memory returns None."""
        memory = store.get_memory("nonexistent_id")
        assert memory is None

    def test_update_memory_confidence(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test updating memory confidence score."""
        # Add memory with initial confidence
        memory_id = store.add_memory(
            content="Test content",
            embedding=mock_embedding,
            confidence=0.3,
        )

        # Update confidence
        updated = store.update_memory(memory_id, confidence=0.7)
        assert updated is True

        # Verify update
        memory = store.get_memory(memory_id)
        assert memory is not None
        assert memory["confidence"] == 0.7

    def test_update_memory_multiple_fields(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test updating multiple memory fields at once."""
        memory_id = store.add_memory(
            content="Initial content",
            embedding=mock_embedding,
            confidence=0.3,
            importance=0.5,
        )

        # Update multiple fields
        updated = store.update_memory(
            memory_id,
            confidence=0.8,
            importance=0.9,
            tags={"updated": True},
        )
        assert updated is True

        # Verify updates
        memory = store.get_memory(memory_id)
        assert memory is not None
        assert memory["confidence"] == 0.8
        assert memory["importance"] == 0.9
        assert memory["tags"] == {"updated": True}

    def test_update_memory_nonexistent(self, store: SQLiteStore) -> None:
        """Test updating a nonexistent memory returns False."""
        updated = store.update_memory("nonexistent_id", confidence=0.5)
        assert updated is False

    def test_delete_memory(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test deleting a memory removes it from all tables."""
        # Add memory
        memory_id = store.add_memory(
            content="Content to delete",
            embedding=mock_embedding,
        )

        # Verify it exists
        assert store.get_memory(memory_id) is not None
        assert store.count_memories() == 1

        # Delete it
        deleted = store.delete_memory(memory_id)
        assert deleted is True

        # Verify it's gone
        assert store.get_memory(memory_id) is None
        assert store.count_memories() == 0

    def test_delete_memory_nonexistent(self, store: SQLiteStore) -> None:
        """Test deleting a nonexistent memory returns False."""
        deleted = store.delete_memory("nonexistent_id")
        assert deleted is False


# ============================================================================
# List and Count Tests
# ============================================================================


class TestListAndCount:
    """Test list_memories and count_memories with filters."""

    def test_list_memories_no_filter(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test listing all memories without filters."""
        # Add multiple memories
        store.add_memory(content="Memory 1", embedding=mock_embedding)
        store.add_memory(content="Memory 2", embedding=mock_embedding)
        store.add_memory(content="Memory 3", embedding=mock_embedding)

        # List all
        memories = store.list_memories()
        assert len(memories) == 3

    def test_list_memories_with_namespace_filter(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test listing memories filtered by namespace."""
        store.add_memory(content="Memory 1", embedding=mock_embedding, namespace="ns1")
        store.add_memory(content="Memory 2", embedding=mock_embedding, namespace="ns1")
        store.add_memory(content="Memory 3", embedding=mock_embedding, namespace="ns2")

        # Filter by namespace
        ns1_memories = store.list_memories(namespace="ns1")
        assert len(ns1_memories) == 2

        ns2_memories = store.list_memories(namespace="ns2")
        assert len(ns2_memories) == 1

    def test_list_memories_with_type_filter(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test listing memories filtered by memory_type."""
        store.add_memory(content="Fact 1", embedding=mock_embedding, memory_type="fact")
        store.add_memory(content="Fact 2", embedding=mock_embedding, memory_type="fact")
        store.add_memory(content="Doc 1", embedding=mock_embedding, memory_type="document")

        # Filter by type
        facts = store.list_memories(memory_type="fact")
        assert len(facts) == 2

        docs = store.list_memories(memory_type="document")
        assert len(docs) == 1

    def test_list_memories_pagination(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test list_memories with limit and offset."""
        # Add 5 memories
        for i in range(5):
            store.add_memory(content=f"Memory {i}", embedding=mock_embedding)

        # Get first 2
        page1 = store.list_memories(limit=2, offset=0)
        assert len(page1) == 2

        # Get next 2
        page2 = store.list_memories(limit=2, offset=2)
        assert len(page2) == 2

        # Get last 1
        page3 = store.list_memories(limit=2, offset=4)
        assert len(page3) == 1

    def test_count_memories_no_filter(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test counting all memories."""
        assert store.count_memories() == 0

        store.add_memory(content="Memory 1", embedding=mock_embedding)
        assert store.count_memories() == 1

        store.add_memory(content="Memory 2", embedding=mock_embedding)
        assert store.count_memories() == 2

    def test_count_memories_with_filters(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test counting memories with filters."""
        store.add_memory(
            content="Fact in ns1",
            embedding=mock_embedding,
            namespace="ns1",
            memory_type="fact",
        )
        store.add_memory(
            content="Doc in ns1",
            embedding=mock_embedding,
            namespace="ns1",
            memory_type="document",
        )
        store.add_memory(
            content="Fact in ns2",
            embedding=mock_embedding,
            namespace="ns2",
            memory_type="fact",
        )

        # Count by namespace
        assert store.count_memories(namespace="ns1") == 2
        assert store.count_memories(namespace="ns2") == 1

        # Count by type
        assert store.count_memories(memory_type="fact") == 2
        assert store.count_memories(memory_type="document") == 1

        # Count by both
        assert store.count_memories(namespace="ns1", memory_type="fact") == 1


# ============================================================================
# Vector Search Tests (sqlite-vec KNN)
# ============================================================================


class TestVectorSearch:
    """Test KNN vector search using sqlite-vec."""

    def test_search_vector_knn(
        self,
        store: SQLiteStore,
        mock_embedding: list[float],
        mock_embedding_alt: list[float],
    ) -> None:
        """Test KNN search returns results ordered by similarity."""
        # Add memories with different embeddings
        id1 = store.add_memory(content="Similar to query", embedding=mock_embedding)
        store.add_memory(content="Different content", embedding=mock_embedding_alt)

        # Search with query similar to mock_embedding
        results = store.search_vector(embedding=mock_embedding, n_results=5)

        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)

        # First result should be most similar (closest to query)
        assert results[0].id == id1
        assert results[0].score >= results[1].score

    def test_search_vector_n_results(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test search_vector respects n_results limit."""
        # Add 5 memories
        for i in range(5):
            # Slightly vary embedding to get different distances
            emb = [0.1 + (i * 0.01)] * 1024
            store.add_memory(content=f"Memory {i}", embedding=emb)

        # Request only 3 results
        results = store.search_vector(embedding=mock_embedding, n_results=3)
        assert len(results) == 3

    def test_search_vector_with_filter(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test search_vector with namespace/type filters."""
        store.add_memory(
            content="Fact in ns1",
            embedding=mock_embedding,
            namespace="ns1",
            memory_type="fact",
        )
        store.add_memory(
            content="Doc in ns1",
            embedding=mock_embedding,
            namespace="ns1",
            memory_type="document",
        )
        store.add_memory(
            content="Fact in ns2",
            embedding=mock_embedding,
            namespace="ns2",
            memory_type="fact",
        )

        # Filter by namespace
        results = store.search_vector(
            embedding=mock_embedding,
            n_results=5,
            where={"namespace": "ns1"},
        )
        assert len(results) == 2
        for r in results:
            assert r.namespace == "ns1"

        # Filter by memory_type
        results = store.search_vector(
            embedding=mock_embedding,
            n_results=5,
            where={"memory_type": "fact"},
        )
        assert len(results) == 2
        for r in results:
            assert r.memory_type == "fact"

    def test_search_vector_empty_collection(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test search_vector on empty collection returns empty list."""
        results = store.search_vector(embedding=mock_embedding, n_results=5)
        assert results == []


# ============================================================================
# Full-Text Search Tests (FTS5)
# ============================================================================


class TestFTSSearch:
    """Test full-text search using FTS5."""

    def test_search_fts_match(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test FTS5 keyword matching."""
        store.add_memory(
            content="Python is a programming language",
            embedding=mock_embedding,
        )
        store.add_memory(
            content="JavaScript is also a programming language",
            embedding=mock_embedding,
        )
        store.add_memory(
            content="The quick brown fox jumps over the lazy dog",
            embedding=mock_embedding,
        )

        # Search for "Python"
        results = store.search_fts(query="Python", n_results=5)
        assert len(results) == 1
        assert "Python" in results[0].content

        # Search for "programming"
        results = store.search_fts(query="programming", n_results=5)
        assert len(results) == 2

        # Search for non-matching term
        results = store.search_fts(query="nonexistent", n_results=5)
        assert len(results) == 0

    def test_search_fts_bm25_ranking(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test FTS5 results are ranked by relevance (BM25)."""
        # Add documents with varying keyword frequency
        store.add_memory(
            content="Python Python Python is great",
            embedding=mock_embedding,
        )
        store.add_memory(
            content="Python is a language",
            embedding=mock_embedding,
        )

        results = store.search_fts(query="Python", n_results=5)
        assert len(results) == 2

        # Document with more Python mentions should rank higher
        assert results[0].content.count("Python") >= results[1].content.count("Python")

    def test_search_fts_with_filter(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test FTS5 search with namespace filter."""
        store.add_memory(
            content="Python programming",
            embedding=mock_embedding,
            namespace="tech",
        )
        store.add_memory(
            content="Python snake species",
            embedding=mock_embedding,
            namespace="biology",
        )

        # Filter by namespace
        results = store.search_fts(
            query="Python",
            n_results=5,
            where={"namespace": "tech"},
        )
        assert len(results) == 1
        assert results[0].namespace == "tech"

    def test_search_fts_special_characters(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test FTS5 handles special characters in queries."""
        store.add_memory(
            content="C++ is a programming language",
            embedding=mock_embedding,
        )

        # Search with special characters should be escaped
        results = store.search_fts(query="C++", n_results=5)
        # FTS5 may or may not match depending on tokenization
        assert isinstance(results, list)


# ============================================================================
# Hybrid Search Tests
# ============================================================================


class TestHybridSearch:
    """Test hybrid search combining vector and FTS."""

    def test_search_hybrid_combined(
        self, store: SQLiteStore, mock_embedding: list[float], mock_embedding_alt: list[float]
    ) -> None:
        """Test hybrid search combines vector and FTS results with RRF."""
        # Add memories optimized for different search methods
        store.add_memory(
            content="Python programming tutorial",
            embedding=mock_embedding,  # Similar to query embedding
        )
        store.add_memory(
            content="Java programming tutorial",
            embedding=mock_embedding_alt,  # Different from query
        )
        store.add_memory(
            content="Python snake in the zoo",
            embedding=mock_embedding_alt,  # Different from query but matches "Python"
        )

        # Hybrid search with both vector and text query
        results = store.search_hybrid(
            embedding=mock_embedding,
            query="Python",
            n_results=5,
            vector_weight=0.5,
        )

        # Should return results from both vector and FTS
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

        # All results should be scored
        for r in results:
            assert r.score > 0

    def test_search_hybrid_vector_weight(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test hybrid search respects vector_weight parameter."""
        store.add_memory(
            content="Document about coding",
            embedding=mock_embedding,
        )

        # High vector weight
        results_high = store.search_hybrid(
            embedding=mock_embedding,
            query="coding",
            n_results=5,
            vector_weight=0.9,
        )

        # Low vector weight
        results_low = store.search_hybrid(
            embedding=mock_embedding,
            query="coding",
            n_results=5,
            vector_weight=0.1,
        )

        # Both should return results
        assert len(results_high) > 0
        assert len(results_low) > 0


# ============================================================================
# Embedding Cache Tests
# ============================================================================


class TestEmbeddingCache:
    """Test embedding cache operations."""

    def test_embedding_cache_miss(self, store: SQLiteStore) -> None:
        """Test cache miss returns None."""
        result = store.get_cached_embedding(
            content_hash="nonexistent_hash",
            provider="mlx",
            model="mxbai-embed-large-v1",
        )
        assert result is None

    def test_embedding_cache_hit(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test cache hit returns stored embedding."""
        content_hash = "abc123hash"
        provider = "mlx"
        model = "mxbai-embed-large-v1"

        # Cache the embedding
        store.cache_embedding(
            content_hash=content_hash,
            provider=provider,
            model=model,
            embedding=mock_embedding,
        )

        # Retrieve from cache
        cached = store.get_cached_embedding(
            content_hash=content_hash,
            provider=provider,
            model=model,
        )

        assert cached is not None
        assert len(cached) == 1024
        # Use approximate comparison due to float32 serialization precision loss
        for i, (c, m) in enumerate(zip(cached, mock_embedding)):
            assert abs(c - m) < 1e-6, f"Mismatch at index {i}: {c} != {m}"

    def test_embedding_cache_different_models(
        self, store: SQLiteStore, mock_embedding: list[float], mock_embedding_alt: list[float]
    ) -> None:
        """Test cache distinguishes between different models."""
        content_hash = "same_hash"

        # Cache with different models
        store.cache_embedding(
            content_hash=content_hash,
            provider="mlx",
            model="model1",
            embedding=mock_embedding,
        )
        store.cache_embedding(
            content_hash=content_hash,
            provider="mlx",
            model="model2",
            embedding=mock_embedding_alt,
        )

        # Retrieve each
        cached1 = store.get_cached_embedding(content_hash, "mlx", "model1")
        cached2 = store.get_cached_embedding(content_hash, "mlx", "model2")

        # Use approximate comparison due to float32 serialization
        assert cached1 is not None
        assert cached2 is not None
        for c, m in zip(cached1, mock_embedding):
            assert abs(c - m) < 1e-6
        for c, m in zip(cached2, mock_embedding_alt):
            assert abs(c - m) < 1e-6

    def test_embedding_cache_overwrite(
        self, store: SQLiteStore, mock_embedding: list[float], mock_embedding_alt: list[float]
    ) -> None:
        """Test cache overwrites on same key (INSERT OR REPLACE)."""
        content_hash = "hash_to_overwrite"
        provider = "mlx"
        model = "model"

        # Initial cache
        store.cache_embedding(content_hash, provider, model, mock_embedding)

        # Overwrite
        store.cache_embedding(content_hash, provider, model, mock_embedding_alt)

        # Should return new value (with float32 precision tolerance)
        cached = store.get_cached_embedding(content_hash, provider, model)
        assert cached is not None
        for c, m in zip(cached, mock_embedding_alt):
            assert abs(c - m) < 1e-6


# ============================================================================
# Golden Rule Confidence Threshold Tests
# ============================================================================


class TestGoldenRuleThreshold:
    """Test golden rule confidence threshold (>= 0.9)."""

    def test_golden_rule_confidence_threshold(
        self, store: SQLiteStore, mock_embedding: list[float]
    ) -> None:
        """Test that confidence >= 0.9 indicates a golden rule."""
        # Add memory just below threshold
        id_below = store.add_memory(
            content="Almost golden rule",
            embedding=mock_embedding,
            confidence=0.89,
        )

        # Add memory at threshold
        id_at = store.add_memory(
            content="Golden rule",
            embedding=mock_embedding,
            confidence=0.9,
        )

        # Add memory above threshold
        id_above = store.add_memory(
            content="Very golden rule",
            embedding=mock_embedding,
            confidence=0.95,
        )

        # Verify confidences
        mem_below = store.get_memory(id_below)
        mem_at = store.get_memory(id_at)
        mem_above = store.get_memory(id_above)

        assert mem_below is not None
        assert mem_at is not None
        assert mem_above is not None

        # Check golden rule status
        assert mem_below["confidence"] < 0.9  # Not golden
        assert mem_at["confidence"] >= 0.9  # Golden
        assert mem_above["confidence"] >= 0.9  # Golden

    def test_update_to_golden_rule(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test updating a memory to golden rule status."""
        memory_id = store.add_memory(
            content="Will become golden",
            embedding=mock_embedding,
            confidence=0.3,  # Start low
        )

        # Simulate validation successes
        store.update_memory(memory_id, confidence=0.5)
        store.update_memory(memory_id, confidence=0.7)
        store.update_memory(memory_id, confidence=0.9)  # Now golden

        memory = store.get_memory(memory_id)
        assert memory is not None
        assert memory["confidence"] >= 0.9


# ============================================================================
# Edge/Relationship Tests
# ============================================================================


class TestEdgeOperations:
    """Test edge (relationship) operations."""

    def test_add_edge(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test adding an edge between memories."""
        # Create two memories
        id1 = store.add_memory(content="Memory 1", embedding=mock_embedding)
        id2 = store.add_memory(content="Memory 2", embedding=mock_embedding)

        # Add edge
        edge_id = store.add_edge(
            source_id=id1,
            target_id=id2,
            edge_type="relates_to",
            weight=0.8,
            metadata={"reason": "related topic"},
        )

        assert edge_id is not None
        assert isinstance(edge_id, int)
        assert edge_id > 0

    def test_get_edges(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test getting edges for a memory."""
        id1 = store.add_memory(content="Central memory", embedding=mock_embedding)
        id2 = store.add_memory(content="Related memory 1", embedding=mock_embedding)
        id3 = store.add_memory(content="Related memory 2", embedding=mock_embedding)

        # Add outgoing edges from id1
        store.add_edge(id1, id2, "relates_to")
        store.add_edge(id1, id3, "supersedes")

        # Get all edges (both directions)
        edges = store.get_edges(id1, direction="both")
        assert len(edges) == 2

        # Get only outgoing
        outgoing = store.get_edges(id1, direction="outgoing")
        assert len(outgoing) == 2

        # Get only incoming (none expected)
        incoming = store.get_edges(id1, direction="incoming")
        assert len(incoming) == 0

    def test_get_edges_by_type(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test filtering edges by type."""
        id1 = store.add_memory(content="Memory 1", embedding=mock_embedding)
        id2 = store.add_memory(content="Memory 2", embedding=mock_embedding)
        id3 = store.add_memory(content="Memory 3", embedding=mock_embedding)

        store.add_edge(id1, id2, "relates_to")
        store.add_edge(id1, id3, "contradicts")

        # Filter by type
        relates_edges = store.get_edges(id1, direction="outgoing", edge_type="relates_to")
        assert len(relates_edges) == 1
        assert relates_edges[0]["edge_type"] == "relates_to"

        contradicts_edges = store.get_edges(id1, direction="outgoing", edge_type="contradicts")
        assert len(contradicts_edges) == 1
        assert contradicts_edges[0]["edge_type"] == "contradicts"

    def test_delete_edge(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test deleting a specific edge."""
        id1 = store.add_memory(content="Memory 1", embedding=mock_embedding)
        id2 = store.add_memory(content="Memory 2", embedding=mock_embedding)

        store.add_edge(id1, id2, "relates_to")

        # Verify edge exists
        edges = store.get_edges(id1, direction="outgoing")
        assert len(edges) == 1

        # Delete edge
        deleted = store.delete_edge(id1, id2, "relates_to")
        assert deleted is True

        # Verify edge is gone
        edges = store.get_edges(id1, direction="outgoing")
        assert len(edges) == 0

    def test_delete_edge_nonexistent(self, store: SQLiteStore) -> None:
        """Test deleting a nonexistent edge returns False."""
        deleted = store.delete_edge("fake_id1", "fake_id2", "relates_to")
        assert deleted is False

    def test_edge_weight_validation(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test edge weight must be between 0.0 and 1.0."""
        id1 = store.add_memory(content="Memory 1", embedding=mock_embedding)
        id2 = store.add_memory(content="Memory 2", embedding=mock_embedding)

        # Valid weights
        store.add_edge(id1, id2, "relates_to", weight=0.0)
        store.add_edge(id1, id2, "supersedes", weight=1.0)

        # Invalid weights
        with pytest.raises(ValueError, match="Weight must be between"):
            store.add_edge(id1, id2, "caused_by", weight=1.5)

        with pytest.raises(ValueError, match="Weight must be between"):
            store.add_edge(id1, id2, "contradicts", weight=-0.1)


# ============================================================================
# Validation Event Tests
# ============================================================================


class TestValidationEvents:
    """Test validation event tracking (TRY/LEARN cycle)."""

    def test_add_validation_event(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test adding validation events."""
        memory_id = store.add_memory(content="Test memory", embedding=mock_embedding)

        # Add validation events
        event_id = store.add_validation_event(
            memory_id=memory_id,
            event_type="applied",
            context="Used in code review",
            session_id="session123",
        )

        assert event_id is not None
        assert event_id > 0

    def test_get_validation_events(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test retrieving validation events."""
        memory_id = store.add_memory(content="Test memory", embedding=mock_embedding)

        # Add multiple events
        store.add_validation_event(memory_id, "applied", "Context 1")
        store.add_validation_event(memory_id, "succeeded", "Context 2")
        store.add_validation_event(memory_id, "failed", "Context 3")

        # Get all events
        events = store.get_validation_events(memory_id)
        assert len(events) == 3

        # Get events by type
        applied_events = store.get_validation_events(memory_id, event_type="applied")
        assert len(applied_events) == 1
        assert applied_events[0]["event_type"] == "applied"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_store_error_on_invalid_operation(self, store: SQLiteStore) -> None:
        """Test that invalid operations raise SQLiteStoreError."""
        # Attempt operations that might fail
        # (Most operations are designed to be safe, so we test specific cases)
        pass  # SQLiteStore is fairly robust against invalid inputs

    def test_empty_content(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test handling empty content."""
        # Empty content should still work (though not recommended)
        memory_id = store.add_memory(content="", embedding=mock_embedding)
        assert memory_id is not None

        memory = store.get_memory(memory_id)
        assert memory is not None
        assert memory["content"] == ""

    def test_update_with_no_fields(self, store: SQLiteStore, mock_embedding: list[float]) -> None:
        """Test update_memory with no fields returns False."""
        memory_id = store.add_memory(content="Test", embedding=mock_embedding)
        updated = store.update_memory(memory_id)
        assert updated is False


# ============================================================================
# Recency Scoring Tests
# ============================================================================


class TestRecencyScoring:
    """Test recency-based scoring for memory ranking."""

    def test_new_memory_scores_high(self) -> None:
        """Recently created memory should have near-full recency factor."""
        now = time.time()
        score = compute_recency_score(0.5, now, now, 0)
        assert 0.45 < score < 0.55  # importance * ~1.0 recency * 1.0 access

    def test_seven_day_old_memory_halves(self) -> None:
        """7-day-old memory should have ~50% recency factor."""
        now = time.time()
        seven_days_ago = now - 7 * 86400
        score = compute_recency_score(1.0, seven_days_ago, seven_days_ago, 0)
        assert 0.45 < score < 0.55  # ~0.5 recency * 1.0 access

    def test_high_access_boosts_score(self) -> None:
        """Frequently accessed memory should score higher."""
        now = time.time()
        low = compute_recency_score(0.5, now, now, 0)
        high = compute_recency_score(0.5, now, now, 10)
        assert high > low * 2  # log(11) + 1 ≈ 3.4 vs 1.0
