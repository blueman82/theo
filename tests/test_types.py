"""Tests for unified type system.

This module tests the unified type system including:
- MemoryDocument creation and validation
- Factory methods (from_chunk, from_memory)
- Storage format conversion (to_storage_format, from_storage_format)
- Confidence and access tracking
- Relationship management
- SearchResult types with match_type
"""

from datetime import datetime

import pytest

from theo.types import (
    Document,
    HybridSearchSummary,
    MatchType,
    MemoryDocument,
    MemoryType,
    Relationship,
    RelationType,
    UnifiedSearchResult,
)


class TestMemoryDocument:
    """Test MemoryDocument Pydantic model."""

    def test_create_basic_memory_document(self) -> None:
        """Test creating a basic MemoryDocument."""
        doc = MemoryDocument(
            id="test_001",
            content="This is test content",
        )

        assert doc.id == "test_001"
        assert doc.content == "This is test content"
        assert doc.memory_type == MemoryType.DOCUMENT
        assert doc.namespace == "default"
        assert doc.confidence == 0.3
        assert doc.importance == 0.5
        assert doc.access_count == 0

    def test_create_with_all_fields(self) -> None:
        """Test creating MemoryDocument with all fields."""
        now = datetime.now()
        doc = MemoryDocument(
            id="test_002",
            content="Full content",
            content_hash="abc123",
            memory_type=MemoryType.PREFERENCE,
            namespace="project:myproject",
            source_file="test.py",
            chunk_index=5,
            start_line=10,
            end_line=20,
            importance=0.8,
            confidence=0.9,
            created_at=now,
            accessed_at=now,
            access_count=3,
            metadata={"key": "value"},
        )

        assert doc.id == "test_002"
        assert doc.content_hash == "abc123"
        assert doc.memory_type == MemoryType.PREFERENCE
        assert doc.namespace == "project:myproject"
        assert doc.source_file == "test.py"
        assert doc.chunk_index == 5
        assert doc.start_line == 10
        assert doc.end_line == 20
        assert doc.importance == 0.8
        assert doc.confidence == 0.9
        assert doc.metadata == {"key": "value"}

    def test_validation_empty_content(self) -> None:
        """Test that empty content raises validation error."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            MemoryDocument(id="test", content="")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            MemoryDocument(id="test", content="   ")

    def test_validation_invalid_namespace(self) -> None:
        """Test that invalid namespace raises validation error."""
        with pytest.raises(ValueError, match="Invalid namespace"):
            MemoryDocument(id="test", content="content", namespace="invalid")

        with pytest.raises(ValueError, match="Invalid namespace"):
            MemoryDocument(id="test", content="content", namespace="project:")

    def test_valid_namespaces(self) -> None:
        """Test that valid namespaces are accepted."""
        # Default namespace
        doc1 = MemoryDocument(id="t1", content="c", namespace="default")
        assert doc1.namespace == "default"

        # Global namespace
        doc2 = MemoryDocument(id="t2", content="c", namespace="global")
        assert doc2.namespace == "global"

        # Project namespace
        doc3 = MemoryDocument(id="t3", content="c", namespace="project:my-app")
        assert doc3.namespace == "project:my-app"

        doc4 = MemoryDocument(id="t4", content="c", namespace="project:my_app_123")
        assert doc4.namespace == "project:my_app_123"

    def test_validation_confidence_range(self) -> None:
        """Test that confidence must be in [0.0, 1.0]."""
        with pytest.raises(ValueError):
            MemoryDocument(id="test", content="content", confidence=1.5)

        with pytest.raises(ValueError):
            MemoryDocument(id="test", content="content", confidence=-0.1)

    def test_validation_importance_range(self) -> None:
        """Test that importance must be in [0.0, 1.0]."""
        with pytest.raises(ValueError):
            MemoryDocument(id="test", content="content", importance=2.0)

        with pytest.raises(ValueError):
            MemoryDocument(id="test", content="content", importance=-0.5)


class TestMemoryDocumentMethods:
    """Test MemoryDocument instance methods."""

    def test_is_golden_rule_by_confidence(self) -> None:
        """Test golden rule detection by confidence."""
        doc = MemoryDocument(id="t1", content="c", confidence=0.9)
        assert doc.is_golden_rule() is True

        doc2 = MemoryDocument(id="t2", content="c", confidence=0.8)
        assert doc2.is_golden_rule() is False

    def test_is_golden_rule_by_type(self) -> None:
        """Test golden rule detection by type."""
        doc = MemoryDocument(
            id="t1", content="c", memory_type=MemoryType.GOLDEN_RULE, confidence=0.5
        )
        assert doc.is_golden_rule() is True

    def test_can_be_promoted(self) -> None:
        """Test promotion eligibility."""
        # Promotable types with high confidence
        doc1 = MemoryDocument(
            id="t1", content="c", memory_type=MemoryType.PREFERENCE, confidence=0.9
        )
        assert doc1.can_be_promoted() is True

        doc2 = MemoryDocument(id="t2", content="c", memory_type=MemoryType.DECISION, confidence=0.9)
        assert doc2.can_be_promoted() is True

        # Non-promotable type
        doc3 = MemoryDocument(id="t3", content="c", memory_type=MemoryType.SESSION, confidence=0.9)
        assert doc3.can_be_promoted() is False

        # Low confidence
        doc4 = MemoryDocument(
            id="t4", content="c", memory_type=MemoryType.PREFERENCE, confidence=0.5
        )
        assert doc4.can_be_promoted() is False

    def test_is_document_chunk(self) -> None:
        """Test document chunk detection."""
        doc1 = MemoryDocument(id="t1", content="c", source_file="test.py")
        assert doc1.is_document_chunk() is True

        doc2 = MemoryDocument(id="t2", content="c")
        assert doc2.is_document_chunk() is False

    def test_record_access(self) -> None:
        """Test access recording."""
        doc = MemoryDocument(id="t1", content="c", access_count=0)
        original_accessed = doc.accessed_at

        # Small sleep to ensure timestamp difference
        import time

        time.sleep(0.01)

        doc.record_access()

        assert doc.access_count == 1
        assert doc.accessed_at > original_accessed

        doc.record_access()
        assert doc.access_count == 2

    def test_update_confidence(self) -> None:
        """Test confidence updates."""
        doc = MemoryDocument(id="t1", content="c", confidence=0.5)

        doc.update_confidence(0.2)
        assert doc.confidence == 0.7

        doc.update_confidence(-0.1)
        assert doc.confidence == 0.6

        # Test clamping at max
        doc.update_confidence(0.5)
        assert doc.confidence == 1.0

        # Test clamping at min
        doc.update_confidence(-2.0)
        assert doc.confidence == 0.0

    def test_add_relationship(self) -> None:
        """Test adding relationships."""
        doc = MemoryDocument(id="t1", content="c")
        assert len(doc.relationships) == 0

        doc.add_relationship("t2", RelationType.RELATES_TO)
        assert len(doc.relationships) == 1
        assert doc.relationships[0].target_id == "t2"
        assert doc.relationships[0].relation == RelationType.RELATES_TO

        doc.add_relationship("t3", RelationType.SUPERSEDES, weight=0.8)
        assert len(doc.relationships) == 2
        assert doc.relationships[1].weight == 0.8


class TestMemoryDocumentFactoryMethods:
    """Test MemoryDocument factory methods."""

    def test_from_chunk(self) -> None:
        """Test from_chunk factory method."""
        doc = MemoryDocument.from_chunk(
            chunk_id="chunk_001",
            content="Function definition",
            source_file="main.py",
            chunk_index=0,
            content_hash="hash123",
            start_line=10,
            end_line=25,
            metadata={"language": "python"},
        )

        assert doc.id == "chunk_001"
        assert doc.content == "Function definition"
        assert doc.memory_type == MemoryType.DOCUMENT
        assert doc.source_file == "main.py"
        assert doc.chunk_index == 0
        assert doc.content_hash == "hash123"
        assert doc.start_line == 10
        assert doc.end_line == 25
        assert doc.confidence == 1.0  # Documents have full confidence
        assert doc.importance == 0.5  # Neutral importance
        assert doc.metadata == {"language": "python"}

    def test_from_memory(self) -> None:
        """Test from_memory factory method."""
        doc = MemoryDocument.from_memory(
            memory_id="mem_001",
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            content_hash="hash456",
            namespace="global",
            importance=0.8,
            confidence=0.6,
            metadata={"source": "conversation"},
        )

        assert doc.id == "mem_001"
        assert doc.content == "User prefers dark mode"
        assert doc.memory_type == MemoryType.PREFERENCE
        assert doc.namespace == "global"
        assert doc.importance == 0.8
        assert doc.confidence == 0.6
        assert doc.source_file is None  # Memory, not document
        assert doc.metadata == {"source": "conversation"}


class TestMemoryDocumentStorageConversion:
    """Test storage format conversion methods."""

    def test_to_storage_format_basic(self) -> None:
        """Test basic to_storage_format conversion."""
        doc = MemoryDocument(
            id="test_001",
            content="Test content",
            memory_type=MemoryType.PREFERENCE,
            confidence=0.7,
        )

        storage = doc.to_storage_format()

        assert storage["id"] == "test_001"
        assert storage["content"] == "Test content"
        assert storage["metadata"]["doc_type"] == "preference"
        assert storage["metadata"]["confidence"] == 0.7
        assert storage["metadata"]["namespace"] == "default"

    def test_to_storage_format_full(self) -> None:
        """Test to_storage_format with all fields."""
        doc = MemoryDocument(
            id="test_002",
            content="Full content",
            content_hash="abc123",
            memory_type=MemoryType.DOCUMENT,
            namespace="project:myapp",
            source_file="test.py",
            chunk_index=5,
            start_line=10,
            end_line=20,
            importance=0.8,
            confidence=0.9,
            metadata={"key": "value", "number": 42},
        )

        storage = doc.to_storage_format()

        assert storage["id"] == "test_002"
        assert storage["content"] == "Full content"
        meta = storage["metadata"]
        assert meta["doc_hash"] == "abc123"
        assert meta["source_file"] == "test.py"
        assert meta["chunk_index"] == 5
        assert meta["start_line"] == 10
        assert meta["end_line"] == 20
        assert meta["importance"] == 0.8
        assert meta["confidence"] == 0.9
        assert meta["meta_key"] == "value"
        assert meta["meta_number"] == 42

    def test_from_storage_format(self) -> None:
        """Test from_storage_format reconstruction."""
        metadata = {
            "doc_type": "preference",
            "doc_hash": "xyz789",
            "namespace": "global",
            "source_file": "test.py",
            "chunk_index": 3,
            "start_line": 5,
            "end_line": 15,
            "importance": 0.7,
            "confidence": 0.6,
            "access_count": 5,
            "created_at": "2024-01-15T10:30:00",
            "accessed_at": "2024-01-20T15:45:00",
            "meta_custom": "data",
        }

        doc = MemoryDocument.from_storage_format(
            doc_id="doc123",
            content="Restored content",
            metadata=metadata,
        )

        assert doc.id == "doc123"
        assert doc.content == "Restored content"
        assert doc.content_hash == "xyz789"
        assert doc.memory_type == MemoryType.PREFERENCE
        assert doc.namespace == "global"
        assert doc.source_file == "test.py"
        assert doc.chunk_index == 3
        assert doc.start_line == 5
        assert doc.end_line == 15
        assert doc.importance == 0.7
        assert doc.confidence == 0.6
        assert doc.access_count == 5
        assert doc.metadata == {"custom": "data"}

    def test_roundtrip_conversion(self) -> None:
        """Test that to_storage_format -> from_storage_format roundtrips."""
        original = MemoryDocument(
            id="roundtrip_001",
            content="Roundtrip test content",
            content_hash="hash123",
            memory_type=MemoryType.DECISION,
            namespace="project:test",
            source_file="roundtrip.py",
            chunk_index=2,
            start_line=100,
            end_line=150,
            importance=0.75,
            confidence=0.85,
            metadata={"test": "value"},
        )

        storage = original.to_storage_format()
        restored = MemoryDocument.from_storage_format(
            doc_id=storage["id"],
            content=storage["content"],
            metadata=storage["metadata"],
        )

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.content_hash == original.content_hash
        assert restored.memory_type == original.memory_type
        assert restored.namespace == original.namespace
        assert restored.source_file == original.source_file
        assert restored.chunk_index == original.chunk_index
        assert restored.start_line == original.start_line
        assert restored.end_line == original.end_line
        assert restored.importance == original.importance
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata


class TestRelationship:
    """Test Relationship model."""

    def test_create_relationship(self) -> None:
        """Test creating a basic relationship."""
        rel = Relationship(
            target_id="target_001",
            relation=RelationType.RELATES_TO,
        )

        assert rel.target_id == "target_001"
        assert rel.relation == RelationType.RELATES_TO
        assert rel.weight == 1.0

    def test_relationship_with_weight(self) -> None:
        """Test relationship with custom weight."""
        rel = Relationship(
            target_id="target_002",
            relation=RelationType.SUPERSEDES,
            weight=0.8,
        )

        assert rel.weight == 0.8

    def test_relationship_immutable(self) -> None:
        """Test that Relationship is immutable (frozen)."""
        rel = Relationship(
            target_id="target",
            relation=RelationType.RELATES_TO,
        )

        with pytest.raises(Exception):  # ValidationError for frozen model
            rel.target_id = "new_target"  # type: ignore


class TestUnifiedSearchResult:
    """Test UnifiedSearchResult model."""

    def test_from_vector_search(self) -> None:
        """Test creating result from vector search."""
        doc = MemoryDocument(id="t1", content="test")
        result = UnifiedSearchResult.from_vector_search(
            document=doc,
            distance=0.4,
            rank=0,
        )

        assert result.document == doc
        assert result.distance == 0.4
        # Cosine distance 0.4 -> similarity 1 - (0.4/2) = 0.8
        assert result.similarity == 0.8
        assert result.match_type == MatchType.VECTOR
        assert result.boosted is False

    def test_from_fts_search(self) -> None:
        """Test creating result from FTS search."""
        doc = MemoryDocument(id="t1", content="test content")
        result = UnifiedSearchResult.from_fts_search(
            document=doc,
            distance=0.2,
            rank=1,
            matched_terms=["test", "content"],
        )

        assert result.match_type == MatchType.FTS
        assert result.match_details == {"matched_terms": ["test", "content"]}

    def test_from_graph_traversal(self) -> None:
        """Test creating result from graph traversal."""
        doc = MemoryDocument(id="t1", content="test")
        result = UnifiedSearchResult.from_graph_traversal(
            document=doc,
            relevance_score=0.75,
            rank=0,
            path=["origin", "intermediate", "t1"],
        )

        assert result.match_type == MatchType.GRAPH
        assert result.similarity == 0.75
        assert result.graph_path == ["origin", "intermediate", "t1"]

    def test_boost(self) -> None:
        """Test boosting a result."""
        doc = MemoryDocument(id="t1", content="test")
        result = UnifiedSearchResult.from_vector_search(
            document=doc,
            distance=0.4,
            rank=0,
        )

        boosted = result.boost(0.1)

        assert boosted.boosted is True
        assert boosted.match_type == MatchType.HYBRID
        assert boosted.similarity == 0.9  # 0.8 + 0.1

    def test_boost_clamps_at_one(self) -> None:
        """Test that boosting clamps similarity at 1.0."""
        doc = MemoryDocument(id="t1", content="test")
        result = UnifiedSearchResult.from_vector_search(
            document=doc,
            distance=0.0,  # Perfect match, similarity = 1.0
            rank=0,
        )

        boosted = result.boost(0.5)
        assert boosted.similarity == 1.0


class TestHybridSearchSummary:
    """Test HybridSearchSummary model."""

    def test_create_empty_summary(self) -> None:
        """Test creating empty search summary."""
        summary = HybridSearchSummary()

        assert summary.results == []
        assert summary.total_results == 0
        assert summary.fts_used is False

    def test_summary_with_results(self) -> None:
        """Test summary with mixed results."""
        doc1 = MemoryDocument(id="t1", content="test1")
        doc2 = MemoryDocument(id="t2", content="test2")

        results = [
            UnifiedSearchResult.from_vector_search(doc1, 0.2, 0),
            UnifiedSearchResult.from_fts_search(doc2, 0.3, 1),
        ]

        summary = HybridSearchSummary(
            results=results,
            total_vector_matches=1,
            total_fts_matches=1,
            fts_used=True,
            fts_triggered_reason="Low vector similarity",
        )

        assert summary.total_results == 2
        assert summary.fts_used is True

    def test_get_by_match_type(self) -> None:
        """Test filtering results by match type."""
        doc1 = MemoryDocument(id="t1", content="test1")
        doc2 = MemoryDocument(id="t2", content="test2")
        doc3 = MemoryDocument(id="t3", content="test3")

        results = [
            UnifiedSearchResult.from_vector_search(doc1, 0.2, 0),
            UnifiedSearchResult.from_fts_search(doc2, 0.3, 1),
            UnifiedSearchResult.from_vector_search(doc3, 0.4, 2),
        ]

        summary = HybridSearchSummary(results=results)

        vector_results = summary.get_by_match_type(MatchType.VECTOR)
        assert len(vector_results) == 2

        fts_results = summary.get_by_match_type(MatchType.FTS)
        assert len(fts_results) == 1

    def test_boosted_count(self) -> None:
        """Test counting boosted results."""
        doc1 = MemoryDocument(id="t1", content="test1")
        doc2 = MemoryDocument(id="t2", content="test2")

        result1 = UnifiedSearchResult.from_vector_search(doc1, 0.2, 0)
        result2 = result1.boost()  # This one is boosted

        summary = HybridSearchSummary(
            results=[
                result1,
                result2,
                UnifiedSearchResult.from_vector_search(doc2, 0.3, 2),
            ]
        )

        assert summary.boosted_count == 1


class TestDocumentCompatibility:
    """Test backward compatibility with Document type."""

    def test_document_import(self) -> None:
        """Test that Document is importable from theo.types."""
        # Document should be importable (re-exported from storage)
        assert Document is not None

    def test_document_creation(self) -> None:
        """Test creating a Document via the compatibility layer."""
        doc = Document(
            id="compat_001",
            content="Compatibility test",
            source_file="test.py",
        )

        assert doc.id == "compat_001"
        assert doc.content == "Compatibility test"


class TestMemoryTypes:
    """Test MemoryType enum."""

    def test_all_memory_types(self) -> None:
        """Test all memory types are defined."""
        assert MemoryType.DOCUMENT.value == "document"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.PATTERN.value == "pattern"
        assert MemoryType.SESSION.value == "session"
        assert MemoryType.FILE_CONTEXT.value == "file_context"
        assert MemoryType.GOLDEN_RULE.value == "golden_rule"
        assert MemoryType.FACT.value == "fact"


class TestRelationTypes:
    """Test RelationType enum."""

    def test_all_relation_types(self) -> None:
        """Test all relation types are defined."""
        assert RelationType.RELATES_TO.value == "relates_to"
        assert RelationType.SUPERSEDES.value == "supersedes"
        assert RelationType.CAUSED_BY.value == "caused_by"
        assert RelationType.CONTRADICTS.value == "contradicts"


class TestMatchTypes:
    """Test MatchType enum."""

    def test_all_match_types(self) -> None:
        """Test all match types are defined."""
        assert MatchType.VECTOR.value == "vector"
        assert MatchType.FTS.value == "fts"
        assert MatchType.HYBRID.value == "hybrid"
        assert MatchType.EXACT.value == "exact"
        assert MatchType.GRAPH.value == "graph"
