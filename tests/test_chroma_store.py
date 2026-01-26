"""Tests for ChromaDB storage layer.

This module tests ChromaStore functionality including:
- Initialization and collection management
- Adding documents with metadata
- Semantic search with and without filters
- Hybrid search combining vector and text
- Document deletion
- Hash-based deduplication
- Confidence score updates
"""

import uuid
from pathlib import Path

import pytest

from theo.storage import (
    ChromaStore,
    Document,
    HybridSearchResult,
    SearchResult,
    StorageError,
    StoreStats,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide temporary database path for testing."""
    return tmp_path / "test_chroma_db"


@pytest.fixture
def chroma_store() -> ChromaStore:
    """Provide ephemeral ChromaStore instance for testing.

    Uses unique collection name to ensure test isolation since
    ChromaDB ephemeral clients share state within a process.
    """
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaStore(ephemeral=True, collection_name=unique_name)


@pytest.fixture
def persistent_store(temp_db_path: Path) -> ChromaStore:
    """Provide persistent ChromaStore instance for testing."""
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaStore(db_path=temp_db_path, collection_name=unique_name)


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Provide sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide sample documents for testing."""
    return [
        "This is the first document about Python programming.",
        "This is the second document about data science.",
        "This is the third document about machine learning.",
    ]


@pytest.fixture
def sample_metadatas() -> list[dict]:
    """Provide sample metadata for testing."""
    return [
        {
            "doc_hash": "hash123",
            "source_file": "file1.py",
            "chunk_index": 0,
            "namespace": "default",
            "doc_type": "document",
            "confidence": 1.0,
        },
        {
            "doc_hash": "hash456",
            "source_file": "file2.md",
            "chunk_index": 0,
            "namespace": "default",
            "doc_type": "document",
            "confidence": 0.8,
        },
        {
            "doc_hash": "hash789",
            "source_file": "file1.py",
            "chunk_index": 1,
            "namespace": "project",
            "doc_type": "memory",
            "confidence": 0.5,
        },
    ]


@pytest.fixture
def sample_document_objects() -> list[Document]:
    """Provide sample Document objects for testing."""
    return [
        Document(
            id="doc1",
            content="This is the first document about Python programming.",
            source_file="file1.py",
            chunk_index=0,
            doc_hash="hash123",
            confidence=1.0,
            namespace="default",
            doc_type="document",
        ),
        Document(
            id="doc2",
            content="This is the second document about data science.",
            source_file="file2.md",
            chunk_index=0,
            doc_hash="hash456",
            confidence=0.8,
            namespace="default",
            doc_type="document",
        ),
        Document(
            id="doc3",
            content="This is the third document about machine learning.",
            source_file="file1.py",
            chunk_index=1,
            doc_hash="hash789",
            confidence=0.5,
            namespace="project",
            doc_type="memory",
        ),
    ]


class TestChromaStoreInitialization:
    """Test ChromaStore initialization and setup."""

    def test_init_ephemeral(self) -> None:
        """Test initialization with ephemeral storage."""
        store = ChromaStore(ephemeral=True)
        assert store.ephemeral is True
        assert store.db_path is None

    def test_init_creates_db_directory(self, temp_db_path: Path) -> None:
        """Test that initialization creates database directory."""
        assert not temp_db_path.exists()
        ChromaStore(db_path=temp_db_path)
        assert temp_db_path.exists()
        assert temp_db_path.is_dir()

    def test_init_with_existing_directory(self, temp_db_path: Path) -> None:
        """Test initialization with existing directory."""
        temp_db_path.mkdir(parents=True, exist_ok=True)
        store = ChromaStore(db_path=temp_db_path)
        assert store.db_path == temp_db_path

    def test_init_with_custom_collection_name(self, temp_db_path: Path) -> None:
        """Test initialization with custom collection name."""
        store = ChromaStore(db_path=temp_db_path, collection_name="custom_name")
        assert store.collection_name == "custom_name"

    def test_init_creates_collection(self, chroma_store: ChromaStore) -> None:
        """Test that initialization creates collection."""
        assert chroma_store._collection is not None
        # Collection name is unique per test (starts with test_)
        assert chroma_store._collection.name.startswith("test_")

    def test_persistent_storage(self, temp_db_path: Path) -> None:
        """Test that data persists across ChromaStore instances."""
        # Create first instance and add data
        store1 = ChromaStore(db_path=temp_db_path)
        store1.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"doc_hash": "test_hash"}],
        )

        # Create second instance and verify data exists
        store2 = ChromaStore(db_path=temp_db_path)
        assert store2.count() == 1


class TestChromaStoreAddRaw:
    """Test adding documents using raw API."""

    def test_add_single_document(self, chroma_store: ChromaStore) -> None:
        """Test adding a single document."""
        ids = chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"doc_hash": "hash1"}],
        )

        assert len(ids) == 1
        assert isinstance(ids[0], str)
        assert chroma_store.count() == 1

    def test_add_multiple_documents(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test adding multiple documents."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert len(ids) == 3
        assert chroma_store.count() == 3

    def test_add_generates_unique_ids(self, chroma_store: ChromaStore) -> None:
        """Test that add generates unique IDs."""
        ids1 = chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["doc1"],
            metadatas=[{"doc_hash": "hash1"}],
        )
        ids2 = chroma_store.add(
            embeddings=[[0.4, 0.5, 0.6]],
            documents=["doc2"],
            metadatas=[{"doc_hash": "hash2"}],
        )

        assert ids1[0] != ids2[0]

    def test_add_empty_list(self, chroma_store: ChromaStore) -> None:
        """Test adding empty list returns empty list."""
        ids = chroma_store.add(embeddings=[], documents=[], metadatas=[])
        assert ids == []
        assert chroma_store.count() == 0

    def test_add_without_metadata(self, chroma_store: ChromaStore) -> None:
        """Test adding documents without metadata."""
        ids = chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["doc without metadata"],
        )

        assert len(ids) == 1
        assert chroma_store.count() == 1

    def test_add_length_mismatch_raises_error(self, chroma_store: ChromaStore) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            chroma_store.add(
                embeddings=[[0.1, 0.2, 0.3]],
                documents=["doc1", "doc2"],
                metadatas=[{"doc_hash": "hash1"}],
            )


class TestChromaStoreAddDocuments:
    """Test adding Document objects."""

    def test_add_documents(
        self,
        chroma_store: ChromaStore,
        sample_document_objects: list[Document],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test adding Document objects."""
        ids = chroma_store.add_documents(
            documents=sample_document_objects,
            embeddings=sample_embeddings,
        )

        assert len(ids) == 3
        assert chroma_store.count() == 3
        # Should use document IDs
        assert ids == ["doc1", "doc2", "doc3"]

    def test_add_documents_preserves_metadata(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test that Document metadata is preserved."""
        doc = Document(
            id="test_doc",
            content="Test content",
            source_file="test.py",
            chunk_index=5,
            doc_hash="abc123",
            confidence=0.75,
            namespace="my_namespace",
            doc_type="memory",
            metadata={"custom_field": "custom_value"},
        )

        chroma_store.add_documents([doc], [sample_embeddings[0]])

        result = chroma_store.get_by_hash("abc123")
        assert result is not None
        meta = result["metadatas"][0]
        assert meta["source_file"] == "test.py"
        assert meta["chunk_index"] == 5
        assert meta["confidence"] == 0.75
        assert meta["namespace"] == "my_namespace"
        assert meta["doc_type"] == "memory"
        assert meta["meta_custom_field"] == "custom_value"

    def test_add_documents_length_mismatch(
        self,
        chroma_store: ChromaStore,
        sample_document_objects: list[Document],
    ) -> None:
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            chroma_store.add_documents(
                documents=sample_document_objects,
                embeddings=[[0.1, 0.2]],  # Wrong length
            )


class TestChromaStoreSearch:
    """Test semantic search functionality."""

    def test_search_returns_results(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test basic search returns SearchResult objects."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4], n_results=2
        )

        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_structure(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test SearchResult has expected attributes."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert len(results) > 0
        result = results[0]
        assert isinstance(result.document, Document)
        assert isinstance(result.distance, float)
        assert isinstance(result.similarity, float)
        assert 0.0 <= result.similarity <= 1.0
        assert result.rank == 0

    def test_search_with_n_results(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test search respects n_results parameter."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(
            query_embedding=[0.5, 0.6, 0.7, 0.8], n_results=1
        )
        assert len(results) == 1

        results = chroma_store.search(
            query_embedding=[0.5, 0.6, 0.7, 0.8], n_results=3
        )
        assert len(results) == 3

    def test_search_with_metadata_filter(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test search with metadata filtering."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        # Search with source_file filter
        results = chroma_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            n_results=5,
            where={"source_file": "file1.py"},
        )

        # Should only return documents from file1.py
        assert len(results) == 2
        for result in results:
            assert result.document.source_file == "file1.py"

    def test_search_empty_collection(self, chroma_store: ChromaStore) -> None:
        """Test search on empty collection returns empty results."""
        results = chroma_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert results == []

    def test_search_raw_returns_dict(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test search_raw returns raw dictionary format."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search_raw(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results


class TestChromaStoreHybridSearch:
    """Test hybrid search combining vector and FTS."""

    def test_hybrid_search_returns_results(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test hybrid search returns HybridSearchResult."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = chroma_store.hybrid_search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            query_text="Python",
            n_results=3,
        )

        assert isinstance(result, HybridSearchResult)
        assert len(result.results) <= 3

    def test_hybrid_search_with_weak_vector_uses_fts(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test that hybrid search uses FTS when vector results are weak."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        # Use a very different embedding to get low similarity
        result = chroma_store.hybrid_search(
            query_embedding=[-1.0, -1.0, -1.0, -1.0],
            query_text="python",
            n_results=3,
            min_vector_similarity=0.99,  # High threshold to trigger FTS
        )

        # FTS should be used when similarity is below threshold
        assert isinstance(result, HybridSearchResult)

    def test_hybrid_search_with_metadata_filter(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test hybrid search respects metadata filters."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = chroma_store.hybrid_search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            query_text="document",
            n_results=5,
            where={"source_file": "file1.py"},
        )

        for sr in result.results:
            assert sr.document.source_file == "file1.py"


class TestChromaStoreDelete:
    """Test document deletion functionality."""

    def test_delete_by_id(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test deleting documents by ID."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert chroma_store.count() == 3
        chroma_store.delete_by_id([ids[0]])
        assert chroma_store.count() == 2

    def test_delete_multiple_documents(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test deleting multiple documents."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        chroma_store.delete_by_id([ids[0], ids[1]])
        assert chroma_store.count() == 1

    def test_delete_by_source(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test deleting documents by source file."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert chroma_store.count() == 3
        deleted_count = chroma_store.delete_by_source("file1.py")

        assert deleted_count == 2
        assert chroma_store.count() == 1

    def test_delete_empty_list(self, chroma_store: ChromaStore) -> None:
        """Test deleting empty list does nothing."""
        chroma_store.delete_by_id([])
        assert chroma_store.count() == 0

    def test_delete_nonexistent_source(self, chroma_store: ChromaStore) -> None:
        """Test deleting nonexistent source returns zero."""
        deleted_count = chroma_store.delete_by_source("nonexistent.py")
        assert deleted_count == 0


class TestChromaStoreGetByHash:
    """Test hash-based deduplication queries."""

    def test_get_by_hash_existing(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test getting document by existing hash."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = chroma_store.get_by_hash("hash123")

        assert result is not None
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["doc_hash"] == "hash123"

    def test_get_by_hash_nonexistent(self, chroma_store: ChromaStore) -> None:
        """Test getting document by nonexistent hash returns None."""
        result = chroma_store.get_by_hash("nonexistent_hash")
        assert result is None


class TestChromaStoreGetBySourceFile:
    """Test querying documents by source file."""

    def test_get_by_source_file_existing(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test getting documents by existing source file."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = chroma_store.get_by_source_file("file1.py")

        assert result is not None
        assert len(result["ids"]) == 2

    def test_get_by_source_file_nonexistent(self, chroma_store: ChromaStore) -> None:
        """Test getting documents by nonexistent source file returns None."""
        result = chroma_store.get_by_source_file("nonexistent.py")
        assert result is None


class TestChromaStoreClear:
    """Test clearing the collection."""

    def test_clear_with_documents(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test clearing a collection with documents."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert chroma_store.count() == 3
        deleted_count = chroma_store.clear()

        assert deleted_count == 3
        assert chroma_store.count() == 0

    def test_clear_empty_collection(self, chroma_store: ChromaStore) -> None:
        """Test clearing an empty collection returns zero."""
        deleted_count = chroma_store.clear()
        assert deleted_count == 0


class TestChromaStoreGetStats:
    """Test collection statistics."""

    def test_get_stats_with_documents(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
        sample_documents: list[str],
        sample_metadatas: list[dict],
    ) -> None:
        """Test getting stats from a populated collection."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        stats = chroma_store.get_stats()

        assert isinstance(stats, StoreStats)
        assert stats.total_documents == 3
        assert stats.unique_sources == 2
        assert "file1.py" in stats.source_files
        assert "file2.md" in stats.source_files
        assert "default" in stats.namespaces
        assert "project" in stats.namespaces

    def test_get_stats_empty_collection(self, chroma_store: ChromaStore) -> None:
        """Test getting stats from an empty collection."""
        stats = chroma_store.get_stats()

        assert stats.total_documents == 0
        assert stats.unique_sources == 0
        assert stats.source_files == []


class TestChromaStoreUpdateConfidence:
    """Test confidence score updates."""

    def test_update_confidence_success(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test updating confidence score."""
        ids = chroma_store.add(
            embeddings=[sample_embeddings[0]],
            documents=["test doc"],
            metadatas=[{"confidence": 0.5}],
        )

        updated = chroma_store.update_confidence(ids[0], 0.9)

        assert updated is True
        result = chroma_store.get(ids=ids)
        assert result["metadatas"][0]["confidence"] == 0.9

    def test_update_confidence_nonexistent(self, chroma_store: ChromaStore) -> None:
        """Test updating confidence on nonexistent document."""
        updated = chroma_store.update_confidence("nonexistent_id", 0.5)
        assert updated is False

    def test_update_confidence_invalid_range(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test that invalid confidence values raise error."""
        ids = chroma_store.add(
            embeddings=[sample_embeddings[0]],
            documents=["test doc"],
            metadatas=[{"confidence": 0.5}],
        )

        with pytest.raises(ValueError, match="Confidence must be between"):
            chroma_store.update_confidence(ids[0], 1.5)

        with pytest.raises(ValueError, match="Confidence must be between"):
            chroma_store.update_confidence(ids[0], -0.1)


class TestDocumentTypes:
    """Test Document dataclass functionality."""

    def test_document_to_chroma_metadata(self) -> None:
        """Test Document.to_chroma_metadata() conversion."""
        doc = Document(
            id="test_id",
            content="Test content",
            source_file="test.py",
            chunk_index=5,
            doc_hash="abc123",
            confidence=0.75,
            namespace="my_ns",
            doc_type="memory",
            metadata={"key": "value", "number": 42},
        )

        meta = doc.to_chroma_metadata()

        assert meta["source_file"] == "test.py"
        assert meta["chunk_index"] == 5
        assert meta["doc_hash"] == "abc123"
        assert meta["confidence"] == 0.75
        assert meta["namespace"] == "my_ns"
        assert meta["doc_type"] == "memory"
        assert meta["meta_key"] == "value"
        assert meta["meta_number"] == 42

    def test_document_from_chroma_result(self) -> None:
        """Test Document.from_chroma_result() construction."""
        metadata = {
            "source_file": "test.py",
            "chunk_index": 3,
            "doc_hash": "xyz789",
            "confidence": 0.6,
            "namespace": "proj",
            "doc_type": "fact",
            "created_at": "2024-01-15T10:30:00",
            "meta_custom": "data",
        }

        doc = Document.from_chroma_result(
            doc_id="doc123",
            content="Restored content",
            metadata=metadata,
        )

        assert doc.id == "doc123"
        assert doc.content == "Restored content"
        assert doc.source_file == "test.py"
        assert doc.chunk_index == 3
        assert doc.doc_hash == "xyz789"
        assert doc.confidence == 0.6
        assert doc.namespace == "proj"
        assert doc.doc_type == "fact"
        assert doc.metadata == {"custom": "data"}


class TestSearchResultTypes:
    """Test SearchResult dataclass functionality."""

    def test_search_result_from_chroma_result(self) -> None:
        """Test SearchResult.from_chroma_result() construction."""
        metadata = {
            "source_file": "test.py",
            "confidence": 0.8,
            "namespace": "default",
            "doc_type": "document",
        }

        result = SearchResult.from_chroma_result(
            doc_id="doc1",
            content="Test content",
            metadata=metadata,
            distance=0.4,
            rank=0,
        )

        assert result.document.id == "doc1"
        assert result.distance == 0.4
        # Cosine distance 0.4 -> similarity 1 - (0.4/2) = 0.8
        assert result.similarity == 0.8
        assert result.rank == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_embedding_dimension(self, chroma_store: ChromaStore) -> None:
        """Test that inconsistent embedding dimensions are handled."""
        # First add with dimension 3
        chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["doc1"],
            metadatas=[{"doc_hash": "hash1"}],
        )

        # Try to add with different dimension
        with pytest.raises(StorageError):
            chroma_store.add(
                embeddings=[[0.1, 0.2]],  # Different dimension
                documents=["doc2"],
                metadatas=[{"doc_hash": "hash2"}],
            )

    def test_large_batch_add(self, chroma_store: ChromaStore) -> None:
        """Test adding a large batch of documents."""
        batch_size = 100
        embeddings = [[0.1, 0.2, 0.3] for _ in range(batch_size)]
        documents = [f"document {i}" for i in range(batch_size)]
        metadatas = [{"doc_hash": f"hash{i}"} for i in range(batch_size)]

        ids = chroma_store.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        assert len(ids) == batch_size
        assert chroma_store.count() == batch_size

    def test_search_with_nonexistent_filter(
        self,
        chroma_store: ChromaStore,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test search with filter that matches nothing."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=["doc1", "doc2", "doc3"],
            metadatas=[{"key": "val1"}, {"key": "val2"}, {"key": "val3"}],
        )

        results = chroma_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            where={"key": "nonexistent"},
        )

        assert results == []
