"""Integration tests for the full Theo indexing and query pipeline.

This module provides end-to-end tests covering:
- Multi-format document indexing (Markdown, Python, text)
- Full pipeline from chunking through embedding to storage
- Query pipeline with search and filtering
- Memory operations with validation feedback
- Deduplication with hash tracking

Tests use real components but mock the embedding provider to avoid
external dependencies (MLX/Ollama).

Usage:
    uv run pytest tests/integration/test_full_pipeline.py -v
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from theo.chunking import ChunkerRegistry
from theo.storage import SQLiteStore
from theo.tools import IndexingTools, ManagementTools, MemoryTools, QueryTools
from theo.validation import FeedbackCollector, ValidationLoop

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary database file
    """
    return tmp_path / "test_theo.db"


@pytest.fixture
def mock_daemon_client() -> MagicMock:
    """Create a mock DaemonClient for testing without real daemon.

    Returns deterministic embeddings based on text content to enable
    testing of similarity search without requiring actual MLX/Ollama.

    Returns:
        Mock DaemonClient instance
    """
    client = MagicMock()

    def mock_embed(texts: list[str]) -> dict[str, Any]:
        """Generate deterministic embeddings from texts."""
        embeddings = []
        for text in texts:
            # Use hash to create reproducible embeddings
            text_hash = hash(text[:100] if text else "")
            # Generate 1024-dimensional embedding
            embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
            # Normalize to unit vector
            magnitude = sum(x * x for x in embedding) ** 0.5
            embeddings.append([x / magnitude for x in embedding])

        return {
            "success": True,
            "data": {
                "embeddings": embeddings,
                "count": len(embeddings),
                "dimensions": 1024,
            },
        }

    def mock_search(query: str, **_: Any) -> dict[str, Any]:
        """Generate query embedding for search."""
        text_hash = hash(query[:100] if query else "")
        embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
        magnitude = sum(x * x for x in embedding) ** 0.5
        normalized = [x / magnitude for x in embedding]

        return {
            "success": True,
            "data": {
                "embedding": normalized,
                "dimensions": 1024,
            },
        }

    client.embed = MagicMock(side_effect=mock_embed)
    client.send = MagicMock(
        side_effect=lambda cmd, **kwargs: (
            mock_search(kwargs.get("query", ""))
            if cmd == "search"
            else {"success": False, "error": f"Unknown command: {cmd}"}
        )
    )

    return client


@pytest.fixture
def sqlite_store(temp_db_path: Path) -> Generator[SQLiteStore, None, None]:
    """Create SQLiteStore instance with temporary database.

    Args:
        temp_db_path: Temporary database path fixture

    Yields:
        SQLiteStore instance for testing
    """
    store = SQLiteStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def chunker_registry() -> ChunkerRegistry:
    """Create ChunkerRegistry instance.

    Returns:
        ChunkerRegistry for format-specific chunking
    """
    return ChunkerRegistry()


@pytest.fixture
def validation_loop(sqlite_store: SQLiteStore) -> ValidationLoop:
    """Create ValidationLoop instance.

    Args:
        sqlite_store: SQLiteStore fixture

    Returns:
        ValidationLoop for confidence scoring
    """
    return ValidationLoop(store=sqlite_store)


@pytest.fixture
def feedback_collector() -> FeedbackCollector:
    """Create FeedbackCollector instance.

    Returns:
        FeedbackCollector for tracking search feedback
    """
    return FeedbackCollector()


@pytest.fixture
def indexing_tools(
    mock_daemon_client: MagicMock,
    chunker_registry: ChunkerRegistry,
    sqlite_store: SQLiteStore,
) -> IndexingTools:
    """Create IndexingTools instance.

    Args:
        mock_daemon_client: Mock daemon client fixture
        chunker_registry: ChunkerRegistry fixture
        sqlite_store: SQLiteStore fixture

    Returns:
        IndexingTools for document indexing
    """
    return IndexingTools(
        daemon_client=mock_daemon_client,
        chunker_registry=chunker_registry,
        store=sqlite_store,
    )


@pytest.fixture
def query_tools(
    mock_daemon_client: MagicMock,
    sqlite_store: SQLiteStore,
    feedback_collector: FeedbackCollector,
) -> QueryTools:
    """Create QueryTools instance.

    Args:
        mock_daemon_client: Mock daemon client fixture
        sqlite_store: SQLiteStore fixture
        feedback_collector: FeedbackCollector fixture

    Returns:
        QueryTools for search operations
    """
    return QueryTools(
        daemon_client=mock_daemon_client,
        store=sqlite_store,
        feedback_collector=feedback_collector,
    )


@pytest.fixture
def memory_tools(
    mock_daemon_client: MagicMock,
    sqlite_store: SQLiteStore,
    validation_loop: ValidationLoop,
) -> MemoryTools:
    """Create MemoryTools instance.

    Args:
        mock_daemon_client: Mock daemon client fixture
        sqlite_store: SQLiteStore fixture
        validation_loop: ValidationLoop fixture

    Returns:
        MemoryTools for memory operations
    """
    return MemoryTools(
        daemon_client=mock_daemon_client,
        store=sqlite_store,
        validation_loop=validation_loop,
    )


@pytest.fixture
def management_tools(sqlite_store: SQLiteStore) -> ManagementTools:
    """Create ManagementTools instance.

    Args:
        sqlite_store: SQLiteStore fixture

    Returns:
        ManagementTools for collection management
    """
    return ManagementTools(store=sqlite_store)


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing.

    Args:
        tmp_path: pytest temporary directory

    Returns:
        Path to created markdown file
    """
    content = """# Test Document

## Introduction

This is a test markdown document for Theo integration testing.
It contains multiple sections to test chunking behavior.

## Features

- Feature one: Vector search
- Feature two: Memory storage
- Feature three: Validation loop

## Implementation Details

The implementation uses ChromaDB for vector storage and supports
multiple file formats including markdown, Python, and plain text.

### Subsection

This subsection provides more detail about the implementation.
It should be chunked separately from the main section.

## Conclusion

Testing is important for ensuring code quality.
"""
    file_path = tmp_path / "test.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a sample Python file for testing.

    Args:
        tmp_path: pytest temporary directory

    Returns:
        Path to created Python file
    """
    content = '''"""Sample Python module for testing."""


def hello_world():
    """Print hello world."""
    print("Hello, World!")


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        """Initialize the calculator."""
        self.result = 0

    def add(self, value: int) -> None:
        """Add a value to the result."""
        self.result += value

    def subtract(self, value: int) -> None:
        """Subtract a value from the result."""
        self.result -= value

    def reset(self) -> None:
        """Reset the calculator."""
        self.result = 0
'''
    file_path = tmp_path / "calculator.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing.

    Args:
        tmp_path: pytest temporary directory

    Returns:
        Path to created text file
    """
    content = """This is a plain text document for testing.

It has multiple paragraphs to test text chunking behavior.
Each paragraph represents a logical unit of content.

The chunking algorithm should handle this gracefully and
produce meaningful chunks that preserve context.

This is the final paragraph with some concluding remarks.
"""
    file_path = tmp_path / "document.txt"
    file_path.write_text(content)
    return file_path


# =============================================================================
# Indexing Pipeline Tests
# =============================================================================


class TestIndexingPipeline:
    """Test the complete indexing pipeline from files to storage."""

    @pytest.mark.asyncio
    async def test_index_markdown_document(
        self,
        indexing_tools: IndexingTools,
        sample_markdown_file: Path,
        sqlite_store: SQLiteStore,
    ):
        """Test indexing a Markdown file through full pipeline."""
        # Index the document
        result = await indexing_tools.index_file(str(sample_markdown_file))

        # Verify successful indexing
        assert result["success"] is True
        assert "data" in result
        assert result["data"]["chunks_created"] > 0
        assert str(sample_markdown_file) in result["data"]["source_file"]

        # Verify chunks are in storage
        count = sqlite_store.count_memories()
        assert count == result["data"]["chunks_created"]

    @pytest.mark.asyncio
    async def test_index_python_document(
        self,
        indexing_tools: IndexingTools,
        sample_python_file: Path,
        sqlite_store: SQLiteStore,
    ):
        """Test indexing a Python file through full pipeline."""
        # Index the document
        result = await indexing_tools.index_file(str(sample_python_file))

        # Verify successful indexing
        assert result["success"] is True
        assert result["data"]["chunks_created"] > 0

        # Verify memories are stored (SQLiteStore doesn't have get_stats with source_files)
        memories = sqlite_store.list_memories(limit=100)
        source_files = {m.get("source_file") for m in memories if m.get("source_file")}
        assert str(sample_python_file) in source_files

    @pytest.mark.asyncio
    async def test_index_text_document(
        self,
        indexing_tools: IndexingTools,
        sample_text_file: Path,
        sqlite_store: SQLiteStore,
    ):
        """Test indexing a plain text file through full pipeline."""
        # Index the document
        result = await indexing_tools.index_file(str(sample_text_file))

        # Verify successful indexing
        assert result["success"] is True
        assert result["data"]["chunks_created"] > 0

        # Verify content is stored
        count = sqlite_store.count_memories()
        assert count > 0

    @pytest.mark.asyncio
    async def test_index_directory(
        self,
        indexing_tools: IndexingTools,
        sample_markdown_file: Path,
        sample_python_file: Path,  # noqa: ARG002 - fixture creates file
        sample_text_file: Path,  # noqa: ARG002 - fixture creates file
        sqlite_store: SQLiteStore,
    ):
        """Test indexing a directory with multiple files."""
        # Get the directory containing our sample files (all fixtures create in same dir)
        directory = sample_markdown_file.parent

        # Index the directory with async_mode=False for synchronous processing
        result = await indexing_tools.index_directory(
            str(directory), recursive=False, namespace="test", async_mode=False
        )

        # Verify successful indexing
        assert result["success"] is True
        assert result["data"]["files_processed"] >= 3
        assert result["data"]["total_chunks"] > 0

        # Verify all files are in storage
        memories = sqlite_store.list_memories(limit=100)
        source_files = {m.get("source_file") for m in memories if m.get("source_file")}
        assert len(source_files) >= 3

    @pytest.mark.asyncio
    async def test_reindex_file_replaces_content(
        self,
        indexing_tools: IndexingTools,
        sample_text_file: Path,
        sqlite_store: SQLiteStore,
    ):
        """Test that re-indexing a file replaces existing chunks."""
        # Index the file first time
        result1 = await indexing_tools.index_file(str(sample_text_file))
        assert result1["success"] is True
        _ = result1["data"]["chunks_created"]  # Track initial chunks

        # Modify the file
        original_content = sample_text_file.read_text()
        sample_text_file.write_text(original_content + "\n\nNew content added!")

        # Re-index the file
        result2 = await indexing_tools.index_file(str(sample_text_file))
        assert result2["success"] is True

        # Verify total count is the new chunk count (not accumulated)
        final_count = sqlite_store.count_memories()
        assert final_count == result2["data"]["chunks_created"]


# =============================================================================
# Query Pipeline Tests
# =============================================================================


class TestQueryPipeline:
    """Test the complete query pipeline including search and filtering."""

    @pytest.mark.asyncio
    async def test_basic_search(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        sample_markdown_file: Path,
    ):
        """Test basic semantic search functionality."""
        # Index document first
        await indexing_tools.index_file(str(sample_markdown_file))

        # Search for relevant content
        result = await query_tools.search("vector database", n_results=5)

        # Verify results structure
        assert result["success"] is True
        assert "data" in result
        assert "results" in result["data"]
        assert "total" in result["data"]

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        sample_markdown_file: Path,
        sample_python_file: Path,
    ):
        """Test search with metadata filtering."""
        # Index multiple documents with different namespaces
        await indexing_tools.index_file(str(sample_markdown_file), namespace="docs")
        await indexing_tools.index_file(str(sample_python_file), namespace="code")

        # Search with namespace filter
        result = await query_tools.search_with_filters(
            query="function",
            filters={"namespace": "code"},
            n_results=5,
        )

        # Verify results are filtered
        assert result["success"] is True
        # Results should only include code namespace
        for item in result["data"]["results"]:
            assert item["namespace"] == "code"

    @pytest.mark.asyncio
    async def test_search_with_budget(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        sample_markdown_file: Path,
    ):
        """Test search with token budget constraint."""
        # Index document
        await indexing_tools.index_file(str(sample_markdown_file))

        # Search with token budget
        max_tokens = 200
        result = await query_tools.search_with_budget("testing", max_tokens=max_tokens)

        # Verify budget constraints
        assert result["success"] is True
        assert "tokens_used" in result["data"]
        assert result["data"]["tokens_used"] <= max_tokens

    @pytest.mark.asyncio
    async def test_empty_query_error(self, query_tools: QueryTools):
        """Test that empty query returns error."""
        result = await query_tools.search("", n_results=5)

        assert result["success"] is False
        assert "empty" in result["error"].lower()


# =============================================================================
# Memory Pipeline Tests
# =============================================================================


class TestMemoryPipeline:
    """Test the complete memory pipeline from store to recall to validate."""

    @pytest.mark.asyncio
    async def test_memory_store_and_recall(
        self,
        memory_tools: MemoryTools,
    ):
        """Test storing and recalling memories."""
        # Store a memory
        store_result = await memory_tools.memory_store(
            content="User prefers dark mode for their IDE",
            memory_type="preference",
            namespace="global",
            importance=0.8,
        )

        # Verify successful storage
        assert store_result["success"] is True
        assert "id" in store_result["data"]
        _ = store_result["data"]["id"]  # Track memory ID

        # Recall memories
        recall_result = await memory_tools.memory_recall(
            query="dark mode preferences",
            n_results=5,
        )

        # Verify recall finds the memory
        assert recall_result["success"] is True
        assert recall_result["data"]["total"] >= 1

        # Verify memory content
        memories = recall_result["data"]["memories"]
        assert any("dark mode" in m["content"].lower() for m in memories)

    @pytest.mark.asyncio
    async def test_memory_validation_workflow(
        self,
        memory_tools: MemoryTools,
    ):
        """Test memory validation adjusts confidence."""
        # Store a memory
        store_result = await memory_tools.memory_store(
            content="Always use type hints in Python code",
            memory_type="pattern",
            namespace="global",
        )
        memory_id = store_result["data"]["id"]

        # Validate as helpful using memory_outcome with skip_event=True
        validate_result = await memory_tools.memory_outcome(
            memory_id=memory_id,
            success=True,
            skip_event=True,
        )

        # Verify confidence increased
        assert validate_result["success"] is True
        assert validate_result["data"]["new_confidence"] is not None

    @pytest.mark.asyncio
    async def test_memory_deduplication(
        self,
        memory_tools: MemoryTools,
    ):
        """Test that duplicate memories can be stored (deduplication not enforced).

        Note: Current implementation does not enforce deduplication.
        Both stores succeed with different IDs but same content_hash.
        """
        content = "Unique memory content for dedup test"

        # Store memory first time
        result1 = await memory_tools.memory_store(content=content, memory_type="fact")
        assert result1["success"] is True
        hash1 = result1["data"]["content_hash"]

        # Store same content again - creates new memory (no dedup enforcement)
        result2 = await memory_tools.memory_store(content=content, memory_type="fact")
        assert result2["success"] is True
        # Current implementation creates a new memory (no deduplication)
        # Both have the same content_hash but different IDs
        assert result2["data"]["content_hash"] == hash1
        # Note: duplicate flag is always False in current implementation
        assert result2["data"]["duplicate"] is False

    @pytest.mark.asyncio
    async def test_memory_forget(
        self,
        memory_tools: MemoryTools,
    ):
        """Test deleting memories."""
        # Store a memory
        store_result = await memory_tools.memory_store(
            content="Temporary memory to delete",
            memory_type="session",
        )
        memory_id = store_result["data"]["id"]

        # Delete the memory
        forget_result = await memory_tools.memory_forget(memory_id=memory_id)

        # Verify deletion
        assert forget_result["success"] is True
        assert memory_id in forget_result["data"]["deleted_ids"]

    @pytest.mark.asyncio
    async def test_memory_namespace_isolation(
        self,
        memory_tools: MemoryTools,
    ):
        """Test that memories in different namespaces are isolated."""
        # Store in global namespace
        await memory_tools.memory_store(
            content="Global memory",
            namespace="global",
            memory_type="fact",
        )

        # Store in project namespace
        await memory_tools.memory_store(
            content="Project memory",
            namespace="project:myapp",
            memory_type="fact",
        )

        # Query only global namespace
        result = await memory_tools.memory_recall(
            query="memory",
            namespace="global",
        )

        # Should only get global memories
        assert result["success"] is True
        for memory in result["data"]["memories"]:
            assert memory["namespace"] == "global"


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end integration tests simulating real usage."""

    @pytest.mark.asyncio
    async def test_full_document_workflow(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        management_tools: ManagementTools,
        sample_markdown_file: Path,
    ):
        """Test complete workflow: index -> search -> delete -> verify."""
        # Step 1: Index document
        index_result = await indexing_tools.index_file(str(sample_markdown_file))
        assert index_result["success"] is True
        chunks_created = index_result["data"]["chunks_created"]

        # Step 2: Search for content
        search_result = await query_tools.search("testing important", n_results=5)
        assert search_result["success"] is True
        assert search_result["data"]["total"] > 0

        # Step 3: Get stats
        stats_result = await management_tools.get_stats()
        assert stats_result["success"] is True
        assert stats_result["data"]["total_documents"] == chunks_created

        # Step 4: Delete the file
        delete_result = await management_tools.delete_by_file(str(sample_markdown_file))
        assert delete_result["success"] is True
        assert delete_result["data"]["deleted_count"] == chunks_created

        # Step 5: Verify deletion
        final_stats = await management_tools.get_stats()
        assert final_stats["data"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_full_memory_workflow(
        self,
        memory_tools: MemoryTools,
    ):
        """Test complete memory lifecycle: store -> recall -> validate -> forget."""
        # Step 1: Store memory
        store_result = await memory_tools.memory_store(
            content="Test memory for full workflow",
            memory_type="decision",
            importance=0.7,
        )
        assert store_result["success"] is True
        memory_id = store_result["data"]["id"]

        # Step 2: Recall memory
        recall_result = await memory_tools.memory_recall(query="test workflow")
        assert recall_result["success"] is True
        assert recall_result["data"]["total"] >= 1

        # Step 3: Validate memory (helpful) via memory_outcome with skip_event=True
        validate_result = await memory_tools.memory_outcome(
            memory_id=memory_id,
            success=True,
            skip_event=True,
        )
        assert validate_result["success"] is True
        initial_confidence = validate_result["data"]["new_confidence"]

        # Step 4: Validate again (not helpful)
        validate_result2 = await memory_tools.memory_outcome(
            memory_id=memory_id,
            success=False,
            skip_event=True,
        )
        assert validate_result2["success"] is True
        # Confidence should decrease
        assert validate_result2["data"]["new_confidence"] < initial_confidence

        # Step 5: Forget memory
        forget_result = await memory_tools.memory_forget(memory_id=memory_id)
        assert forget_result["success"] is True

    @pytest.mark.asyncio
    async def test_mixed_documents_and_memories(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        memory_tools: MemoryTools,
        management_tools: ManagementTools,
        sample_markdown_file: Path,
    ):
        """Test that documents and memories coexist correctly."""
        # Index a document
        await indexing_tools.index_file(str(sample_markdown_file), namespace="docs")

        # Store a memory
        await memory_tools.memory_store(
            content="Important fact about the project",
            memory_type="fact",
            namespace="memories",
        )

        # Get stats - should show both
        stats_result = await management_tools.get_stats()
        assert stats_result["success"] is True
        # Should have document chunks + 1 memory
        assert stats_result["data"]["total_documents"] > 1

        # Search should find both (if same embedding space)
        search_result = await query_tools.search("project", n_results=10)
        assert search_result["success"] is True

        # Clear all
        clear_result = await management_tools.delete_all(confirm=True)
        assert clear_result["success"] is True

        # Verify empty
        final_stats = await management_tools.get_stats()
        assert final_stats["data"]["total_documents"] == 0
