"""Integration tests for Theo MCP tool handlers.

This module provides end-to-end tests for MCP tool invocations:
- Tool registration with FastMCP
- Request -> Handler -> Response flow
- Error handling and validation
- Component integration through tools

Tests use real tool instances with mock embedding providers
to ensure integration between all components.

Usage:
    uv run pytest tests/integration/test_mcp_tools.py -v
"""

import asyncio
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
    """Create temporary database path."""
    return tmp_path / "test_mcp.db"


@pytest.fixture
def mock_daemon_client() -> MagicMock:
    """Create mock DaemonClient for testing.

    Returns deterministic embeddings for testing MCP tool integration.
    """
    client = MagicMock()

    def mock_embed(texts: list[str]) -> dict[str, Any]:
        """Generate deterministic embeddings."""
        embeddings = []
        for text in texts:
            text_hash = hash(text[:100] if text else "")
            embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
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
        """Generate query embedding."""
        text_hash = hash(query[:100] if query else "")
        embedding = [((text_hash + i) % 100) / 100.0 for i in range(1024)]
        magnitude = sum(x * x for x in embedding) ** 0.5

        return {
            "success": True,
            "data": {
                "embedding": [x / magnitude for x in embedding],
                "dimensions": 1024,
            },
        }

    client.embed = MagicMock(side_effect=mock_embed)
    client.send = MagicMock(
        side_effect=lambda cmd, **kwargs: (
            mock_search(kwargs.get("query", ""))
            if cmd == "search"
            else {"success": False, "error": f"Unknown: {cmd}"}
        )
    )

    return client


@pytest.fixture
def sqlite_store(temp_db_path: Path) -> Generator[SQLiteStore, None, None]:
    """Create SQLiteStore with temporary database."""
    store = SQLiteStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def chunker_registry() -> ChunkerRegistry:
    """Create ChunkerRegistry."""
    return ChunkerRegistry()


@pytest.fixture
def validation_loop(sqlite_store: SQLiteStore) -> ValidationLoop:
    """Create ValidationLoop."""
    return ValidationLoop(store=sqlite_store)


@pytest.fixture
def feedback_collector() -> FeedbackCollector:
    """Create FeedbackCollector."""
    return FeedbackCollector()


@pytest.fixture
def all_tools(
    mock_daemon_client: MagicMock,
    chunker_registry: ChunkerRegistry,
    sqlite_store: SQLiteStore,
    validation_loop: ValidationLoop,
    feedback_collector: FeedbackCollector,
) -> dict[str, Any]:
    """Create all tool instances for testing."""
    return {
        "indexing": IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=chunker_registry,
            store=sqlite_store,
        ),
        "query": QueryTools(
            daemon_client=mock_daemon_client,
            store=sqlite_store,
            feedback_collector=feedback_collector,
        ),
        "memory": MemoryTools(
            daemon_client=mock_daemon_client,
            store=sqlite_store,
            validation_loop=validation_loop,
        ),
        "management": ManagementTools(store=sqlite_store),
        "store": sqlite_store,
    }


@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """Create sample test files."""
    files = {}

    # Markdown file
    md_path = tmp_path / "readme.md"
    md_path.write_text(
        """# Project README

## Overview

This is a sample project for testing MCP tools.

## Features

- Feature A: Document indexing
- Feature B: Semantic search
- Feature C: Memory storage

## Usage

Run the tests with pytest.
"""
    )
    files["markdown"] = md_path

    # Python file
    py_path = tmp_path / "example.py"
    py_path.write_text(
        '''"""Example Python module."""


def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def calculate(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
'''
    )
    files["python"] = py_path

    # Text file
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text(
        """Important notes for the project.

First point: Always run tests before committing.

Second point: Document all public APIs.

Third point: Use type hints consistently.
"""
    )
    files["text"] = txt_path

    return files


# =============================================================================
# MCP Server Registration Tests
# =============================================================================


class TestMCPServerRegistration:
    """Test MCP server module and tool registration."""

    def test_mcp_server_imports(self):
        """Test mcp_server module imports without errors."""
        from theo.mcp_server import mcp, set_tool_instances

        assert mcp is not None
        assert set_tool_instances is not None

    def test_set_tool_instances(self, all_tools: dict[str, Any]):
        """Test setting global tool instances."""
        from theo import mcp_server
        from theo.mcp_server import set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        assert mcp_server.indexing_tools is all_tools["indexing"]
        assert mcp_server.query_tools is all_tools["query"]
        assert mcp_server.memory_tools is all_tools["memory"]
        assert mcp_server.management_tools is all_tools["management"]

    def test_tools_not_initialized_returns_error(self):
        """Test tools return error when not initialized."""
        from theo import mcp_server

        # Save and clear
        saved = {
            "indexing": mcp_server.indexing_tools,
            "query": mcp_server.query_tools,
            "memory": mcp_server.memory_tools,
            "management": mcp_server.management_tools,
        }

        try:
            mcp_server.indexing_tools = None
            mcp_server.query_tools = None
            mcp_server.memory_tools = None
            mcp_server.management_tools = None

            # Tools should return error
            from theo.mcp_server import (
                get_index_stats,
                index_file,
                memory_store,
                search,
            )

            # Run async tools
            result = asyncio.run(index_file("/test/path.md"))
            assert result["success"] is False
            assert "not initialized" in result["error"].lower()

            result = asyncio.run(search("test query"))
            assert result["success"] is False
            assert "not initialized" in result["error"].lower()

            result = asyncio.run(memory_store(content="test"))
            assert result["success"] is False
            assert "not initialized" in result["error"].lower()

            result = asyncio.run(get_index_stats())
            assert result["success"] is False
            assert "not initialized" in result["error"].lower()

        finally:
            # Restore
            mcp_server.indexing_tools = saved["indexing"]
            mcp_server.query_tools = saved["query"]
            mcp_server.memory_tools = saved["memory"]
            mcp_server.management_tools = saved["management"]


# =============================================================================
# Indexing Tool Tests
# =============================================================================


class TestIndexingTools:
    """Test indexing tool handlers."""

    @pytest.mark.asyncio
    async def test_index_file_success(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test successful file indexing."""
        from theo.mcp_server import index_file, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await index_file(str(sample_files["markdown"]))

        assert result["success"] is True
        assert result["data"]["chunks_created"] > 0
        assert str(sample_files["markdown"]) in result["data"]["source_file"]

    @pytest.mark.asyncio
    async def test_index_file_not_found(self, all_tools: dict[str, Any]):
        """Test indexing non-existent file returns error."""
        from theo.mcp_server import index_file, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await index_file("/nonexistent/path/file.md")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_file_unsupported_type(
        self,
        all_tools: dict[str, Any],
        tmp_path: Path,
    ):
        """Test indexing unsupported file type returns error."""
        from theo.mcp_server import index_file, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Create unsupported file
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("content")

        result = await index_file(str(unsupported))

        assert result["success"] is False
        assert "unsupported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_directory_success(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test successful directory indexing."""
        from theo.mcp_server import index_directory, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        directory = sample_files["markdown"].parent
        # Use async_mode=False for synchronous processing with expected keys
        result = await index_directory(str(directory), recursive=False, async_mode=False)

        assert result["success"] is True
        assert result["data"]["files_processed"] >= 3
        assert result["data"]["total_chunks"] > 0


# =============================================================================
# Query Tool Tests
# =============================================================================


class TestQueryTools:
    """Test query tool handlers."""

    @pytest.mark.asyncio
    async def test_search_with_indexed_content(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test search after indexing content."""
        from theo.mcp_server import index_file, search, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index first
        await index_file(str(sample_files["markdown"]))

        # Then search
        result = await search("project features", n_results=5)

        assert result["success"] is True
        assert "results" in result["data"]
        assert "total" in result["data"]

    @pytest.mark.asyncio
    async def test_search_empty_query_error(self, all_tools: dict[str, Any]):
        """Test search with empty query returns error."""
        from theo.mcp_server import search, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await search("", n_results=5)

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test search with metadata filters."""
        from theo.mcp_server import (
            index_file,
            search_with_filters,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index with namespace
        await index_file(str(sample_files["markdown"]), namespace="docs")

        # Search with filter
        result = await search_with_filters(
            query="features",
            filters={"namespace": "docs"},
            n_results=5,
        )

        assert result["success"] is True
        # All results should match namespace
        for item in result["data"]["results"]:
            assert item["namespace"] == "docs"

    @pytest.mark.asyncio
    async def test_search_with_budget(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test search with token budget constraint."""
        from theo.mcp_server import index_file, search_with_budget, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index content
        await index_file(str(sample_files["markdown"]))

        # Search with budget
        max_tokens = 100
        result = await search_with_budget("readme features", max_tokens=max_tokens)

        assert result["success"] is True
        assert result["data"]["tokens_used"] <= max_tokens


# =============================================================================
# Memory Tool Tests
# =============================================================================


class TestMemoryTools:
    """Test memory tool handlers."""

    @pytest.mark.asyncio
    async def test_memory_store_success(self, all_tools: dict[str, Any]):
        """Test successful memory storage."""
        from theo.mcp_server import memory_store, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await memory_store(
            content="User prefers dark mode",
            memory_type="preference",
            namespace="global",
            importance=0.8,
        )

        assert result["success"] is True
        assert "id" in result["data"]
        assert result["data"]["memory_type"] == "preference"

    @pytest.mark.asyncio
    async def test_memory_store_invalid_type(self, all_tools: dict[str, Any]):
        """Test memory storage with invalid type returns error."""
        from theo.mcp_server import memory_store, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await memory_store(
            content="Test memory",
            memory_type="invalid_type",
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_memory_store_empty_content(self, all_tools: dict[str, Any]):
        """Test memory storage with empty content returns error."""
        from theo.mcp_server import memory_store, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await memory_store(content="", memory_type="fact")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_memory_recall_success(self, all_tools: dict[str, Any]):
        """Test successful memory recall."""
        from theo.mcp_server import memory_recall, memory_store, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Store first
        await memory_store(
            content="Important decision about architecture",
            memory_type="decision",
        )

        # Recall
        result = await memory_recall(query="architecture decision")

        assert result["success"] is True
        assert "memories" in result["data"]
        assert "total" in result["data"]

    @pytest.mark.asyncio
    async def test_memory_outcome_direct_validation(self, all_tools: dict[str, Any]):
        """Test memory_outcome with skip_event=True for direct validation."""
        from theo.mcp_server import (
            memory_outcome,
            memory_store,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Store memory
        store_result = await memory_store(content="Test pattern", memory_type="pattern")
        memory_id = store_result["data"]["id"]

        # Validate using memory_outcome with skip_event=True
        result = await memory_outcome(memory_id=memory_id, success=True, skip_event=True)

        assert result["success"] is True
        assert result["data"]["new_confidence"] is not None

    @pytest.mark.asyncio
    async def test_memory_forget_by_id(self, all_tools: dict[str, Any]):
        """Test memory deletion by ID."""
        from theo.mcp_server import memory_forget, memory_store, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Store memory
        store_result = await memory_store(
            content="Memory to delete",
            memory_type="session",
        )
        memory_id = store_result["data"]["id"]

        # Forget
        result = await memory_forget(memory_id=memory_id)

        assert result["success"] is True
        assert memory_id in result["data"]["deleted_ids"]

    @pytest.mark.asyncio
    async def test_memory_context_generation(self, all_tools: dict[str, Any]):
        """Test memory context generation for LLM injection."""
        from theo.mcp_server import (
            memory_context,
            memory_store,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Store some memories
        await memory_store(
            content="User prefers vim keybindings",
            memory_type="preference",
            importance=0.9,
        )
        await memory_store(
            content="Project uses FastAPI for backend",
            memory_type="decision",
            importance=0.8,
        )

        # Generate context
        result = await memory_context(token_budget=2000)

        assert result["success"] is True
        assert "context" in result["data"]
        assert result["data"]["token_estimate"] <= 2000


# =============================================================================
# Management Tool Tests
# =============================================================================


class TestManagementTools:
    """Test management tool handlers."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, all_tools: dict[str, Any]):
        """Test getting stats on empty collection."""
        from theo.mcp_server import get_index_stats, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        result = await get_index_stats()

        assert result["success"] is True
        assert result["data"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_content(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test getting stats after indexing."""
        from theo.mcp_server import get_index_stats, index_file, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index file
        await index_file(str(sample_files["markdown"]))

        # Get stats
        result = await get_index_stats()

        assert result["success"] is True
        assert result["data"]["total_documents"] > 0
        assert result["data"]["unique_sources"] >= 1

    @pytest.mark.asyncio
    async def test_delete_chunks_by_ids(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test deleting specific chunks by IDs."""
        from theo.mcp_server import (
            delete_chunks,
            get_index_stats,
            index_file,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index file
        index_result = await index_file(str(sample_files["markdown"]))
        chunk_ids = index_result["data"]["chunk_ids"]

        # Delete first chunk
        delete_result = await delete_chunks(ids=[chunk_ids[0]])

        assert delete_result["success"] is True
        assert delete_result["data"]["deleted_count"] == 1

        # Verify count decreased
        stats = await get_index_stats()
        assert stats["data"]["total_documents"] == len(chunk_ids) - 1

    @pytest.mark.asyncio
    async def test_delete_file(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test deleting all chunks from a file."""
        from theo.mcp_server import (
            delete_file,
            get_index_stats,
            index_file,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index file
        index_result = await index_file(str(sample_files["markdown"]))
        chunks_created = index_result["data"]["chunks_created"]

        # Delete file
        delete_result = await delete_file(source_file=str(sample_files["markdown"]))

        assert delete_result["success"] is True
        assert delete_result["data"]["deleted_count"] == chunks_created

        # Verify empty
        stats = await get_index_stats()
        assert stats["data"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_clear_index_requires_confirmation(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test clear_index requires confirmation."""
        from theo.mcp_server import clear_index, index_file, set_tool_instances

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index file
        await index_file(str(sample_files["markdown"]))

        # Try to clear without confirmation
        result = await clear_index(confirm=False)

        assert result["success"] is False
        assert "confirm" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_clear_index_with_confirmation(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test clear_index with confirmation."""
        from theo.mcp_server import (
            clear_index,
            get_index_stats,
            index_file,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index file
        await index_file(str(sample_files["markdown"]))

        # Clear with confirmation
        result = await clear_index(confirm=True)

        assert result["success"] is True
        assert result["data"]["deleted_count"] > 0

        # Verify empty
        stats = await get_index_stats()
        assert stats["data"]["total_documents"] == 0


# =============================================================================
# Integration Workflow Tests
# =============================================================================


class TestMCPToolWorkflows:
    """Test complete workflows using MCP tools."""

    @pytest.mark.asyncio
    async def test_complete_document_workflow(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test complete document workflow through MCP tools."""
        from theo.mcp_server import (
            clear_index,
            get_index_stats,
            index_directory,
            search,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        directory = sample_files["markdown"].parent

        # 1. Index directory (use async_mode=False for synchronous processing)
        index_result = await index_directory(str(directory), recursive=False, async_mode=False)
        assert index_result["success"] is True
        assert index_result["data"]["files_processed"] >= 3

        # 2. Search
        search_result = await search("python function", n_results=5)
        assert search_result["success"] is True

        # 3. Get stats
        stats_result = await get_index_stats()
        assert stats_result["success"] is True
        assert stats_result["data"]["total_documents"] > 0

        # 4. Clear
        clear_result = await clear_index(confirm=True)
        assert clear_result["success"] is True

        # 5. Verify empty
        final_stats = await get_index_stats()
        assert final_stats["data"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_complete_memory_workflow(self, all_tools: dict[str, Any]):
        """Test complete memory workflow through MCP tools."""
        from theo.mcp_server import (
            memory_context,
            memory_forget,
            memory_outcome,
            memory_recall,
            memory_store,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # 1. Store memories
        store1 = await memory_store(
            content="Always use type hints",
            memory_type="pattern",
            importance=0.9,
        )
        assert store1["success"] is True
        id1 = store1["data"]["id"]

        store2 = await memory_store(
            content="Prefer composition over inheritance",
            memory_type="pattern",
            importance=0.8,
        )
        assert store2["success"] is True
        id2 = store2["data"]["id"]

        # 2. Recall
        recall_result = await memory_recall(query="coding patterns")
        assert recall_result["success"] is True
        assert recall_result["data"]["total"] >= 2

        # 3. Validate
        validate_result = await memory_validate(memory_id=id1, was_helpful=True)
        assert validate_result["success"] is True

        # 4. Get context
        context_result = await memory_context(token_budget=1000)
        assert context_result["success"] is True
        assert "context" in context_result["data"]

        # 5. Forget
        forget_result = await memory_forget(memory_id=id2)
        assert forget_result["success"] is True

    @pytest.mark.asyncio
    async def test_mixed_document_and_memory_workflow(
        self,
        all_tools: dict[str, Any],
        sample_files: dict[str, Path],
    ):
        """Test workflow mixing documents and memories."""
        from theo.mcp_server import (
            clear_index,
            get_index_stats,
            index_file,
            memory_recall,
            memory_store,
            search,
            set_tool_instances,
        )

        set_tool_instances(
            indexing=all_tools["indexing"],
            query=all_tools["query"],
            memory=all_tools["memory"],
            management=all_tools["management"],
        )

        # Index document
        await index_file(str(sample_files["markdown"]), namespace="docs")

        # Store memory
        await memory_store(
            content="Project uses semantic search",
            memory_type="fact",
            namespace="memories",
        )

        # Stats should show both
        stats = await get_index_stats()
        assert stats["data"]["total_documents"] > 1

        # Search documents
        doc_search = await search("project features")
        assert doc_search["success"] is True

        # Recall memories
        mem_recall = await memory_recall(query="semantic search")
        assert mem_recall["success"] is True

        # Clear all
        await clear_index(confirm=True)
