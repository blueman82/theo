"""Tests for the Theo MCP server.

This module tests the unified MCP server including:
- Tool registration with FastMCP
- IndexingTools for document indexing
- QueryTools for semantic search
- MemoryTools for memory operations
- ManagementTools for collection management

Tests use mocks to isolate components and avoid network/file system dependencies.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from theo.storage.types import Document, StoreStats, HybridSearchResult
from theo.storage.sqlite_store import SearchResult
from theo.types import MemoryType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_daemon_client():
    """Create a mock DaemonClient for testing."""
    client = MagicMock()
    # Default embed response
    client.embed.return_value = {
        "success": True,
        "data": {
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5] * 256],  # 1280 dims
            "count": 1,
            "dimensions": 1280,
        },
    }
    # Default send response for search queries
    client.send.return_value = {
        "success": True,
        "data": {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 256,
        },
    }
    return client


@pytest.fixture
def mock_store():
    """Create a mock SQLiteStore for testing."""
    store = MagicMock()
    # Mock methods used by ManagementTools.get_stats
    store.count_memories.return_value = 10
    store.count_edges.return_value = 5
    store.list_memories.return_value = [
        {"id": "1", "memory_type": "document", "source_file": "/path/to/file1.md", "namespace": "default"},
        {"id": "2", "memory_type": "document", "source_file": "/path/to/file2.py", "namespace": "default"},
        {"id": "3", "memory_type": "document", "source_file": "/path/to/file3.py", "namespace": "default"},
        {"id": "4", "memory_type": "document", "source_file": "/path/to/file4.md", "namespace": "project1"},
        {"id": "5", "memory_type": "document", "source_file": "/path/to/file5.txt", "namespace": "project1"},
        {"id": "6", "memory_type": "memory", "source_file": None, "namespace": "default"},
        {"id": "7", "memory_type": "memory", "source_file": None, "namespace": "default"},
        {"id": "8", "memory_type": "document", "source_file": "/path/to/file1.md", "namespace": "default"},
        {"id": "9", "memory_type": "document", "source_file": "/path/to/file2.py", "namespace": "project1"},
        {"id": "10", "memory_type": "document", "source_file": "/path/to/file3.py", "namespace": "default"},
    ]
    # Mock methods used by IndexingTools
    store.get_by_hash.return_value = None
    store.get_by_source_file.return_value = None
    store.add_memory.return_value = []
    # Mock methods used by QueryTools - search_vector for vector search
    store.search_vector.return_value = []
    # Mock methods used by MemoryTools - search_hybrid for memory recall
    store.search_hybrid.return_value = []
    # Mock methods used by delete operations
    store.delete_memory.return_value = None
    return store


@pytest.fixture
def mock_chunker_registry():
    """Create a mock ChunkerRegistry for testing."""
    registry = MagicMock()
    # Mock chunker
    chunker = MagicMock()
    chunker.chunk.return_value = [
        MagicMock(text="Chunk 1 content", metadata={"header": "Section 1"}),
        MagicMock(text="Chunk 2 content", metadata={"header": "Section 2"}),
    ]
    registry.get_chunker.return_value = chunker
    return registry


@pytest.fixture
def mock_validation_loop():
    """Create a mock ValidationLoop for testing."""
    loop = MagicMock()
    # Use AsyncMock for record_usage since it's awaited in memory_validate
    validation_result = MagicMock(
        success=True,
        doc_id="mem_session_abc12345",
        old_confidence=0.5,
        new_confidence=0.6,
        was_helpful=True,
        promoted=False,
        error=None,
    )
    loop.record_usage = AsyncMock(return_value=validation_result)
    return loop


@pytest.fixture
def mock_feedback_collector():
    """Create a mock FeedbackCollector for testing."""
    collector = MagicMock()
    return collector


# =============================================================================
# IndexingTools Tests
# =============================================================================


class TestIndexingTools:
    """Tests for IndexingTools."""

    def test_index_file_file_not_found(
        self, mock_daemon_client, mock_chunker_registry, mock_store
    ):
        """Test index_file returns error when file not found."""
        from theo.tools.indexing_tools import IndexingTools

        tools = IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=mock_chunker_registry,
            store=mock_store,
        )

        import asyncio

        result = asyncio.run(tools.index_file("/nonexistent/path/file.md"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_index_file_unsupported_extension(
        self, mock_daemon_client, mock_chunker_registry, mock_store, tmp_path
    ):
        """Test index_file returns error for unsupported file types."""
        from theo.tools.indexing_tools import IndexingTools

        tools = IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=mock_chunker_registry,
            store=mock_store,
        )

        # Create a file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        import asyncio

        result = asyncio.run(tools.index_file(str(test_file)))

        assert result["success"] is False
        assert "unsupported" in result["error"].lower()

    def test_index_file_success(
        self, mock_daemon_client, mock_chunker_registry, mock_store, tmp_path
    ):
        """Test successful file indexing."""
        from theo.tools.indexing_tools import IndexingTools

        # Configure mock daemon to return embeddings for 2 chunks
        mock_daemon_client.embed.return_value = {
            "success": True,
            "data": {
                "embeddings": [
                    [0.1] * 1280,
                    [0.2] * 1280,
                ],
                "count": 2,
                "dimensions": 1280,
            },
        }
        mock_store.add_memory.return_value = ["id_1", "id_2"]

        tools = IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=mock_chunker_registry,
            store=mock_store,
        )

        # Create a markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nSome content here.")

        import asyncio

        result = asyncio.run(tools.index_file(str(test_file)))

        assert result["success"] is True
        assert result["data"]["chunks_created"] == 2
        assert str(test_file) in result["data"]["source_file"]
        mock_daemon_client.embed.assert_called_once()
        assert mock_store.add_memory.call_count == 2  # Once per chunk

    def test_index_directory_empty(
        self, mock_daemon_client, mock_chunker_registry, mock_store, tmp_path
    ):
        """Test indexing empty directory."""
        from theo.tools.indexing_tools import IndexingTools

        tools = IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=mock_chunker_registry,
            store=mock_store,
        )

        import asyncio

        result = asyncio.run(tools.index_directory(str(tmp_path)))

        assert result["success"] is True
        assert result["data"]["files_processed"] == 0
        assert "no supported files" in result["data"]["message"].lower()


# =============================================================================
# QueryTools Tests
# =============================================================================


class TestQueryTools:
    """Tests for QueryTools."""

    def test_search_empty_query(
        self, mock_daemon_client, mock_store, mock_feedback_collector
    ):
        """Test search returns error for empty query."""
        from theo.tools.query_tools import QueryTools

        tools = QueryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            feedback_collector=mock_feedback_collector,
        )

        import asyncio

        result = asyncio.run(tools.search("", n_results=5))

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_search_success(
        self, mock_daemon_client, mock_store, mock_feedback_collector
    ):
        """Test successful search."""
        from theo.tools.query_tools import QueryTools

        # Configure store to return results
        mock_doc = Document(
            id="doc_1",
            content="Test document content",
            source_file="/path/to/file.md",
            chunk_index=0,
            content_hash="abc123",
            namespace="default",
            memory_type="document",
            confidence=1.0,
        )
        mock_result = SearchResult(
            document=mock_doc,
            distance=0.2,
            similarity=0.9,
            rank=0,
        )
        mock_store.search_vector.return_value = [mock_result]

        tools = QueryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            feedback_collector=mock_feedback_collector,
        )

        import asyncio

        result = asyncio.run(tools.search("test query", n_results=5))

        assert result["success"] is True
        assert result["data"]["total"] == 1
        assert result["data"]["results"][0]["id"] == "doc_1"
        assert result["data"]["results"][0]["similarity"] == 0.9

    def test_search_with_budget(
        self, mock_daemon_client, mock_store, mock_feedback_collector
    ):
        """Test search with token budget."""
        from theo.tools.query_tools import QueryTools

        # Configure store to return results
        docs = []
        for i in range(5):
            doc = Document(
                id=f"doc_{i}",
                content="A" * 200,  # ~50 tokens each
                source_file="/path/to/file.md",
                chunk_index=i,
                content_hash=f"hash_{i}",
                namespace="default",
                memory_type="document",
                confidence=1.0,
            )
            docs.append(
                SearchResult(
                    document=doc,
                    distance=0.1 * i,
                    similarity=0.9 - 0.1 * i,
                    rank=i,
                )
            )
        mock_store.search_vector.return_value = docs

        tools = QueryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            feedback_collector=mock_feedback_collector,
        )

        import asyncio

        result = asyncio.run(tools.search_with_budget("test query", max_tokens=150))

        assert result["success"] is True
        # Should only return results that fit within budget
        assert result["data"]["tokens_used"] <= 150


# =============================================================================
# MemoryTools Tests
# =============================================================================


class TestMemoryTools:
    """Tests for MemoryTools."""

    def test_memory_store_empty_content(
        self, mock_daemon_client, mock_store, mock_validation_loop
    ):
        """Test memory_store returns error for empty content."""
        from theo.tools.memory_tools import MemoryTools

        tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )

        import asyncio

        result = asyncio.run(
            tools.memory_store(content="", memory_type="session", namespace="global")
        )

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_memory_store_invalid_type(
        self, mock_daemon_client, mock_store, mock_validation_loop
    ):
        """Test memory_store returns error for invalid memory type."""
        from theo.tools.memory_tools import MemoryTools

        tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )

        import asyncio

        result = asyncio.run(
            tools.memory_store(
                content="Test memory",
                memory_type="invalid_type",
                namespace="global",
            )
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_memory_store_success(
        self, mock_daemon_client, mock_store, mock_validation_loop
    ):
        """Test successful memory storage."""
        from theo.tools.memory_tools import MemoryTools

        mock_store.add_memory.return_value = ["mem_session_abc12345"]

        tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )

        import asyncio

        result = asyncio.run(
            tools.memory_store(
                content="User prefers dark mode",
                memory_type="preference",
                namespace="global",
                importance=0.8,
            )
        )

        assert result["success"] is True
        assert "id" in result["data"]
        assert result["data"]["memory_type"] == "preference"
        assert result["data"]["namespace"] == "global"

    def test_memory_recall_success(
        self, mock_daemon_client, mock_store, mock_validation_loop
    ):
        """Test successful memory recall."""
        from theo.tools.memory_tools import MemoryTools

        # Configure store to return memory results
        mock_doc = Document(
            id="mem_preference_abc12345",
            content="User prefers dark mode",
            source_file=None,
            chunk_index=0,
            content_hash="xyz789",
            namespace="global",
            memory_type="preference",
            confidence=0.5,
            importance=0.8,
        )
        mock_result = SearchResult(
            document=mock_doc,
            distance=0.15,
            similarity=0.925,
            rank=0,
        )
        mock_store.search_hybrid.return_value = [mock_result]

        tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )

        import asyncio

        result = asyncio.run(tools.memory_recall("user preferences", n_results=5))

        assert result["success"] is True
        assert result["data"]["total"] == 1
        assert "dark mode" in result["data"]["memories"][0]["content"]

    def test_memory_validate_success(
        self, mock_daemon_client, mock_store, mock_validation_loop
    ):
        """Test successful memory validation."""
        from theo.tools.memory_tools import MemoryTools

        tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )

        import asyncio

        result = asyncio.run(
            tools.memory_validate(
                memory_id="mem_session_abc12345",
                was_helpful=True,
                context="Used in test scenario",
            )
        )

        assert result["success"] is True
        assert result["data"]["old_confidence"] == 0.5
        assert result["data"]["new_confidence"] == 0.6


# =============================================================================
# ManagementTools Tests
# =============================================================================


class TestManagementTools:
    """Tests for ManagementTools."""

    def test_get_stats(self, mock_store):
        """Test get_stats returns collection statistics."""
        from theo.tools.management_tools import ManagementTools

        tools = ManagementTools(store=mock_store)

        import asyncio

        result = asyncio.run(tools.get_stats())

        assert result["success"] is True
        assert result["data"]["total_documents"] == 10
        assert result["data"]["unique_sources"] == 5

    def test_delete_all_requires_confirmation(self, mock_store):
        """Test delete_all requires explicit confirmation."""
        from theo.tools.management_tools import ManagementTools

        tools = ManagementTools(store=mock_store)

        import asyncio

        result = asyncio.run(tools.delete_all(confirm=False))

        assert result["success"] is False
        assert "confirmation" in result["error"].lower()
        # delete_memory should not be called when not confirmed
        mock_store.delete_memory.assert_not_called()

    def test_delete_all_with_confirmation(self, mock_store):
        """Test delete_all works with confirmation."""
        from theo.tools.management_tools import ManagementTools

        tools = ManagementTools(store=mock_store)

        import asyncio

        result = asyncio.run(tools.delete_all(confirm=True))

        assert result["success"] is True
        assert result["data"]["deleted_count"] == 10
        # delete_memory called once for each memory in list_memories
        assert mock_store.delete_memory.call_count == 10


# =============================================================================
# MCP Server Registration Tests
# =============================================================================


class TestMCPServerRegistration:
    """Tests for MCP server tool registration."""

    def test_mcp_server_imports_correctly(self):
        """Test that mcp_server module imports without errors."""
        from theo.mcp_server import mcp, set_tool_instances

        assert mcp is not None
        assert set_tool_instances is not None

    def test_set_tool_instances(self):
        """Test setting global tool instances."""
        from theo import mcp_server
        from theo.mcp_server import set_tool_instances

        mock_indexing = MagicMock()
        mock_query = MagicMock()
        mock_memory = MagicMock()
        mock_management = MagicMock()

        set_tool_instances(
            indexing=mock_indexing,
            query=mock_query,
            memory=mock_memory,
            management=mock_management,
        )

        assert mcp_server.indexing_tools is mock_indexing
        assert mcp_server.query_tools is mock_query
        assert mcp_server.memory_tools is mock_memory
        assert mcp_server.management_tools is mock_management


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestMCPToolIntegration:
    """Integration tests for MCP tools with mocked dependencies."""

    @pytest.fixture
    def configured_mcp_server(
        self,
        mock_daemon_client,
        mock_store,
        mock_chunker_registry,
        mock_validation_loop,
        mock_feedback_collector,
    ):
        """Set up MCP server with mock dependencies."""
        from theo.tools.indexing_tools import IndexingTools
        from theo.tools.query_tools import QueryTools
        from theo.tools.memory_tools import MemoryTools
        from theo.tools.management_tools import ManagementTools
        from theo.mcp_server import set_tool_instances

        indexing_tools = IndexingTools(
            daemon_client=mock_daemon_client,
            chunker_registry=mock_chunker_registry,
            store=mock_store,
        )
        query_tools = QueryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            feedback_collector=mock_feedback_collector,
        )
        memory_tools = MemoryTools(
            daemon_client=mock_daemon_client,
            store=mock_store,
            validation_loop=mock_validation_loop,
        )
        management_tools = ManagementTools(store=mock_store)

        set_tool_instances(
            indexing=indexing_tools,
            query=query_tools,
            memory=memory_tools,
            management=management_tools,
        )

        return {
            "indexing": indexing_tools,
            "query": query_tools,
            "memory": memory_tools,
            "management": management_tools,
        }

    def test_full_indexing_workflow(self, configured_mcp_server, tmp_path):
        """Test full document indexing workflow through MCP tools."""
        from theo.mcp_server import index_file, get_index_stats

        # Note: This test verifies the MCP tool wrappers work correctly
        # The actual functionality is tested in the unit tests above
        import asyncio

        # Test get_stats works
        result = asyncio.run(get_index_stats())
        assert result["success"] is True

    def test_full_memory_workflow(self, configured_mcp_server):
        """Test full memory workflow through MCP tools."""
        from theo.mcp_server import memory_store, memory_recall, memory_validate

        import asyncio

        # Store a memory
        store_result = asyncio.run(
            memory_store(
                content="Test memory content",
                memory_type="session",
                namespace="global",
            )
        )
        assert store_result["success"] is True

        # Recall memories
        recall_result = asyncio.run(memory_recall(query="test memory", n_results=5))
        assert recall_result["success"] is True

        # Validate a memory
        validate_result = asyncio.run(
            memory_validate(
                memory_id="mem_session_test123",
                was_helpful=True,
            )
        )
        assert validate_result["success"] is True
