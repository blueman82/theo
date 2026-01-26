"""MCP tools module for Theo.

This module provides the tool implementations for the Theo MCP server,
organized by functional category:

- indexing_tools: Document indexing (index_file, index_directory)
- query_tools: Semantic search (search, search_with_filters, search_with_budget)
- memory_tools: Memory operations (memory_store, memory_recall, memory_validate)
- management_tools: Collection management (delete_chunks, delete_file, clear_index, get_stats)

Each tool module provides a class that encapsulates tool implementations
with dependencies injected via constructor.

Example:
    >>> from theo.tools import IndexingTools, QueryTools, MemoryTools
    >>> indexing = IndexingTools(daemon_client=client, chunker_registry=registry, store=store)
    >>> result = await indexing.index_file("/path/to/file.md")
"""

from theo.tools.indexing_tools import IndexingTools
from theo.tools.management_tools import ManagementTools
from theo.tools.memory_tools import MemoryTools
from theo.tools.query_tools import QueryTools

__all__ = [
    "IndexingTools",
    "QueryTools",
    "MemoryTools",
    "ManagementTools",
]
