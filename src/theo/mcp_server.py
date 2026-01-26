"""Theo MCP server module.

This module provides the FastMCP server instance and tool registration for Theo,
a unified AI memory and document retrieval system combining DocVec and Recall.

The server exposes tools for:
- Document indexing (index_file, index_directory)
- Semantic search (search, search_with_filters, search_with_budget)
- Memory operations (memory_store, memory_recall, memory_validate, memory_forget)
- Collection management (delete_chunks, delete_file, clear_index, get_stats)

Architectural justification:
- Uses FastMCP for MCP protocol compliance
- All logging goes to stderr (never stdout) for stdio transport
- DaemonClient integration for non-blocking embedding operations
- ValidationLoop integration for confidence scoring

CRITICAL: MCP servers using stdio transport must NEVER write to stdout
as it corrupts JSON-RPC messages. All logging goes to stderr.
"""

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("theo")


# ==============================================================================
# Global Tool Instances (initialized in main/__main__.py)
# ==============================================================================

# These are set by the main entry point after component initialization
indexing_tools: Optional[Any] = None
query_tools: Optional[Any] = None
memory_tools: Optional[Any] = None
management_tools: Optional[Any] = None


def set_tool_instances(
    indexing: Any,
    query: Any,
    memory: Any,
    management: Any,
) -> None:
    """Set global tool instances after initialization.

    Called by __main__.py after components are initialized.

    Args:
        indexing: IndexingTools instance
        query: QueryTools instance
        memory: MemoryTools instance
        management: ManagementTools instance
    """
    global indexing_tools, query_tools, memory_tools, management_tools
    indexing_tools = indexing
    query_tools = query
    memory_tools = memory
    management_tools = management


# ==============================================================================
# Document Indexing Tools (from DocVec)
# ==============================================================================


@mcp.tool()
async def index_file(file_path: str, namespace: str = "default") -> dict[str, Any]:
    """Index a single document file.

    Reads the file, chunks it using the appropriate chunker (based on extension),
    generates embeddings, and stores in the vector database.

    Supported file types: .md, .markdown, .txt, .py, .pdf

    Args:
        file_path: Absolute or relative path to the file to index
        namespace: Namespace for organizing documents (default: "default")

    Returns:
        Result dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with chunks_created, source_file, namespace
        - error: Error message if operation failed
    """
    if indexing_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await indexing_tools.index_file(file_path, namespace=namespace)


@mcp.tool()
async def index_directory(
    dir_path: str,
    recursive: bool = True,
    namespace: str = "default",
    async_mode: bool = True,
) -> dict[str, Any]:
    """Index all supported files in a directory.

    Walks the directory (optionally recursively), indexing all files
    with supported extensions.

    Args:
        dir_path: Absolute or relative path to the directory to index
        recursive: Whether to recursively index subdirectories (default: True)
        namespace: Namespace for organizing documents (default: "default")
        async_mode: If True (default), queue chunks for fast background processing.
                   If False, wait for all embeddings synchronously.

    Returns:
        Result dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with files_indexed, total_chunks, failed_files, etc.
        - error: Error message if operation failed
    """
    if indexing_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await indexing_tools.index_directory(
        dir_path, recursive=recursive, namespace=namespace, async_mode=async_mode
    )


# ==============================================================================
# Search Tools (from DocVec)
# ==============================================================================


@mcp.tool()
async def search(query: str, n_results: int = 5) -> dict[str, Any]:
    """Perform semantic search for a query.

    Generates query embedding and searches for similar documents
    in the vector database.

    Args:
        query: Search query string
        n_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with results, total, query, total_tokens
        - error: Error message if operation failed
    """
    if query_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await query_tools.search(query, n_results)


@mcp.tool()
async def search_with_filters(
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
    if query_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await query_tools.search_with_filters(query, filters, n_results)


@mcp.tool()
async def search_with_budget(query: str, max_tokens: int) -> dict[str, Any]:
    """Search and return results within a token budget.

    Fetches results and truncates to fit within the specified
    token budget.

    Args:
        query: Search query string
        max_tokens: Maximum total tokens allowed in results

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with results, tokens_used, budget_remaining
        - error: Error message if operation failed
    """
    if query_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await query_tools.search_with_budget(query, max_tokens)


# ==============================================================================
# Memory Tools (from Recall)
# ==============================================================================


@mcp.tool()
async def memory_store(
    content: str,
    memory_type: str = "session",
    namespace: str = "global",
    importance: float = 0.5,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Store a new memory with semantic indexing.

    Creates a new memory with the specified content, type, and namespace.
    Deduplication: If a memory with the same content already exists,
    the existing memory ID is returned.

    Args:
        content: The memory content text
        memory_type: Type of memory (preference, decision, pattern, session, fact)
        namespace: Scope of the memory (global, default, or project:{name})
        importance: Importance score from 0.0 to 1.0 (default: 0.5)
        metadata: Optional additional metadata

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with id, content_hash, namespace, duplicate flag
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_store(
        content=content,
        memory_type=memory_type,
        namespace=namespace,
        importance=importance,
        metadata=metadata,
    )


@mcp.tool()
async def memory_recall(
    query: str,
    n_results: int = 5,
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_importance: Optional[float] = None,
    min_confidence: Optional[float] = None,
    include_related: bool = True,
    max_depth: int = 1,
) -> dict[str, Any]:
    """Recall memories using semantic search with optional graph expansion.

    Searches for memories similar to the query, with optional filtering.

    Args:
        query: Search query text
        n_results: Maximum number of results (default: 5)
        namespace: Filter by namespace (optional)
        memory_type: Filter by memory type (optional)
        min_importance: Minimum importance score filter (optional)
        min_confidence: Minimum confidence score filter (optional)
        include_related: If True, expand results via graph edges (default: False)
        max_depth: Maximum graph traversal depth (default: 1)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with memories, total, query, filters, and optionally expanded
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_recall(
        query=query,
        n_results=n_results,
        namespace=namespace,
        memory_type=memory_type,
        min_importance=min_importance,
        min_confidence=min_confidence,
        include_related=include_related,
        max_depth=max_depth,
    )


@mcp.tool()
async def memory_validate(
    memory_id: str,
    was_helpful: bool,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """Validate a memory and adjust its confidence score.

    Records whether a memory was helpful and adjusts confidence:
    - Helpful: confidence += 0.1 (max 1.0)
    - Not helpful: confidence -= 0.15 (min 0.0)

    Memories reaching confidence >= 0.9 become golden rules.

    Args:
        memory_id: ID of the memory to validate
        was_helpful: Whether the memory was helpful
        context: Optional context describing the usage

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with old_confidence, new_confidence, promoted
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_validate(
        memory_id=memory_id,
        was_helpful=was_helpful,
        context=context,
    )


@mcp.tool()
async def memory_forget(
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    input_value: Optional[str] = None,
    namespace: Optional[str] = None,
    n_results: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Delete memories by ID or semantic search.

    Golden rules (type=golden_rule or confidence >= 0.9) are protected
    from deletion unless force=True.

    Args:
        memory_id: Specific memory ID to delete
        query: Search query to find memories to delete
        input_value: Smart parameter - auto-detects if ID or query
        namespace: Filter deletion to specific namespace
        n_results: Number of search results to delete in query mode
        force: If True, allow deletion of golden rules

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with deleted_ids, deleted_count, protected_ids
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_forget(
        memory_id=memory_id,
        query=query,
        input_value=input_value,
        namespace=namespace,
        n_results=n_results,
        force=force,
    )


@mcp.tool()
async def memory_context(
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    token_budget: int = 4000,
) -> dict[str, Any]:
    """Fetch relevant memories and format for context injection.

    Retrieves memories relevant to the query and formats them
    as markdown suitable for injection into an LLM context.

    Args:
        query: Optional search query to filter relevant memories
        namespace: Project namespace (optional)
        token_budget: Maximum tokens for context (default: 4000)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with context markdown and token_estimate
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_context(
        query=query,
        namespace=namespace,
        token_budget=token_budget,
    )


# ==============================================================================
# TRY/LEARN Cycle Tools
# ==============================================================================


@mcp.tool()
async def memory_apply(
    memory_id: str,
    context: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Record that a memory is being applied.

    Creates a validation event to track when a memory is used in practice.
    This starts the TRY phase of the validation loop.

    Args:
        memory_id: ID of the memory being applied
        context: Description of how/where the memory is being applied
        session_id: Optional session identifier

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with memory_id and event_id
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_apply(
        memory_id=memory_id,
        context=context,
        session_id=session_id,
    )


@mcp.tool()
async def memory_outcome(
    memory_id: str,
    success: bool,
    error_msg: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Record the outcome of a memory application and adjust confidence.

    Records whether applying a memory succeeded or failed, creating a
    validation event and adjusting confidence accordingly.

    Args:
        memory_id: ID of the memory that was applied
        success: Whether the application was successful
        error_msg: Optional error message if failed
        session_id: Optional session identifier

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with memory_id, outcome_success, new_confidence, promoted
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_outcome(
        memory_id=memory_id,
        success=success,
        error_msg=error_msg,
        session_id=session_id,
    )


# ==============================================================================
# Graph Relationship Tools
# ==============================================================================


@mcp.tool()
async def memory_relate(
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Create a relationship between two memories.

    Args:
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation: Type of relationship (relates_to, supersedes, caused_by, contradicts)
        weight: Edge weight (default: 1.0)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with edge_id
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_relate(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        weight=weight,
    )


@mcp.tool()
async def memory_edge_forget(
    edge_id: Optional[int] = None,
    memory_id: Optional[str] = None,
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    relation: Optional[str] = None,
    direction: str = "both",
) -> dict[str, Any]:
    """Delete edges (relationships) between memories.

    Supports three deletion modes:
    1. Direct ID: Delete a specific edge by its ID
    2. Memory-based: Delete all edges connected to a memory
    3. Pair: Delete edge(s) between two specific memories

    Args:
        edge_id: Specific edge ID to delete
        memory_id: Memory ID to delete all connected edges
        source_id: Source memory ID for pair deletion
        target_id: Target memory ID for pair deletion
        relation: Filter by relation type (optional)
        direction: For memory_id mode: 'outgoing', 'incoming', or 'both'

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with deleted_ids and deleted_count
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_edge_forget(
        edge_id=edge_id,
        memory_id=memory_id,
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        direction=direction,
    )


@mcp.tool()
async def memory_inspect_graph(
    memory_id: str,
    max_depth: int = 2,
    direction: str = "both",
    edge_types: Optional[list[str]] = None,
    include_scores: bool = True,
    decay_factor: float = 0.7,
    output_format: str = "json",
) -> dict[str, Any]:
    """Inspect the graph structure around a memory node.

    Performs read-only breadth-first search from the origin memory,
    collecting all nodes and edges within max_depth hops.

    Args:
        memory_id: ID of the memory to start inspection from
        max_depth: Maximum number of hops to traverse (default: 2)
        direction: Edge traversal direction - "outgoing", "incoming", or "both"
        edge_types: Optional list of edge types to include
        include_scores: If True, compute relevance scores for paths
        decay_factor: Factor by which relevance decays per hop
        output_format: Output format - "json" or "mermaid"

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - origin_id: The starting memory ID
        - nodes/mermaid: Graph structure
        - edges: List of edge dicts
        - paths: List of path dicts
        - stats: Summary statistics
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_inspect_graph(
        memory_id=memory_id,
        max_depth=max_depth,
        direction=direction,
        edge_types=edge_types,
        include_scores=include_scores,
        decay_factor=decay_factor,
        output_format=output_format,
    )


# ==============================================================================
# Memory Inspection Tools
# ==============================================================================


@mcp.tool()
async def memory_count(
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> dict[str, Any]:
    """Count memories with optional filters.

    Args:
        namespace: Filter by namespace (optional)
        memory_type: Filter by type (optional)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with count and filters
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_count(
        namespace=namespace,
        memory_type=memory_type,
    )


@mcp.tool()
async def memory_list(
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at",
    descending: bool = True,
) -> dict[str, Any]:
    """List memories with filtering and pagination.

    Args:
        namespace: Filter by namespace (optional)
        memory_type: Filter by type (optional)
        limit: Maximum number of results (default: 100, max: 1000)
        offset: Number of results to skip for pagination
        order_by: Field to sort by (default: 'created_at')
        descending: Sort in descending order (default: True)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with memories list and pagination info
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_list(
        namespace=namespace,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
        order_by=order_by,
        descending=descending,
    )


@mcp.tool()
async def validation_history(
    memory_id: str,
    event_type: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get validation event history for a memory.

    Shows the history of validation events (applied, succeeded, failed)
    for a specific memory.

    Args:
        memory_id: ID of the memory to get history for
        event_type: Filter by event type (optional)
        limit: Maximum number of events to return (default: 50)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with events list and summary statistics
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.validation_history(
        memory_id=memory_id,
        event_type=event_type,
        limit=limit,
    )


# ==============================================================================
# Validation Analysis Tools
# ==============================================================================


@mcp.tool()
async def memory_detect_contradictions(
    memory_id: str,
    similarity_threshold: float = 0.7,
    create_edges: bool = True,
) -> dict[str, Any]:
    """Detect memories that contradict a given memory.

    Uses semantic search to find similar memories, then checks for
    contradictions based on content analysis.

    Args:
        memory_id: ID of the memory to check for contradictions
        similarity_threshold: Minimum similarity for considering (default: 0.7)
        create_edges: Whether to create CONTRADICTS edges (default: True)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with contradictions list and edges_created count
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_detect_contradictions(
        memory_id=memory_id,
        similarity_threshold=similarity_threshold,
        create_edges=create_edges,
    )


@mcp.tool()
async def memory_check_supersedes(
    memory_id: str,
    create_edge: bool = True,
) -> dict[str, Any]:
    """Check if a memory should supersede another.

    A newer memory supersedes an older one when it consistently succeeds
    where the older one failed on similar topics.

    Args:
        memory_id: ID of the (potentially newer) memory to check
        create_edge: Whether to create SUPERSEDES edge (default: True)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with superseded_id and edge_created
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_check_supersedes(
        memory_id=memory_id,
        create_edge=create_edge,
    )


@mcp.tool()
async def memory_analyze_health(
    namespace: Optional[str] = None,
    include_contradictions: bool = True,
    include_low_confidence: bool = True,
    include_stale: bool = True,
    stale_days: int = 30,
) -> dict[str, Any]:
    """Analyze the health of memories in the system.

    Checks for unresolved contradictions, low-confidence memories,
    and stale memories that haven't been validated recently.

    Args:
        namespace: Limit analysis to specific namespace (optional)
        include_contradictions: Check for contradictions (default: True)
        include_low_confidence: Find low-confidence memories (default: True)
        include_stale: Find stale memories (default: True)
        stale_days: Days without validation to consider stale (default: 30)

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with categorized issues and recommendations
        - error: Error message if operation failed
    """
    if memory_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await memory_tools.memory_analyze_health(
        namespace=namespace,
        include_contradictions=include_contradictions,
        include_low_confidence=include_low_confidence,
        include_stale=include_stale,
        stale_days=stale_days,
    )


# ==============================================================================
# Management Tools
# ==============================================================================


@mcp.tool()
async def delete_chunks(ids: list[str]) -> dict[str, Any]:
    """Delete specific chunks by their IDs.

    Args:
        ids: List of chunk IDs to delete

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with deleted_count, deleted_ids
        - error: Error message if operation failed
    """
    if management_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await management_tools.delete_by_ids(ids)


@mcp.tool()
async def delete_file(source_file: str) -> dict[str, Any]:
    """Delete all chunks from a specific source file.

    Args:
        source_file: Source file path to delete chunks for

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with deleted_count, source_file
        - error: Error message if operation failed
    """
    if management_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await management_tools.delete_by_file(source_file)


@mcp.tool()
async def clear_index(confirm: bool) -> dict[str, Any]:
    """Delete all documents from the collection.

    Requires explicit confirmation to prevent accidental data loss.

    Args:
        confirm: Must be True to proceed with deletion

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with deleted_count
        - error: Error message if not confirmed or operation failed
    """
    if management_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await management_tools.delete_all(confirm)


@mcp.tool()
async def get_index_stats() -> dict[str, Any]:
    """Get collection statistics.

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - data: Dictionary with total_documents, unique_sources, source_files, etc.
        - error: Error message if operation failed
    """
    if management_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await management_tools.get_stats()
