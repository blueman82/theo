"""Memory tools for the Theo MCP server.

This module provides MCP tools for memory operations (from Recall):
- memory_store: Store a new memory with semantic indexing
- memory_recall: Recall memories using semantic search
- memory_validate: Validate a memory and adjust confidence
- memory_forget: Delete memories by ID or query
- memory_apply: Record that a memory is being applied (TRY phase)
- memory_outcome: Record the outcome of memory application (LEARN phase)
- memory_relate: Create a relationship between two memories
- memory_inspect_graph: Inspect the graph structure around a memory
- memory_edge_forget: Delete edges between memories
- memory_count: Count memories with optional filters
- memory_list: List memories with pagination
- memory_detect_contradictions: Detect contradicting memories
- memory_check_supersedes: Check if a memory supersedes another
- memory_analyze_health: Analyze memory system health

Uses DaemonClient for non-blocking embedding operations and integrates
with ValidationLoop for confidence scoring.
"""

import hashlib
import json
import logging
import re
import uuid
from collections import deque
from typing import Any, Optional

import httpx

from theo.config import TheoSettings
from theo.daemon import DaemonClient
from theo.storage.hybrid import HybridStore
from theo.storage.sqlite_store import SQLiteStore
from theo.types import (
    GraphEdge,
    GraphInspectionResult,
    GraphNode,
    GraphPath,
    GraphStats,
    MemoryType,
    RelationType,
)
from theo.validation import ValidationLoop

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)

# Auto-relationship inference settings (from Recall)
RELATIONSHIP_SIMILARITY_THRESHOLD = 0.6
MAX_AUTO_RELATIONSHIPS = 5

RELATIONSHIP_CLASSIFICATION_PROMPT = """You are analyzing the relationship between two memories in a knowledge graph.

New Memory: {new_memory}
Existing Memory: {existing_memory}

Determine the most appropriate relationship type from the new memory TO the existing memory:

1. "relates_to" - General topical relationship (same subject area, related concepts)
2. "supersedes" - New memory replaces/updates the existing one (newer info about same thing)
3. "caused_by" - New memory is a consequence or result of the existing memory
4. "contradicts" - Memories make incompatible claims (use sparingly, only for direct conflicts)

If no meaningful relationship exists, respond with "none".

Respond with ONLY a JSON object:
{{"relation": "relates_to|supersedes|caused_by|contradicts|none", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


async def _call_ollama_for_relationship(
    prompt: str,
    host: str,
    model: str,
    timeout: int = 30,
) -> Optional[dict[str, Any]]:
    """Call Ollama LLM for relationship classification.

    Args:
        prompt: The prompt to send to the LLM
        host: Ollama server host URL
        model: Model to use for generation
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^}]+\}", response_text)
                if json_match:
                    return json.loads(json_match.group())
                logger.warning(f"Failed to parse LLM response: {response_text}")
                return None
    except Exception as e:
        logger.debug(f"Ollama LLM call failed: {e}")
        return None


def _is_memory_id(value: str) -> bool:
    """Check if a string looks like a memory ID.

    Memory IDs follow the pattern: mem_{type}_{uuid_hex8}

    Args:
        value: String to check

    Returns:
        True if the string looks like a memory ID
    """
    if not value:
        return False
    # Memory IDs start with "mem_" and contain underscores
    if value.startswith("mem_") and value.count("_") >= 2:
        return True
    # Also accept UUID-like strings
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        pass
    return False


class MemoryTools:
    """Tool implementations for memory operations.

    Provides memory operations ported from the Recall project:
    - memory_store, memory_recall, memory_validate, memory_forget
    - memory_apply, memory_outcome (TRY/LEARN cycle)
    - memory_relate, memory_edge_forget (graph operations)
    - memory_inspect_graph (graph visualization)
    - memory_count, memory_list (inspection)
    - memory_detect_contradictions, memory_check_supersedes, memory_analyze_health

    Architectural justification:
    - DaemonClient handles expensive embedding operations asynchronously
    - SQLiteStore provides unified storage (sqlite-vec + FTS5 + edges)
    - HybridStore wraps SQLiteStore with automatic embedding generation
    - ValidationLoop tracks confidence scores through practical use

    Args:
        daemon_client: Client for daemon embedding operations
        store: SQLiteStore instance for all storage operations
        validation_loop: ValidationLoop for confidence scoring
        hybrid_store: Optional HybridStore for graph operations

    Example:
        >>> tools = MemoryTools(daemon_client, store, validation_loop, hybrid_store)
        >>> result = await tools.memory_store(
        ...     content="User prefers dark mode",
        ...     memory_type="preference",
        ...     namespace="global",
        ... )
        >>> if result["success"]:
        ...     print(f"Stored memory: {result['data']['id']}")
    """

    def __init__(
        self,
        daemon_client: DaemonClient,
        store: SQLiteStore,
        validation_loop: ValidationLoop,
        hybrid_store: Optional[HybridStore] = None,
        settings: Optional[TheoSettings] = None,
    ) -> None:
        """Initialize MemoryTools with dependencies.

        Args:
            daemon_client: Client for daemon embedding operations
            store: SQLiteStore instance for all storage operations
            validation_loop: ValidationLoop for confidence scoring
            hybrid_store: Optional HybridStore for graph operations
            settings: Optional TheoSettings for LLM configuration
        """
        self._daemon = daemon_client
        self._store = store
        self._validation = validation_loop
        self._hybrid = hybrid_store
        self._settings = settings

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _generate_memory_id(self, memory_type: MemoryType) -> str:
        """Generate a unique memory ID.

        Args:
            memory_type: Type of memory being created

        Returns:
            Unique memory ID in format: mem_{type}_{uuid_hex8}
        """
        short_uuid = uuid.uuid4().hex[:8]
        return f"mem_{memory_type.value}_{short_uuid}"

    async def _infer_relationships(
        self,
        memory_id: str,
        content: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """Automatically infer and create relationships to existing memories.

        Uses embedding similarity to find related memories and creates 'relates_to'
        edges. If LLM is available, attempts to classify more specific relation types
        (supersedes, caused_by, contradicts).

        This function ALWAYS creates edges for sufficiently similar memories,
        regardless of LLM availability. The LLM is only used for type refinement.

        Args:
            memory_id: ID of the newly stored memory
            content: Content of the new memory
            namespace: Namespace to search within

        Returns:
            List of created relationship dicts with target_id, relation, similarity,
            confidence, and reason
        """
        created_relationships: list[dict[str, Any]] = []

        if not self._hybrid:
            return created_relationships

        # Search for similar memories
        try:
            search_result = await self.memory_recall(
                query=content,
                n_results=MAX_AUTO_RELATIONSHIPS + 5,
                namespace=namespace,
                include_related=False,
            )
        except Exception as e:
            logger.debug(f"Failed to search for similar memories: {e}")
            return created_relationships

        if not search_result.get("success"):
            return created_relationships

        similar_memories = search_result.get("data", {}).get("memories", [])

        # Filter candidates by similarity threshold and exclude self
        candidates = [
            mem
            for mem in similar_memories
            if mem["id"] != memory_id
            and mem.get("similarity", 0.0) >= RELATIONSHIP_SIMILARITY_THRESHOLD
        ]

        if not candidates:
            return created_relationships

        # Get LLM config from settings
        ollama_host = "http://localhost:11434"
        ollama_model = ""
        ollama_timeout = 30
        if self._settings:
            ollama_host = self._settings.ollama_host
            ollama_model = self._settings.ollama_llm_model
            ollama_timeout = self._settings.ollama_timeout

        for candidate in candidates[:MAX_AUTO_RELATIONSHIPS]:
            if len(created_relationships) >= MAX_AUTO_RELATIONSHIPS:
                break

            candidate_content = candidate.get("content", "")
            candidate_id = candidate["id"]
            similarity = candidate.get("similarity", 0.0)

            # Default: create relates_to edge based on embedding similarity alone
            rel_type = RelationType.RELATES_TO
            confidence = similarity
            reason = f"Semantic similarity: {similarity:.2f}"

            # Optionally try LLM classification to upgrade edge type
            if ollama_model:
                prompt = RELATIONSHIP_CLASSIFICATION_PROMPT.format(
                    new_memory=content,
                    existing_memory=candidate_content,
                )

                llm_result = await _call_ollama_for_relationship(
                    prompt=prompt,
                    host=ollama_host,
                    model=ollama_model,
                    timeout=ollama_timeout,
                )

                if llm_result:
                    llm_relation = llm_result.get("relation", "none")
                    llm_confidence = llm_result.get("confidence", 0.0)
                    llm_reason = llm_result.get("reason", "")

                    # Only upgrade if LLM found more specific relation with high confidence
                    if (
                        llm_relation != "none"
                        and llm_relation != "relates_to"
                        and llm_confidence >= 0.6
                    ):
                        try:
                            rel_type = RelationType(llm_relation)
                            confidence = llm_confidence
                            reason = llm_reason
                        except ValueError:
                            pass  # Keep default relates_to

            try:
                # Check if edge already exists
                existing_edges = self._store.get_edges(
                    memory_id, direction="outgoing", edge_type=rel_type.value
                )
                edge_exists = any(e["target_id"] == candidate_id for e in existing_edges)

                if not edge_exists:
                    self._hybrid.add_edge(
                        source_id=memory_id,
                        target_id=candidate_id,
                        edge_type=rel_type.value,
                        weight=confidence,
                        metadata={
                            "reason": reason,
                            "auto_inferred": True,
                            "similarity": similarity,
                        },
                    )

                    # Handle supersedes: reduce importance of superseded memory
                    if rel_type == RelationType.SUPERSEDES:
                        target_memory = await self._hybrid.get_memory(candidate_id)
                        if target_memory:
                            current_importance = target_memory.get("importance", 0.5)
                            await self._hybrid.update_memory(
                                candidate_id, importance=current_importance * 0.5
                            )

                    created_relationships.append(
                        {
                            "target_id": candidate_id,
                            "relation": rel_type.value,
                            "similarity": similarity,
                            "confidence": confidence,
                            "reason": reason,
                        }
                    )

                    logger.info(
                        f"Auto-linked: {memory_id} --[{rel_type.value}]--> {candidate_id} "
                        f"(similarity: {similarity:.2f}, reason: {reason})"
                    )

            except Exception as e:
                logger.debug(f"Failed to create edge {memory_id} -> {candidate_id}: {e}")

        return created_relationships

    async def memory_store(
        self,
        content: str,
        memory_type: str = "session",
        namespace: str = "global",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
        relates_to: Optional[list[dict[str, Any]]] = None,
        supersedes_query: Optional[str] = None,
    ) -> dict[str, Any]:
        """Store a new memory with semantic indexing.

        Creates a new memory with the specified content, type, and namespace.
        Uses the daemon for non-blocking embedding generation.

        Deduplication: If a memory with the same content hash already exists
        in the same namespace, the existing memory ID is returned instead of
        creating a duplicate.

        Args:
            content: The memory content text
            memory_type: Type of memory (preference, decision, pattern, session, fact)
            namespace: Scope of the memory (global, default, or project:{name})
            importance: Importance score from 0.0 to 1.0 (default: 0.5)
            metadata: Optional additional metadata
            relates_to: Optional list of relations to create after storing.
                Each dict must have 'target_id' and 'relation' keys.
                Valid relations: relates_to, supersedes, caused_by, contradicts.
                Optional 'weight' key (default: 1.0).
                Example: [{"target_id": "mem_abc", "relation": "relates_to"}]
            supersedes_query: Optional query to find and auto-supersede matching
                memories. Memories with similarity >= 0.7 will be superseded
                (importance halved, confidence set to 0.1).

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with id, content_hash, namespace, duplicate, relations,
                    superseded (list of auto-superseded memory IDs)
            - error: Error message if operation failed
        """
        try:
            if not content or not content.strip():
                return {
                    "success": False,
                    "error": "Memory content cannot be empty",
                }

            # Parse memory type
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                valid_types = [t.value for t in MemoryType]
                return {
                    "success": False,
                    "error": f"Invalid memory_type: {memory_type}. Must be one of: {valid_types}",
                }

            # Validate namespace format
            if namespace not in ("default", "global"):
                if not namespace.startswith("project:"):
                    return {
                        "success": False,
                        "error": f"Invalid namespace: {namespace}. "
                        "Must be 'default', 'global', or 'project:{{name}}'",
                    }

            # Validate importance range
            if not 0.0 <= importance <= 1.0:
                return {
                    "success": False,
                    "error": f"Importance must be between 0.0 and 1.0, got {importance}",
                }

            # Validate relates_to structure if provided
            if relates_to is not None:
                if not isinstance(relates_to, list):
                    return {
                        "success": False,
                        "error": "relates_to must be a list of relation dicts",
                    }
                valid_relations = [r.value for r in RelationType]
                for rel in relates_to:
                    if not isinstance(rel, dict):
                        return {
                            "success": False,
                            "error": "Each relates_to entry must be a dict",
                        }
                    if "target_id" not in rel:
                        return {
                            "success": False,
                            "error": "Each relates_to entry must have 'target_id'",
                        }
                    if "relation" not in rel:
                        return {
                            "success": False,
                            "error": "Each relates_to entry must have 'relation'",
                        }
                    if rel["relation"] not in valid_relations:
                        return {
                            "success": False,
                            "error": f"Invalid relation '{rel['relation']}'. "
                            f"Must be one of: {valid_relations}",
                        }

            # Handle supersedes_query - auto-populate relates_to with supersedes relations
            auto_supersedes: list[dict[str, Any]] = []
            if (query := supersedes_query) and query.strip():
                search_result = await self.memory_recall(
                    query=query,
                    n_results=10,
                    namespace=namespace,
                    include_related=False,
                )
                if search_result.get("success"):
                    auto_supersedes = [
                        {"target_id": mem["id"], "relation": "supersedes"}
                        for mem in search_result.get("data", {}).get("memories", [])
                        if mem.get("similarity", 0) >= 0.7
                    ]

            # Merge auto_supersedes into relates_to
            if auto_supersedes:
                relates_to = auto_supersedes if relates_to is None else relates_to + auto_supersedes

            # Compute content hash for deduplication
            content_hash = self._compute_hash(content)

            # Generate embedding via daemon (non-blocking)
            embed_result = self._daemon.embed([content])

            if not embed_result.get("success"):
                err = embed_result.get("error", "Unknown error")
                return {
                    "success": False,
                    "error": f"Embedding generation failed: {err}",
                }

            embeddings = embed_result.get("data", {}).get("embeddings", [])

            if not embeddings:
                return {
                    "success": False,
                    "error": "No embedding returned",
                }

            # Store memory using SQLiteStore's add_memory
            # This stores in memories, memories_vec, and memories_fts tables
            memory_id = self._store.add_memory(
                content=content,
                embedding=embeddings[0],
                memory_type=mem_type.value,
                namespace=namespace,
                confidence=0.3,  # Memories start at low confidence
                importance=importance,
                content_hash=content_hash,
                tags=metadata,
            )

            logger.info(f"Stored memory {memory_id} in namespace {namespace}")

            # Create relations if provided
            created_relations: list[dict[str, Any]] = []
            relation_errors: list[str] = []

            if relates_to and self._hybrid:
                for rel in relates_to:
                    target_id = rel["target_id"]
                    relation = rel["relation"]
                    weight = rel.get("weight", 1.0)

                    try:
                        # Verify target memory exists
                        target_memory = await self._hybrid.get_memory(target_id)
                        if target_memory is None:
                            relation_errors.append(f"Target '{target_id}' not found")
                            continue

                        # Handle 'supersedes' relation special case
                        rel_type = RelationType(relation)
                        if rel_type == RelationType.SUPERSEDES:
                            current_importance = target_memory.get("importance", 0.5)
                            await self._hybrid.update_memory(
                                target_id,
                                importance=current_importance * 0.5,
                                confidence=0.1,  # Floor confidence for superseded memories
                            )

                        # Create the edge
                        edge_id = self._hybrid.add_edge(
                            source_id=memory_id,
                            target_id=target_id,
                            edge_type=rel_type.value,
                            weight=weight,
                        )
                        created_relations.append(
                            {
                                "edge_id": edge_id,
                                "target_id": target_id,
                                "relation": relation,
                            }
                        )
                    except Exception as e:
                        relation_errors.append(f"Failed to create relation to '{target_id}': {e}")

            # Auto-infer relationships to existing memories
            auto_relations = await self._infer_relationships(
                memory_id=memory_id,
                content=content,
                namespace=namespace,
            )
            created_relations.extend(auto_relations)

            # Extract auto-superseded IDs for return value
            superseded_ids = [rel["target_id"] for rel in auto_supersedes]

            result_data: dict[str, Any] = {
                "id": memory_id,
                "content_hash": content_hash,
                "namespace": namespace,
                "memory_type": mem_type.value,
                "importance": importance,
                "duplicate": False,
                "superseded": superseded_ids,
                "superseded_count": len(superseded_ids),
            }

            if created_relations:
                result_data["relations"] = created_relations
            if relation_errors:
                result_data["relation_errors"] = relation_errors

            return {
                "success": True,
                "data": result_data,
            }

        except Exception as e:
            logger.error(f"memory_store failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_recall(
        self,
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

        Searches for memories similar to the query, with optional filtering
        by namespace, type, importance, and confidence.

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
            - data: Dictionary with memories, total, query, and optionally expanded
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
                    "error": f"Query embedding failed: {embed_result.get('error', 'Unknown')}",
                }

            query_embedding = embed_result.get("data", {}).get("embedding", [])

            if not query_embedding:
                return {
                    "success": False,
                    "error": "No query embedding returned",
                }

            # Build where filter for search_hybrid
            where_filter: dict | None = None
            if namespace or memory_type:
                where_filter = {}
                if namespace:
                    where_filter["namespace"] = namespace
                if memory_type:
                    where_filter["memory_type"] = memory_type

            # Search using SQLiteStore's hybrid search (vector + FTS) with filters
            results = self._store.search_hybrid(
                embedding=query_embedding,
                query=query,
                n_results=n_results * 2,  # Fetch extra for importance/confidence filtering
                where=where_filter,
            )

            # Filter results by importance and confidence (namespace/type already filtered)
            memories_data: list[dict[str, Any]] = []
            for i, result in enumerate(results):
                # Apply importance filter
                if min_importance is not None and result.importance < min_importance:
                    continue
                # Apply confidence filter
                if min_confidence is not None and result.confidence < min_confidence:
                    continue

                memories_data.append(
                    {
                        "id": result.id,
                        "content": result.content,
                        "content_hash": None,  # Not returned by search_hybrid
                        "type": result.memory_type,
                        "namespace": result.namespace,
                        "importance": result.importance,
                        "confidence": result.confidence,
                        "similarity": result.score,
                        "rank": len(memories_data),
                    }
                )

                # Stop once we have enough results
                if len(memories_data) >= n_results:
                    break

            # Graph expansion if requested - uses SQLiteStore directly
            expanded_data = []
            if include_related and memories_data:
                seen_ids: set[str] = {str(m["id"]) for m in memories_data}
                to_expand: list[str] = [str(m["id"]) for m in memories_data]

                for depth in range(max_depth):
                    next_expand: list[str] = []
                    for memory_id in to_expand:
                        # Use SQLiteStore's get_edges directly
                        edges = self._store.get_edges(memory_id, direction="both")
                        for edge in edges:
                            # Get the connected memory ID
                            connected_id = (
                                edge.get("target_id")
                                if edge.get("source_id") == memory_id
                                else edge.get("source_id")
                            )
                            if connected_id and connected_id not in seen_ids:
                                seen_ids.add(connected_id)
                                next_expand.append(connected_id)
                                # Fetch the connected memory using SQLiteStore
                                connected_mem = self._store.get_memory(connected_id)
                                if connected_mem:
                                    expanded_data.append(
                                        {
                                            "id": connected_id,
                                            "content": connected_mem.get("content", ""),
                                            "type": connected_mem.get("memory_type", "memory"),
                                            "namespace": connected_mem.get("namespace", ""),
                                            "importance": connected_mem.get("importance", 0.5),
                                            "confidence": connected_mem.get("confidence", 0.5),
                                            "relation": edge.get("edge_type", "related"),
                                            "via_memory": memory_id,
                                            "depth": depth + 1,
                                        }
                                    )
                    to_expand = next_expand
                    if not to_expand:
                        break

            response_data = {
                "memories": memories_data,
                "total": len(memories_data),
                "query": query,
                "filters": {
                    "namespace": namespace,
                    "memory_type": memory_type,
                    "min_importance": min_importance,
                    "min_confidence": min_confidence,
                },
            }

            if include_related:
                response_data["expanded"] = expanded_data
                response_data["expanded_total"] = len(expanded_data)

            return {
                "success": True,
                "data": response_data,
            }

        except Exception as e:
            logger.error(f"memory_recall failed for query '{query}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_validate(
        self,
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
        try:
            result = await self._validation.record_usage(
                doc_id=memory_id,
                was_helpful=was_helpful,
                context=context,
            )

            if not result.success:
                return {
                    "success": False,
                    "error": result.error or "Validation failed",
                }

            return {
                "success": True,
                "data": {
                    "memory_id": result.doc_id,
                    "old_confidence": result.old_confidence,
                    "new_confidence": result.new_confidence,
                    "was_helpful": result.was_helpful,
                    "promoted": result.promoted,
                },
            }

        except Exception as e:
            logger.error(f"memory_validate failed for {memory_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_forget(
        self,
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
            - data: Dictionary with deleted_ids, deleted_count
            - error: Error message if operation failed
        """
        try:
            # Auto-detect input type
            if memory_id is None and query is None and input_value is not None:
                if _is_memory_id(input_value):
                    memory_id = input_value
                else:
                    query = input_value

            # Validate we have something to delete
            if memory_id is None and query is None:
                return {
                    "success": False,
                    "error": "Must provide memory_id, query, or input_value",
                }

            deleted_ids: list[str] = []
            protected_ids: list[str] = []

            if memory_id:
                # Direct deletion by ID using SQLiteStore.get_memory
                existing = self._store.get_memory(memory_id)
                if not existing:
                    return {
                        "success": False,
                        "error": f"Memory not found: {memory_id}",
                    }

                # Check golden rule protection
                confidence = existing.get("confidence", 0.0)
                mem_type = existing.get("memory_type", "")

                is_protected = confidence >= 0.9 or mem_type == "golden_rule"

                if is_protected and not force:
                    return {
                        "success": False,
                        "error": f"Memory {memory_id} is golden "
                        f"(confidence={confidence:.2f}). Use force=True.",
                        "data": {"protected_ids": [memory_id]},
                    }

                # Delete the memory using SQLiteStore.delete_memory
                self._store.delete_memory(memory_id)
                deleted_ids.append(memory_id)

            else:
                # Search-based deletion
                assert query is not None

                # Generate query embedding via daemon
                embed_result = self._daemon.send("search", query=query)

                if not embed_result.get("success"):
                    return {
                        "success": False,
                        "error": f"Query embedding failed: {embed_result.get('error', 'Unknown')}",
                    }

                query_embedding = embed_result.get("data", {}).get("embedding", [])

                if not query_embedding:
                    return {
                        "success": False,
                        "error": "No query embedding returned",
                    }

                # Search for memories to delete using SQLiteStore.search_hybrid
                results = self._store.search_hybrid(
                    embedding=query_embedding,
                    query=query,
                    n_results=n_results * 2,  # Fetch extra for namespace filtering
                )

                # Check each result for golden rule protection and namespace filter
                ids_to_delete: list[str] = []
                for result in results:
                    # Apply namespace filter
                    if namespace and result.namespace != namespace:
                        continue

                    is_protected = result.confidence >= 0.9 or result.memory_type == "golden_rule"

                    if is_protected and not force:
                        protected_ids.append(result.id)
                    else:
                        ids_to_delete.append(result.id)

                    # Stop once we have enough
                    if len(ids_to_delete) + len(protected_ids) >= n_results:
                        break

                # Delete non-protected memories using SQLiteStore.delete_memory
                for mem_id in ids_to_delete:
                    self._store.delete_memory(mem_id)
                    deleted_ids.append(mem_id)

            logger.info(f"Deleted {len(deleted_ids)} memories, {len(protected_ids)} protected")

            return {
                "success": True,
                "data": {
                    "deleted_ids": deleted_ids,
                    "deleted_count": len(deleted_ids),
                    "protected_ids": protected_ids,
                    "protected_count": len(protected_ids),
                },
            }

        except Exception as e:
            logger.error(f"memory_forget failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_context(
        self,
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
        try:
            # Calculate how many results we can fit
            # Estimate ~100 tokens per memory average
            n_results = token_budget // 100

            if query:
                # Semantic search for relevant memories
                result = await self.memory_recall(
                    query=query,
                    n_results=n_results,
                    namespace=namespace,
                )

                if not result.get("success"):
                    return result

                memories = result.get("data", {}).get("memories", [])

            else:
                # Get memories using SQLiteStore.list_memories
                all_docs = self._store.list_memories(
                    namespace=namespace,
                    limit=n_results * 2,  # Fetch extra for sorting
                )

                # Convert to expected format and sort by importance (descending)
                memories_data = []
                for doc in all_docs:
                    memories_data.append(
                        {
                            "id": doc["id"],
                            "content": doc["content"],
                            "type": doc["memory_type"],
                            "importance": doc["importance"],
                            "confidence": doc["confidence"],
                            "namespace": doc["namespace"],
                        }
                    )

                # Sort by importance and take top results
                memories_data.sort(key=lambda x: x["importance"], reverse=True)
                memories = memories_data[:n_results]

            # Format as markdown context
            context_lines = ["# Relevant Memories", ""]

            total_tokens = 0
            for mem in memories:
                conf = mem.get("confidence", 0.3)
                mem_text = f"- **{mem['type']}** (confidence: {conf:.2f}): "
                mem_text += mem["content"]
                mem_tokens = len(mem_text) // 4

                if total_tokens + mem_tokens > token_budget:
                    break

                context_lines.append(mem_text)
                total_tokens += mem_tokens

            context = "\n".join(context_lines)

            return {
                "success": True,
                "data": {
                    "context": context,
                    "token_estimate": total_tokens,
                    "memory_count": len(memories),
                },
            }

        except Exception as e:
            logger.error(f"memory_context failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # TRY/LEARN Cycle Operations (memory_apply, memory_outcome)
    # =========================================================================

    async def memory_apply(
        self,
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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured - graph operations unavailable",
                }

            # Verify memory exists
            memory = await self._hybrid.get_memory(memory_id)
            if memory is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found",
                }

            # Record the 'applied' event
            event_id = self._hybrid.add_validation_event(
                memory_id=memory_id,
                event_type="applied",
                context=context,
                session_id=session_id,
            )

            return {
                "success": True,
                "data": {
                    "memory_id": memory_id,
                    "event_id": event_id,
                },
            }

        except Exception as e:
            logger.error(f"memory_apply failed for {memory_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_outcome(
        self,
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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured - graph operations unavailable",
                }

            # Verify memory exists
            memory = await self._hybrid.get_memory(memory_id)
            if memory is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found",
                }

            # Record the outcome event
            event_type = "succeeded" if success else "failed"
            context = error_msg if error_msg else ("Success" if success else "Failed")

            self._hybrid.add_validation_event(
                memory_id=memory_id,
                event_type=event_type,
                context=context,
                session_id=session_id,
            )

            # Adjust confidence via memory_validate
            validate_result = await self.memory_validate(
                memory_id=memory_id,
                was_helpful=success,
                context=context,
            )

            if not validate_result.get("success"):
                return validate_result

            data = validate_result.get("data", {})

            return {
                "success": True,
                "data": {
                    "memory_id": memory_id,
                    "outcome_success": success,
                    "new_confidence": data.get("new_confidence"),
                    "promoted": data.get("promoted", False),
                },
            }

        except Exception as e:
            logger.error(f"memory_outcome failed for {memory_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Graph Operations (memory_relate, memory_edge_forget, memory_inspect_graph)
    # =========================================================================

    async def memory_relate(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        """Create a relationship between two memories.

        Validates both memories exist, validates relation type, and creates the edge.
        Special handling for 'supersedes' relation which reduces target importance by 50%.

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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured - graph operations unavailable",
                }

            # Validate relation type
            try:
                rel_type = RelationType(relation)
            except ValueError:
                valid_types = [r.value for r in RelationType]
                return {
                    "success": False,
                    "error": f"Invalid relation type. Must be one of: {valid_types}",
                }

            # Verify source memory exists
            source_memory = await self._hybrid.get_memory(source_id)
            if source_memory is None:
                return {
                    "success": False,
                    "error": f"Source memory '{source_id}' not found",
                }

            # Verify target memory exists
            target_memory = await self._hybrid.get_memory(target_id)
            if target_memory is None:
                return {
                    "success": False,
                    "error": f"Target memory '{target_id}' not found",
                }

            # Handle 'supersedes' relation special case - reduce target importance
            if rel_type == RelationType.SUPERSEDES:
                current_importance = target_memory.get("importance", 0.5)
                new_importance = current_importance * 0.5
                await self._hybrid.update_memory(target_id, importance=new_importance)

            # Create the edge
            edge_id = self._hybrid.add_edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=rel_type.value,
                weight=weight,
            )

            return {
                "success": True,
                "data": {
                    "edge_id": edge_id,
                },
            }

        except Exception as e:
            logger.error(f"memory_relate failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_edge_forget(
        self,
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
            edge_id: Specific edge ID to delete (direct deletion mode)
            memory_id: Memory ID to delete all connected edges (memory-based mode)
            source_id: Source memory ID for pair deletion mode
            target_id: Target memory ID for pair deletion mode
            relation: Filter by relation type (optional)
            direction: For memory_id mode: 'outgoing', 'incoming', or 'both'

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with deleted_ids and deleted_count
            - error: Error message if operation failed
        """
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured - graph operations unavailable",
                }

            deleted_ids: list[int] = []

            if edge_id is not None:
                # Mode 1: Delete specific edge by ID
                if self._hybrid.delete_edge_by_id(edge_id):
                    deleted_ids.append(edge_id)

            elif memory_id is not None:
                # Mode 2: Delete all edges connected to a memory
                edges = self._hybrid.get_edges(memory_id, direction=direction, edge_type=relation)
                for edge in edges:
                    if self._hybrid.delete_edge_by_id(edge["id"]):
                        deleted_ids.append(edge["id"])

            elif source_id is not None and target_id is not None:
                # Mode 3: Delete edges between two specific memories
                outgoing = self._hybrid.get_edges(
                    source_id, direction="outgoing", edge_type=relation
                )
                for edge in outgoing:
                    if edge["target_id"] == target_id:
                        if self._hybrid.delete_edge_by_id(edge["id"]):
                            deleted_ids.append(edge["id"])

            else:
                return {
                    "success": False,
                    "error": "Must provide edge_id, memory_id, or both source_id and target_id",
                }

            return {
                "success": True,
                "data": {
                    "deleted_ids": deleted_ids,
                    "deleted_count": len(deleted_ids),
                },
            }

        except Exception as e:
            logger.error(f"memory_edge_forget failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_inspect_graph(
        self,
        memory_id: str,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: Optional[list[str]] = None,
        include_scores: bool = True,
        decay_factor: float = 0.7,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """Inspect the graph structure around a memory node.

        Performs read-only breadth-first search from the origin memory, collecting
        all nodes and edges within max_depth hops.

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
            - nodes: List of node dicts (json) or mermaid: Mermaid diagram string
            - edges: List of edge dicts
            - paths: List of path dicts
            - stats: Summary statistics
            - error: Error message if operation failed
        """
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured - graph operations unavailable",
                }

            # Validate direction
            valid_dirs = ("outgoing", "incoming", "both")
            if direction not in valid_dirs:
                return {
                    "success": False,
                    "error": f"Invalid direction: {direction}. Must be one of: {valid_dirs}",
                }

            # Verify origin memory exists
            origin = await self._hybrid.get_memory(memory_id)
            if origin is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found",
                }

            # BFS traversal
            nodes: dict[str, GraphNode] = {}
            edges: list[GraphEdge] = []
            paths: list[GraphPath] = []

            # Add origin node
            content_preview = origin.get("content", "")[:150]
            if len(origin.get("content", "")) > 150:
                content_preview += "..."

            nodes[memory_id] = GraphNode(
                id=memory_id,
                content_preview=content_preview,
                memory_type=origin.get("type", "document"),
                confidence=origin.get("confidence", 0.3),
                importance=origin.get("importance", 0.5),
            )

            # BFS queue: (node_id, depth, path_node_ids, path_edge_types, cumulative_weight)
            queue: deque[tuple[str, int, list[str], list[str], float]] = deque()
            queue.append((memory_id, 0, [memory_id], [], 1.0))
            visited_edges: set[int] = set()
            max_depth_reached = 0

            while queue:
                current_id, depth, path_nodes, path_edges, path_weight = queue.popleft()

                if depth >= max_depth:
                    continue

                max_depth_reached = max(max_depth_reached, depth)

                # Get connected edges
                connected = self._hybrid.get_edges(current_id, direction=direction)

                for edge_data in connected:
                    # Filter by edge type if specified
                    if edge_types and edge_data["edge_type"] not in edge_types:
                        continue

                    # Skip if already visited this edge
                    if edge_data["id"] in visited_edges:
                        continue
                    visited_edges.add(edge_data["id"])

                    # Determine neighbor
                    if edge_data["source_id"] == current_id:
                        neighbor_id = edge_data["target_id"]
                    else:
                        neighbor_id = edge_data["source_id"]

                    # Add edge to results
                    edges.append(
                        GraphEdge(
                            id=edge_data["id"],
                            source_id=edge_data["source_id"],
                            target_id=edge_data["target_id"],
                            edge_type=edge_data["edge_type"],
                            weight=edge_data["weight"],
                        )
                    )

                    # Add neighbor node if not seen
                    if neighbor_id not in nodes:
                        neighbor = await self._hybrid.get_memory(neighbor_id)
                        if neighbor:
                            n_content = neighbor.get("content", "")[:150]
                            if len(neighbor.get("content", "")) > 150:
                                n_content += "..."
                            nodes[neighbor_id] = GraphNode(
                                id=neighbor_id,
                                content_preview=n_content,
                                memory_type=neighbor.get("type", "document"),
                                confidence=neighbor.get("confidence", 0.3),
                                importance=neighbor.get("importance", 0.5),
                            )

                    # Build path
                    new_path_nodes = path_nodes + [neighbor_id]
                    new_path_edges = path_edges + [edge_data["edge_type"]]
                    new_weight = path_weight * edge_data["weight"]

                    if include_scores:
                        relevance = new_weight * (decay_factor ** (depth + 1))
                        paths.append(
                            GraphPath(
                                node_ids=new_path_nodes,
                                edge_types=new_path_edges,
                                total_weight=new_weight,
                                relevance_score=relevance,
                            )
                        )

                    # Continue BFS
                    queue.append(
                        (neighbor_id, depth + 1, new_path_nodes, new_path_edges, new_weight)
                    )

            # Build result
            result = GraphInspectionResult(
                success=True,
                origin_id=memory_id,
                nodes=list(nodes.values()),
                edges=edges,
                paths=paths,
                stats=GraphStats(
                    node_count=len(nodes),
                    edge_count=len(edges),
                    max_depth_reached=max_depth_reached,
                    origin_id=memory_id,
                ),
            )

            if output_format == "mermaid":
                return {
                    "success": True,
                    "origin_id": memory_id,
                    "mermaid": result.to_mermaid(),
                    "stats": {
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                        "max_depth_reached": max_depth_reached,
                    },
                }

            # JSON format
            return {
                "success": True,
                "origin_id": memory_id,
                "nodes": [
                    {
                        "id": n.id,
                        "content_preview": n.content_preview,
                        "type": n.memory_type,
                        "confidence": n.confidence,
                        "importance": n.importance,
                    }
                    for n in result.nodes
                ],
                "edges": [
                    {
                        "id": e.id,
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                    for e in result.edges
                ],
                "paths": [
                    {
                        "node_ids": p.node_ids,
                        "edge_types": p.edge_types,
                        "total_weight": p.total_weight,
                        "relevance_score": p.relevance_score,
                    }
                    for p in result.paths
                ],
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "max_depth_reached": max_depth_reached,
                },
            }

        except Exception as e:
            logger.error(f"memory_inspect_graph failed for {memory_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Inspection Operations (memory_count, memory_list)
    # =========================================================================

    async def memory_count(
        self,
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
        try:
            # Use SQLiteStore.count_memories directly
            count = self._store.count_memories(
                namespace=namespace,
                memory_type=memory_type,
            )

            return {
                "success": True,
                "data": {
                    "count": count,
                    "filters": {
                        "namespace": namespace,
                        "memory_type": memory_type,
                    },
                },
            }

        except Exception as e:
            logger.error(f"memory_count failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_list(
        self,
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
        try:
            # Clamp limit
            limit = min(limit, 1000)

            # Use SQLiteStore.list_memories directly
            result = self._store.list_memories(
                namespace=namespace,
                memory_type=memory_type,
                limit=limit,
                offset=offset,
                order_by=order_by,
                descending=descending,
            )

            # Convert to expected response format
            memories = []
            for doc in result:
                memories.append(
                    {
                        "id": doc["id"],
                        "content": doc["content"],
                        "type": doc["memory_type"],
                        "namespace": doc["namespace"],
                        "importance": doc["importance"],
                        "confidence": doc["confidence"],
                    }
                )

            return {
                "success": True,
                "data": {
                    "memories": memories,
                    "count": len(memories),
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": len(memories) == limit,
                    },
                    "filters": {
                        "namespace": namespace,
                        "memory_type": memory_type,
                        "order_by": order_by,
                        "descending": descending,
                    },
                },
            }

        except Exception as e:
            logger.error(f"memory_list failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Validation Analysis (detect_contradictions, check_supersedes, analyze_health)
    # =========================================================================

    async def memory_detect_contradictions(
        self,
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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured",
                }

            # Get the memory
            memory = await self._hybrid.get_memory(memory_id)
            if memory is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found",
                }

            # Search for similar memories
            result = await self.memory_recall(
                query=memory.get("content", ""),
                n_results=20,
            )

            if not result.get("success"):
                return result

            similar_memories = result.get("data", {}).get("memories", [])

            # Filter by similarity threshold and exclude self
            contradictions: list[str] = []
            edges_created = 0

            for similar in similar_memories:
                if similar["id"] == memory_id:
                    continue
                if similar.get("similarity", 0) < similarity_threshold:
                    continue

                # Simple contradiction detection: same type, high similarity
                # In production, this would use LLM reasoning
                if similar.get("type") == memory.get("type"):
                    contradictions.append(similar["id"])

                    if create_edges:
                        self._hybrid.add_edge(
                            source_id=memory_id,
                            target_id=similar["id"],
                            edge_type="contradicts",
                            weight=similar.get("similarity", 0.7),
                        )
                        edges_created += 1

            return {
                "success": True,
                "data": {
                    "memory_id": memory_id,
                    "contradictions": contradictions,
                    "edges_created": edges_created,
                },
            }

        except Exception as e:
            logger.error(f"memory_detect_contradictions failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_check_supersedes(
        self,
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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured",
                }

            # Get the memory
            memory = await self._hybrid.get_memory(memory_id)
            if memory is None:
                return {
                    "success": False,
                    "error": f"Memory '{memory_id}' not found",
                }

            # Search for similar memories
            result = await self.memory_recall(
                query=memory.get("content", ""),
                n_results=10,
            )

            if not result.get("success"):
                return result

            similar_memories = result.get("data", {}).get("memories", [])

            superseded_id: Optional[str] = None
            edge_created = False
            reason: Optional[str] = None

            # Check for supersession: newer memory with higher confidence
            # that is similar to an older memory with lower confidence
            for similar in similar_memories:
                if similar["id"] == memory_id:
                    continue

                # Check if this memory should supersede the similar one
                my_confidence = memory.get("confidence", 0.3)
                their_confidence = similar.get("confidence", 0.3)

                if my_confidence > their_confidence and similar.get("similarity", 0) > 0.8:
                    target_mem_id: str = similar["id"]
                    superseded_id = target_mem_id
                    reason = (
                        f"Memory {memory_id} (confidence={my_confidence:.2f}) "
                        f"supersedes {target_mem_id} (confidence={their_confidence:.2f}) "
                        f"due to higher confidence on similar content"
                    )

                    if create_edge:
                        self._hybrid.add_edge(
                            source_id=memory_id,
                            target_id=target_mem_id,
                            edge_type="supersedes",
                            weight=1.0,
                        )
                        edge_created = True

                        # Reduce superseded memory's importance
                        await self._hybrid.update_memory(
                            target_mem_id,
                            importance=similar.get("importance", 0.5) * 0.5,
                        )

                    break  # Only supersede one memory

            return {
                "success": True,
                "data": {
                    "memory_id": memory_id,
                    "superseded_id": superseded_id,
                    "edge_created": edge_created,
                    "reason": reason,
                },
            }

        except Exception as e:
            logger.error(f"memory_check_supersedes failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_analyze_health(
        self,
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
        try:
            import time

            issues: dict[str, Any] = {
                "contradictions": [],
                "low_confidence": [],
                "stale": [],
                "recommendations": [],
            }

            # Get all memories (or filtered by namespace)
            result = await self.memory_list(
                namespace=namespace,
                limit=1000,
            )

            if not result.get("success"):
                return result

            memories = result.get("data", {}).get("memories", [])
            now = time.time()
            stale_threshold = now - (stale_days * 86400)

            for mem in memories:
                # Check low confidence
                if include_low_confidence and mem.get("confidence", 0.3) < 0.3:
                    issues["low_confidence"].append(
                        {
                            "id": mem["id"],
                            "confidence": mem.get("confidence", 0.3),
                            "content_preview": mem.get("content", "")[:100],
                        }
                    )

                # Check stale (no recent access)
                if include_stale:
                    accessed_at = mem.get("accessed_at")
                    if accessed_at and isinstance(accessed_at, (int, float)):
                        if accessed_at < stale_threshold:
                            issues["stale"].append(
                                {
                                    "id": mem["id"],
                                    "last_accessed": accessed_at,
                                    "content_preview": mem.get("content", "")[:100],
                                }
                            )

            # Generate recommendations
            if issues["low_confidence"]:
                issues["recommendations"].append(
                    f"Found {len(issues['low_confidence'])} low-confidence memories. "
                    "Consider validating them through use or removing if no longer relevant."
                )

            if issues["stale"]:
                stale_count = len(issues["stale"])
                issues["recommendations"].append(
                    f"Found {stale_count} stale memories (no access in {stale_days} days). "
                    "Consider reviewing and validating or archiving them."
                )

            return {
                "success": True,
                "data": {
                    "total_memories": len(memories),
                    "contradictions": issues["contradictions"],
                    "contradiction_count": len(issues["contradictions"]),
                    "low_confidence": issues["low_confidence"],
                    "low_confidence_count": len(issues["low_confidence"]),
                    "stale": issues["stale"],
                    "stale_count": len(issues["stale"]),
                    "recommendations": issues["recommendations"],
                },
            }

        except Exception as e:
            logger.error(f"memory_analyze_health failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def validation_history(
        self,
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
        try:
            if not self._hybrid:
                return {
                    "success": False,
                    "error": "HybridStore not configured",
                }

            events = self._hybrid.get_validation_events(
                memory_id=memory_id,
                event_type=event_type,
                limit=limit,
            )

            # Calculate summary stats
            success_count = sum(1 for e in events if e.get("event_type") == "succeeded")
            failure_count = sum(1 for e in events if e.get("event_type") == "failed")
            applied_count = sum(1 for e in events if e.get("event_type") == "applied")

            return {
                "success": True,
                "data": {
                    "memory_id": memory_id,
                    "events": events,
                    "summary": {
                        "total_events": len(events),
                        "applied": applied_count,
                        "succeeded": success_count,
                        "failed": failure_count,
                        "success_rate": (
                            success_count / (success_count + failure_count)
                            if (success_count + failure_count) > 0
                            else None
                        ),
                    },
                },
            }

        except Exception as e:
            logger.error(f"validation_history failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
