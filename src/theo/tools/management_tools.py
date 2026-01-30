"""Management tools for the Theo MCP server.

This module provides MCP tools for collection management operations:
- delete_chunks: Delete specific chunks by their IDs
- delete_file: Delete all chunks from a specific source file
- clear_index: Delete all documents from the collection
- get_stats: Get collection statistics

These tools provide administrative control over the stored documents.
"""

import logging
from typing import Any, Optional

from theo.storage.sqlite_store import SQLiteStore

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)


class ManagementTools:
    """Tool implementations for collection management operations.

    Provides delete_chunks, delete_file, clear_index, and get_stats
    operations for managing the document/memory collection.

    Architectural justification:
    - Thin wrapper around SQLiteStore for MCP compatibility
    - Consistent error handling and response format
    - Confirmation required for destructive operations

    Args:
        store: SQLiteStore instance for all storage operations

    Example:
        >>> tools = ManagementTools(store)
        >>> result = await tools.get_stats()
        >>> print(f"Total documents: {result['data']['total_documents']}")
    """

    def __init__(self, store: SQLiteStore) -> None:
        """Initialize ManagementTools with storage dependency.

        Args:
            store: SQLiteStore instance for all storage operations
        """
        self._store = store

    async def delete_by_ids(self, ids: list[str]) -> dict[str, Any]:
        """Delete specific chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with deleted_count, deleted_ids
            - error: Error message if operation failed
        """
        try:
            if not ids:
                return {
                    "success": False,
                    "error": "No IDs provided for deletion",
                }

            # Verify IDs exist and delete using SQLiteStore
            existing_ids = set()
            for mem_id in ids:
                mem = self._store.get_memory(mem_id)
                if mem:
                    existing_ids.add(mem_id)

            requested_ids = set(ids)
            missing_ids = requested_ids - existing_ids
            found_ids = list(requested_ids & existing_ids)

            if not found_ids:
                return {
                    "success": False,
                    "error": f"No matching documents found. Missing IDs: {list(missing_ids)}",
                }

            # Delete found documents using SQLiteStore.delete_memory
            for mem_id in found_ids:
                self._store.delete_memory(mem_id)

            logger.info(f"Deleted {len(found_ids)} documents by ID")

            result: dict[str, Any] = {
                "success": True,
                "data": {
                    "deleted_count": len(found_ids),
                    "deleted_ids": found_ids,
                },
            }

            if missing_ids:
                result["data"]["missing_ids"] = list(missing_ids)

            return result

        except Exception as e:
            logger.error(f"delete_by_ids failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def delete_by_file(self, source_file: str) -> dict[str, Any]:
        """Delete all chunks from a specific source file.

        Args:
            source_file: Source file path to delete chunks for

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with deleted_count, source_file
            - error: Error message if operation failed
        """
        try:
            if not source_file:
                return {
                    "success": False,
                    "error": "No source_file provided",
                }

            # Find documents from this source file using SQLiteStore.list_memories
            all_memories = self._store.list_memories(limit=10000)
            existing = [m for m in all_memories if m.get("source_file") == source_file]

            if not existing:
                return {
                    "success": True,
                    "data": {
                        "deleted_count": 0,
                        "source_file": source_file,
                        "message": "No documents found for this source file",
                    },
                }

            # Delete documents by source file using SQLiteStore.delete_memory
            for mem in existing:
                self._store.delete_memory(mem["id"])
            deleted_count = len(existing)

            logger.info(f"Deleted {deleted_count} documents from {source_file}")

            return {
                "success": True,
                "data": {
                    "deleted_count": deleted_count,
                    "source_file": source_file,
                },
            }

        except Exception as e:
            logger.error(f"delete_by_file failed for {source_file}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def delete_all(self, confirm: bool) -> dict[str, Any]:
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
        try:
            if not confirm:
                return {
                    "success": False,
                    "error": "Confirmation required. Set confirm=True to delete all documents.",
                }

            # Get current count before deletion using SQLiteStore.count_memories
            count_before = self._store.count_memories()

            if count_before == 0:
                return {
                    "success": True,
                    "data": {
                        "deleted_count": 0,
                        "message": "Collection was already empty",
                    },
                }

            # Clear the collection by listing and deleting all memories
            all_memories = self._store.list_memories(limit=100000)
            for mem in all_memories:
                self._store.delete_memory(mem["id"])
            deleted_count = len(all_memories)

            logger.info(f"Cleared collection: deleted {deleted_count} documents")

            return {
                "success": True,
                "data": {
                    "deleted_count": deleted_count,
                },
            }

        except Exception as e:
            logger.error(f"delete_all failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with total_documents, unique_sources,
              source_files, namespaces, doc_types
            - error: Error message if operation failed
        """
        try:
            # Get basic counts using SQLiteStore methods
            total_documents = self._store.count_memories()
            edge_count = self._store.count_edges()

            # Get unique namespaces and memory types from list_memories
            all_memories = self._store.list_memories(limit=10000)
            namespaces = list(set(m["namespace"] for m in all_memories if m.get("namespace")))
            memory_types: dict[str, int] = {}
            source_files = set()
            for m in all_memories:
                mem_type = m.get("memory_type", "document")
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                if m.get("source_file"):
                    source_files.add(m["source_file"])

            return {
                "success": True,
                "data": {
                    "total_documents": total_documents,
                    "unique_sources": len(source_files),
                    "source_files": list(source_files),
                    "namespaces": namespaces,
                    "doc_types": memory_types,
                    "edge_count": edge_count,
                },
            }

        except Exception as e:
            logger.error(f"get_stats failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def count(self, namespace: Optional[str] = None) -> dict[str, Any]:
        """Get count of documents, optionally filtered by namespace.

        Args:
            namespace: Optional namespace filter

        Returns:
            Dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with count and namespace filter
            - error: Error message if operation failed
        """
        try:
            # Use SQLiteStore.count_memories with optional namespace filter
            count = self._store.count_memories(namespace=namespace)

            return {
                "success": True,
                "data": {
                    "count": count,
                    "namespace": namespace,
                },
            }

        except Exception as e:
            logger.error(f"count failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
