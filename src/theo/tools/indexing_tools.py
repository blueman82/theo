"""Indexing tools for the Theo MCP server.

This module provides MCP tools for document indexing operations:
- index_file: Index a single document file
- index_directory: Index all supported files in a directory

Uses DaemonClient for non-blocking embedding operations to avoid
MCP timeout issues with expensive embedding operations.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from theo.chunking import ChunkerRegistry
from theo.daemon import DaemonClient
from theo.storage.sqlite_store import SQLiteStore

# MCP servers must never write to stdout (corrupts JSON-RPC)
logger = logging.getLogger(__name__)

# Supported file extensions for indexing
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".py", ".pdf"}


class IndexingTools:
    """Tool implementations for document indexing operations.

    Provides index_file and index_directory operations that use the
    DaemonClient for non-blocking embedding generation.

    Architectural justification:
    - DaemonClient handles expensive embedding operations asynchronously
    - ChunkerRegistry provides format-aware document splitting
    - SQLiteStore manages storage with sqlite-vec + FTS5 + deduplication

    Args:
        daemon_client: Client for daemon embedding operations
        chunker_registry: Registry for format-specific chunkers
        store: SQLiteStore instance for all storage operations

    Example:
        >>> tools = IndexingTools(daemon_client, chunker_registry, store)
        >>> result = await tools.index_file("/path/to/document.md")
        >>> if result["success"]:
        ...     print(f"Indexed {result['data']['chunks_created']} chunks")
    """

    def __init__(
        self,
        daemon_client: DaemonClient,
        chunker_registry: ChunkerRegistry,
        store: SQLiteStore,
    ) -> None:
        """Initialize IndexingTools with dependencies.

        Args:
            daemon_client: Client for daemon embedding operations
            chunker_registry: Registry for format-specific chunkers
            store: SQLiteStore instance for all storage operations
        """
        self._daemon = daemon_client
        self._chunker_registry = chunker_registry
        self._store = store

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file has a supported extension.

        Args:
            file_path: Path to check

        Returns:
            True if file extension is supported
        """
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    async def index_file(
        self,
        file_path: str,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Index a single document file.

        Reads the file, chunks it using the appropriate chunker,
        generates embeddings via the daemon, and stores in ChromaDB.

        Args:
            file_path: Absolute or relative path to the file to index
            namespace: Namespace for organizing documents (default: "default")

        Returns:
            Result dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with chunks_created, source_file, namespace
            - error: Error message if operation failed

        Raises:
            No exceptions raised - errors returned in result dict
        """
        try:
            path = Path(file_path).resolve()

            # Validate file exists and is readable
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                }

            if not path.is_file():
                return {
                    "success": False,
                    "error": f"Not a file: {file_path}",
                }

            if not self._is_supported_file(path):
                return {
                    "success": False,
                    "error": f"Unsupported file extension: {path.suffix}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                }

            # Read file content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try binary read for PDFs
                if path.suffix.lower() == ".pdf":
                    # PDF chunker handles binary content differently
                    content = path.read_bytes().decode("utf-8", errors="replace")
                else:
                    return {
                        "success": False,
                        "error": f"Cannot read file as text: {file_path}",
                    }

            if not content.strip():
                return {
                    "success": False,
                    "error": f"File is empty: {file_path}",
                }

            # Get appropriate chunker and split content
            chunker = self._chunker_registry.get_chunker(path)
            chunks = chunker.chunk(content, str(path))

            if not chunks:
                return {
                    "success": False,
                    "error": f"No chunks generated from file: {file_path}",
                }

            # Check for existing documents from this file (for re-indexing)
            # List memories to find existing chunks from this source file
            existing = self._store.list_memories(limit=1000)
            existing_from_file = [m for m in existing if m.get("source_file") == str(path)]
            if existing_from_file:
                # Delete existing chunks before re-indexing
                for mem in existing_from_file:
                    self._store.delete_memory(mem["id"])
                logger.info(f"Deleted {len(existing_from_file)} existing chunks from {path}")

            # Extract text for embedding
            chunk_texts = [chunk.text for chunk in chunks]

            # Generate embeddings via daemon (non-blocking)
            embed_result = self._daemon.embed(chunk_texts)

            if not embed_result.get("success"):
                return {
                    "success": False,
                    "error": f"Embedding generation failed: {embed_result.get('error', 'Unknown error')}",
                }

            embeddings = embed_result.get("data", {}).get("embeddings", [])

            if len(embeddings) != len(chunks):
                return {
                    "success": False,
                    "error": f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}",
                }

            # Store each chunk using SQLiteStore.add_memory
            ids = []
            for i, chunk in enumerate(chunks):
                doc_hash = self._compute_hash(chunk.text)
                memory_id = self._store.add_memory(
                    content=chunk.text,
                    embedding=embeddings[i],
                    memory_type="document",
                    namespace=namespace,
                    confidence=1.0,  # Documents have full confidence
                    importance=0.5,
                    source_file=str(path),
                    chunk_index=i,
                    content_hash=doc_hash,
                    tags=chunk.metadata,
                )
                ids.append(memory_id)

            logger.info(f"Indexed {len(ids)} chunks from {path}")

            return {
                "success": True,
                "data": {
                    "chunks_created": len(ids),
                    "source_file": str(path),
                    "namespace": namespace,
                    "chunk_ids": ids,
                },
            }

        except Exception as e:
            logger.error(f"index_file failed for {file_path}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def index_directory(
        self,
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
            async_mode: If True (default), queue chunks for background processing
                       and return immediately. If False, wait for all embeddings.

        Returns:
            Result dictionary with:
            - success: Boolean indicating operation success
            - data: Dictionary with files processed, chunks queued/created, etc.
            - error: Error message if operation failed

        Raises:
            No exceptions raised - errors returned in result dict
        """
        try:
            path = Path(dir_path).resolve()

            # Validate directory exists
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {dir_path}",
                }

            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Not a directory: {dir_path}",
                }

            # Collect files to index
            if recursive:
                files = [f for f in path.rglob("*") if f.is_file() and self._is_supported_file(f)]
            else:
                files = [f for f in path.iterdir() if f.is_file() and self._is_supported_file(f)]

            if not files:
                return {
                    "success": True,
                    "data": {
                        "files_processed": 0,
                        "total_chunks": 0,
                        "failed_files": [],
                        "directory": str(path),
                        "message": "No supported files found in directory",
                    },
                }

            # Async mode: Chunk all files and queue for background processing
            if async_mode:
                return await self._index_directory_async(path, files, namespace)

            # Sync mode: Process each file and wait for completion
            files_indexed = 0
            total_chunks = 0
            failed_files: list[dict[str, str]] = []
            indexed_files: list[str] = []

            for file_path in files:
                result = await self.index_file(str(file_path), namespace=namespace)

                if result.get("success"):
                    files_indexed += 1
                    total_chunks += result.get("data", {}).get("chunks_created", 0)
                    indexed_files.append(str(file_path))
                else:
                    failed_files.append({
                        "file": str(file_path),
                        "error": result.get("error", "Unknown error"),
                    })

            logger.info(
                f"Indexed directory {path}: {files_indexed} files, "
                f"{total_chunks} chunks, {len(failed_files)} failed"
            )

            return {
                "success": True,
                "data": {
                    "files_processed": files_indexed,
                    "total_chunks": total_chunks,
                    "failed_files": failed_files,
                    "indexed_files": indexed_files,
                    "directory": str(path),
                    "namespace": namespace,
                    "recursive": recursive,
                    "async_mode": False,
                },
            }

        except Exception as e:
            logger.error(f"index_directory failed for {dir_path}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def _index_directory_async(
        self,
        path: Path,
        files: list[Path],
        namespace: str,
    ) -> dict[str, Any]:
        """Async implementation: chunk files and queue for background processing.

        This method collects all chunks from all files (fast, I/O only) and
        queues them via the daemon's "index" command for background embedding
        and storage. Returns immediately with queue statistics.

        Args:
            path: Resolved directory path
            files: List of files to index
            namespace: Document namespace

        Returns:
            Result dict with queued chunk count
        """
        all_chunks: list[dict[str, Any]] = []
        files_processed = 0
        failed_files: list[dict[str, str]] = []

        for file_path in files:
            try:
                # Read file content
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    if file_path.suffix.lower() == ".pdf":
                        content = file_path.read_bytes().decode("utf-8", errors="replace")
                    else:
                        failed_files.append({
                            "file": str(file_path),
                            "error": "Cannot read file as text",
                        })
                        continue

                if not content.strip():
                    failed_files.append({
                        "file": str(file_path),
                        "error": "File is empty",
                    })
                    continue

                # Get chunker and split content
                chunker = self._chunker_registry.get_chunker(file_path)
                chunks = chunker.chunk(content, str(file_path))

                if not chunks:
                    failed_files.append({
                        "file": str(file_path),
                        "error": "No chunks generated",
                    })
                    continue

                # Convert chunks to dicts for daemon
                for i, chunk in enumerate(chunks):
                    doc_hash = self._compute_hash(chunk.text)
                    all_chunks.append({
                        "content": chunk.text,
                        "source_file": str(file_path),
                        "chunk_index": i,
                        "namespace": namespace,
                        "doc_hash": doc_hash,
                        "metadata": chunk.metadata,
                    })

                files_processed += 1

            except Exception as e:
                failed_files.append({
                    "file": str(file_path),
                    "error": str(e),
                })

        if not all_chunks:
            return {
                "success": True,
                "data": {
                    "files_processed": files_processed,
                    "chunks_queued": 0,
                    "failed_files": failed_files,
                    "directory": str(path),
                    "message": "No chunks to index",
                    "async_mode": True,
                },
            }

        # Queue chunks via daemon in batches to avoid buffer overflow
        # Batch size of 50 chunks keeps message size reasonable (~1-2MB)
        BATCH_SIZE = 50
        total_queued = 0
        pending_total = 0

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            index_result = self._daemon.send("index", chunks=batch, namespace=namespace)

            if not index_result.get("success"):
                logger.warning(
                    f"Daemon index batch {i // BATCH_SIZE + 1} failed: {index_result.get('error')}"
                )
                # Continue with remaining batches
                continue

            total_queued += index_result.get("data", {}).get("queued", 0)
            pending_total = index_result.get("data", {}).get("pending_total", 0)

        queued = total_queued

        logger.info(
            f"Queued {queued} chunks from {files_processed} files for async indexing "
            f"(total pending: {pending_total})"
        )

        return {
            "success": True,
            "data": {
                "files_processed": files_processed,
                "chunks_queued": queued,
                "pending_total": pending_total,
                "failed_files": failed_files,
                "directory": str(path),
                "namespace": namespace,
                "async_mode": True,
                "message": f"Queued {queued} chunks for background indexing. "
                          f"Use get_index_stats to monitor progress.",
            },
        }
