"""Storage layer for Theo.

This module provides vector and metadata storage using ChromaDB with:
- Semantic search with metadata filtering
- Hybrid search combining vector and FTS
- Hash-based deduplication
- Confidence scoring for memory validation

Additionally, SQLite storage provides:
- Relationship graphs (edges between memories)
- Graph traversal for memory expansion during recall

Example:
    >>> from theo.storage import ChromaStore, SQLiteStore, Document, SearchResult
    >>> chroma = ChromaStore(ephemeral=True)
    >>> sqlite = SQLiteStore()
    >>> # Add documents with embeddings
    >>> ids = chroma.add_documents(documents, embeddings)
    >>> # Add relationships
    >>> sqlite.add_edge("mem1", "mem2", "supersedes")
    >>> # Search with graph expansion
    >>> results = chroma.search(query_embedding, n_results=5)
    >>> related_ids = sqlite.get_related(results[0].document.id, max_depth=2)
"""

from theo.storage.chroma_store import ChromaStore, StorageError
from theo.storage.hybrid import HybridStore, HybridStoreError
from theo.storage.sqlite_store import SQLiteStore, SQLiteStoreError
from theo.storage.types import Document, HybridSearchResult, SearchResult, StoreStats

__all__ = [
    "ChromaStore",
    "StorageError",
    "HybridStore",
    "HybridStoreError",
    "SQLiteStore",
    "SQLiteStoreError",
    "Document",
    "SearchResult",
    "HybridSearchResult",
    "StoreStats",
]
