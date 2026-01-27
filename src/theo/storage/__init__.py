"""Storage layer for Theo.

This module provides unified vector and metadata storage using SQLite with:
- sqlite-vec for semantic search with cosine similarity
- FTS5 for full-text keyword search
- Hybrid search combining vector and FTS with reciprocal rank fusion
- Hash-based deduplication
- Confidence scoring for memory validation

Additionally, SQLite storage provides:
- Relationship graphs (edges between memories)
- Graph traversal for memory expansion during recall
- Embedding cache for performance optimization

Example:
    >>> from theo.storage import SQLiteStore, Document, SearchResult
    >>> store = SQLiteStore()
    >>> # Add memories with embeddings
    >>> mem_id = store.add_memory(content="Python is great", embedding=[0.1]*1024)
    >>> # Add relationships
    >>> store.add_edge("mem1", "mem2", "supersedes")
    >>> # Search with vector similarity
    >>> results = store.search_vector(embedding=[0.1]*1024, n_results=5)
    >>> # Search with FTS
    >>> fts_results = store.search_fts("Python", n_results=5)
    >>> # Hybrid search
    >>> hybrid_results = store.search_hybrid(embedding, "Python", n_results=5)
"""

from theo.storage.hybrid import HybridStore, HybridStoreError
from theo.storage.sqlite_store import SQLiteStore, SQLiteStoreError
from theo.storage.types import Document, HybridSearchResult, SearchResult, StoreStats

# Backwards compatibility alias for code that imported StorageError
StorageError = SQLiteStoreError

__all__ = [
    "HybridStore",
    "HybridStoreError",
    "SQLiteStore",
    "SQLiteStoreError",
    "StorageError",
    "Document",
    "SearchResult",
    "HybridSearchResult",
    "StoreStats",
]
