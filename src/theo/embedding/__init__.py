"""Embedding generation module for Theo.

This module provides pluggable embedding backends for document indexing
and semantic search. The default backend is MLX (optimized for Apple Silicon).

Available backends:
    - mlx: Local embedding using mlx-embeddings (Apple Silicon)
    - ollama: Remote embedding using Ollama server

Usage:
    >>> from theo.embedding import create_embedding_provider
    >>> provider = create_embedding_provider("mlx")  # or "ollama"
    >>> query_emb = provider.embed_query("What is Python?")
    >>> doc_embs = provider.embed_texts(["doc1", "doc2"])
"""

from .factory import EmbeddingBackend, create_embedding_provider
from .ollama_provider import EmbeddingError, OllamaProvider
from .provider import EmbeddingProvider

# MLXProvider is imported lazily in factory to handle missing mlx-embeddings

__all__ = [
    "EmbeddingProvider",
    "EmbeddingBackend",
    "create_embedding_provider",
    "OllamaProvider",
    "EmbeddingError",
]
