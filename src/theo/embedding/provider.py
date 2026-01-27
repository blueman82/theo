"""Embedding provider protocol for pluggable backends.

This module defines the EmbeddingProvider protocol that allows different
embedding backends (Ollama, MLX) to be used interchangeably.

API Contract (per success criteria):
    - embed_texts(texts: list[str]) -> list[list[float]] - batch embedding for documents
    - embed_query(text: str) -> list[float] - single query embedding with prefix
"""

from typing import Protocol, runtime_checkable

__all__ = ["EmbeddingProvider"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers.

    This protocol enables structural typing for embedding backends,
    allowing any class implementing these methods to be used as a provider
    without explicit inheritance.

    Required Methods:
        embed_texts: Generate embeddings for multiple texts (batch)
        embed_query: Generate embedding for a single query (with prefix)
        health_check: Check if the provider is ready
        close: Release resources

    Example:
        >>> class CustomProvider:
        ...     def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...         ...
        ...     def embed_query(self, text: str) -> list[float]:
        ...         ...
        ...     def health_check(self) -> bool:
        ...         ...
        ...     def close(self) -> None:
        ...         ...
        >>>
        >>> # Runtime check works due to @runtime_checkable
        >>> isinstance(CustomProvider(), EmbeddingProvider)
        True
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (documents).

        This is the primary batch embedding method for indexing documents.
        Documents are embedded without query-specific prefixes.

        Args:
            texts: List of input texts to embed. Must not be empty.

        Returns:
            List of embedding vectors, one per input text.
            Order is preserved: result[i] corresponds to texts[i].
            Each embedding is a list of 1024 floats (for mxbai-embed-large).

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text.

        This method applies query-specific preprocessing for asymmetric
        retrieval. For mxbai-embed models, this adds the query prefix
        "Represent this sentence for searching relevant passages: ".

        Args:
            text: Query text to embed. Must not be empty.

        Returns:
            Embedding vector as list of 1024 floats (for mxbai-embed-large).

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.
        """
        ...

    def health_check(self) -> bool:
        """Check if the provider is available and ready.

        Returns:
            True if provider is ready, False otherwise.
        """
        ...

    def close(self) -> None:
        """Release resources held by the provider.

        Should be called when the provider is no longer needed.
        Implementations should be idempotent (safe to call multiple times).
        """
        ...
