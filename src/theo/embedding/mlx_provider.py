"""MLX-based embedding provider for Apple Silicon.

This module provides embedding generation using the mlx-embeddings library,
optimized for Apple Silicon Macs.

Default model: mlx-community/mxbai-embed-large-v1 (1024 dimensions)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Query prefix for mxbai-embed models (asymmetric retrieval)
EMBED_PREFIX = "Represent this sentence for searching relevant passages: "

__all__ = ["MLXProvider", "MLXNotAvailableError", "EmbeddingError"]


class MLXNotAvailableError(Exception):
    """Raised when mlx-embeddings is not installed or unavailable."""

    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


def _check_mlx_available() -> None:
    """Check if mlx-embeddings is available.

    Raises:
        MLXNotAvailableError: If mlx-embeddings is not installed.
    """
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError as e:
        raise MLXNotAvailableError(
            "mlx-embeddings is not installed. Install with: uv add mlx-embeddings"
        ) from e


class MLXProvider:
    """MLX-based embedding provider for Apple Silicon.

    Implements the EmbeddingProvider protocol using mlx-embeddings for
    local embedding generation. Optimized for Apple Silicon Macs.

    The provider lazily loads the model on first use to avoid startup
    overhead when the provider is instantiated but not used.

    Default model: mlx-community/mxbai-embed-large-v1 (1024 dimensions)

    Args:
        model: HuggingFace model path (default: "mlx-community/mxbai-embed-large-v1")

    Example:
        >>> provider = MLXProvider()
        >>> # Query embedding (with prefix)
        >>> query_emb = provider.embed_query("What is Python?")
        >>> # Document embeddings (no prefix)
        >>> doc_embs = provider.embed_texts(["Python is a language.", "Java too."])

    Raises:
        MLXNotAvailableError: If mlx-embeddings is not installed.
    """

    def __init__(
        self,
        model: str = "mlx-community/mxbai-embed-large-v1",
    ):
        """Initialize MLX provider.

        Args:
            model: HuggingFace model path for mlx-embeddings.
                   Default uses mxbai-embed-large-v1 which produces 1024-dim embeddings.
        """
        self.model = model
        self._model_instance: Any = None
        self._tokenizer: Any = None
        self._is_mxbai = "mxbai" in model.lower()

    def _ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded.

        Lazily loads the model on first use.

        Raises:
            MLXNotAvailableError: If mlx-embeddings is not installed.
            EmbeddingError: If model loading fails.
        """
        if self._model_instance is not None:
            return

        _check_mlx_available()

        try:
            from mlx_embeddings import load

            logger.info(f"Loading MLX embedding model: {self.model}")
            self._model_instance, self._tokenizer = load(self.model)
            logger.info(f"Successfully loaded MLX model: {self.model}")
        except ImportError:
            raise MLXNotAvailableError(
                "mlx-embeddings is not installed. Install with: uv add mlx-embeddings"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load MLX model {self.model}: {e}") from e

    def _generate_embedding_sync(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings synchronously using MLX.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self._ensure_loaded()

        try:
            from mlx_embeddings import generate

            # mlx_embeddings.generate returns an output object with text_embeds attribute
            # Note: texts must be passed as keyword argument
            output = generate(self._model_instance, self._tokenizer, texts=texts)

            # Extract embeddings from the output object
            # output.text_embeds contains the normalized embeddings as MLX array
            if hasattr(output, "text_embeds"):
                embeddings = output.text_embeds
            elif hasattr(output, "embeddings"):
                embeddings = output.embeddings
            else:
                raise EmbeddingError(
                    f"Unexpected MLX output format: {type(output)}. "
                    f"Available attributes: {[a for a in dir(output) if not a.startswith('_')]}"
                )

            # Convert MLX array to Python list of floats
            if hasattr(embeddings, "tolist"):
                result = embeddings.tolist()
            else:
                result = [
                    emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    for emb in embeddings
                ]

            # Clear MLX cache to prevent GPU memory accumulation across batches
            # Without this, intermediate tensors accumulate and can consume 18+ GB
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass  # Non-critical - just memory optimization

            return result
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"MLX embedding generation failed: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (documents).

        Documents are embedded without query-specific prefixes.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors (1024-dim each for mxbai-embed-large).

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.

        Example:
            >>> provider = MLXProvider()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> embeddings = provider.embed_texts(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])  # 1024-dimensional
            1024
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # For documents, no prefix is applied
        batch_embeddings = self._generate_embedding_sync(texts)

        if not batch_embeddings or len(batch_embeddings) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, "
                f"got {len(batch_embeddings) if batch_embeddings else 0}"
            )

        return batch_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text.

        Applies query-specific prefix for mxbai models (asymmetric retrieval).

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as list of 1024 floats.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.

        Example:
            >>> provider = MLXProvider()
            >>> # Automatically prefixes with "Represent this sentence..."
            >>> query_emb = provider.embed_query("What is Python?")
            >>> len(query_emb)
            1024
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Apply query prefix for mxbai models
        if self._is_mxbai:
            prefixed_text = f"{EMBED_PREFIX}{text}"
        else:
            prefixed_text = text

        embeddings = self._generate_embedding_sync([prefixed_text])

        if not embeddings or len(embeddings) == 0:
            raise EmbeddingError("No embedding returned from MLX")

        return embeddings[0]

    def health_check(self) -> bool:
        """Check if MLX provider is available and ready.

        Returns:
            True if mlx-embeddings is available, False otherwise.
        """
        try:
            _check_mlx_available()
            return True
        except MLXNotAvailableError:
            return False

    def close(self) -> None:
        """Release resources held by the provider.

        Clears the model from memory and releases GPU resources.
        """
        if self._model_instance is not None:
            # Clear references to allow garbage collection
            self._model_instance = None
            self._tokenizer = None

            # Clear MLX Metal cache to release GPU memory and multiprocessing resources
            # This prevents semaphore leaks on process exit
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass  # Non-critical if MLX not available

            logger.debug("MLX provider resources released")

    async def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Async batch embedding for daemon workers.

        NOTE: MLX uses Metal GPU which is NOT thread-safe. This method runs
        synchronously on the main thread. The brief blocking (~50-100ms) is
        acceptable because MLX is fast and the daemon isn't latency-critical.

        Args:
            texts: List of texts to embed.
            is_query: If True, apply query prefix (for search queries).
                      If False, embed as documents (no prefix).

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.
        """
        # MLX Metal is NOT thread-safe - must run on main thread synchronously
        if is_query:
            prefixed_texts = [
                f"{EMBED_PREFIX}{t}" if self._is_mxbai else t for t in texts
            ]
            return self._generate_embedding_sync(prefixed_texts)
        else:
            return self.embed_texts(texts)
