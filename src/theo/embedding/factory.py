"""Factory for creating embedding providers.

This module provides a factory function for creating embedding providers
based on the configured backend.

IMPORTANT: Both backends default to mxbai-embed-large models which produce
1024-dimensional embeddings. This ensures consistency across backends.
"""

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from theo.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

# Backend type for type checking
EmbeddingBackend = Literal["ollama", "mlx"]

__all__ = ["EmbeddingBackend", "create_embedding_provider"]


def create_embedding_provider(
    backend: EmbeddingBackend = "mlx",
    *,
    host: str = "http://localhost:11434",
    model: str | None = None,
    timeout: int = 30,
    mlx_model: str = "mlx-community/mxbai-embed-large-v1",
) -> "EmbeddingProvider":
    """Create an embedding provider based on the backend configuration.

    This factory function centralizes provider instantiation, handling
    the conditional import of MLXProvider and providing helpful error
    messages when dependencies are missing.

    IMPORTANT: Both backends default to mxbai-embed-large models which produce
    1024-dimensional embeddings. This ensures consistency across backends.

    Args:
        backend: The embedding backend to use ('mlx' or 'ollama').
                 Default is 'mlx' for Apple Silicon optimization.
        host: Ollama server host URL (only used for 'ollama' backend).
        model: Embedding model name for Ollama. Defaults to 'mxbai-embed-large'
               for 1024-dim embeddings matching MLX. For MLX, use mlx_model instead.
        timeout: Request timeout in seconds (only used for 'ollama' backend).
        mlx_model: HuggingFace model path (only used for 'mlx' backend).
                   Default: mlx-community/mxbai-embed-large-v1 (1024-dim).

    Returns:
        An instance implementing the EmbeddingProvider protocol.

    Raises:
        ValueError: If an unknown backend is specified.
        ImportError: If 'mlx' backend is selected but mlx-embeddings is not installed.

    Example:
        >>> # Create MLX provider (default)
        >>> provider = create_embedding_provider()

        >>> # Create MLX provider explicitly
        >>> provider = create_embedding_provider("mlx")

        >>> # Create Ollama provider
        >>> provider = create_embedding_provider("ollama")

        >>> # Create with custom settings
        >>> provider = create_embedding_provider(
        ...     "ollama",
        ...     host="http://custom:11434",
        ...     model="mxbai-embed-large",
        ...     timeout=60,
        ... )
    """
    match backend:
        case "mlx":
            # Lazy import to handle missing mlx-embeddings gracefully
            try:
                from theo.embedding.mlx_provider import MLXProvider
            except ImportError as e:
                raise ImportError(
                    "MLX embedding backend requires mlx-embeddings. "
                    "Install with: uv add mlx-embeddings\n"
                    "Note: mlx-embeddings only works on Apple Silicon Macs."
                ) from e

            logger.info(f"Creating MLXProvider with model={mlx_model}")
            return MLXProvider(model=mlx_model)

        case "ollama":
            from theo.embedding.ollama_provider import OllamaProvider

            # Default to mxbai-embed-large for 1024-dim consistency with MLX
            ollama_model = model if model else "mxbai-embed-large"
            logger.info(
                f"Creating OllamaProvider with host={host}, "
                f"model={ollama_model}, timeout={timeout}"
            )
            return OllamaProvider(
                host=host,
                model=ollama_model,
                timeout=timeout,
            )

        case _:
            # This should never happen with proper type hints, but provides
            # a clear error message if it does
            raise ValueError(
                f"Unknown embedding backend: {backend!r}. "
                f"Valid options are: 'mlx', 'ollama'"
            )
