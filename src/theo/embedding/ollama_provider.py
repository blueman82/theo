"""Ollama embedding provider with retry logic and health checks.

This module provides an HTTP client for the Ollama embeddings API with:
- Exponential backoff retry logic for network resilience
- Health checks to validate model availability
- Batch embedding support to reduce API calls
- Configurable timeout handling

IMPORTANT: Uses mxbai-embed-large by default (1024 dimensions) to match MLXProvider.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable

import requests

logger = logging.getLogger(__name__)

# Query prefix for mxbai-embed models (asymmetric retrieval)
MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    pass


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 10.0)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3)
        >>> def fetch_data():
        ...     return requests.get("http://example.com")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, EmbeddingError):
                    if attempt == max_retries - 1:
                        raise

                    delay = min(base_delay * (2**attempt), max_delay)
                    time.sleep(delay)

            return func(*args, **kwargs)

        return wrapper

    return decorator


class OllamaProvider:
    """HTTP client for Ollama embeddings API.

    Implements the EmbeddingProvider protocol using Ollama for remote
    embedding generation with retry logic and health checks.

    IMPORTANT: Default model is mxbai-embed-large (1024 dimensions) to ensure
    consistency with MLXProvider. Both providers produce identical embedding
    dimensions for interoperability.

    Args:
        host: Ollama server host URL (default: "http://localhost:11434")
        model: Embedding model name (default: "mxbai-embed-large" - 1024d)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> provider = OllamaProvider()
        >>> query_emb = provider.embed_query("What is Python?")
        >>> doc_embs = provider.embed_texts(["Python is a language.", "Java too."])
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        timeout: int = 30,
    ):
        """Initialize Ollama provider.

        Args:
            host: Ollama server host URL
            model: Embedding model name (default: mxbai-embed-large for 1024d)
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        self._is_mxbai = "mxbai" in model.lower()

    def health_check(self) -> bool:
        """Check if Ollama server is available and model is loaded.

        Validates that:
        1. Ollama server is responding
        2. The specified model exists and is available

        Returns:
            True if health check passes, False otherwise

        Example:
            >>> provider = OllamaProvider()
            >>> if provider.health_check():
            ...     print("Ollama is ready")
        """
        try:
            # Check if server is responding
            response = self._session.get(f"{self.host}/api/tags", timeout=self.timeout)
            response.raise_for_status()

            # Verify model exists in available models
            models_data = response.json()
            available_models = [
                m.get("name", "") for m in models_data.get("models", [])
            ]

            # Check if our model is in the list
            model_available = any(
                self.model in model_name for model_name in available_models
            )

            if not model_available:
                # Try to pull/load the model by making a test embedding request
                try:
                    test_response = self._request_with_retry(
                        {"model": self.model, "prompt": "test"}
                    )
                    return bool(test_response.status_code == 200)
                except Exception:
                    return False

            return True

        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Check if the configured model is available locally.

        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models_data = response.json()
            available_models = [
                m.get("name", "") for m in models_data.get("models", [])
            ]
            return any(self.model in model_name for model_name in available_models)
        except Exception:
            return False

    def pull_model(self, stream: bool = True) -> bool:
        """Pull/download the configured model from Ollama registry.

        This is a blocking operation that downloads the model if not present.
        For mxbai-embed-large (~670MB), this may take several minutes.

        Args:
            stream: If True, streams progress to logger (default: True)

        Returns:
            True if model was pulled successfully, False otherwise
        """
        logger.info(f"Pulling model '{self.model}' from Ollama registry...")

        try:
            response = self._session.post(
                f"{self.host}/api/pull",
                json={"name": self.model, "stream": stream},
                timeout=None,  # No timeout for model pulls (can take minutes)
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                # Process streaming response to show progress
                last_status = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            import json

                            data = json.loads(line)
                            status = data.get("status", "")

                            # Log progress updates (deduplicated)
                            if status != last_status:
                                if "pulling" in status:
                                    total = data.get("total", 0)
                                    completed = data.get("completed", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        logger.info(
                                            f"Pulling {self.model}: {pct:.1f}% "
                                            f"({completed}/{total} bytes)"
                                        )
                                    else:
                                        logger.info(f"Pulling {self.model}: {status}")
                                elif status == "success":
                                    logger.info(
                                        f"Successfully pulled model '{self.model}'"
                                    )
                                else:
                                    logger.info(f"Pull status: {status}")
                                last_status = status
                        except (ValueError, KeyError):
                            continue
            else:
                logger.info(f"Successfully pulled model '{self.model}'")

            return True

        except requests.RequestException as e:
            logger.error(f"Failed to pull model '{self.model}': {e}")
            return False

    def ensure_model(self) -> bool:
        """Ensure the model is available, pulling it if necessary.

        Returns:
            True if model is available, False if it could not be made available
        """
        if self.is_model_available():
            logger.info(f"Model '{self.model}' is already available")
            return True

        logger.info(f"Model '{self.model}' not found locally, attempting to pull...")
        return self.pull_model()

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
    def _request_with_retry(self, payload: dict) -> requests.Response:
        """Make HTTP request to Ollama API with retry logic.

        Args:
            payload: Request payload dictionary

        Returns:
            HTTP response object

        Raises:
            EmbeddingError: If request fails after all retries
        """
        try:
            response = self._session.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response

        except requests.Timeout as e:
            raise EmbeddingError(
                f"Request timeout after {self.timeout}s. "
                f"Consider increasing timeout for model {self.model}"
            ) from e

        except requests.RequestException as e:
            raise EmbeddingError(f"Ollama API request failed: {e}") from e

    def _embed_single(self, text: str, apply_query_prefix: bool = False) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            apply_query_prefix: If True, apply query-specific prefix

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Apply mxbai query prefix if needed
            if apply_query_prefix and self._is_mxbai:
                prefixed_text = f"{MXBAI_QUERY_PREFIX}{text}"
            else:
                prefixed_text = text

            payload = {"model": self.model, "prompt": prefixed_text}
            response = self._request_with_retry(payload)

            data = response.json()
            embedding = data.get("embedding")

            if not embedding:
                raise EmbeddingError("No embedding returned from Ollama API")

            return list(embedding) if embedding else []

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (documents).

        Documents are embedded without query-specific prefixes.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors (1024-dim each for mxbai-embed-large)

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts list is empty

        Example:
            >>> provider = OllamaProvider()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> embeddings = provider.embed_texts(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])  # 1024-dimensional for mxbai-embed-large
            1024
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings = []

        try:
            for text in texts:
                # Documents: no query prefix
                embedding = self._embed_single(text, apply_query_prefix=False)
                embeddings.append(embedding)

            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text.

        Applies query-specific prefix for mxbai models (asymmetric retrieval).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of 1024 floats (for mxbai-embed-large)

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Example:
            >>> provider = OllamaProvider()
            >>> # Automatically prefixes with "Represent this sentence..."
            >>> query_emb = provider.embed_query("What is Python?")
            >>> len(query_emb)
            1024
        """
        return self._embed_single(text, apply_query_prefix=True)

    def close(self) -> None:
        """Close the HTTP session and release resources.

        Should be called when the provider is no longer needed.
        This method is idempotent (safe to call multiple times).
        """
        if self._session is not None:
            self._session.close()
            logger.debug("OllamaProvider session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close()

    async def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Async batch embedding for daemon workers.

        This method provides an async interface for the daemon's EmbeddingBatcher.
        Since Ollama operations are I/O-bound, we run them in a thread pool to avoid
        blocking the event loop.

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
        if is_query:
            # For queries, embed each with query prefix
            return await asyncio.to_thread(
                lambda: [self._embed_single(t, apply_query_prefix=True) for t in texts]
            )
        else:
            return await asyncio.to_thread(self.embed_texts, texts)
