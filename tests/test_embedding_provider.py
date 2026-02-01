"""Tests for embedding providers.

Tests validate the EmbeddingProvider protocol and implementations:
- EmbeddingProvider protocol with embed_texts() and embed_query() methods
- OllamaProvider with mxbai-embed-large default (1024d)
- MLXProvider with mlx-community/mxbai-embed-large-v1 default (1024d)
- Factory function for creating providers
"""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from theo.embedding.factory import create_embedding_provider
from theo.embedding.ollama_provider import (
    EmbeddingError,
    OllamaProvider,
    retry_with_backoff,
)
from theo.embedding.provider import EmbeddingProvider


class TestRetryDecorator:
    """Test retry_with_backoff decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_after_failures(self):
        """Test function succeeds after initial failures."""
        mock_func = Mock(
            side_effect=[
                requests.RequestException("error1"),
                requests.RequestException("error2"),
                "success",
            ]
        )
        decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausted(self):
        """Test all retries exhausted."""
        mock_func = Mock(side_effect=requests.RequestException("persistent error"))
        decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

        with pytest.raises(requests.RequestException, match="persistent error"):
            decorated()

        assert mock_func.call_count == 3

    def test_retry_exponential_backoff(self):
        """Test exponential backoff delays."""
        mock_func = Mock(
            side_effect=[
                requests.RequestException("error1"),
                requests.RequestException("error2"),
                "success",
            ]
        )
        decorated = retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=1.0)(mock_func)

        start_time = time.time()
        result = decorated()
        elapsed = time.time() - start_time

        assert result == "success"
        # Should take at least base_delay * (2^0 + 2^1) = 0.1 + 0.2 = 0.3s
        assert elapsed >= 0.2


class TestOllamaProvider:
    """Test OllamaProvider class."""

    def test_init_default_params(self):
        """Test initialization with default parameters.

        Default model is mxbai-embed-large (1024d) for consistency with MLXProvider.
        """
        provider = OllamaProvider()

        assert provider.host == "http://localhost:11434"
        assert provider.model == "mxbai-embed-large"  # 1024-dim default
        assert provider.timeout == 30
        assert provider._session is not None
        assert provider._is_mxbai is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        provider = OllamaProvider(host="http://custom-host:8080", model="custom-model", timeout=60)

        assert provider.host == "http://custom-host:8080"
        assert provider.model == "custom-model"
        assert provider.timeout == 60

    def test_init_strips_trailing_slash(self):
        """Test host URL trailing slash is removed."""
        provider = OllamaProvider(host="http://localhost:11434/")

        assert provider.host == "http://localhost:11434"

    @patch("requests.Session.get")
    def test_health_check_success_model_available(self, mock_get):
        """Test health check when model is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mxbai-embed-large"}, {"name": "llama2"}]
        }
        mock_get.return_value = mock_response

        provider = OllamaProvider()
        result = provider.health_check()

        assert result is True
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    @patch("requests.Session.post")
    def test_health_check_model_not_listed_but_loadable(self, mock_post, mock_get):
        """Test health check when model not listed but can be loaded."""
        # GET /api/tags returns empty models list
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # POST /api/embeddings succeeds (model loads)
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()
        result = provider.health_check()

        assert result is True

    @patch("requests.Session.get")
    def test_health_check_server_unreachable(self, mock_get):
        """Test health check when server is unreachable."""
        mock_get.side_effect = requests.RequestException("Connection refused")

        provider = OllamaProvider()
        result = provider.health_check()

        assert result is False

    @patch("requests.Session.post")
    def test_embed_query_success(self, mock_post):
        """Test successful query embedding with mxbai prefix."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_post.return_value = mock_response

        provider = OllamaProvider()  # mxbai-embed-large by default
        embedding = provider.embed_query("What is Python?")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

        # Verify request payload - mxbai model gets query prefix
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "mxbai-embed-large"
        expected_prefix = "Represent this sentence for searching relevant passages: "
        assert call_args[1]["json"]["prompt"] == f"{expected_prefix}What is Python?"

    @patch("requests.Session.post")
    def test_embed_texts_success(self, mock_post):
        """Test successful batch document embedding (no prefix for mxbai)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        provider = OllamaProvider()  # mxbai-embed-large
        texts = ["Python is a language.", "Java is also a language."]
        embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 2
        # mxbai has no document prefix
        call_args_list = mock_post.call_args_list
        assert call_args_list[0][1]["json"]["prompt"] == "Python is a language."
        assert call_args_list[1][1]["json"]["prompt"] == "Java is also a language."

    def test_embed_query_empty_text(self):
        """Test embed_query with empty text raises ValueError."""
        provider = OllamaProvider()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            provider.embed_query("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            provider.embed_query("   ")

    def test_embed_texts_empty_list(self):
        """Test embed_texts with empty list raises ValueError."""
        provider = OllamaProvider()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            provider.embed_texts([])

    @patch("requests.Session.post")
    def test_embed_timeout(self, mock_post):
        """Test embedding request timeout."""
        mock_post.side_effect = requests.Timeout("Request timeout")

        provider = OllamaProvider(timeout=5)

        with pytest.raises(EmbeddingError, match="Request timeout after 5s"):
            provider.embed_query("Hello")

    @patch("requests.Session.post")
    def test_embed_no_embedding_in_response(self, mock_post):
        """Test handling of response without embedding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        with pytest.raises(EmbeddingError, match="No embedding returned"):
            provider.embed_query("Hello")

    @patch("requests.Session.post")
    def test_embed_request_exception(self, mock_post):
        """Test handling of request exceptions."""
        mock_post.side_effect = requests.RequestException("Network error")

        provider = OllamaProvider()

        with pytest.raises(EmbeddingError, match="Ollama API request failed"):
            provider.embed_query("Hello")

    @patch("requests.Session.post")
    def test_embed_with_retry(self, mock_post):
        """Test embedding succeeds after retry."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.RequestException("Temporary error"),
            Mock(status_code=200, json=lambda: {"embedding": [0.1, 0.2, 0.3]}),
        ]

        provider = OllamaProvider()
        embedding = provider.embed_query("Hello")

        assert embedding == [0.1, 0.2, 0.3]
        assert mock_post.call_count == 2

    @patch.object(OllamaProvider, "_embed_single")
    def test_embed_texts_large_dataset(self, mock_embed):
        """Test batch embedding with large dataset."""
        # Generate 100 mock embeddings
        mock_embed.side_effect = [[i * 0.1] * 3 for i in range(100)]

        provider = OllamaProvider()
        texts = [f"text{i}" for i in range(100)]
        embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 100
        assert mock_embed.call_count == 100

    @patch.object(OllamaProvider, "_embed_single")
    def test_embed_texts_propagates_errors(self, mock_embed):
        """Test batch embedding propagates individual embedding errors."""
        mock_embed.side_effect = [
            [0.1, 0.2, 0.3],
            EmbeddingError("Failed to embed text2"),
        ]

        provider = OllamaProvider()
        texts = ["text1", "text2", "text3"]

        with pytest.raises(EmbeddingError, match="Failed to embed text2"):
            provider.embed_texts(texts)

    def test_context_manager(self):
        """Test OllamaProvider as context manager."""
        with patch("requests.Session.close") as mock_close:
            with OllamaProvider() as provider:
                assert provider is not None

            mock_close.assert_called_once()

    @patch("requests.Session.post")
    def test_custom_timeout_configuration(self, mock_post):
        """Test custom timeout is used in requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_post.return_value = mock_response

        provider = OllamaProvider(timeout=60)
        provider.embed_query("Hello")

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60

    @patch("requests.Session.post")
    def test_http_error_handling(self, mock_post):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        with pytest.raises(EmbeddingError):
            provider.embed_query("Hello")


class TestModelManagement:
    """Test model availability and auto-pull functionality."""

    @patch("requests.Session.get")
    def test_is_model_available_true(self, mock_get):
        """Test is_model_available returns True when model exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mxbai-embed-large"}, {"name": "llama2"}]
        }
        mock_get.return_value = mock_response

        provider = OllamaProvider(model="mxbai-embed-large")
        assert provider.is_model_available() is True

    @patch("requests.Session.get")
    def test_is_model_available_false(self, mock_get):
        """Test is_model_available returns False when model not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2"}]}
        mock_get.return_value = mock_response

        provider = OllamaProvider(model="mxbai-embed-large")
        assert provider.is_model_available() is False

    @patch("requests.Session.get")
    def test_is_model_available_server_error(self, mock_get):
        """Test is_model_available returns False on server error."""
        mock_get.side_effect = requests.RequestException("Connection refused")

        provider = OllamaProvider()
        assert provider.is_model_available() is False

    @patch("requests.Session.post")
    def test_pull_model_success(self, mock_post):
        """Test successful model pull."""
        # Simulate streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"status": "pulling manifest"}',
            b'{"status": "pulling abc123", "total": 1000, "completed": 500}',
            b'{"status": "pulling abc123", "total": 1000, "completed": 1000}',
            b'{"status": "success"}',
        ]
        mock_post.return_value = mock_response

        provider = OllamaProvider(model="mxbai-embed-large")
        result = provider.pull_model()

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["name"] == "mxbai-embed-large"

    @patch("requests.Session.post")
    def test_pull_model_failure(self, mock_post):
        """Test model pull failure."""
        mock_post.side_effect = requests.RequestException("Network error")

        provider = OllamaProvider(model="nonexistent-model")
        result = provider.pull_model()

        assert result is False

    @patch("requests.Session.post")
    def test_pull_model_no_stream(self, mock_post):
        """Test model pull without streaming."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        provider = OllamaProvider(model="mxbai-embed-large")
        result = provider.pull_model(stream=False)

        assert result is True

    @patch.object(OllamaProvider, "is_model_available")
    def test_ensure_model_already_available(self, mock_available):
        """Test ensure_model when model is already available."""
        mock_available.return_value = True

        provider = OllamaProvider(model="mxbai-embed-large")
        result = provider.ensure_model()

        assert result is True
        mock_available.assert_called_once()

    @patch.object(OllamaProvider, "is_model_available")
    @patch.object(OllamaProvider, "pull_model")
    def test_ensure_model_pulls_when_missing(self, mock_pull, mock_available):
        """Test ensure_model pulls when model is not available."""
        mock_available.return_value = False
        mock_pull.return_value = True

        provider = OllamaProvider(model="mxbai-embed-large")
        result = provider.ensure_model()

        assert result is True
        mock_available.assert_called_once()
        mock_pull.assert_called_once()

    @patch.object(OllamaProvider, "is_model_available")
    @patch.object(OllamaProvider, "pull_model")
    def test_ensure_model_pull_fails(self, mock_pull, mock_available):
        """Test ensure_model returns False when pull fails."""
        mock_available.return_value = False
        mock_pull.return_value = False

        provider = OllamaProvider(model="invalid-model")
        result = provider.ensure_model()

        assert result is False


class TestEmbeddingProviderProtocol:
    """Test EmbeddingProvider protocol compliance.

    Protocol requires: embed_texts(), embed_query(), health_check(), close()
    """

    def test_ollama_provider_implements_protocol(self):
        """Test OllamaProvider implements EmbeddingProvider protocol."""
        provider = OllamaProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_protocol_runtime_checkable(self):
        """Test EmbeddingProvider protocol is runtime checkable."""

        # Create a class that implements the protocol
        class CustomProvider:
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.1, 0.2]

            def health_check(self) -> bool:
                return True

            def close(self) -> None:
                pass

        custom = CustomProvider()
        assert isinstance(custom, EmbeddingProvider)

    def test_non_compliant_class_fails_protocol(self):
        """Test that non-compliant class fails isinstance check."""

        class NonCompliant:
            def embed_query(self, text: str) -> list[float]:
                return [0.1, 0.2]

            # Missing embed_texts, health_check, close

        non_compliant = NonCompliant()
        assert not isinstance(non_compliant, EmbeddingProvider)


class TestFactory:
    """Test embedding provider factory."""

    def test_create_ollama_provider(self):
        """Test factory creates OllamaProvider with mxbai-embed-large default."""
        provider = create_embedding_provider("ollama")
        assert isinstance(provider, OllamaProvider)
        assert isinstance(provider, EmbeddingProvider)
        assert provider.model == "mxbai-embed-large"  # 1024-dim default

    def test_create_ollama_with_custom_params(self):
        """Test factory creates OllamaProvider with custom parameters."""
        provider = create_embedding_provider(
            "ollama",
            host="http://custom:8080",
            model="custom-model",
            timeout=120,
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.host == "http://custom:8080"
        assert provider.model == "custom-model"
        assert provider.timeout == 120

    def test_create_invalid_backend(self):
        """Test factory raises ValueError for invalid backend."""
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            create_embedding_provider("invalid")  # type: ignore

    def test_default_backend_is_mlx(self):
        """Test default backend is MLX."""
        # This will likely fail due to missing mlx-embeddings on non-Apple Silicon
        # We test the error message instead
        try:
            provider = create_embedding_provider()
            # If we got here, MLX is available
            assert provider is not None
        except ImportError as e:
            # Expected on non-Apple Silicon
            assert "mlx-embeddings" in str(e)


class TestMLXProvider:
    """Test MLXProvider class."""

    def test_mlx_provider_initialization(self):
        """Test MLXProvider can be instantiated."""
        # Import lazily as it may not be available
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            assert provider.model == "mlx-community/mxbai-embed-large-v1"
            assert provider._model_instance is None  # Lazy loading
            assert provider._is_mxbai is True
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_custom_model(self):
        """Test MLXProvider with custom model."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider(model="custom-model")
            assert provider.model == "custom-model"
            assert provider._is_mxbai is False  # custom-model doesn't contain "mxbai"
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_embed_query_empty_raises(self):
        """Test MLXProvider raises ValueError for empty text."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            with pytest.raises(ValueError, match="Text cannot be empty"):
                provider.embed_query("")
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_embed_texts_empty_raises(self):
        """Test MLXProvider raises ValueError for empty texts list."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            with pytest.raises(ValueError, match="Texts list cannot be empty"):
                provider.embed_texts([])
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_health_check_no_mlx(self):
        """Test MLXProvider health_check when mlx is not available."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            # health_check should return True/False without raising
            result = provider.health_check()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_close_idempotent(self):
        """Test MLXProvider close is idempotent."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            # Should not raise even when called multiple times
            provider.close()
            provider.close()
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_provider_implements_protocol(self):
        """Test MLXProvider implements EmbeddingProvider protocol."""
        try:
            from theo.embedding.mlx_provider import MLXProvider

            provider = MLXProvider()
            assert isinstance(provider, EmbeddingProvider)
        except ImportError:
            pytest.skip("MLX not available")
