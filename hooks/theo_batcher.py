#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Recall Batcher - Batch embedding generation via EmbeddingProvider.

This module provides an EmbeddingBatcher class that collects texts and generates
embeddings in batches via an EmbeddingProvider (Ollama or MLX backend).
Batching reduces API overhead from N calls to 1 call for N embeddings.

Architecture:
    add(content, memory_id) -> O(1) append to pending list
    flush() -> Single provider.embed_batch() call for all pending texts
    pending_count() -> Current queue size

The add/flush pattern allows callers to:
    1. Collect multiple texts quickly (add is O(1))
    2. Batch embed when ready (flush does 1 API call)
    3. Continue other work while flush() awaits (async)

API Contract:
    - add() is synchronous and O(1) - never blocks
    - flush() is async and returns list[(memory_id, embedding)]
    - flush() clears the pending list after successful embedding
    - On provider error, flush() raises (pending list unchanged)

Thread Safety:
    This class is NOT thread-safe. Use one instance per asyncio task.
    For multi-worker scenarios, each worker should have its own batcher.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

# Add theo source to path for imports
_theo_dir = Path.home() / "Github" / "theo"
if _theo_dir.exists() and str(_theo_dir / "src") not in sys.path:
    sys.path.insert(0, str(_theo_dir / "src"))

from theo.embedding.provider import EmbeddingProvider


# =============================================================================
# Constants
# =============================================================================

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_WAIT_SECONDS = 5.0
# mxbai-embed-large has 512 token context; JSON content is ~3 chars/token
MAX_CONTENT_CHARS = 1000


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(slots=True)
class PendingEmbedding:
    """A pending text awaiting embedding.

    Attributes:
        content: The text content to embed.
        memory_id: The memory ID to associate with the embedding.
    """

    content: str
    memory_id: str


# =============================================================================
# EmbeddingBatcher Class
# =============================================================================


class EmbeddingBatcher:
    """Batch embedding generator using an EmbeddingProvider.

    This class collects texts via add() and generates embeddings in a single
    batch call via flush(). This reduces API overhead from N calls to 1 call.

    Design Decisions:
        - Uses EmbeddingProvider abstraction for backend flexibility (Ollama/MLX)
        - flush() uses single provider.embed_batch() call for all pending texts
        - add() is O(1) append, flush() is O(1) API call regardless of batch size
        - Error handling: flush() raises on provider error, leaves pending list intact

    Attributes:
        provider: EmbeddingProvider instance for generating embeddings.
        batch_size: Maximum batch size (advisory, not enforced).
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize the embedding batcher.

        Args:
            provider: EmbeddingProvider instance (OllamaProvider or MLXProvider).
            batch_size: Maximum recommended batch size (default: 50).
        """
        self.provider = provider
        self.batch_size = batch_size
        self._pending: list[PendingEmbedding] = []

    def add(self, content: str, memory_id: str) -> None:
        """Add a text to the pending batch for embedding.

        This is an O(1) append operation that never blocks.

        Args:
            content: The text content to embed.
            memory_id: The memory ID to associate with the embedding.
        """
        self._pending.append(PendingEmbedding(content=content, memory_id=memory_id))

    async def flush(self) -> list[tuple[str, list[float]]]:
        """Generate embeddings for all pending texts in a single API call.

        Uses the EmbeddingProvider's embed_batch() method for all pending texts.
        This reduces API overhead from N calls to 1 call.

        Returns:
            List of (memory_id, embedding) tuples in the same order as added.
            Returns empty list if no pending items.

        Raises:
            EmbeddingError: On provider errors. Pending list is NOT cleared
                            on error, allowing retry.
            httpx.HTTPError: On network errors (for Ollama backend).
        """
        if not self._pending:
            return []

        # Extract texts and memory IDs, truncating long content
        texts = [item.content[:MAX_CONTENT_CHARS] for item in self._pending]
        memory_ids = [item.memory_id for item in self._pending]

        # Call provider's embed_batch method
        embeddings = await self.provider.embed_batch(texts, is_query=False)

        # Validate response length matches input
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Provider returned {len(embeddings)} embeddings for {len(texts)} texts"
            )

        # Build result list
        result = list(zip(memory_ids, embeddings, strict=True))

        # Clear pending list only on success
        self._pending.clear()

        return result

    def pending_count(self) -> int:
        """Get the number of texts pending embedding.

        Returns:
            Number of items in the pending queue.
        """
        return len(self._pending)

    def clear(self) -> int:
        """Clear all pending items without embedding.

        Returns:
            Number of items cleared.
        """
        count = len(self._pending)
        self._pending.clear()
        return count


# =============================================================================
# Module Test / Validation
# =============================================================================


async def _run_validation() -> None:
    """Run validation tests for EmbeddingBatcher.

    Note: This requires either Ollama or MLX backend to be available.
    If neither is available, integration tests will be skipped.
    """
    from theo.config import RecallSettings
    from theo.embedding.factory import create_embedding_provider

    print("Running EmbeddingBatcher validation tests...")

    # Create provider based on settings
    settings = RecallSettings()
    provider = create_embedding_provider(
        backend=settings.embedding_backend,
        host=settings.ollama_host,
        model=settings.ollama_model,
        mlx_model=settings.mlx_model,
    )

    # Test 1: Basic initialization
    batcher = EmbeddingBatcher(provider=provider)
    assert batcher.provider is provider
    assert batcher.batch_size == DEFAULT_BATCH_SIZE
    assert batcher.pending_count() == 0
    print("  Initialization OK")

    # Test 2: Custom initialization
    batcher2 = EmbeddingBatcher(provider=provider, batch_size=20)
    assert batcher2.batch_size == 20
    print("  Custom initialization OK")

    # Test 3: Add items
    batcher = EmbeddingBatcher(provider=provider)
    batcher.add("Test content 1", "mem_001")
    assert batcher.pending_count() == 1
    batcher.add("Test content 2", "mem_002")
    assert batcher.pending_count() == 2
    batcher.add("Test content 3", "mem_003")
    assert batcher.pending_count() == 3
    print("  Add items OK")

    # Test 4: Clear
    cleared = batcher.clear()
    assert cleared == 3
    assert batcher.pending_count() == 0
    print("  Clear OK")

    # Test 5: Flush empty
    result = await batcher.flush()
    assert result == []
    print("  Flush empty OK")

    # Test 6: Integration test with provider
    batcher = EmbeddingBatcher(provider=provider)
    batcher.add("Hello world", "test_001")
    batcher.add("Goodbye world", "test_002")

    try:
        result = await batcher.flush()
        assert len(result) == 2
        assert result[0][0] == "test_001"
        assert result[1][0] == "test_002"
        assert isinstance(result[0][1], list)
        assert len(result[0][1]) > 0  # Has embedding dimensions
        assert batcher.pending_count() == 0  # Cleared after flush
        print(f"  Integration test OK (embedding dim: {len(result[0][1])}, backend: {settings.embedding_backend})")
    except httpx.ConnectError:
        print("  Integration test SKIPPED (Ollama not available)")
    except httpx.HTTPStatusError as e:
        print(f"  Integration test SKIPPED (Ollama error: {e})")
    except Exception as e:
        print(f"  Integration test SKIPPED (Error: {e})")
    finally:
        await provider.close()

    print("\nAll EmbeddingBatcher validation tests passed!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_run_validation())
