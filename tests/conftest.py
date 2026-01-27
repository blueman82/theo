"""Pytest configuration and shared fixtures for theo tests.

This module provides reusable fixtures for testing:
- mock_embedder: Returns fake embeddings without MLX/Ollama
- temp_dir: Temporary directory for file operations

Usage:
    def test_something(mock_embedder, temp_dir):
        # Tests run with mock dependencies
        pass
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test file operations.

    Yields:
        Path: Path to the temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedding provider for testing without real embeddings.

    The mock returns deterministic fake embeddings based on input hash,
    allowing tests to run without MLX or Ollama dependencies.

    The mock implements the EmbeddingProvider protocol:
    - embed_texts(texts: list[str]) -> list[list[float]] - batch embedding
    - embed_query(text: str) -> list[float] - query embedding
    - health_check() -> bool
    - close() -> None

    Returns:
        MagicMock: A mock embedder implementing EmbeddingProvider protocol
    """
    embedder = MagicMock()

    def fake_embed(text: str) -> list[float]:
        """Generate deterministic fake embedding from text hash."""
        # Use hash to create reproducible embeddings
        hash_val = hash(text)
        # Return 1024-dimensional embedding (mxbai-embed-large-v1 size)
        return [(hash_val >> (i % 64) & 0xFF) / 255.0 for i in range(1024)]

    # Protocol-compliant methods (EmbeddingProvider protocol)
    embedder.embed_query = MagicMock(side_effect=fake_embed)
    embedder.embed_texts = MagicMock(side_effect=lambda texts: [fake_embed(t) for t in texts])
    embedder.health_check = MagicMock(return_value=True)
    embedder.close = MagicMock()

    return embedder


@pytest.fixture
def temp_chroma(temp_dir: Path) -> Generator[Any, None, None]:
    """Create a temporary ChromaDB instance for isolated testing.

    This fixture creates a ChromaDB client with ephemeral storage,
    ensuring tests don't affect production data.

    Args:
        temp_dir: Temporary directory fixture for ChromaDB storage

    Yields:
        chromadb.Client: A ChromaDB client configured for testing
    """
    try:
        import chromadb

        # Use ephemeral client for testing (in-memory)
        client = chromadb.Client()
        yield client
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture
def sample_documents(temp_dir: Path) -> dict[str, Path]:
    """Create sample test documents in various formats.

    Args:
        temp_dir: Temporary directory to create files in

    Returns:
        dict: Mapping of format name to file path
    """
    docs = {}

    # Markdown document
    md_content = """# Test Document

## Introduction

This is a test markdown document for theo.

## Features

- Feature one
- Feature two
- Feature three

## Conclusion

Testing is important.
"""
    md_path = temp_dir / "test.md"
    md_path.write_text(md_content)
    docs["markdown"] = md_path

    # Plain text document
    txt_content = """This is a plain text document.

It has multiple paragraphs for testing chunking.

Each paragraph should be handled appropriately.
"""
    txt_path = temp_dir / "test.txt"
    txt_path.write_text(txt_content)
    docs["text"] = txt_path

    # Python code document
    py_content = '''"""Sample Python module for testing."""

def hello_world():
    """Print hello world."""
    print("Hello, World!")

class TestClass:
    """A test class."""

    def method_one(self):
        """First method."""
        pass

    def method_two(self):
        """Second method."""
        pass
'''
    py_path = temp_dir / "test.py"
    py_path.write_text(py_content)
    docs["python"] = py_path

    return docs
