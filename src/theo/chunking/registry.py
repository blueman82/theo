"""Chunker registry for automatic format detection and selection.

This module provides a centralized registry for mapping file extensions to
appropriate chunker implementations. The ChunkerRegistry encapsulates the
format detection logic that was previously embedded in the Indexer class.
"""

from pathlib import Path
from typing import Optional

from theo.chunking.base import AbstractChunker
from theo.chunking.code_chunker import CodeChunker
from theo.chunking.markdown_chunker import MarkdownChunker
from theo.chunking.pdf_chunker import PDFChunker
from theo.chunking.text_chunker import TextChunker


class ChunkerRegistry:
    """Registry for file extension to chunker mapping.

    This class provides automatic chunker selection based on file extension,
    with support for registering custom chunkers and configurable defaults.

    The registry uses a hierarchical approach:
    1. Exact extension match (e.g., .md -> MarkdownChunker)
    2. Language-specific chunkers (e.g., .py -> CodeChunker)
    3. Default fallback (TextChunker)

    Attributes:
        _chunker_map: Mapping of file extensions to chunker classes
        _default_chunker: Default chunker class for unknown extensions
        chunk_size: Default chunk size passed to chunker constructors
        max_tokens: Maximum tokens per chunk passed to chunker constructors
    """

    # Class-level default mappings
    DEFAULT_MAPPINGS: dict[str, type[AbstractChunker]] = {
        ".md": MarkdownChunker,
        ".markdown": MarkdownChunker,
        ".pdf": PDFChunker,
        ".txt": TextChunker,
        ".py": CodeChunker,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        max_tokens: int = 256,
        custom_mappings: Optional[dict[str, type[AbstractChunker]]] = None,
    ):
        """Initialize the chunker registry.

        Args:
            chunk_size: Default chunk size for chunkers (characters)
            max_tokens: Maximum tokens per chunk
            custom_mappings: Optional custom extension to chunker mappings
                            that override or extend the defaults

        Raises:
            ValueError: If chunk_size or max_tokens are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.chunk_size = chunk_size
        self.max_tokens = max_tokens

        # Start with default mappings
        self._chunker_map: dict[str, type[AbstractChunker]] = dict(self.DEFAULT_MAPPINGS)

        # Override with custom mappings if provided
        if custom_mappings:
            self._chunker_map.update(custom_mappings)

        self._default_chunker: type[AbstractChunker] = TextChunker

    def get_chunker(self, file_path: str | Path) -> AbstractChunker:
        """Get appropriate chunker for a file based on its extension.

        Args:
            file_path: Path to the file (string or Path object)

        Returns:
            Instantiated chunker appropriate for the file type
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = path.suffix.lower()

        chunker_class = self._chunker_map.get(suffix, self._default_chunker)
        return self._create_chunker(chunker_class)

    def _create_chunker(self, chunker_class: type[AbstractChunker]) -> AbstractChunker:
        """Create a chunker instance with appropriate parameters.

        Different chunker types have different constructor signatures,
        so we customize the instantiation accordingly.

        Args:
            chunker_class: The chunker class to instantiate

        Returns:
            Instantiated chunker
        """
        if chunker_class == CodeChunker:
            # CodeChunker uses chunk_size for line count, not characters
            # Default to 100 lines per chunk for fallback mode
            return CodeChunker(chunk_size=100, max_tokens=self.max_tokens)
        elif chunker_class in (MarkdownChunker, PDFChunker, TextChunker):
            # These chunkers accept chunk_size, chunk_overlap, and max_tokens
            return chunker_class(
                chunk_size=self.chunk_size,
                chunk_overlap=min(200, self.chunk_size // 5),
                max_tokens=self.max_tokens,
            )
        else:
            # Custom chunker - try with both params, fall back if needed
            try:
                return chunker_class(
                    chunk_size=self.chunk_size,
                    max_tokens=self.max_tokens,
                )
            except TypeError:
                # Custom chunker might have different signature
                return chunker_class()

    def register(self, extension: str, chunker_class: type[AbstractChunker]) -> None:
        """Register a custom chunker for a file extension.

        Args:
            extension: File extension including dot (e.g., ".rst")
            chunker_class: Chunker class to use for this extension

        Raises:
            ValueError: If extension doesn't start with '.'
        """
        if not extension.startswith("."):
            raise ValueError(f"Extension must start with '.', got: {extension}")

        self._chunker_map[extension.lower()] = chunker_class

    def unregister(self, extension: str) -> bool:
        """Unregister a chunker for a file extension.

        Args:
            extension: File extension to unregister

        Returns:
            True if extension was registered and removed, False otherwise
        """
        extension = extension.lower()
        if extension in self._chunker_map:
            del self._chunker_map[extension]
            return True
        return False

    def set_default_chunker(self, chunker_class: type[AbstractChunker]) -> None:
        """Set the default chunker for unknown extensions.

        Args:
            chunker_class: Chunker class to use as default
        """
        self._default_chunker = chunker_class

    def get_supported_extensions(self) -> list[str]:
        """Get list of all registered file extensions.

        Returns:
            Sorted list of supported file extensions
        """
        return sorted(self._chunker_map.keys())

    def is_supported(self, file_path: str | Path) -> bool:
        """Check if a file extension has a specific chunker registered.

        Note: This returns False for files that would use the default chunker.
        All files can be chunked (using the default), but this method indicates
        whether there's a specialized chunker for the file type.

        Args:
            file_path: Path to check

        Returns:
            True if a specific chunker is registered for this extension
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.suffix.lower() in self._chunker_map

    def __repr__(self) -> str:
        """Return string representation of registry state."""
        extensions = ", ".join(sorted(self._chunker_map.keys()))
        return f"ChunkerRegistry(extensions=[{extensions}], default={self._default_chunker.__name__})"
