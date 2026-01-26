"""Chunking module for document splitting and processing.

This module provides format-aware document chunking for the Theo memory system.
Different chunkers handle different file types while preserving semantic meaning:

- MarkdownChunker: Header-aware chunking for markdown files
- PDFChunker: Page-aware chunking for PDF documents
- CodeChunker: AST-based chunking for Python source code
- TextChunker: Paragraph-based chunking for plain text

The ChunkerRegistry provides automatic format detection and chunker selection
based on file extension.

Example:
    >>> from theo.chunking import ChunkerRegistry, Chunk
    >>> registry = ChunkerRegistry()
    >>> chunker = registry.get_chunker("document.md")
    >>> chunks = chunker.chunk(content, "document.md")
"""

from theo.chunking.base import AbstractChunker, Chunk
from theo.chunking.code_chunker import CodeChunker
from theo.chunking.markdown_chunker import MarkdownChunker
from theo.chunking.pdf_chunker import PDFChunker
from theo.chunking.registry import ChunkerRegistry
from theo.chunking.text_chunker import TextChunker

__all__ = [
    "AbstractChunker",
    "Chunk",
    "ChunkerRegistry",
    "CodeChunker",
    "MarkdownChunker",
    "PDFChunker",
    "TextChunker",
]
