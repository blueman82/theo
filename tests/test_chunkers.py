"""Tests for the chunking module.

This module contains comprehensive tests for all chunker implementations
including base functionality, format-specific chunkers, and the registry.
"""

from unittest.mock import Mock, patch

import pytest

from theo.chunking import (
    AbstractChunker,
    Chunk,
    ChunkerRegistry,
    CodeChunker,
    MarkdownChunker,
    PDFChunker,
    TextChunker,
)

# =============================================================================
# Chunk Dataclass Tests
# =============================================================================


class TestChunk:
    """Test suite for Chunk dataclass."""

    def test_chunk_creation_valid(self):
        """Test creating a valid chunk with all required fields."""
        chunk = Chunk(
            text="This is test content",
            metadata={"key": "value"},
            start_line=1,
            end_line=5,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunk.text == "This is test content"
        assert chunk.metadata == {"key": "value"}
        assert chunk.start_line == 1
        assert chunk.end_line == 5
        assert chunk.source_file == "/path/to/file.txt"
        assert chunk.chunk_index == 0
        assert chunk.token_count is None

    def test_chunk_has_required_fields(self):
        """Test that Chunk dataclass includes text, metadata, start_line, end_line fields."""
        # Verify the dataclass has these as explicit fields (not just in metadata)
        chunk = Chunk(
            text="Test",
            metadata={},
            start_line=1,
            end_line=10,
        )
        # Check these are real attributes, not just metadata keys
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "metadata")
        assert hasattr(chunk, "start_line")
        assert hasattr(chunk, "end_line")

        # Verify they can be accessed directly as dataclass fields
        assert chunk.text == "Test"
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert isinstance(chunk.metadata, dict)

    def test_chunk_creation_with_metadata(self):
        """Test creating chunk with additional metadata."""
        metadata = {"page": 1, "section": "introduction"}
        chunk = Chunk(
            text="Content with metadata",
            metadata=metadata,
            start_line=5,
            end_line=15,
            source_file="/path/to/doc.pdf",
            chunk_index=5,
        )
        assert chunk.metadata == metadata
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["section"] == "introduction"

    def test_chunk_creation_with_token_count(self):
        """Test creating chunk with token count."""
        chunk = Chunk(
            text="Content with tokens",
            metadata={},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
            token_count=42,
        )
        assert chunk.token_count == 42

    def test_chunk_immutability(self):
        """Test that chunks are immutable (frozen)."""
        chunk = Chunk(
            text="Immutable content",
            metadata={},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        with pytest.raises(AttributeError):
            chunk.text = "Modified content"

    def test_chunk_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk(
                text="",
                metadata={},
                start_line=1,
                end_line=1,
                source_file="/path/to/file.txt",
                chunk_index=0,
            )

    def test_chunk_whitespace_only_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk(
                text="   \n\t  ",
                metadata={},
                start_line=1,
                end_line=1,
                source_file="/path/to/file.txt",
                chunk_index=0,
            )

    def test_chunk_negative_index_raises_error(self):
        """Test that negative chunk index raises ValueError."""
        with pytest.raises(ValueError, match="Chunk index must be non-negative"):
            Chunk(
                text="Valid content",
                metadata={},
                start_line=1,
                end_line=1,
                source_file="/path/to/file.txt",
                chunk_index=-1,
            )

    def test_chunk_negative_start_line_raises_error(self):
        """Test that negative start_line raises ValueError."""
        with pytest.raises(ValueError, match="start_line must be non-negative"):
            Chunk(
                text="Valid content",
                metadata={},
                start_line=-1,
                end_line=1,
            )

    def test_chunk_end_line_before_start_line_raises_error(self):
        """Test that end_line before start_line raises ValueError."""
        with pytest.raises(ValueError, match="end_line.*must be >= start_line"):
            Chunk(
                text="Valid content",
                metadata={},
                start_line=10,
                end_line=5,
            )

    def test_chunk_equality(self):
        """Test chunk equality comparison."""
        chunk1 = Chunk(
            text="Same content",
            metadata={},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        chunk2 = Chunk(
            text="Same content",
            metadata={},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunk1 == chunk2


# =============================================================================
# AbstractChunker Tests
# =============================================================================


class ConcreteChunker(AbstractChunker):
    """Concrete implementation of AbstractChunker for testing."""

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Simple implementation that splits on newlines."""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        lines = content.split("\n")
        result = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                result.append(
                    Chunk(
                        text=stripped,
                        metadata={},
                        start_line=idx + 1,
                        end_line=idx + 1,
                        source_file=source_file,
                        chunk_index=len(result),
                    )
                )
        return result


class TestAbstractChunker:
    """Test suite for AbstractChunker interface."""

    def test_concrete_chunker_implementation(self):
        """Test that concrete implementation works."""
        chunker = ConcreteChunker()
        content = "Line 1\nLine 2\nLine 3"
        chunks = chunker.chunk(content, "/path/to/file.txt")

        assert len(chunks) == 3
        assert chunks[0].text == "Line 1"
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1
        assert chunks[1].text == "Line 2"
        assert chunks[1].chunk_index == 1
        assert chunks[2].text == "Line 3"
        assert chunks[2].chunk_index == 2

    def test_chunker_empty_content_raises_error(self):
        """Test that chunker raises error for empty content."""
        chunker = ConcreteChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/path/to/file.txt")

    def test_validate_chunk_quality_valid(self):
        """Test validate_chunk_quality with valid chunk."""
        chunker = ConcreteChunker()
        chunk = Chunk(
            text="Valid content",
            metadata={},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunker.validate_chunk_quality(chunk) is True

    def test_abstract_chunker_cannot_instantiate(self):
        """Test that AbstractChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractChunker()


class TestSplitOversizedChunk:
    """Test suite for split_oversized_chunk method."""

    def test_chunk_under_limit_returns_unchanged(self):
        """Test that a chunk under the token limit is returned unchanged."""
        chunker = ConcreteChunker()
        chunk = Chunk(
            text="Short content",
            metadata={"existing": "data"},
            start_line=1,
            end_line=1,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=100)

        assert len(result) == 1
        assert result[0].text == "Short content"
        assert result[0].chunk_index == 0
        assert result[0].metadata["existing"] == "data"
        assert "split_part" not in result[0].metadata

    def test_split_at_paragraph_boundaries(self):
        """Test that oversized chunks are split at paragraph boundaries."""
        chunker = ConcreteChunker()
        content = (
            "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        )
        chunk = Chunk(
            text=content,
            metadata={},
            start_line=1,
            end_line=5,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=15)

        assert len(result) > 1
        for i, part in enumerate(result):
            assert part.metadata["split_part"] == i
            assert part.source_file == "/path/to/file.txt"

    def test_split_preserves_source_file(self):
        """Test that split chunks preserve the source file."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two.\n\nPara three."
        chunk = Chunk(
            text=content,
            metadata={},
            start_line=1,
            end_line=5,
            source_file="/important/document.txt",
            chunk_index=5,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        for part in result:
            assert part.source_file == "/important/document.txt"

    def test_split_uses_base_index_for_chunk_indices(self):
        """Test that split uses base_index parameter for chunk indices."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two.\n\nPara three."
        chunk = Chunk(
            text=content,
            metadata={},
            start_line=1,
            end_line=5,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5, base_index=10)

        assert result[0].chunk_index == 10
        if len(result) > 1:
            assert result[1].chunk_index == 11


# =============================================================================
# MarkdownChunker Tests
# =============================================================================


class TestMarkdownChunker:
    """Test suite for MarkdownChunker."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        chunker = MarkdownChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_invalid_chunk_size(self):
        """Test that negative or zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MarkdownChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MarkdownChunker(chunk_size=-100)

    def test_init_invalid_overlap(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            MarkdownChunker(chunk_overlap=-50)

    def test_chunk_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = MarkdownChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/path/to/file.md")

    def test_chunk_simple_markdown_single_header(self):
        """Test chunking markdown with single header."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Introduction

This is the introduction section.
It has multiple lines of content."""

        chunks = chunker.chunk(content, "/docs/readme.md")

        assert len(chunks) == 1
        assert (
            chunks[0].text == "This is the introduction section.\nIt has multiple lines of content."
        )
        assert chunks[0].source_file == "/docs/readme.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["header_path"] == "Introduction"
        assert chunks[0].metadata["header_level"] == 1
        # Verify start_line and end_line are set
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= chunks[0].start_line

    def test_chunk_multiple_h1_headers(self):
        """Test chunking markdown with multiple top-level headers."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Chapter 1

Content of chapter 1.

# Chapter 2

Content of chapter 2."""

        chunks = chunker.chunk(content, "/docs/book.md")

        assert len(chunks) == 2
        assert chunks[0].metadata["header_path"] == "Chapter 1"
        assert chunks[0].text == "Content of chapter 1."
        assert chunks[1].metadata["header_path"] == "Chapter 2"
        assert chunks[1].text == "Content of chapter 2."

    def test_chunk_nested_headers(self):
        """Test chunking markdown with nested header hierarchy."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Chapter 1

Introduction to chapter 1.

## Section 1.1

Content of section 1.1.

## Section 1.2

Content of section 1.2.

### Subsection 1.2.1

Content of subsection 1.2.1."""

        chunks = chunker.chunk(content, "/docs/manual.md")

        assert len(chunks) == 4
        assert chunks[0].metadata["header_path"] == "Chapter 1"
        assert chunks[1].metadata["header_path"] == "Chapter 1 > Section 1.1"
        assert chunks[2].metadata["header_path"] == "Chapter 1 > Section 1.2"
        assert chunks[3].metadata["header_path"] == "Chapter 1 > Section 1.2 > Subsection 1.2.1"

    def test_chunk_no_headers(self):
        """Test chunking plain text without headers."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """This is plain text without any headers.

It should still be chunked properly."""

        chunks = chunker.chunk(content, "/docs/plain.md")

        assert len(chunks) == 1
        assert chunks[0].metadata["header_path"] == "Document"
        assert chunks[0].metadata["header_level"] == 0


# =============================================================================
# CodeChunker Tests
# =============================================================================


class TestCodeChunker:
    """Test suite for CodeChunker."""

    def test_chunk_simple_function(self):
        """Test chunking file with single function."""
        chunker = CodeChunker()
        content = '''def hello():
    """Say hello."""
    print("Hello, world!")
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].text.strip() == content.strip()
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["type"] == "function"
        # Verify start_line and end_line are set
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= chunks[0].start_line

    def test_chunk_multiple_functions(self):
        """Test chunking file with multiple functions."""
        chunker = CodeChunker()
        content = """def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 3
        assert "def func1():" in chunks[0].text
        assert "def func2():" in chunks[1].text
        assert "def func3():" in chunks[2].text

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.metadata["type"] == "function"

    def test_chunk_class_definition(self):
        """Test chunking file with class definition."""
        chunker = CodeChunker()
        content = '''class MyClass:
    """A simple class."""

    def __init__(self):
        self.value = 42

    def method(self):
        return self.value
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert "class MyClass:" in chunks[0].text
        assert chunks[0].metadata["type"] == "class"
        assert chunks[0].metadata["class_name"] == "MyClass"

    def test_chunk_with_imports(self):
        """Test that imports are preserved in first chunk."""
        chunker = CodeChunker()
        content = """import os
import sys
from typing import Optional

def func():
    return os.path.join("a", "b")
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 2
        assert chunks[0].metadata["type"] == "imports"
        assert "import os" in chunks[0].text
        assert "import sys" in chunks[0].text

        assert chunks[1].metadata["type"] == "function"

    def test_chunk_async_function(self):
        """Test chunking async function."""
        chunker = CodeChunker()
        content = '''async def async_func():
    """An async function."""
    await something()
    return 42
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "async_function"

    def test_chunk_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = CodeChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "test.py")

    def test_fallback_chunking_syntax_error(self):
        """Test that syntax errors trigger fallback chunking."""
        chunker = CodeChunker(chunk_size=3)
        content = """def broken(
    this is not valid python
    syntax error here
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "fallback"

    def test_chunk_decorator_with_function(self):
        """Test that decorators are included with function."""
        chunker = CodeChunker()
        content = """@decorator
@another_decorator(arg=True)
def decorated_func():
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "function"
        assert "@decorator" in chunks[0].text
        assert "@another_decorator" in chunks[0].text


# =============================================================================
# TextChunker Tests
# =============================================================================


class TestTextChunker:
    """Test suite for TextChunker."""

    def test_default_initialization(self):
        """Test chunker with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_custom_chunk_size(self):
        """Test chunker with custom chunk size."""
        chunker = TextChunker(chunk_size=500)
        assert chunker.chunk_size == 500

    def test_invalid_chunk_size_zero(self):
        """Test that zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

    def test_invalid_chunk_overlap_negative(self):
        """Test that negative chunk_overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_overlap=-50)

    def test_single_paragraph_fits_in_chunk(self):
        """Test single paragraph that fits in one chunk."""
        chunker = TextChunker(chunk_size=1000)
        content = "This is a single paragraph. It has multiple sentences. But it fits in one chunk."
        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 1
        assert chunks[0].text == content
        assert chunks[0].source_file == "/test/file.txt"
        assert chunks[0].chunk_index == 0
        # Verify start_line and end_line are set as explicit fields
        assert chunks[0].start_line >= 0
        assert chunks[0].end_line >= chunks[0].start_line

    def test_multiple_paragraphs(self):
        """Test splitting multiple paragraphs."""
        chunker = TextChunker(chunk_size=1000)
        content = """First paragraph here.

Second paragraph here.

Third paragraph here."""
        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text
        assert "Third paragraph" in chunks[2].text

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = TextChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/test/file.txt")

    def test_sequential_chunk_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        content = """Paragraph one.

Paragraph two.

Paragraph three.

Paragraph four."""
        chunks = chunker.chunk(content, "/test/file.txt")

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx


# =============================================================================
# PDFChunker Tests
# =============================================================================


class TestPDFChunker:
    """Test suite for PDFChunker."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        chunker = PDFChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_invalid_chunk_size_zero(self):
        """Test that chunk_size of 0 raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            PDFChunker(chunk_size=0)

    def test_init_invalid_overlap_negative(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            PDFChunker(chunk_overlap=-50)

    def test_chunk_file_not_found(self, tmp_path):
        """Test that non-existent file raises error."""
        chunker = PDFChunker()
        non_existent = tmp_path / "does_not_exist.pdf"

        with pytest.raises(ValueError, match="PDF file does not exist"):
            chunker.chunk("", str(non_existent))

    def test_chunk_non_pdf_file(self, tmp_path):
        """Test that non-PDF file raises error."""
        chunker = PDFChunker()
        text_file = tmp_path / "document.txt"
        text_file.write_text("Not a PDF")

        with pytest.raises(ValueError, match="File is not a PDF"):
            chunker.chunk("", str(text_file))

    @patch("theo.chunking.pdf_chunker.PdfReader")
    def test_extract_single_page(self, mock_reader, tmp_path):
        """Test extraction from single-page PDF."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "single.pdf"
        pdf_file.write_text("fake pdf")

        mock_page = Mock()
        mock_page.extract_text.return_value = "This is page one content."
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) == 1
        assert chunks[0].text == "This is page one content."
        assert chunks[0].metadata["page_number"] == 1
        # PDF chunks use page numbers as line references
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    @patch("theo.chunking.pdf_chunker.PdfReader")
    def test_extract_multiple_pages(self, mock_reader, tmp_path):
        """Test extraction from multi-page PDF."""
        chunker = PDFChunker(chunk_size=100, chunk_overlap=20)
        pdf_file = tmp_path / "multi.pdf"
        pdf_file.write_text("fake pdf")

        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one text."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page two text."
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page three text."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source_file == str(pdf_file)


# =============================================================================
# ChunkerRegistry Tests
# =============================================================================


class TestChunkerRegistry:
    """Test suite for ChunkerRegistry."""

    def test_default_initialization(self):
        """Test registry with default parameters."""
        registry = ChunkerRegistry()
        assert registry.chunk_size == 1000
        assert registry.max_tokens == 256

    def test_custom_initialization(self):
        """Test registry with custom parameters."""
        registry = ChunkerRegistry(chunk_size=500, max_tokens=512)
        assert registry.chunk_size == 500
        assert registry.max_tokens == 512

    def test_invalid_chunk_size(self):
        """Test that invalid chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkerRegistry(chunk_size=0)

    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ChunkerRegistry(max_tokens=0)

    def test_get_chunker_markdown(self):
        """Test getting chunker for markdown files."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker("document.md")
        assert isinstance(chunker, MarkdownChunker)

    def test_get_chunker_python(self):
        """Test getting chunker for Python files."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker("script.py")
        assert isinstance(chunker, CodeChunker)

    def test_get_chunker_pdf(self):
        """Test getting chunker for PDF files."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker("document.pdf")
        assert isinstance(chunker, PDFChunker)

    def test_get_chunker_text(self):
        """Test getting chunker for text files."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker("document.txt")
        assert isinstance(chunker, TextChunker)

    def test_get_chunker_unknown_extension(self):
        """Test that unknown extensions fall back to TextChunker."""
        registry = ChunkerRegistry()
        chunker = registry.get_chunker("document.xyz")
        assert isinstance(chunker, TextChunker)

    def test_register_custom_chunker(self):
        """Test registering a custom chunker."""
        registry = ChunkerRegistry()
        registry.register(".rst", MarkdownChunker)

        chunker = registry.get_chunker("document.rst")
        assert isinstance(chunker, MarkdownChunker)

    def test_register_invalid_extension(self):
        """Test that registering without dot raises error."""
        registry = ChunkerRegistry()
        with pytest.raises(ValueError, match="Extension must start with '.'"):
            registry.register("md", MarkdownChunker)

    def test_unregister_chunker(self):
        """Test unregistering a chunker."""
        registry = ChunkerRegistry()
        assert registry.unregister(".md") is True
        chunker = registry.get_chunker("document.md")
        assert isinstance(chunker, TextChunker)

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent extension."""
        registry = ChunkerRegistry()
        assert registry.unregister(".xyz") is False

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        registry = ChunkerRegistry()
        extensions = registry.get_supported_extensions()

        assert ".md" in extensions
        assert ".py" in extensions
        assert ".pdf" in extensions
        assert ".txt" in extensions

    def test_is_supported(self):
        """Test checking if extension is supported."""
        registry = ChunkerRegistry()
        assert registry.is_supported("document.md") is True
        assert registry.is_supported("script.py") is True
        assert registry.is_supported("unknown.xyz") is False

    def test_custom_mappings_initialization(self):
        """Test initialization with custom mappings."""
        custom = {".rst": MarkdownChunker}
        registry = ChunkerRegistry(custom_mappings=custom)

        assert registry.is_supported("document.rst") is True
        chunker = registry.get_chunker("document.rst")
        assert isinstance(chunker, MarkdownChunker)

    def test_set_default_chunker(self):
        """Test setting a custom default chunker."""
        registry = ChunkerRegistry()
        registry.set_default_chunker(MarkdownChunker)

        chunker = registry.get_chunker("document.unknown")
        assert isinstance(chunker, MarkdownChunker)

    def test_repr(self):
        """Test string representation."""
        registry = ChunkerRegistry()
        repr_str = repr(registry)
        assert "ChunkerRegistry" in repr_str
        assert "TextChunker" in repr_str

    def test_pathlib_path_support(self):
        """Test that registry accepts pathlib.Path objects."""
        from pathlib import Path

        registry = ChunkerRegistry()
        path = Path("/some/path/document.md")
        chunker = registry.get_chunker(path)
        assert isinstance(chunker, MarkdownChunker)


# =============================================================================
# Integration Tests
# =============================================================================


class TestChunkingIntegration:
    """Integration tests for the chunking system."""

    def test_chunker_registry_produces_valid_chunks(self, sample_documents):
        """Test that registry-selected chunkers produce valid chunks."""
        registry = ChunkerRegistry()

        for format_name, doc_path in sample_documents.items():
            if format_name == "python":
                # Skip PDF for now as it needs the actual file
                chunker = registry.get_chunker(doc_path)
                content = doc_path.read_text()
                chunks = chunker.chunk(content, str(doc_path))

                assert len(chunks) > 0
                for chunk in chunks:
                    assert chunk.text.strip()
                    assert chunk.source_file == str(doc_path)
                    assert chunk.chunk_index >= 0
                    # Verify start_line and end_line are set
                    assert chunk.start_line >= 0
                    assert chunk.end_line >= chunk.start_line

    def test_markdown_chunker_real_world(self):
        """Test markdown chunker with realistic content."""
        chunker = MarkdownChunker(chunk_size=500)
        content = """# User Guide

Welcome to our application.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

Run pip install our-package.

## Usage

Import and use the package.
"""
        chunks = chunker.chunk(content, "/docs/guide.md")

        assert len(chunks) >= 2
        # Verify hierarchy is preserved
        header_paths = [c.metadata["header_path"] for c in chunks]
        assert "User Guide" in header_paths[0]

    def test_code_chunker_real_world(self):
        """Test code chunker with realistic content."""
        chunker = CodeChunker()
        content = '''"""Sample module."""

import os
from typing import Optional


def main():
    """Entry point."""
    print("Hello")


class Handler:
    """Request handler."""

    def __init__(self):
        self.count = 0

    def handle(self):
        self.count += 1
'''
        chunks = chunker.chunk(content, "app.py")

        # Should have imports, function, and class
        types = [c.metadata.get("type") for c in chunks]
        assert "imports" in types
        assert "function" in types
        assert "class" in types
