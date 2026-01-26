"""Base chunking interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Chunk:
    """Immutable chunk of document content with metadata.

    Attributes:
        text: The actual text content of the chunk
        metadata: Additional metadata about the chunk
        start_line: 1-indexed line number where chunk begins in source file
        end_line: 1-indexed line number where chunk ends in source file
        source_file: Path to the original source file
        chunk_index: Zero-based index of this chunk in the document
        token_count: Optional token count (computed during embedding prep)
    """

    text: str
    metadata: dict[str, Any]
    start_line: int
    end_line: int
    source_file: str = ""
    chunk_index: int = 0
    token_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate chunk content after initialization."""
        self.validate_content()

    def validate_content(self) -> None:
        """Validate that chunk content meets quality constraints.

        Raises:
            ValueError: If text is empty or invalid
        """
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")

        if self.chunk_index < 0:
            raise ValueError(f"Chunk index must be non-negative, got {self.chunk_index}")

        if self.start_line < 0:
            raise ValueError(f"start_line must be non-negative, got {self.start_line}")

        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )


class AbstractChunker(ABC):
    """Abstract base class for document chunking strategies.

    Concrete implementations must provide a chunking strategy that splits
    documents into semantically meaningful pieces suitable for embedding.
    """

    @abstractmethod
    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split content into chunks.

        Args:
            content: The document content to chunk
            source_file: Path to the source file for provenance tracking

        Returns:
            List of Chunk objects with sequential indexing

        Raises:
            ValueError: If content is empty or invalid
        """
        pass

    def validate_chunk_quality(self, chunk: Chunk) -> bool:
        """Validate that a chunk meets quality standards.

        Args:
            chunk: The chunk to validate

        Returns:
            True if chunk passes quality checks, False otherwise
        """
        try:
            chunk.validate_content()
            return True
        except ValueError:
            return False

    def split_oversized_chunk(
        self, chunk: Chunk, max_tokens: int, base_index: int = 0
    ) -> list[Chunk]:
        """Split an oversized chunk into smaller chunks that fit within token limits.

        Uses character-based estimation (chars/4 ≈ tokens) consistent with
        existing codebase patterns. Splits at paragraph boundaries first,
        then falls back to line boundaries if needed.

        Args:
            chunk: The chunk to potentially split
            max_tokens: Maximum tokens per chunk
            base_index: Starting index for the resulting chunks

        Returns:
            List of Chunk objects. If the original chunk is within limits,
            returns a single-element list with the chunk (with updated index).
            If split, all chunks have "split_part" added to metadata.
        """
        # Estimate tokens using chars/4 approximation
        max_chars = max_tokens * 4
        text = chunk.text

        # If within limit, return unchanged (with base_index applied)
        if len(text) <= max_chars:
            return [
                Chunk(
                    text=text,
                    metadata=dict(chunk.metadata),
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    source_file=chunk.source_file,
                    chunk_index=base_index,
                    token_count=chunk.token_count,
                )
            ]

        # Split at paragraph boundaries first (double newline)
        paragraphs = text.split("\n\n")

        # If we only got one paragraph, fall back to line boundaries
        if len(paragraphs) == 1:
            paragraphs = text.split("\n")

        # Build chunks from segments, combining small ones and splitting large ones
        result_chunks: list[Chunk] = []
        current_text = ""
        part_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph would exceed the limit
            if current_text:
                combined = current_text + "\n\n" + para
            else:
                combined = para

            if len(combined) <= max_chars:
                current_text = combined
            else:
                # Flush current text if we have any
                if current_text:
                    new_metadata = dict(chunk.metadata)
                    new_metadata["split_part"] = part_index
                    result_chunks.append(
                        Chunk(
                            text=current_text,
                            metadata=new_metadata,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            source_file=chunk.source_file,
                            chunk_index=base_index + part_index,
                        )
                    )
                    part_index += 1

                # Handle the paragraph that didn't fit
                if len(para) <= max_chars:
                    current_text = para
                else:
                    # Paragraph itself is too large, split by words
                    word_chunks = self._split_by_words(para, max_chars)
                    for word_chunk in word_chunks:
                        new_metadata = dict(chunk.metadata)
                        new_metadata["split_part"] = part_index
                        result_chunks.append(
                            Chunk(
                                text=word_chunk,
                                metadata=new_metadata,
                                start_line=chunk.start_line,
                                end_line=chunk.end_line,
                                source_file=chunk.source_file,
                                chunk_index=base_index + part_index,
                            )
                        )
                        part_index += 1
                    current_text = ""

        # Flush remaining text
        if current_text:
            new_metadata = dict(chunk.metadata)
            new_metadata["split_part"] = part_index
            result_chunks.append(
                Chunk(
                    text=current_text,
                    metadata=new_metadata,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    source_file=chunk.source_file,
                    chunk_index=base_index + part_index,
                )
            )

        return result_chunks

    def _split_by_words(self, text: str, max_chars: int) -> list[str]:
        """Split text by word boundaries to fit within character limit.

        Args:
            text: Text to split
            max_chars: Maximum characters per segment

        Returns:
            List of text segments
        """
        words = text.split()
        segments: list[str] = []
        current_segment = ""

        for word in words:
            if current_segment:
                test_segment = current_segment + " " + word
            else:
                test_segment = word

            if len(test_segment) <= max_chars:
                current_segment = test_segment
            else:
                if current_segment:
                    segments.append(current_segment)
                # Handle words longer than max_chars by truncating
                if len(word) > max_chars:
                    # Split long word at max_chars boundaries
                    for i in range(0, len(word), max_chars):
                        segments.append(word[i : i + max_chars])
                    current_segment = ""
                else:
                    current_segment = word

        if current_segment:
            segments.append(current_segment)

        return segments
