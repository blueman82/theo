"""Paragraph-based text chunker for plain text files."""

import re
from typing import Optional

from theo.chunking.base import AbstractChunker, Chunk


class TextChunker(AbstractChunker):
    """Simple paragraph-based text chunker.

    Splits text on blank lines (paragraph boundaries) and further splits
    large paragraphs by sentences to maintain chunk size constraints.

    Attributes:
        chunk_size: Maximum character length for a chunk
        chunk_overlap: Number of characters to overlap between chunks
        max_tokens: Maximum tokens per chunk (enforced via split_oversized_chunk)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, max_tokens: int = 256):
        """Initialize text chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
            max_tokens: Maximum tokens per chunk (uses 4 chars ≈ 1 token approximation)

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0 or max_tokens <= 0
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split content into chunks.

        Strategy:
        1. Split by paragraph boundaries (blank lines)
        2. For paragraphs exceeding chunk_size, split by sentences
        3. Track line numbers in metadata
        4. Apply overlap strategy between chunks

        Args:
            content: The document content to chunk
            source_file: Path to the source file

        Returns:
            List of Chunk objects with sequential indexing

        Raises:
            ValueError: If content is empty or invalid
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        chunks: list[Chunk] = []
        paragraphs = self._split_paragraphs(content)

        # Track line numbers
        current_line = 0
        previous_sentences: list[str] = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                current_line += 1
                continue

            # Count lines in this paragraph
            paragraph_lines = paragraph.count("\n") + 1

            # Chunk this paragraph
            paragraph_chunks = self._chunk_paragraph(paragraph, current_line, previous_sentences)

            # Add chunks with proper indexing
            for para_chunk in paragraph_chunks:
                chunks.append(
                    Chunk(
                        text=para_chunk["content"],
                        metadata=para_chunk["metadata"],
                        start_line=para_chunk["metadata"]["start_line"],
                        end_line=para_chunk["metadata"]["end_line"],
                        source_file=source_file,
                        chunk_index=len(chunks),
                    )
                )

            # Update previous sentences for overlap
            if paragraph_chunks:
                last_content = paragraph_chunks[-1]["content"]
                previous_sentences = (
                    self._split_sentences(last_content)[-2:] if last_content else []
                )

            current_line += paragraph_lines

        # Post-process: split any oversized chunks and rebuild indices
        return self._split_oversized_chunks(chunks)

    def _split_paragraphs(self, content: str) -> list[str]:
        """Split content by blank lines (paragraph boundaries).

        Args:
            content: Text content to split

        Returns:
            List of paragraph strings
        """
        # Split on one or more blank lines
        paragraphs = re.split(r"\n\s*\n", content)
        return [p for p in paragraphs if p.strip()]

    def _chunk_paragraph(
        self,
        paragraph: str,
        start_line: int,
        previous_sentences: Optional[list[str]] = None,
    ) -> list[dict]:
        """Chunk a single paragraph, splitting by sentences if needed.

        Args:
            paragraph: The paragraph text
            start_line: Starting line number
            previous_sentences: Sentences from previous chunk for overlap

        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        previous_sentences = previous_sentences or []

        # If paragraph fits in chunk_size, return as-is
        if len(paragraph) <= self.chunk_size:
            return [
                {
                    "content": paragraph,
                    "metadata": {
                        "start_line": start_line,
                        "end_line": start_line + paragraph.count("\n"),
                    },
                }
            ]

        # Split by sentences
        sentences = self._split_sentences(paragraph)
        chunks = []
        current_chunk: list[str] = []
        current_length = 0
        chunk_start_line = start_line

        # Add overlap from previous paragraph
        if previous_sentences and self.chunk_overlap > 0:
            overlap_text = " ".join(previous_sentences)
            if len(overlap_text) <= self.chunk_overlap:
                current_chunk.extend(previous_sentences)
                current_length = len(overlap_text) + 1  # +1 for space

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, start new chunk
            if current_chunk and current_length + sentence_len > self.chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                line_count = chunk_text.count("\n")
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "start_line": chunk_start_line,
                            "end_line": chunk_start_line + line_count,
                        },
                    }
                )

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) >= 2:
                    # Include last 2 sentences for overlap
                    overlap_sentences = current_chunk[-2:]
                    overlap_text = " ".join(overlap_sentences)

                    if len(overlap_text) <= self.chunk_overlap:
                        current_chunk = overlap_sentences.copy()
                        current_length = len(overlap_text) + 1
                    else:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[0]) + 1
                else:
                    current_chunk = []
                    current_length = 0

                chunk_start_line = chunk_start_line + line_count

            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space between sentences

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            line_count = chunk_text.count("\n")
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "start_line": chunk_start_line,
                        "end_line": chunk_start_line + line_count,
                    },
                }
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentence strings
        """
        # Split on sentence boundaries (., !, ?)
        # Keep the punctuation with the sentence
        sentences = re.split(r"([.!?]+(?:\s|$))", text)

        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            combined = (sentence + punctuation).strip()
            if combined:
                result.append(combined)

        # Handle last sentence if no punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result

    def _split_oversized_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Split any chunks exceeding max_tokens and rebuild indices.

        Args:
            chunks: Original list of chunks

        Returns:
            New list with oversized chunks split, indices rebuilt
        """
        result: list[Chunk] = []
        for chunk in chunks:
            split_chunks = self.split_oversized_chunk(
                chunk, self.max_tokens, base_index=len(result)
            )
            result.extend(split_chunks)
        return result
