"""Page-aware PDF chunker using pypdf for text extraction."""

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from theo.chunking.base import AbstractChunker, Chunk

logger = logging.getLogger(__name__)


class PDFChunker(AbstractChunker):
    """Page-aware PDF chunker with pypdf text extraction.

    Extracts text from PDF files page by page, then chunks the content
    while maintaining page metadata. Handles multi-page documents and
    allows chunks to span page boundaries with proper overlap.

    Attributes:
        chunk_size: Maximum character length for a chunk
        chunk_overlap: Number of characters to overlap between chunks
        max_tokens: Maximum tokens per chunk (enforced via split_oversized_chunk)
    """

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, max_tokens: int = 256
    ):
        """Initialize PDF chunker.

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
        """Split PDF content into chunks.

        For PDF files, this method expects the source_file to be a path
        to the PDF file. The content parameter is ignored as we extract
        text directly from the PDF file.

        Strategy:
        1. Extract text from each page using pypdf
        2. Track page numbers for each text segment
        3. Concatenate pages and chunk with overlap
        4. Store page metadata in each chunk

        Args:
            content: Ignored for PDF files
            source_file: Path to the PDF file

        Returns:
            List of Chunk objects with sequential indexing and page metadata

        Raises:
            ValueError: If file doesn't exist, is not a PDF, or has no text
            PdfReadError: If PDF is corrupted or cannot be read
        """
        file_path = Path(source_file)

        if not file_path.exists():
            raise ValueError(f"PDF file does not exist: {source_file}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {source_file}")

        # Extract text from PDF
        try:
            page_texts = self._extract_from_file(file_path)
        except PdfReadError as e:
            logger.error(f"Failed to read PDF {source_file}: {e}")
            raise ValueError(f"Corrupted or invalid PDF file: {source_file}") from e

        if not page_texts:
            logger.warning(f"PDF has no readable text: {source_file}")
            raise ValueError(f"PDF contains no extractable text: {source_file}")

        # Create chunks from extracted pages
        chunks = self._create_page_chunks(page_texts, source_file)

        # Post-process: split any oversized chunks and rebuild indices
        return self._split_oversized_chunks(chunks)

    def _extract_from_file(self, file_path: Path) -> list[tuple[str, int]]:
        """Extract text from PDF file page by page.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of tuples (page_text, page_number)
            Page numbers are 1-indexed for user reference

        Raises:
            PdfReadError: If PDF cannot be read
        """
        page_texts: list[tuple[str, int]] = []

        try:
            reader = PdfReader(str(file_path))

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()

                    # Only include pages with actual text content
                    if text and text.strip():
                        page_texts.append((text.strip(), page_num))
                    else:
                        logger.debug(f"Page {page_num} has no text content")

                except Exception as e:
                    # Log page-level extraction errors but continue
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

        except PdfReadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading PDF: {e}")
            raise PdfReadError(f"Failed to read PDF: {e}") from e

        return page_texts

    def _create_page_chunks(
        self, page_texts: list[tuple[str, int]], source_file: str
    ) -> list[Chunk]:
        """Create chunks from extracted page texts.

        Chunks can span multiple pages. Each chunk stores the page range
        it covers in metadata.

        Args:
            page_texts: List of (text, page_number) tuples
            source_file: Path to source PDF file

        Returns:
            List of Chunk objects with page metadata
        """
        chunks: list[Chunk] = []
        current_chunk_text = ""
        current_pages: list[int] = []
        previous_text = ""

        for page_text, page_num in page_texts:
            # Add overlap from previous page if configured
            if previous_text and self.chunk_overlap > 0 and not current_chunk_text:
                # Get last N characters from previous text for overlap
                overlap_text = previous_text[-self.chunk_overlap :].lstrip()
                if overlap_text:
                    current_chunk_text = overlap_text
                    # Note: overlap pages are not added to current_pages

            # Process this page's text
            remaining_text = page_text

            while remaining_text:
                # Calculate space available in current chunk
                space_available = self.chunk_size - len(current_chunk_text)

                if space_available <= 0:
                    # Current chunk is full, save it
                    if current_chunk_text.strip():
                        chunks.append(
                            self._create_chunk(
                                current_chunk_text.strip(),
                                source_file,
                                len(chunks),
                                current_pages,
                            )
                        )

                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        current_chunk_text = current_chunk_text[-self.chunk_overlap :].lstrip()
                    else:
                        current_chunk_text = ""
                    current_pages = []
                    continue

                # Add text to current chunk
                if len(remaining_text) <= space_available:
                    # All remaining text fits
                    if current_chunk_text and not current_chunk_text.endswith(" "):
                        current_chunk_text += " "
                    current_chunk_text += remaining_text
                    if page_num not in current_pages:
                        current_pages.append(page_num)
                    remaining_text = ""
                else:
                    # Take what we can fit
                    text_to_add = remaining_text[:space_available]

                    # Try to break at sentence boundary
                    last_period = max(
                        text_to_add.rfind(". "),
                        text_to_add.rfind("! "),
                        text_to_add.rfind("? "),
                    )

                    if last_period > space_available // 2:
                        # Good break point found
                        text_to_add = text_to_add[: last_period + 1]

                    if current_chunk_text and not current_chunk_text.endswith(" "):
                        current_chunk_text += " "
                    current_chunk_text += text_to_add
                    if page_num not in current_pages:
                        current_pages.append(page_num)
                    remaining_text = remaining_text[len(text_to_add) :].lstrip()

            previous_text = page_text

        # Save final chunk
        if current_chunk_text.strip():
            chunks.append(
                self._create_chunk(
                    current_chunk_text.strip(), source_file, len(chunks), current_pages
                )
            )

        return chunks

    def _create_chunk(
        self, content: str, source_file: str, chunk_index: int, pages: list[int]
    ) -> Chunk:
        """Create a Chunk with page metadata.

        Args:
            content: The chunk text content
            source_file: Path to source file
            chunk_index: Index of this chunk
            pages: List of page numbers this chunk spans

        Returns:
            Chunk object with page metadata
        """
        metadata: dict[str, int | str] = {}

        if pages:
            if len(pages) == 1:
                metadata["page_number"] = pages[0]
            else:
                metadata["page_start"] = pages[0]
                metadata["page_end"] = pages[-1]
                # Convert pages list to comma-separated string for storage
                metadata["pages"] = ",".join(str(p) for p in pages)

        # For PDFs, use page numbers as logical line references
        # start_line/end_line refer to the first and last page numbers
        start_line = pages[0] if pages else 0
        end_line = pages[-1] if pages else 0

        return Chunk(
            text=content,
            metadata=metadata,
            start_line=start_line,
            end_line=end_line,
            source_file=source_file,
            chunk_index=chunk_index,
        )

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
