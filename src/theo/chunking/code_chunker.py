"""AST-based code chunker for Python files."""

import ast
import logging
from typing import Optional

from theo.chunking.base import AbstractChunker, Chunk

logger = logging.getLogger(__name__)


class CodeChunker(AbstractChunker):
    """AST-based chunker that preserves function and class boundaries.

    Uses Python's ast module to parse code structure and chunk by semantic
    units (functions, classes, methods). Falls back to line-based chunking
    for unparseable code. Oversized chunks are split using the base class
    utility, with classes attempting method-level splitting first.
    """

    def __init__(self, chunk_size: int = 100, max_tokens: int = 512):
        """Initialize code chunker.

        Args:
            chunk_size: Maximum number of lines per chunk for fallback chunking
            max_tokens: Maximum tokens per chunk (default 512 for mxbai-embed-large)
        """
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split Python code into semantic chunks.

        Args:
            content: Python source code to chunk
            source_file: Path to source file for provenance tracking

        Returns:
            List of Chunk objects with sequential indexing

        Raises:
            ValueError: If content is empty or invalid
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Try to parse with AST
        ast_module = self._parse_ast(content)

        if ast_module is not None:
            # Successfully parsed, use AST-based chunking
            return self._chunk_by_definitions(content, source_file, ast_module)
        else:
            # Parsing failed, fall back to line-based chunking
            logger.warning(f"Failed to parse {source_file} as Python, using line-based chunking")
            return self._fallback_line_chunking(content, source_file)

    def _parse_ast(self, content: str) -> Optional[ast.Module]:
        """Parse Python code into AST.

        Args:
            content: Python source code

        Returns:
            AST Module if parsing succeeds, None otherwise
        """
        try:
            return ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error during AST parsing: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error during AST parsing: {e}")
            return None

    def _extract_definitions(
        self, ast_module: ast.Module
    ) -> list[tuple[str, int, int, Optional[str]]]:
        """Extract top-level function and class definitions.

        Args:
            ast_module: Parsed AST module

        Returns:
            List of tuples (definition_type, start_line, end_line, class_name)
            where definition_type is 'function', 'class', or 'async_function'
            and class_name is set for class definitions
        """
        definitions: list[tuple[str, int, int, str | None]] = []

        for node in ast_module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line, end_line = self._get_node_lines(node)
                def_type = (
                    "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                )
                definitions.append((def_type, start_line, end_line, None))

            elif isinstance(node, ast.ClassDef):
                start_line, end_line = self._get_node_lines(node)
                definitions.append(("class", start_line, end_line, node.name))

        return definitions

    def _get_node_lines(self, node: ast.AST) -> tuple[int, int]:
        """Get start and end line numbers for an AST node.

        For function and class definitions, includes decorators if present.

        Args:
            node: AST node

        Returns:
            Tuple of (start_line, end_line) (1-indexed)
        """
        start_line = node.lineno if hasattr(node, "lineno") else 1
        end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line

        # Include decorators for functions and classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "decorator_list") and node.decorator_list:
                # Get the first decorator's line number
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, "lineno"):
                    start_line = first_decorator.lineno

        return start_line, end_line

    def _chunk_by_definitions(
        self, content: str, source_file: str, ast_module: ast.Module
    ) -> list[Chunk]:
        """Chunk code by AST definitions.

        Args:
            content: Python source code
            source_file: Path to source file
            ast_module: Parsed AST module

        Returns:
            List of chunks based on code structure
        """
        lines = content.splitlines(keepends=False)
        definitions = self._extract_definitions(ast_module)
        chunks = []
        chunk_index = 0

        # Extract module-level imports and docstring
        imports = []
        module_docstring = None
        import_start_line = 1
        import_end_line = 1

        for node in ast_module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start, end = self._get_node_lines(node)
                import_lines = lines[start - 1 : end]
                imports.extend(import_lines)
                if import_end_line < end:
                    import_end_line = end
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                # Module docstring
                if node.lineno == 1 or (node.lineno == 2 and lines[0].startswith("#")):
                    module_docstring = node.value.value

        # Create first chunk with imports and module docstring if present
        if imports or module_docstring:
            import_content_parts = []
            if module_docstring:
                docstring_str = (
                    module_docstring.decode()
                    if isinstance(module_docstring, bytes)
                    else str(module_docstring)
                )
                import_content_parts.append(f'"""{docstring_str}"""')
            if imports:
                import_content_parts.extend(imports)

            import_content = "\n".join(import_content_parts)
            if import_content.strip():
                chunks.append(
                    Chunk(
                        text=import_content,
                        metadata={
                            "type": "imports",
                            "has_docstring": module_docstring is not None,
                        },
                        start_line=import_start_line,
                        end_line=import_end_line,
                        source_file=source_file,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        # Chunk each definition
        for def_type, start_line, end_line, class_name in definitions:
            # Extract definition lines (AST uses 1-based indexing)
            definition_lines = lines[start_line - 1 : end_line]
            definition_content = "\n".join(definition_lines)

            if definition_content.strip():
                metadata = {"type": def_type}
                if class_name:
                    metadata["class_name"] = class_name

                chunk = Chunk(
                    text=definition_content,
                    metadata=metadata,
                    start_line=start_line,
                    end_line=end_line,
                    source_file=source_file,
                    chunk_index=chunk_index,
                )

                # Check if chunk exceeds max_tokens and needs splitting
                max_chars = self.max_tokens * 4
                if len(definition_content) > max_chars:
                    # For classes, try method-level splitting first
                    if def_type == "class" and class_name:
                        split_chunks = self._split_class_by_methods(chunk, class_name, chunk_index)
                    else:
                        # For functions/other, use base class line-based splitting
                        split_chunks = self.split_oversized_chunk(
                            chunk, self.max_tokens, chunk_index
                        )

                    chunks.extend(split_chunks)
                    chunk_index += len(split_chunks)
                else:
                    chunks.append(chunk)
                    chunk_index += 1

        # If no chunks were created (empty file or only comments), create one chunk
        if not chunks:
            chunks.append(
                Chunk(
                    text=content,
                    metadata={"type": "other"},
                    start_line=1,
                    end_line=len(lines),
                    source_file=source_file,
                    chunk_index=0,
                )
            )

        return chunks

    def _split_class_by_methods(
        self, chunk: Chunk, class_name: str, base_index: int
    ) -> list[Chunk]:
        """Split an oversized class chunk by its methods.

        Attempts to parse the class body and split by method boundaries.
        Falls back to base class line-based splitting if method extraction fails.

        Args:
            chunk: The class chunk to split
            class_name: Name of the class
            base_index: Starting index for resulting chunks

        Returns:
            List of chunks from the split class
        """
        text = chunk.text
        max_chars = self.max_tokens * 4

        # Try to parse the class body to extract methods
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # If parsing fails, fall back to base class splitting
            return self.split_oversized_chunk(chunk, self.max_tokens, base_index)

        if not tree.body or not isinstance(tree.body[0], ast.ClassDef):
            return self.split_oversized_chunk(chunk, self.max_tokens, base_index)

        class_node = tree.body[0]
        lines = text.splitlines(keepends=False)

        # Extract class header (class definition line + docstring if present)
        class_header_end = class_node.lineno
        first_method_line = None

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                if hasattr(node, "decorator_list") and node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, "lineno"):
                        start_line = first_decorator.lineno
                if first_method_line is None or start_line < first_method_line:
                    first_method_line = start_line

        # If we have methods, the header ends just before the first method
        if first_method_line:
            class_header_end = first_method_line - 1

        # Extract methods with their line ranges
        method_chunks: list[tuple[str, int, int]] = []
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = node.end_lineno if node.end_lineno is not None else node.lineno

                # Include decorators
                if hasattr(node, "decorator_list") and node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, "lineno"):
                        start_line = first_decorator.lineno

                method_content = "\n".join(lines[start_line - 1 : end_line])
                method_chunks.append((method_content, start_line, end_line))

        # If no methods found or only one method, use base class splitting
        if len(method_chunks) <= 1:
            return self.split_oversized_chunk(chunk, self.max_tokens, base_index)

        # Build result chunks by grouping methods that fit together
        result_chunks: list[Chunk] = []
        current_text = ""
        part_index = 0

        # Include class header in first chunk
        header_lines = lines[:class_header_end]
        header_content = "\n".join(header_lines).strip()
        if header_content:
            current_text = header_content

        for method_content, _, _ in method_chunks:
            method_content = method_content.strip()
            if not method_content:
                continue

            # Check if adding this method would exceed the limit
            if current_text:
                combined = current_text + "\n\n" + method_content
            else:
                combined = method_content

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

                # Handle the method that didn't fit
                if len(method_content) <= max_chars:
                    current_text = method_content
                else:
                    # Method itself is too large, use base class splitting
                    temp_chunk = Chunk(
                        text=method_content,
                        metadata=dict(chunk.metadata),
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        source_file=chunk.source_file,
                        chunk_index=0,
                    )
                    method_splits = self.split_oversized_chunk(
                        temp_chunk, self.max_tokens, base_index + part_index
                    )
                    result_chunks.extend(method_splits)
                    part_index += len(method_splits)
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

        # If method-based splitting produced no better result, fall back
        if not result_chunks:
            return self.split_oversized_chunk(chunk, self.max_tokens, base_index)

        return result_chunks

    def _fallback_line_chunking(self, content: str, source_file: str) -> list[Chunk]:
        """Fall back to simple line-based chunking.

        Args:
            content: Source code content
            source_file: Path to source file

        Returns:
            List of chunks split by line count
        """
        lines = content.splitlines(keepends=False)
        chunks = []
        chunk_index = 0

        for i in range(0, len(lines), self.chunk_size):
            chunk_lines = lines[i : i + self.chunk_size]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():
                start_line = i + 1
                end_line = min(i + self.chunk_size, len(lines))
                chunks.append(
                    Chunk(
                        text=chunk_content,
                        metadata={"type": "fallback"},
                        start_line=start_line,
                        end_line=end_line,
                        source_file=source_file,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        # If no chunks were created, create one with all content
        if not chunks:
            chunks.append(
                Chunk(
                    text=content,
                    metadata={"type": "fallback"},
                    start_line=1,
                    end_line=len(lines) if lines else 1,
                    source_file=source_file,
                    chunk_index=0,
                )
            )

        return chunks
