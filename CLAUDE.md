# CLAUDE.md
Last updated: 2026-01-26

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Theo** - A unified AI memory and document retrieval system, merging the capabilities of DocVec (semantic document indexing) and Recall (long-term memory) into a single MCP server.

### Core Capabilities

1. **Document Indexing** (from DocVec): Semantic document chunking and retrieval using vector embeddings
2. **Long-term Memory** (from Recall): Persistent memory storage with validation loops and confidence scoring
3. **Unified MCP Interface**: Single server exposing both capabilities to MCP-compatible clients

### Design Principles

- **Local-first processing**: MLX embeddings on Apple Silicon, no external API calls
- **Privacy-preserving**: All data stays local
- **Token-efficient**: Smart chunking and retrieval to minimize context usage
- **Validation-driven**: Memories build confidence through practical use

## Validation Loop (ELF-Inspired)

Theo implements a validation loop to build confidence in memories through practical use:

```
TRY → BREAK → ANALYZE → LEARN
 ↑                        ↓
 └────────────────────────┘
```

## Development Commands

### Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/theo --cov-report=html

# Run specific test file
uv run pytest tests/test_indexer.py -v

# Run single test function
uv run pytest tests/test_indexer.py::test_index_document -v
```

### Running the MCP Server

```bash
# Run with default configuration (MLX backend)
uv run python -m theo

# View available CLI options
uv run python -m theo --help
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Sort imports
uv run isort src/ tests/
```

## Architecture

### Component Hierarchy

```
src/theo/
├── __init__.py          # Package entry point with __version__
├── __main__.py          # MCP server entry point (planned)
├── config.py            # Pydantic Settings configuration (planned)
├── indexing/            # Document indexing (from DocVec)
│   ├── indexer.py       # Core document indexer
│   ├── batch_processor.py
│   └── chunking/        # Format-specific chunkers
│       ├── base.py
│       ├── markdown_chunker.py
│       ├── pdf_chunker.py
│       ├── code_chunker.py
│       └── text_chunker.py
├── memory/              # Long-term memory (from Recall)
│   ├── types.py         # Memory, Edge, MemoryType, RelationType
│   ├── operations.py    # store, recall, relate, forget, validate
│   └── validation.py    # Contradiction detection, confidence
├── storage/
│   ├── chroma_store.py  # ChromaDB vector storage
│   ├── sqlite.py        # SQLite metadata (memory system)
│   └── hybrid.py        # Coordinated storage layer
└── embedding/
    ├── provider.py      # Abstract embedding provider
    ├── factory.py       # Backend selection
    ├── mlx_provider.py  # MLX embeddings (Apple Silicon)
    └── ollama.py        # Ollama fallback
```

### Key Data Flows

**Document Indexing Pipeline:**
1. File path → Indexer validates and reads content
2. File extension → Appropriate chunker selected
3. Content → Chunker splits into semantically meaningful chunks
4. Chunks → Embedded via MLX/Ollama
5. Chunks + Embeddings → Stored in ChromaDB

**Memory Operations:**
1. Memory content → Embedded and deduplicated
2. Storage → SQLite (metadata) + ChromaDB (vectors)
3. Recall → Semantic search with optional graph expansion
4. Validation → Confidence adjusted through use

### Important Design Patterns

**Dependency Injection**: All components receive dependencies via constructor
- Makes testing easier (mock dependencies)
- Makes initialization order explicit

**Format-Specific Chunking**: File extension determines chunker strategy
- `.md` → MarkdownChunker (preserves header hierarchy)
- `.pdf` → PDFChunker (preserves page boundaries)
- `.py` → CodeChunker (uses AST for function/class boundaries)
- `.txt` or unknown → TextChunker (paragraph-based)

**Validation Loop** (for memories):
```
TRY → BREAK → ANALYZE → LEARN
 ↑                        ↓
 └────────────────────────┘
```
- Memories start at confidence 0.3
- Success increases confidence (+0.1), failure decreases (-0.15)
- Memories at confidence >= 0.9 become "golden rules"

## Critical Implementation Details

### MCP Server stdio Protocol
- **NEVER write to stdout** - it corrupts JSON-RPC messages
- All logging MUST go to stderr
- This is a hard requirement for stdio-based MCP servers

### Embedding Backends
- **MLX** (default on Apple Silicon): Fast local embeddings (~5-10x faster than Ollama)
- **Ollama** (fallback): Requires running Ollama server

### Testing Philosophy
- **Unit tests**: Test each component in isolation with mocks
- **Integration tests**: Test full pipelines
- **Fixtures**: Use pytest fixtures for shared test setup
  - `mock_embedder`: Returns fake embeddings without MLX/Ollama
  - `temp_chroma`: In-memory ChromaDB for isolated tests
  - `temp_dir`: Temporary directories for file operations

Example pattern:
```python
def test_something(mock_embedder, temp_chroma):
    indexer = Indexer(mock_embedder, temp_chroma)
    # Test without external dependencies
```

## Configuration

Settings via environment variables with `THEO_` prefix (planned):

| Variable | Default | Description |
|----------|---------|-------------|
| THEO_EMBEDDING_BACKEND | mlx | Embedding backend: `mlx` or `ollama` |
| THEO_MLX_MODEL | mlx-community/mxbai-embed-large-v1 | MLX model |
| THEO_CHROMA_PATH | ~/.theo/chroma_db | ChromaDB storage |
| THEO_SQLITE_PATH | ~/.theo/theo.db | SQLite database |
| THEO_LOG_LEVEL | INFO | Logging level |

## File Paths and Structure

- Always use `pathlib.Path` objects, not strings
- Resolve paths with `.resolve()` for absolute paths
- Use `Path.exists()`, `Path.is_file()`, `Path.is_dir()` for validation

## Source Repositories

Theo merges code from:
- **DocVec**: `/Users/harrison/Documents/Github/docvec` - Document indexing
- **Recall**: `/Users/harrison/Documents/Github/recall` - Long-term memory
