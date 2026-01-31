# CLAUDE.md
Last updated: 2026-01-31

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Theo** - A unified AI memory and document retrieval system, merging the capabilities of DocVec (semantic document indexing) and Recall (long-term memory) into a single MCP server.

### Core Capabilities

1. **Document Indexing** (from DocVec): Semantic document chunking and retrieval using vector embeddings
2. **Long-term Memory** (from Recall): Persistent memory storage with validation loops and confidence scoring
3. **Unified MCP Interface**: Single server exposing both capabilities to MCP-compatible clients
4. **Voice Transcription**: TUI for voice-to-text with batch mode support

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

- Memories start at confidence 0.3
- Success increases confidence (+0.1), failure decreases (-0.15)
- Memories at confidence >= 0.9 become "golden rules"

## Development Commands

### Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev

# Copy and configure environment variables (ALL are required)
cp .env.example .env
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/theo --cov-report=html

# Run specific test file
uv run pytest tests/test_chunkers.py -v

# Run single test function
uv run pytest tests/test_chunkers.py::test_markdown_chunker -v
```

### Running the MCP Server

```bash
# Run with default configuration (MLX backend)
uv run python -m theo

# View available CLI options
uv run python -m theo --help

# Run the transcription TUI
uv run python -m theo.transcription
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
├── __init__.py              # Package entry point with __version__
├── __main__.py              # MCP server entry point
├── config.py                # Pydantic Settings configuration
├── constants.py             # Application constants
├── mcp_server.py            # MCP server implementation
│
├── chunking/                # Document chunking (format-specific)
│   ├── base.py              # Abstract base chunker
│   ├── registry.py          # Chunker registry and selection
│   ├── markdown_chunker.py  # Markdown (preserves header hierarchy)
│   ├── pdf_chunker.py       # PDF (preserves page boundaries)
│   ├── code_chunker.py      # Python (uses AST for function/class boundaries)
│   └── text_chunker.py      # Plain text (paragraph-based)
│
├── daemon/                  # Background processing daemon
│   ├── client.py            # Daemon client interface
│   ├── server.py            # Daemon server
│   ├── protocol.py          # Communication protocol
│   └── worker.py            # Worker processes
│
├── embedding/               # Embedding providers
│   ├── provider.py          # Abstract embedding provider protocol
│   ├── factory.py           # Backend selection factory
│   ├── mlx_provider.py      # MLX embeddings (Apple Silicon)
│   └── ollama_provider.py   # Ollama fallback
│
├── storage/                 # Data persistence
│   ├── sqlite_store.py      # SQLite + sqlite-vec vector storage
│   ├── types.py             # Storage type definitions
│   └── hybrid.py            # Coordinated storage layer
│
├── tools/                   # MCP tool implementations
│   ├── indexing_tools.py    # index_file, index_directory
│   ├── query_tools.py       # search, search_with_filters, search_with_budget
│   ├── memory_tools.py      # memory_store, memory_recall, memory_validate
│   └── management_tools.py  # delete_chunks, delete_file, clear_index, get_stats
│
├── transcription/           # Voice transcription module
│   ├── __main__.py          # TUI entry point
│   ├── tui.py               # Terminal user interface
│   ├── audio.py             # Audio capture and processing
│   ├── transcriber.py       # Speech-to-text conversion
│   ├── storage.py           # Transcription persistence
│   └── types.py             # AudioChunk, TranscriptionSegment, TranscriptionSession
│
├── types/                   # Core type definitions
│   ├── document.py          # Document and chunk types
│   ├── memory.py            # Memory, Edge, MemoryType, RelationType
│   └── search_result.py     # Search result types
│
└── validation/              # Memory validation system
    ├── loop.py              # Validation loop implementation
    ├── feedback.py          # Feedback processing
    └── golden_rules.py      # High-confidence memory rules
```

### Key Data Flows

**Document Indexing Pipeline:**
1. File path → Chunker registry selects appropriate chunker by extension
2. Content → Chunker splits into semantically meaningful chunks
3. Chunks → Embedded via MLX/Ollama (daemon handles batch processing)
4. Chunks + Embeddings → Stored in SQLite (sqlite-vec)

**Memory Operations:**
1. Memory content → Embedded and deduplicated
2. Storage → SQLite with sqlite-vec (unified storage)
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

**Tool Organization**: MCP tools are organized by functional category
- `IndexingTools`: Document indexing operations
- `QueryTools`: Semantic search operations
- `MemoryTools`: Memory CRUD and validation
- `ManagementTools`: Collection and index management

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
- **Integration tests**: Test full pipelines (in `tests/integration/`)
- **Fixtures**: Use pytest fixtures for shared test setup
  - `mock_embedder`: Returns fake embeddings without MLX/Ollama
  - `temp_dir`: Temporary directories for file operations
  - `sample_documents`: Pre-created test documents in various formats

Example pattern:
```python
def test_something(mock_embedder, temp_dir):
    # Test without external dependencies
    pass
```

## Configuration

Settings via environment variables with `THEO_` prefix. **All variables are required** - there are no defaults. Copy `.env.example` to `.env` and configure all values.

| Variable | Example Value | Description |
|----------|---------------|-------------|
| THEO_EMBEDDING_BACKEND | mlx | Embedding backend: `mlx` or `ollama` |
| THEO_MLX_MODEL | mlx-community/mxbai-embed-large-v1 | MLX model (HuggingFace path) |
| THEO_OLLAMA_HOST | http://localhost:11434 | Ollama server host URL |
| THEO_OLLAMA_MODEL | nomic-embed-text | Ollama embedding model |
| THEO_OLLAMA_LLM_MODEL | gemma3:12b | Ollama LLM for relationship classification |
| THEO_OLLAMA_TIMEOUT | 30 | Ollama request timeout in seconds |
| THEO_SQLITE_PATH | ~/.theo/theo.db | SQLite database path |
| THEO_LOG_LEVEL | INFO | Logging level |
| THEO_DEFAULT_NAMESPACE | global | Default namespace for memories |
| THEO_DEFAULT_IMPORTANCE | 0.5 | Default importance score (0.0-1.0) |
| THEO_DEFAULT_TOKEN_BUDGET | 4000 | Default token budget for context |

## File Paths and Structure

- Always use `pathlib.Path` objects, not strings
- Resolve paths with `.resolve()` for absolute paths
- Use `Path.exists()`, `Path.is_file()`, `Path.is_dir()` for validation

## Source Repositories

Theo merges code from:
- **DocVec**: `/Users/harrison/Documents/Github/docvec` - Document indexing
- **Recall**: `/Users/harrison/Documents/Github/recall` - Long-term memory
