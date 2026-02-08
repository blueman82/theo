# CLAUDE.md
Last updated: 2026-02-08

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Theo** - AI memory and document retrieval system with semantic indexing, validation loops, and confidence scoring.

### Core Capabilities

1. **Document Indexing**: Semantic document chunking and retrieval using vector embeddings
2. **Long-term Memory**: Persistent memory storage with validation loops and confidence scoring
3. **MCP Interface**: Single server exposing all capabilities to MCP-compatible clients

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
├── __main__.py          # MCP server entry point
├── config.py            # Pydantic Settings configuration
├── constants.py         # Shared constants
├── mcp_server.py        # FastMCP server with tool registration
├── chunking/            # Document chunking (format-specific)
│   ├── base.py          # AbstractChunker interface
│   ├── code_chunker.py  # AST-based Python chunking
│   ├── markdown_chunker.py  # Header-aware chunking
│   ├── pdf_chunker.py   # Page-aware chunking
│   ├── registry.py      # Extension → chunker mapping
│   └── text_chunker.py  # Paragraph-based fallback
├── daemon/              # Non-blocking embedding daemon
│   ├── client.py        # DaemonClient for IPC
│   ├── protocol.py      # JSON-RPC protocol definitions
│   ├── server.py        # Unix socket server
│   └── worker.py        # Background job processor
├── embedding/           # Embedding providers
│   ├── provider.py      # EmbeddingProvider protocol
│   ├── factory.py       # Backend selection (mlx/ollama)
│   ├── mlx_provider.py  # MLX embeddings (Apple Silicon)
│   └── ollama_provider.py  # Ollama HTTP client
├── storage/             # Unified storage layer
│   ├── sqlite_store.py  # SQLite + sqlite-vec (vectors + metadata)
│   ├── types.py         # Storage type definitions
│   └── hybrid.py        # HybridStore coordination layer
├── tools/               # MCP tool implementations
│   ├── indexing_tools.py   # index_file, index_directory
│   ├── query_tools.py      # search, search_with_filters
│   ├── memory_tools.py     # memory_store, memory_recall, etc.
│   └── management_tools.py # delete_*, clear_index, stats
├── types/               # Type definitions
│   ├── document.py      # Document, Chunk types
│   ├── memory.py        # MemoryDocument, MemoryType, RelationType
│   └── search_result.py # SearchResult types
├── validation/          # Memory validation loop
│   ├── feedback.py      # Feedback tracking
│   ├── golden_rules.py  # Golden rule promotion
│   └── loop.py          # TRY → BREAK → ANALYZE → LEARN
└── transcription/       # Voice transcription (MLX Whisper)
    ├── __main__.py      # CLI entry point
    ├── audio.py         # Audio capture with silence detection
    ├── transcriber.py   # MLX Whisper transcription
    ├── tts.py           # Text-to-speech
    ├── tui.py           # Terminal UI
    ├── storage.py       # Transcription storage
    └── types.py         # Type definitions
```

### Key Data Flows

**Document Indexing Pipeline:**
1. File path → Indexer validates and reads content
2. File extension → Appropriate chunker selected
3. Content → Chunker splits into semantically meaningful chunks
4. Chunks → Embedded via MLX/Ollama
5. Chunks + Embeddings → Stored in SQLite (sqlite-vec)

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
  - `temp_sqlite`: Temporary SQLite database for isolated tests
  - `temp_dir`: Temporary directories for file operations

Example pattern:
```python
def test_something(mock_embedder, temp_sqlite):
    store = SQLiteStore(temp_sqlite)
    # Test without external dependencies
```

## Configuration

Settings via environment variables with `THEO_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| THEO_EMBEDDING_BACKEND | mlx | Embedding backend: `mlx` or `ollama` |
| THEO_MLX_MODEL | mlx-community/mxbai-embed-large-v1 | MLX model |
| THEO_SQLITE_PATH | ~/.theo/theo.db | SQLite database (vectors + metadata) |
| THEO_LOG_LEVEL | INFO | Logging level |

## Agent Trace (AI Code Attribution)

Theo implements the [agent-trace.dev](https://agent-trace.dev) open standard for AI code attribution.

### How It Works

1. `auto_commit.py` hook captures Write/Edit/MultiEdit operations
2. After commit, extracts line ranges via `git diff HEAD~1..HEAD --unified=0`
3. Model ID auto-detected from Claude Code transcript (`~/.claude/projects/<path>/<session>.jsonl`)
4. Trace stored in SQLite with spec-compliant JSON structure

### CLI Commands

```bash
# Query attribution for a file/line
uv run python -m theo trace query <file> [--line N]
```

### MCP Tools

- `trace_query(file, line)` - Query AI attribution via git blame
- `trace_list(conversation_url, limit)` - List recorded traces

### Storage Schema

Traces table stores spec-compliant JSON with:
- `version`, `id`, `timestamp` (required)
- `files[].conversations[].ranges[]` with `start_line`, `end_line`
- `vcs.type`, `vcs.revision` (git commit SHA)
- `tool.name`, `tool.version` (claude-code)
- `contributor.type`, `contributor.model_id`

## Claude Code Hooks

Theo provides hooks for Claude Code integration. Install from `hooks/` to `~/.claude/hooks/`.

### Hook Responsibilities

| Hook | Claude Event | Responsibility | Blocking? |
|------|--------------|----------------|-----------|
| `theo-daemon-ctl.py` | PrePromptSubmit | Ensure daemon is running | No |
| `theo-context.py` | SessionStart | Fetch + curate memories → inject session context | No |
| `theo-prompt.py` | UserPromptSubmit | Search memories by prompt → inject RFC 2119 context | No |
| `theo-precontext.py` | PreToolUse | Search memories → inject reminders, can modify input | Can modify |
| `theo-recall.py` | PostToolUse | Autonomous memory recall based on tool output (errors, file reads) | No |
| `auto_commit.py` | PostToolUse | Auto-commit Write/Edit/MultiEdit + capture Agent Trace | No |
| `theo-failure.py` | PostToolUseFailure | Store tool failure patterns for learning | No |
| `theo-capture.py` | SessionEnd | Summarize session with Ollama → store memories | No (background) |
| `theo-stop.py` | Stop | Enforce memory storage before allowing stop | Yes |

### Context Injection Flow

Four hooks inject context at different lifecycle points:

```
SessionStart          UserPromptSubmit       PreToolUse            PostToolUse
     │                      │                     │                     │
     ▼                      ▼                     ▼                     ▼
theo-context.py      theo-prompt.py      theo-precontext.py    theo-recall.py
     │                      │                     │                     │
     ▼                      ▼                     ▼                     ▼
Curated memories     Prompt-specific      Tool-specific         Reactive recall
(broad context)      memories (targeted)  reminders (precise)   (output-based)
```

The first three hooks are **proactive** (inject before actions). The fourth is
**reactive** (recalls memories based on what actually happened — errors, file
reads, test failures). This closes the loop: context flows in both directions.

### Supporting Files

| File | Purpose |
|------|---------|
| `theo_client.py` | DaemonClient for fast IPC with subprocess fallback |
| `theo_worker.py` | Background worker for async operations |
| `theo_batcher.py` | Batch processing for embeddings |
| `theo_queue.py` | Queue management for async operations |
| `theo_session_state.py` | Session state tracking |
| `theo-daemon.py` | Unix socket server (background service) |
| `theo-daemon-ctl.py` | Daemon lifecycle management |

## Autonomous Memory Behavior

Theo hooks automatically inject memory context at multiple lifecycle points.
Claude MUST treat injected memories according to their RFC 2119 priority:

- **MUST** items: Follow unconditionally. These are golden rules or high-confidence memories.
- **SHOULD** items: Follow unless there is a specific, documented reason not to.
- **MAY** items: Consider as helpful context. Apply when relevant.

### Proactive Memory Use

When working on tasks, Claude should proactively use Theo's memory tools:

1. **Store learnings**: Use `memory_store` when discovering preferences, patterns, or decisions.
   Do not wait for the user to ask — store anything that would be useful in future sessions.
2. **Validate memories**: When applying knowledge from a surfaced memory, use `memory_apply`
   before and `memory_outcome` after to record whether it worked. This builds confidence
   and enables golden rule promotion.
3. **Check memories first**: Before making architectural decisions or choosing between
   approaches, search Theo for relevant past decisions with `memory_recall`.

### Reactive Recall (theo-recall.py)

The `theo-recall.py` PostToolUse hook automatically recalls memories when:

- **Bash commands fail**: Searches for similar error patterns and known fixes
- **Files are read**: Searches for file/module-specific decisions and patterns

This means Claude does not need to be asked to recall memories — relevant context
is injected automatically after errors and file reads. When recalled memories appear
in the context, apply them immediately rather than asking the user for confirmation.

## File Paths and Structure

- Always use `pathlib.Path` objects, not strings
- Resolve paths with `.resolve()` for absolute paths
- Use `Path.exists()`, `Path.is_file()`, `Path.is_dir()` for validation

