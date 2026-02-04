# Theo

**Unified AI memory and document retrieval system** - merging the capabilities of DocVec (semantic document indexing) and Recall (long-term memory) into a single MCP server.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [MCP Configuration](#mcp-configuration)
- [Basic Usage](#basic-usage)
  - [Document Indexing](#document-indexing)
  - [Semantic Search](#semantic-search)
  - [Memory Operations](#memory-operations)
  - [Managing the Index](#managing-the-index)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Claude Code Skills](#claude-code-skills)
- [Migration Guide](#migration-guide)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

### Document Indexing (from DocVec)
- **Multi-format Support**: Index markdown, PDF, text, and Python code files
- **Smart Chunking**:
  - Header-aware chunking for markdown documents
  - Page-aware chunking for PDFs
  - AST-based chunking for Python code
  - Paragraph-based chunking for plain text
- **Hash-based Deduplication**: Automatic detection and skipping of duplicate documents

### Long-term Memory (from Recall)
- **Memory Types**: Store preferences, decisions, patterns, facts, and session context
- **Validation Loop**: Memories build confidence through practical use
- **Golden Rules**: High-confidence memories become protected principles
- **Namespace Scoping**: Organize memories by project or globally

### Voice Transcription
- **MLX Whisper**: Local speech-to-text using MLX-optimized Whisper models
- **Streaming Transcription**: Real-time transcription with silence detection
- **Text-to-Speech**: Local TTS for voice responses
- **Memory Integration**: Transcriptions stored as searchable memories

### Agent Trace (AI Code Attribution)
- **Spec Compliance**: Full [agent-trace.dev](https://agent-trace.dev) v0.1 compliance
- **Auto-Capture**: Line-level attribution on every commit via Claude Code hooks
- **Model Detection**: Auto-detects model from session transcript (opus/sonnet/haiku)
- **Query Tools**: CLI (`theo trace query`) and MCP tools (`trace_query`, `trace_list`)

### Unified Capabilities
- **Local Embeddings**: Privacy-first using MLX (Apple Silicon) or Ollama
- **Daemon Service**: Non-blocking embedding via Unix socket IPC
- **Token-aware Retrieval**: Control result size to fit within token budgets
- **MCP Integration**: Seamless integration with Claude Code and other MCP clients

[↑ Back to top](#table-of-contents)

## Quick Start

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Apple Silicon Mac (for MLX backend, default) OR [Ollama](https://ollama.ai) installed
- 1GB+ free disk space for SQLite database and model cache

### Installation

```bash
# Clone the repository
git clone https://github.com/blueman82/theo.git
cd theo

# Install dependencies using uv
uv sync

# Install dev dependencies (optional)
uv sync --dev
```

### MCP Configuration

Add to your MCP config:

```json
{
  "mcpServers": {
    "theo": {
      "command": "uv",
      "args": ["run", "python", "-m", "theo"],
      "cwd": "/path/to/theo",
      "env": {
        "THEO_SQLITE_PATH": "~/.theo/theo.db"
      }
    }
  }
}
```

**Important:** Replace `/path/to/theo` with the actual path where you cloned the repo.

Verify the setup:
```bash
# Restart Claude Code to load the MCP server
# Then ask Claude:
"What's in the Theo index? Use get_index_stats"
```

[↑ Back to top](#table-of-contents)

## Basic Usage

### Document Indexing

Index a single file:
```
"Index /path/to/document.md"
```

Index a directory:
```
"Index all files in /path/to/docs recursively"
```

Index with namespace:
```
"Index /path/to/project using namespace 'project:myproject'"
```

### Semantic Search

Basic search:
```
"Search for authentication configuration"
```

Search with filters:
```
"Search for API endpoints in the docs namespace"
```

Token-budget search:
```
"Search for deployment guide within 2000 tokens"
```

### Memory Operations

Store a memory:
```
"Remember that I prefer using FastAPI for Python APIs"
```

Store with explicit type:
```
"Store a decision: We chose PostgreSQL for the database"
```

Recall memories:
```
"What do you remember about my coding preferences?"
```

Validate a memory:
```
"That memory about FastAPI was helpful - validate it"
```

Delete a memory:
```
"Forget the memory about dark mode preferences"
```

### Managing the Index

Get statistics:
```
"Show me the Theo index stats"
```

Delete a file from index:
```
"Remove /path/to/old_document.md from the index"
```

Clear everything (requires confirmation):
```
"Clear the entire Theo index (confirm=true)"
```

[↑ Back to top](#table-of-contents)

## Configuration

Configuration via CLI arguments (highest priority) or environment variables with `THEO_` prefix:

| Environment Variable | CLI Argument | Default | Description |
|---------------------|--------------|---------|-------------|
| `THEO_EMBEDDING_BACKEND` | `--embedding-backend` | `mlx` | Backend: `mlx` or `ollama` |
| `THEO_MLX_MODEL` | `--mlx-model` | `mlx-community/mxbai-embed-large-v1` | MLX model |
| `THEO_OLLAMA_HOST` | `--ollama-host` | `http://localhost:11434` | Ollama server URL |
| `THEO_OLLAMA_MODEL` | `--ollama-model` | `nomic-embed-text` | Ollama model |
| `THEO_OLLAMA_TIMEOUT` | `--ollama-timeout` | `30` | Timeout in seconds |
| `THEO_SQLITE_PATH` | `--sqlite-path` | `~/.theo/theo.db` | SQLite database path |
| `THEO_LOG_LEVEL` | `--log-level` | `INFO` | Logging level |

[↑ Back to top](#table-of-contents)

## Architecture

Theo combines three key capabilities:

1. **Document Indexing Pipeline**: File → Chunker → Embedder → SQLite (sqlite-vec)
2. **Memory System**: Store → Validate → Recall with confidence scoring
3. **Daemon Service**: Non-blocking IPC for embedding operations

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│   MCP Server    │────▶│   Tool Layer    │
│  (MCP Client)   │     │   (FastMCP)     │     │  (Async Handlers)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────┐
                    │                                    │                    │
              ┌─────▼─────┐                      ┌───────▼───────┐    ┌───────▼───────┐
              │ Indexing  │                      │   Memory      │    │   Query       │
              │   Tools   │                      │   Tools       │    │   Tools       │
              └─────┬─────┘                      └───────┬───────┘    └───────┬───────┘
                    │                                    │                    │
                    └────────────────┬───────────────────┴────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │   Daemon    │
                              │   Client    │
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
        ┌─────▼─────┐         ┌──────▼──────┐        ┌──────▼──────┐
        │ Chunkers  │         │  Embedding  │        │   SQLite    │
        │ Registry  │         │  Provider   │        │   Store     │
        └───────────┘         └─────────────┘        └─────────────┘
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

[↑ Back to top](#table-of-contents)

## API Reference

Theo exposes 24 MCP tools:

### Document Tools (2)
- `index_file(file_path, namespace)` - Index a single document
- `index_directory(dir_path, recursive, namespace)` - Batch index documents

### Search Tools (3)
- `search(query, n_results)` - Basic semantic search
- `search_with_filters(query, filters, n_results)` - Filtered search
- `search_with_budget(query, max_tokens)` - Token-budget search

### Memory Tools (13)
- `memory_store(content, memory_type, namespace, importance, supersedes_query)` - Store memory (use `supersedes_query` to auto-replace old memories)
- `memory_recall(query, n_results, namespace, memory_type)` - Recall memories
- `memory_forget(memory_id, query, force)` - Delete memories
- `memory_context(query, namespace, token_budget)` - Generate LLM context
- `memory_apply(memory_id, context)` - Record memory usage (TRY phase)
- `memory_outcome(memory_id, success, skip_event)` - Record result + adjust confidence (use `skip_event=True` for direct validation)
- `memory_relate(source_id, target_id, relation_type)` - Create relationships
- `memory_edge_forget(source_id, target_id)` - Delete relationship edges
- `memory_inspect_graph(memory_id, max_depth, output_format)` - Visualize graph
- `memory_count(namespace, memory_type)` - Count memories with filters
- `memory_list(namespace, memory_type, limit)` - List memories with pagination
- `validation_history(memory_id, event_type, limit)` - Get validation timeline
- `memory_analyze_health(namespace, include_contradictions)` - Analyze memory system health (includes contradiction detection)

### Management Tools (4)
- `delete_chunks(ids)` - Delete specific chunks
- `delete_file(source_file)` - Delete file's chunks
- `clear_index(confirm)` - Clear everything (requires confirmation)
- `get_index_stats()` - Get collection statistics

### Trace Tools (2)
- `trace_query(file, line)` - Query AI attribution for code via git blame
- `trace_list(conversation_url, limit)` - List recorded traces

See [docs/API.md](docs/API.md) for complete API specifications.

[↑ Back to top](#table-of-contents)

## Claude Code Skills

Theo provides 15 Claude Code skills for convenient CLI access:

| Skill | Description | Example |
|-------|-------------|---------|
| `/index` | Index files or directories | `/index ~/Documents/project` |
| `/search` | Semantic search across indexed docs | `/search authentication flow` |
| `/store` | Store new memories | `/store Always use TypeScript --type=pattern` |
| `/recall` | Recall memories via semantic search | `/recall coding preferences --expand` |
| `/forget` | Delete memories by ID or query | `/forget mem_abc123` |
| `/list` | Browse memories with pagination | `/list --type=preference --limit=10` |
| `/relate` | Manage memory relationships | `/relate mem_a supersedes mem_b` |
| `/clean` | Clean up indexed documents | `/clean file ./old-doc.md` |
| `/stats` | Show index and memory statistics | `/stats` |
| `/validate` | TRY-LEARN validation cycle | `/validate apply mem_abc123 "testing"` |
| `/context` | Get formatted context for LLM injection | `/context authentication --budget 2000` |
| `/health` | Analyze memory system health (includes contradictions) | `/health --include-contradictions` |
| `/history` | View validation event timeline | `/history mem_abc123` |
| `/graph` | Visualize memory relationships | `/graph mem_abc123 --format mermaid` |
| `/contradictions` | Detect contradicting memories | `/contradictions --namespace project:theo` |

### Installing Skills

Skills are located in `skills/` directory. Copy to your Claude Code skills folder:

```bash
cp -r skills/* ~/.claude/skills/
```

[↑ Back to top](#table-of-contents)

## Migration Guide

If you're migrating from DocVec or Recall, see [docs/migration.md](docs/migration.md) for:
- Configuration changes
- Tool name mappings
- Data migration strategies
- Breaking changes

[↑ Back to top](#table-of-contents)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/theo --cov-report=html

# Run integration tests
uv run pytest tests/integration/ -v
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

### Running the MCP Server

```bash
# Run with default configuration (MLX backend)
uv run python -m theo

# Run with Ollama backend
uv run python -m theo --embedding-backend ollama

# Run with debug logging
uv run python -m theo --log-level DEBUG

# View all options
uv run python -m theo --help
```

[↑ Back to top](#table-of-contents)

## Troubleshooting

### MLX Model Download Failed

**Error**: `Failed to download MLX model` or slow first-run

**Solution**:
1. First run downloads ~500MB model from HuggingFace
2. Ensure internet connectivity and sufficient disk space
3. Model cached at `~/.cache/huggingface/` after first download

### Ollama Connection Failed

**Error**: `Failed to connect to Ollama`

**Solution** (only applies when using `--embedding-backend ollama`):
1. Check if Ollama is running: `ollama list`
2. Start Ollama if needed: `ollama serve`
3. Pull the embedding model: `ollama pull nomic-embed-text`

### SQLite Permission Error

**Error**: `Permission denied: ~/.theo/theo.db`

**Solution**:
1. Create directory: `mkdir -p ~/.theo`
2. Fix permissions: `chmod 755 ~/.theo`
3. Or specify different path via `THEO_SQLITE_PATH`

### Search Returns No Results

**Issue**: Search returns empty despite indexed documents

**Solution**:
1. Verify documents were indexed: `get_index_stats`
2. Check namespace matches: search may be filtered by namespace
3. Try broader queries with fewer terms

### Can't Delete Golden Rules

**Issue**: Memory deletion fails for high-confidence memories

**Solution**:
Golden rules (confidence >= 0.9) are protected. Use `force=true`:
```
"Forget memory mem_123 with force"
```

### MLX Metal Threading Crash

**Error**: `-[_MTLCommandBuffer addCompletedHandler:]:976: failed assertion` or SIGSEGV (exit 139)

**Cause**: MLX Metal GPU operations are NOT thread-safe. Using `asyncio.to_thread()` with MLX causes Metal command buffer race conditions.

**Solution**:
1. MLX embedding operations MUST run on the main thread
2. The `embed_batch()` method runs synchronously by design - do NOT wrap with `asyncio.to_thread()`
3. Never call `mx.clear_cache()` during embedding operations - it clears Metal buffers while other operations expect them
4. If running as a daemon, ensure the embedding worker uses the main asyncio event loop

**Prevention**: The daemon's embed_worker is designed to briefly block the event loop (~50-100ms per batch) rather than use thread pools. This is the only reliable approach without process isolation.

### SQLite FTS5 Issues

**Error**: Full-text search returns unexpected results or `PRAGMA integrity_check` shows warnings

**Cause**: The FTS5 index can become stale after crashes or improper shutdowns.

**Solution**:
```bash
# Connect to the Theo SQLite database
sqlite3 ~/.theo/theo.db

# Rebuild the FTS5 index
INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

# Verify the fix
PRAGMA integrity_check;

# Exit
.quit
```

**Prevention**:
1. Always gracefully shutdown the daemon
2. After system crashes, run the FTS5 rebuild command above
3. If corruption persists, delete `~/.theo/theo.db` and re-index

[↑ Back to top](#table-of-contents)

## License

MIT License - see LICENSE file for details.

[↑ Back to top](#table-of-contents)
