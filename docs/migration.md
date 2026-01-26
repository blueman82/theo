# Migration Guide

This guide helps existing DocVec and Recall users migrate to Theo.

## Overview

Theo unifies DocVec (semantic document indexing) and Recall (long-term memory) into a single MCP server. If you're currently using either or both of these tools, this guide covers the migration process.

## What Changed

### Architecture Changes

| Aspect | DocVec | Recall | Theo |
|--------|--------|--------|------|
| MCP Server | Separate server | Separate server | **Single unified server** |
| Socket Path | `/tmp/docvec.sock` | `/tmp/recall.sock` | **`/tmp/theo.sock`** |
| DB Path | `~/.docvec/chroma_db` | `~/.recall/memories.db` | **`~/.theo/chroma_db`** |
| Config Prefix | `DOCVEC_` | `RECALL_` | **`THEO_`** |

### Unified Data Model

Theo introduces a unified `MemoryDocument` type that combines:
- DocVec's `Chunk` (document chunks with provenance)
- Recall's `Memory` (memories with confidence scoring)

Both are now stored in the same ChromaDB collection with a `doc_type` field to distinguish them:
- `doc_type: "document"` - Document chunks
- `doc_type: "preference"`, `"decision"`, `"pattern"`, `"session"`, `"fact"` - Memories

### New Features

Theo includes several new features not available in the original tools:

1. **Daemon Service**: Non-blocking embedding via Unix socket IPC
2. **Unified Search**: Search across documents AND memories simultaneously
3. **Validation Loop**: Memories build confidence through practical use
4. **Golden Rules**: High-confidence memories become constitutional principles
5. **Namespace Support**: Better organization with `global`, `default`, `project:{name}`

---

## Migrating from DocVec

### Step 1: Update MCP Configuration

**Old configuration** (`~/.claude/config/mcp.json`):
```json
{
  "mcpServers": {
    "docvec": {
      "command": "uv",
      "args": ["run", "python", "-m", "docvec"],
      "cwd": "/path/to/docvec",
      "env": {
        "DOCVEC_DB_PATH": "/path/to/chroma_db"
      }
    }
  }
}
```

**New configuration**:
```json
{
  "mcpServers": {
    "theo": {
      "command": "uv",
      "args": ["run", "python", "-m", "theo"],
      "cwd": "/path/to/theo",
      "env": {
        "THEO_DB_PATH": "/path/to/chroma_db"
      }
    }
  }
}
```

### Step 2: Update Environment Variables

| DocVec Variable | Theo Variable |
|-----------------|---------------|
| `DOCVEC_EMBEDDING_BACKEND` | `THEO_EMBEDDING_BACKEND` |
| `DOCVEC_MLX_MODEL` | `THEO_MLX_MODEL` |
| `DOCVEC_DB_PATH` | `THEO_DB_PATH` |
| `DOCVEC_HOST` | `THEO_OLLAMA_HOST` |
| `DOCVEC_MODEL` | `THEO_OLLAMA_MODEL` |
| `DOCVEC_TIMEOUT` | `THEO_OLLAMA_TIMEOUT` |
| `DOCVEC_COLLECTION` | `THEO_COLLECTION` |
| `DOCVEC_LOG_LEVEL` | `THEO_LOG_LEVEL` |

### Step 3: Tool Name Changes

**No changes required!** DocVec tools have the same names and signatures in Theo:

| Tool | Status |
|------|--------|
| `index_file` | ✅ Same (added `namespace` parameter) |
| `index_directory` | ✅ Same (added `namespace` parameter) |
| `search` | ✅ Same |
| `search_with_filters` | ✅ Same |
| `search_with_budget` | ✅ Same |
| `delete_chunks` | ✅ Same |
| `delete_file` | ✅ Same |
| `clear_index` | ✅ Same |
| `get_index_stats` | ✅ Same (returns additional fields) |

### Step 4: Data Migration

**Option A: Re-index (Recommended)**

If your document set isn't too large, re-indexing is the cleanest approach:

```bash
# Start with fresh database
rm -rf ~/.theo/chroma_db

# Or point to a new location
export THEO_DB_PATH="~/.theo/chroma_db"

# Re-index your documents via Claude
"Index all my documents in /path/to/docs using Theo"
```

**Option B: Copy Database**

If you want to preserve existing data:

```bash
# Copy DocVec database to Theo location
cp -r ~/.docvec/chroma_db ~/.theo/chroma_db

# Theo will use existing embeddings and add new fields as needed
```

Note: Copied data won't have the new `namespace` or `doc_type` fields. They'll be added on first access with default values.

### Step 5: New Features to Try

Once migrated, explore Theo's new capabilities:

```
# Use namespaces to organize documents
"Index /path/to/project using namespace 'project:myproject'"

# Search across namespaces
"Search for authentication in the docs namespace"

# Store memories
"Remember that I prefer using FastAPI for Python APIs"
```

---

## Migrating from Recall

### Step 1: Update MCP Configuration

**Old configuration**:
```json
{
  "mcpServers": {
    "recall": {
      "command": "uv",
      "args": ["run", "python", "-m", "recall"],
      "cwd": "/path/to/recall",
      "env": {
        "RECALL_DB_PATH": "/path/to/memories.db"
      }
    }
  }
}
```

**New configuration**:
```json
{
  "mcpServers": {
    "theo": {
      "command": "uv",
      "args": ["run", "python", "-m", "theo"],
      "cwd": "/path/to/theo",
      "env": {
        "THEO_DB_PATH": "/path/to/chroma_db"
      }
    }
  }
}
```

### Step 2: Tool Name Changes

Memory tools have a `memory_` prefix in Theo:

| Recall Tool | Theo Tool | Changes |
|-------------|-----------|---------|
| `store` | `memory_store` | Added `importance` parameter |
| `recall` | `memory_recall` | Added `min_confidence` filter |
| `validate` | `memory_validate` | Same signature |
| `forget` | `memory_forget` | Added `force` for golden rules |
| N/A | `memory_context` | **New!** Generate LLM context |

### Step 3: Memory Type Mapping

| Recall Type | Theo Type | Notes |
|-------------|-----------|-------|
| `preference` | `preference` | Same |
| `decision` | `decision` | Same |
| `pattern` | `pattern` | Same |
| `session` | `session` | Same |
| `fact` | `fact` | Same |
| N/A | `golden_rule` | **New!** Auto-promoted from high confidence |

### Step 4: Data Migration

Recall used SQLite, Theo uses ChromaDB. Data migration requires manual transfer:

**Export memories from Recall** (if you have access to the old database):

```python
# Script to export Recall memories
import sqlite3
import json

conn = sqlite3.connect("~/.recall/memories.db")
cursor = conn.cursor()
cursor.execute("SELECT content, type, namespace, importance FROM memories")
memories = [
    {"content": row[0], "type": row[1], "namespace": row[2], "importance": row[3]}
    for row in cursor.fetchall()
]

with open("memories_export.json", "w") as f:
    json.dump(memories, f, indent=2)
```

**Import into Theo**:

After starting Theo, use Claude to re-import:

```
# For each exported memory:
"Store a memory: {content} with type {type} in namespace {namespace}"
```

Or write a script to call `memory_store` for each exported memory.

### Step 5: New Features

Theo enhances Recall with:

1. **Confidence Scoring**: Memories start at 0.3 confidence and adjust through validation
2. **Golden Rules**: At confidence >= 0.9, memories become protected principles
3. **Unified Search**: Search documents and memories together
4. **Context Generation**: `memory_context` tool formats memories for LLM injection

Example usage:
```
# Validate a memory after use
"That memory about FastAPI was helpful - validate it"

# Generate context for a task
"Get relevant memories for authentication implementation"
```

---

## Migrating from Both DocVec AND Recall

If you're using both tools, Theo replaces both with a single server.

### Step 1: Unified Configuration

Remove both old servers, add Theo:

```json
{
  "mcpServers": {
    "theo": {
      "command": "uv",
      "args": ["run", "python", "-m", "theo"],
      "cwd": "/path/to/theo",
      "env": {
        "THEO_DB_PATH": "~/.theo/chroma_db",
        "THEO_EMBEDDING_BACKEND": "mlx"
      }
    }
  }
}
```

### Step 2: Migrate Data

1. **Re-index documents** (from DocVec):
   ```
   "Index all documents in /path/to/docs"
   ```

2. **Re-store memories** (from Recall):
   ```
   "Remember that I prefer dark mode"
   "Remember the decision to use FastAPI"
   ```

### Step 3: Unified Workflow

With Theo, you can now:

1. **Search across everything**:
   ```
   "Search for authentication"
   # Returns both document chunks AND relevant memories
   ```

2. **Use namespaces consistently**:
   ```
   "Index project docs in namespace project:myapp"
   "Store memory about myapp preferences in namespace project:myapp"
   ```

3. **Build knowledge through use**:
   - Validate helpful memories → confidence increases
   - High-confidence memories become golden rules
   - Golden rules inform future interactions

---

## Real-World Migration Learnings

Based on actual migrations from DocVec and Recall to Theo, here are important lessons learned:

### Migration Statistics (Actual Results)

A complete migration from both DocVec and Recall yielded:
- **12,947 total items migrated** (10,723 DocVec documents + 3,942 Recall memories - 1,658 duplicates)
- **5,510 relationship edges preserved** in SQLite
- **Storage reduced**: ChromaDB went from 351GB (bloated) to 110MB (fresh)
- **Deduplication**: ~13% of items were duplicates across systems

### Critical: ChromaDB HNSW Index Bloat

**Problem**: Copying old ChromaDB databases can result in massive index bloat. HNSW indexes can grow to 600GB+ if:
- Many documents were added and deleted over time
- The index was never compacted
- Multiple embedding models were used

**Solution**: Always regenerate embeddings fresh with MLX rather than copying old databases.

```bash
# DON'T do this (can inherit bloated indexes):
cp -r ~/.docvec/chroma_db ~/.theo/chroma_db

# DO this instead (fresh start with clean indexes):
rm -rf ~/.theo/chroma_db
# Then re-index via Claude: "Index all my documents"
```

### Recommended Migration Approach

1. **Start Fresh** - Don't copy old ChromaDB databases
2. **Use MLX Embeddings** - Regenerate all embeddings with MLX for consistency
3. **Migrate Memories First** - Import Recall memories before documents (they're smaller)
4. **Preserve Edges** - Export and re-import relationship edges from Recall's SQLite

### Automated Migration Script

For large migrations, use the migration script:

```bash
# Run from theo directory
uv run python scripts/migrate_from_legacy.py \
  --docvec-path ~/.docvec/chroma_db \
  --recall-path ~/.recall/memories.db \
  --theo-path ~/.theo/chroma_db \
  --fresh-embeddings  # Recommended: regenerate with MLX
```

The script handles:
- Reading from old DocVec ChromaDB collection ("documents")
- Reading from old Recall SQLite database
- Deduplication across both sources
- Fresh MLX embedding generation
- Edge preservation in Theo's SQLite

### Post-Migration Verification

After migration, verify data integrity:

```
# Check stats via Claude:
"Show me Theo index stats"

# Expected output should show:
# - Total documents matching your DocVec count
# - Total memories matching your Recall count
# - Edges count matching your relationship edges
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THEO_EMBEDDING_BACKEND` | `mlx` | Embedding backend: `mlx` or `ollama` |
| `THEO_MLX_MODEL` | `mlx-community/mxbai-embed-large-v1` | MLX model |
| `THEO_OLLAMA_HOST` | `http://localhost:11434` | Ollama server host |
| `THEO_OLLAMA_MODEL` | `nomic-embed-text` | Ollama model name |
| `THEO_OLLAMA_TIMEOUT` | `30` | Ollama request timeout |
| `THEO_DB_PATH` | `~/.theo/chroma_db` | ChromaDB storage path |
| `THEO_COLLECTION` | `documents` | Collection name |
| `THEO_LOG_LEVEL` | `INFO` | Logging level |

### CLI Arguments

All environment variables can be overridden via CLI:

```bash
uv run python -m theo \
  --embedding-backend mlx \
  --mlx-model mlx-community/mxbai-embed-large-v1 \
  --db-path ~/.theo/chroma_db \
  --collection documents \
  --log-level INFO
```

---

## Breaking Changes Summary

### From DocVec

| Change | Impact | Migration |
|--------|--------|-----------|
| Config prefix `DOCVEC_` → `THEO_` | Low | Update env vars |
| Module name `docvec` → `theo` | Low | Update MCP config |
| Added `namespace` parameter | None | Optional parameter |

### From Recall

| Change | Impact | Migration |
|--------|--------|-----------|
| Config prefix `RECALL_` → `THEO_` | Low | Update env vars |
| Module name `recall` → `theo` | Low | Update MCP config |
| Tool names: `store` → `memory_store`, etc. | Medium | Update tool calls |
| SQLite → ChromaDB | High | Re-import memories |
| Added confidence scoring | None | Auto-applied |
| Added golden rule protection | Low | Use `force=true` to delete |

---

## Migrating Claude Code Hooks

If you have Claude Code hooks using DocVec or Recall, they need to be updated to use Theo.

### Hook File Mapping

| Old Hook | New Hook | Purpose |
|----------|----------|---------|
| `docvec-track.py` | `theo-track.py` | Track file activity |
| `recall-track.py` | `theo-track.py` | Track file activity (merged) |
| `docvec-context.py` | `theo-context.py` | Inject context into prompts |
| `recall-capture.py` | `theo-capture.py` | Capture session learnings |
| `recall-compact.py` | `theo-compact.py` | Store on context compaction |

### Shared Client Pattern

All Theo hooks use a shared client from `theo_client.py`:

```python
# ~/.claude/hooks/theo_client.py
import socket
import json

THEO_SOCKET = "/tmp/theo.sock"

def get_shared_client():
    """Get shared daemon client for all hooks."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(THEO_SOCKET)
    return sock

def call_theo(method: str, params: dict) -> dict:
    """Call Theo daemon method."""
    sock = get_shared_client()
    try:
        request = {"method": method, "params": params}
        sock.sendall(json.dumps(request).encode() + b"\n")
        response = sock.recv(65536).decode()
        return json.loads(response)
    finally:
        sock.close()
```

### Updating Hook Imports

**Before (DocVec/Recall)**:
```python
from docvec_client import get_client
# or
from recall_client import get_client
```

**After (Theo)**:
```python
from theo_client import get_shared_client, call_theo
```

### Example: Context Hook Migration

**Before** (`docvec-context.py`):
```python
result = docvec.search(query=context_query, n_results=5)
```

**After** (`theo-context.py`):
```python
result = call_theo("search", {"query": context_query, "n_results": 5})
# Or for memories:
result = call_theo("memory_context", {"query": context_query, "token_budget": 2000})
```

### Hook Installation

Copy hooks to your Claude Code hooks directory:

```bash
# Copy all theo hooks
cp ~/.claude/hooks/theo-*.py ~/.claude/hooks/
cp ~/.claude/hooks/theo_client.py ~/.claude/hooks/

# Remove old hooks (optional, after verifying theo hooks work)
rm ~/.claude/hooks/docvec-*.py
rm ~/.claude/hooks/recall-*.py
```

---

## Troubleshooting

### "Server not initialized" Error

The server needs time to initialize embedding models. If you see this error:
1. Wait a few seconds after server start
2. Check logs for initialization errors: `THEO_LOG_LEVEL=DEBUG`

### Embeddings Failing

If MLX embeddings fail on Apple Silicon:
1. Ensure you have sufficient memory (~1GB for model)
2. Try Ollama backend: `THEO_EMBEDDING_BACKEND=ollama`

If Ollama embeddings fail:
1. Ensure Ollama is running: `ollama serve`
2. Pull the embedding model: `ollama pull nomic-embed-text`

### Can't Delete Golden Rules

Golden rules (confidence >= 0.9) are protected by default:
```
"Delete memory mem_123 with force"
# Or in code: memory_forget(memory_id="mem_123", force=True)
```

### Data Not Found After Migration

If you copied a DocVec database but data seems missing:
1. Check the collection name matches: `THEO_COLLECTION`
2. Verify the database path: `THEO_DB_PATH`
3. Try `get_index_stats` to see what's indexed

---

## Getting Help

- **Documentation**: See [architecture.md](./architecture.md) and [API.md](./API.md)
- **Issues**: Report problems on the GitHub repository
- **CLAUDE.md**: Check the project's CLAUDE.md for development guidance
