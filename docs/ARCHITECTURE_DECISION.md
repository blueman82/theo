# Theo Architecture Decision: sqlite-vec Migration

**Date**: 2026-01-27
**Status**: APPROVED
**Impact**: Major refactor - removes daemon, replaces ChromaDB

## Context

Theo experienced repeated ChromaDB/HNSW corruption (exit code 139) after unifying docvec and recall. Root cause: daemon and MCP server both write to the same ChromaDB database, and ChromaDB is not designed for concurrent writers.

## Decision

**Replace ChromaDB with sqlite-vec + FTS5. Remove the daemon entirely.**

## Architecture: Before vs After

### Before (Broken)
```
┌─────────────────────┐     ┌─────────────────────┐
│   THEO DAEMON       │     │   THEO MCP SERVER   │
│  (HybridStore)      │     │  (ChromaStore)      │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          └─────────┬─────────────────┘
                    ↓
         ┌─────────────────────┐
         │  ChromaDB (HNSW)    │  ← CORRUPTION
         └─────────────────────┘
```

### After (New)
```
┌─────────────────────────────────────────────────┐
│              THEO MCP SERVER                    │
│         (single process, HTTP transport)        │
│                                                 │
│  ┌──────────────┐       ┌──────────────────┐   │
│  │  MCP Tools   │       │  Embed Worker    │   │
│  │  (handlers)  │       │  (asyncio task)  │   │
│  └──────┬───────┘       └────────┬─────────┘   │
│         │                        │             │
│         └──────────┬─────────────┘             │
│                    ↓                           │
│  ┌─────────────────────────────────────────┐   │
│  │            ~/.theo/theo.db              │   │
│  │                                         │   │
│  │  sqlite-vec (vectors) + FTS5 (text)     │   │
│  │  memories, documents, edges, validation │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Why This Works

**SQLite handles concurrent access safely:**
- WAL mode: multiple readers, serialized writers
- File locking: kernel-level protection
- ACID: transactions are atomic, no partial writes
- Self-healing: crash recovery built-in

**ChromaDB did not:**
- HNSW index in separate binary files
- No cross-process coordination
- Partial write = corrupted index

## Schema Design

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

-- Core tables
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    memory_type TEXT,
    namespace TEXT DEFAULT 'global',
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.3,
    created_at REAL,
    accessed_at REAL,
    access_count INTEGER DEFAULT 0,
    metadata TEXT  -- JSON
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT,
    chunk_index INTEGER,
    doc_hash TEXT,
    namespace TEXT DEFAULT 'default',
    created_at REAL,
    metadata TEXT  -- JSON
);

-- Vector search (sqlite-vec)
CREATE VIRTUAL TABLE memory_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[1024]
);

CREATE VIRTUAL TABLE document_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[1024]
);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE memory_fts USING fts5(
    id, content, memory_type, namespace,
    tokenize='porter'
);

CREATE VIRTUAL TABLE document_fts USING fts5(
    id, content, source_file,
    tokenize='porter'
);

-- Existing tables (unchanged)
CREATE TABLE edges (...);
CREATE TABLE validation_events (...);
```

## Components to DELETE

```
hooks/theo-daemon.py          # Main daemon
hooks/theo-daemon-ctl.py      # Daemon control
hooks/theo_worker.py          # Embed worker
hooks/theo_queue.py           # Queue management
hooks/theo_batcher.py         # Batch processing
hooks/theo_client.py          # Daemon client
src/theo/daemon/              # Entire daemon module
```

## Components to MODIFY

```
src/theo/storage/             # Replace ChromaStore with SQLiteVecStore
src/theo/__main__.py          # Remove ChromaDB init, add SQLite
src/theo/mcp_server.py        # Update storage references
src/theo/tools/               # Update to use new storage
```

## Components UNCHANGED

- MCP tool interface (all 25 tools)
- Memory types, validation loop, confidence scoring
- MLX/Ollama embedding providers
- Chunking logic (markdown, PDF, code, text)
- Edge/graph relationships

## MCP Transport Change

**Current**: stdio (mcp-exec spawns new process each session)
- 2-5s model cold start each time
- Process dies after session

**Recommended**: HTTP streamable transport
- Theo runs as persistent service
- Model stays warm in memory
- Fast responses, no cold start
- mcp-exec connects via HTTP

## Migration Steps

1. Export data from corrupted ChromaDB backup (SQLite readable)
2. Create new schema with sqlite-vec + FTS5
3. Re-embed all ~70K records (MLX, ~30 min)
4. Delete daemon code and hooks
5. Update MCP server to use SQLite directly
6. Switch to HTTP transport (optional but recommended)

## Trade-offs

| Aspect | ChromaDB | sqlite-vec |
|--------|----------|------------|
| Concurrent writers | ❌ Corruption | ✅ Safe |
| Search speed (10K) | ~5ms | ~10-20ms |
| Search speed (100K+) | ~20ms | ~50-100ms |
| Index type | HNSW | Brute/IVF |
| Dependencies | Heavy | Light |
| Recovery | Wipe & reindex | Automatic |
| Single file | No | Yes |

## Embedding Bottleneck Solution

**Problem**: Embedding is slow (~30-100ms per text). Large index operations block MCP and timeout.

**The daemon solved TWO problems:**
1. Concurrent writes → sqlite-vec fixes this
2. Non-blocking async operations → Still need job queue pattern

**Solution: In-Process Job Queue**

```python
# MCP tool returns immediately with job_id
@mcp.tool()
async def index_directory(dir_path: str) -> dict:
    job_id = str(uuid.uuid4())
    chunks = await chunk_directory(dir_path)  # Fast
    job_queue.put(Job(job_id, chunks))         # Queue
    return {"job_id": job_id, "status": "queued"}  # Return fast

# Background worker (asyncio.Task, same process)
async def embed_worker():
    while True:
        job = await job_queue.get()
        for batch in batched(job.chunks, 100):
            embeddings = embed_batch(batch)      # MLX
            await store_vectors(batch, embeddings)  # SQLite (safe!)
        job.status = "complete"

# Status check
@mcp.tool()
async def get_job_status(job_id: str) -> dict:
    return {"status": jobs[job_id].status, "progress": jobs[job_id].progress}
```

**Why this works with sqlite-vec but not ChromaDB:**
- SQLite handles concurrent writes from main thread + background task safely
- ChromaDB required separate daemon process, which caused corruption

**What we still need:**
- Job queue (in-memory asyncio.Queue, or SQLite table for persistence)
- Background worker (asyncio.Task)
- `get_job_status` tool

**What we can remove:**
- Separate daemon process
- Unix socket IPC protocol
- Daemon lifecycle management (launchd, pid files)

## Conclusion

The slight performance trade-off (~2x slower searches) is acceptable given:
- No more corruption risk
- Dramatically simpler architecture
- Single process, single database file
- Standard SQLite tooling for debugging/backup
- In-process job queue replaces complex daemon IPC
