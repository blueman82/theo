# Theo ChromaDB Corruption Root Cause Analysis

**Date**: 2026-01-27
**Status**: CRITICAL - Architecture flaw identified

## Problem

Theo experiences repeated ChromaDB/HNSW corruption (exit code 139/SIGSEGV) after unifying docvec and recall into a single system.

## Root Cause: Concurrent Writers

```
┌─────────────────────┐     ┌─────────────────────┐
│   THEO DAEMON       │     │   THEO MCP SERVER   │
│   (always running)  │     │   (spawned by mcp)  │
│                     │     │                     │
│  HybridStore        │     │  ChromaStore        │
│       ↓             │     │       ↓             │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          └─────────┬─────────────────┘
                    ↓
         ┌─────────────────────┐
         │  ~/.theo/chroma_db  │  ← TWO CONCURRENT WRITERS
         │  (HNSW index)       │  ← CORRUPTION!
         └─────────────────────┘
```

The theo daemon (hooks/theo-daemon.py line 99) imports `HybridStore` which opens ChromaDB.
The theo MCP server (__main__.py) also initializes `ChromaStore` on startup.

**Both processes write to the same ChromaDB database concurrently.**

ChromaDB's HNSW index is NOT designed for concurrent writers - this causes corruption.

## Why Old Architecture Worked

| Component | Old (Separate) | New (Theo) |
|-----------|----------------|------------|
| **docvec** | MCP → ChromaDB (docs only) | Combined into theo |
| **recall** | MCP → **SQLite** (memories) | Combined into theo |
| **Writers** | Single writer per DB | Two writers, same DB |

**Recall used SQLite for memory metadata** - SQLite handles concurrent access gracefully with WAL mode and proper locking.

**ChromaDB/HNSW does NOT** - it's designed for single-writer scenarios.

## Evidence

1. Memory 55608: "ChromaDB only allows one writer - a standalone process holding the SQLite lock will prevent meta-mcp from spawning its own instance"

2. Memory 55738: "Exit code 139 (segfault)... malformed inverted index for FTS5 table"

3. Memory 56026: "When FTS5 rebuild doesn't fix segfaults and collection is corrupted, full database wipe is needed"

## Solutions

### Option 1: Revert to Separate Architecture (SAFEST)
- docvec MCP → ChromaDB (documents)
- recall MCP → SQLite (memories) with separate ChromaDB for embeddings
- This worked reliably for months

### Option 2: Single Writer Design
- Daemon becomes the ONLY ChromaDB writer
- MCP server routes ALL writes through daemon socket
- MCP only reads from ChromaDB
- Requires architectural changes to theo

### Option 3: Different Database
- LanceDB: Supports concurrent writers, 100x faster at scale
- PostgreSQL + pgvector: Full ACID, concurrent access
- Requires migration

## Immediate Mitigation

Until fixed, do NOT run theo daemon and theo MCP server simultaneously:

1. Stop daemon before using MCP: `theo-daemon-ctl.py stop`
2. Or disable daemon auto-start: `theo-daemon-ctl.py launchd-unload`

## Files Involved

- `hooks/theo-daemon.py:99` - imports HybridStore (opens ChromaDB)
- `src/theo/__main__.py:227` - initializes ChromaStore (opens ChromaDB)
- `src/theo/storage/hybrid.py` - HybridStore class
- `src/theo/storage/chroma_store.py` - ChromaStore class
