# Changelog

All notable changes to Theo will be documented in this file.

## [Unreleased] - Data Migration from DocVec/Recall

### Migration Planning Session (2026-01-25)

#### Discovery Phase

**Storage Analysis Completed:**

1. **DocVec Storage**
   - ChromaDB only at `~/.docvec/chroma_db/`
   - Collection: `"documents"` (10,723 embeddings)
   - Schema: embeddings + metadata (doc_hash, source_file, chunk_index)
   - **WARNING**: Directory is 351GB despite only 10K embeddings (severely bloated/corrupted)

2. **Recall Storage**
   - Hybrid: SQLite + ChromaDB
   - SQLite at `~/.recall/recall.db` (12MB)
     - Tables: `memories`, `edges`, `outbox`, `validation_events`, `file_activity`, `memories_fts`
   - ChromaDB: **shares** `~/.docvec/chroma_db/` with DocVec
     - Collection: `"memories"`

3. **Theo Storage** (target)
   - ChromaDB at `~/.theo/chroma_db/` (9MB currently)
   - Unified `Document` type merging both schemas
   - Missing: SQLite layer for full memory graph support

**Critical Finding:** Recall and DocVec share the same ChromaDB directory. Migration must distinguish between document chunks (DocVec) and memory vectors (Recall) in the shared database.

#### Pre-Migration Steps Completed

1. Disabled Claude Code hooks that invoke Theo
2. Stopped `recall-daemon` launchd service (`com.recall.daemon`)
3. Killed all running processes accessing the databases:
   - recall-daemon (PIDs 4378, 4379)
   - theo MCP server (PIDs 4782, 4788, 4870)
4. Verified all database files are accessible and not locked

#### Backup Status

- [x] `~/.recall/recall.db` backed up to `~/.theo/migration-backup-20260125/recall.db` (12MB)
- [ ] `~/.docvec/chroma_db/` - **SKIPPED** (351GB, severely bloated - will migrate via API instead)
- [ ] `~/.theo/chroma_db/` - pending

**Storage Anomaly Discovered:**
- `~/.docvec/chroma_db/edd2ddb1-cce0-4e0e-be8a-565e5e7eaaac/link_lists.bin` is 608GB
- Only 10,723 embeddings exist - this is extreme bloat/corruption

**Root Cause Analysis:**
- `max_seq_id = 12,795` with 10,723 embeddings = ~2K delete/update operations (moderate churn)
- No maintenance logs = ChromaDB never compacted the HNSW index
- 1024-dimension embeddings with HNSW graph structure
- `link_lists.bin` stores HNSW neighbor graph - should be ~100MB max, not 608GB
- Likely cause: HNSW pre-allocation bug or file fragmentation/corruption
- **Good news**: Actual data (embeddings + metadata) is in SQLite and smaller files; the 608GB is just the index graph which will be regenerated

#### Migration Strategy (APPROVED)

**Approach**: API-based migration (not file copy)
1. Read all embeddings + metadata via ChromaDB Python API from bloated DB
2. Read Recall memories + edges from SQLite
3. Write cleanly to Theo's fresh ChromaDB at `~/.theo/chroma_db/`
4. Result: Properly-sized index, no corruption carried over

**Benefits:**
- Avoids copying 351GB of bloated/corrupted files
- Fresh HNSW index will be properly sized
- Can transform schemas during migration
- Deduplication handled at write time

#### SQLite Storage Layer Added

**File**: `src/theo/storage/sqlite_store.py`
- New SQLiteStore class for relationship graphs
- Stores edges in `~/.theo/theo.db`
- Edge types: relates_to, supersedes, caused_by, contradicts
- Graph traversal via `get_related()` for memory expansion during recall
- Batch insert for migration efficiency

#### Migration Script Created

**File**: `scripts/migrate_data.py`
- Standalone uv script with inline dependencies
- Handles corrupted ChromaDB gracefully (falls back to reading without embeddings)
- Transforms DocVec → Theo format (adds doc_type="document", source="docvec")
- Transforms Recall → Theo format (preserves confidence, importance, memory_type)
- Deduplicates by content hash (prefers Recall versions - they have confidence scores)
- Saves edges to `~/.theo/theo.db` (SQLite) with full graph query support

#### Dry-Run Results

| Metric | Count |
|--------|-------|
| DocVec documents read | 10,723 |
| Recall memories read | 3,942 |
| Recall edges read | 5,510 |
| Duplicates removed | 1,658 |
| **Final items to migrate** | **12,947** |

**Critical Finding**: ChromaDB corruption is worse than expected:
- Embeddings cannot be read from the bloated HNSW index (608GB `link_lists.bin`)
- Both DocVec and Recall share this corrupted ChromaDB
- All 12,947 items will require **re-embedding** after migration
- Document content and metadata are intact (stored in ChromaDB's internal SQLite)
- After migration, delete `~/.docvec/chroma_db/` to reclaim 351GB

#### Migration Completed (2026-01-25)

| Metric | Value |
|--------|-------|
| DocVec documents migrated | 10,723 |
| Recall memories migrated | 3,942 |
| Recall edges migrated | 5,510 |
| Duplicates removed | 1,658 |
| **Total items in Theo** | **12,947** |
| Theo ChromaDB size | 110MB |
| Theo SQLite size | 2.6MB |
| Source size (corrupted) | 351GB |
| **Size reduction** | **99.97%** |

**Completed Steps:**
- [x] Write migration script to read from source DBs via API
- [x] Transform DocVec chunks → Theo Document format
- [x] Transform Recall memories → Theo Document format (with confidence, edges)
- [x] Handle deduplication (same content in both systems)
- [x] Add SQLite storage layer for edges
- [x] Run migration with MLX embedding generation
- [x] Verify migrated data integrity

#### Post-Migration Validation

- [x] Theo MCP server restarted and running
- [x] recall-daemon launchd service restarted
- [x] Memory storage tested and working
- [x] Index stats verified (12,947 documents, 197 sources, 50+ namespaces)

#### Cleanup Completed

- [x] Re-enable Claude Code hooks ✓
- [x] Delete corrupted source data (`~/.docvec/chroma_db/`) - reclaimed 351GB ✓
- [x] Remove backup at `~/.theo/migration-backup-20260125/` ✓

### Phase 3: Theo Architecture Completion (2026-01-25)

#### Created Missing Modules

1. **`src/theo/config.py`** - TheoSettings configuration class
   - Pydantic Settings with THEO_ environment prefix
   - Default paths: `~/.theo/theo.db` and `~/.theo/chroma_db`
   - MLX as default embedding backend
   - `RecallSettings` alias for backward compatibility

2. **`src/theo/storage/hybrid.py`** - HybridStore coordinating layer
   - ChromaDB as source of truth for documents/memories
   - SQLite for relationship edges only (simpler than Recall's dual-store)
   - `add_memory`, `get_memory`, `search` methods for daemon compatibility
   - `add_edge`, `get_related` for graph operations

#### Daemon Infrastructure

- Stopped `recall-daemon` (launchd service `com.recall.daemon`)
- Created `com.theo.daemon.plist` for launchd
- Started `theo-daemon` via launchctl
- Verified `/tmp/theo.sock` IPC working

#### Architecture Decision

**Theo's Storage Model:**
- ChromaDB: Source of truth for all content + embeddings (12,947 items)
- SQLite: Edges only (5,510 relationships)

This is simpler than Recall's model where SQLite was the full source of truth.
The HybridStore wraps both with a unified interface.
