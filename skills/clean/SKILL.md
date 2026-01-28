---
name: clean
description: Clean up indexed documents and chunks from Theo. Use when user wants to remove indexed files, clear namespaces, or reset the index. Triggers on "clean index", "remove indexed", "clear documents", or explicit /clean command.
---

# Clean Index

Remove indexed documents and chunks from Theo's search index.

## Arguments

```
/clean file <path>                    # Remove all chunks from a file
/clean chunks <id1> <id2> ...         # Remove specific chunk IDs
/clean all --confirm                  # Clear entire index (dangerous!)
```

- `path`: File path that was indexed
- `id1, id2, ...`: Specific chunk IDs to delete
- `--confirm`: Required for destructive operations

## Instructions

### Mode 1: Clean File

If arguments start with "file":

1. Extract the file path
2. Call:
   ```javascript
   await theo.delete_file({
     source_file: "<path>"
   });
   ```
3. Report:
   ```
   Cleaned: <path>
   Chunks deleted: <count>
   ```

### Mode 2: Clean Specific Chunks

If arguments start with "chunks":

1. Extract chunk IDs (space-separated)
2. Call:
   ```javascript
   await theo.delete_chunks({
     chunk_ids: ["<id1>", "<id2>", ...]
   });
   ```
3. Report:
   ```
   Deleted: <count> chunk(s)
   IDs: <id1>, <id2>, ...
   ```

### Mode 3: Clear All

If arguments start with "all":

1. Check for `--confirm` flag
2. If no --confirm, show warning:
   ```
   This will delete ALL indexed documents!

   Current index:
   - Documents: <count>
   - Unique sources: <count>

   Run with --confirm to proceed:
   /clean all --confirm
   ```

3. If --confirm present, call:
   ```javascript
   await theo.clear_index({
     confirm: true
   });
   ```
4. Report:
   ```
   Index cleared.
   Deleted: <count> documents
   ```

### Mode 4: Show Stats (no arguments)

If no arguments provided:

1. Call:
   ```javascript
   await theo.get_index_stats({});
   ```
2. Report:
   ```
   Index Status:
   - Total documents: <count>
   - Unique sources: <count>
   - Namespaces: <list>

   Use /clean file <path> to remove specific files.
   Use /clean all --confirm to clear everything.
   ```

## Safety

- `clear_index` requires `confirm: true` and `--confirm` flag
- No undo - deleted chunks cannot be recovered
- Re-index files after accidental deletion

## Examples

```
/clean
/clean file ./docs/old-readme.md
/clean file /Users/harrison/notes/draft.txt
/clean chunks chunk_abc123 chunk_def456
/clean all --confirm
```
