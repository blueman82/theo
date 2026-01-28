---
name: forget
description: Delete memories from Theo by ID or semantic search. Use when user wants to remove outdated, incorrect, or unwanted memories. Triggers on "forget this", "delete memory", "remove memory", or explicit /forget command.
---

# Forget Memory

Delete memories from Theo's memory system.

## Arguments

```
/forget <id_or_query> [--force] [--confirm]
```

- `id_or_query`: Memory ID (mem_xxx) or semantic search query (required)
- `--force`: Force deletion of golden rules (confidence >= 0.9)
- `--confirm`: Skip confirmation prompt

## Instructions

1. Parse the arguments from `$ARGUMENTS`:
   - Extract id_or_query (everything not a flag)
   - Check for `--force` flag
   - Check for `--confirm` flag

2. Determine if input is an ID or query:
   - If starts with "mem_" → treat as memory ID
   - Otherwise → treat as semantic search query

3. If semantic search (not ID), first show what will be deleted:
   ```javascript
   const results = await theo.memory_recall({
     query: "<query>",
     n_results: 5
   });
   ```

   Show matches and ask for confirmation (unless --confirm):
   ```
   Found memories matching "<query>":

   [1] <memory_id>: <content preview>
   [2] <memory_id>: <content preview>

   Delete these memories? (Use --confirm to skip this prompt)
   ```

4. Call theo.memory_forget:
   ```javascript
   await theo.memory_forget({
     memory_id: "<id>",           // if specific ID
     query: "<query>",            // if semantic search
     force: <force>               // default: false
   });
   ```

5. Format the result:

   **Success:**
   ```
   Deleted: <count> memory(ies)
   IDs: <id1>, <id2>, ...
   ```

   **Protected (golden rules):**
   ```
   Protected: <count> golden rule(s) not deleted
   IDs: <id1>, <id2>

   Use --force to delete golden rules.
   ```

   **No matches:**
   ```
   No memories found matching "<query>"
   ```

   **Failure:**
   ```
   Failed to delete memory
   Error: <message>
   ```

## Golden Rules

Memories with confidence >= 0.9 are "golden rules" - proven reliable through validation. They are protected from accidental deletion.

Use `--force` to delete golden rules if truly needed.

## Examples

```
/forget mem_abc123
/forget "outdated API endpoint" --confirm
/forget old preferences --force
/forget mem_xyz789 --force
```
