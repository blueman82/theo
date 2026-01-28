---
name: recall
description: Recall memories from Theo via semantic search with optional graph expansion. Use when user wants to find relevant memories, check what was remembered, or query knowledge. Triggers on "what do I know about", "recall", "find memories", or explicit /recall command.
---

# Recall Memory

Search and retrieve memories from Theo's semantic memory system.

## Arguments

```
/recall <query> [--type=<type>] [--namespace=<ns>] [--expand] [--limit=<n>]
```

- `query`: Semantic search query (required)
- `--type=<type>`: Filter by memory type (preference, decision, pattern, session, fact)
- `--namespace=<ns>`: Filter by namespace
- `--expand`: Include related memories via graph expansion
- `--limit=<n>`: Number of results (default: 5)

## Instructions

1. Parse the arguments from `$ARGUMENTS`:
   - Extract query (everything not a flag)
   - Check for `--type=X` flag
   - Check for `--namespace=X` flag
   - Check for `--expand` flag
   - Check for `--limit=X` flag

2. Call theo.memory_recall:
   ```javascript
   await theo.memory_recall({
     query: "<query>",
     memory_type: "<type>",        // optional filter
     namespace: "<namespace>",      // optional filter
     include_related: <expand>,     // default: false
     n_results: <limit>             // default: 5
   });
   ```

3. Format the result:

   **Success:**
   ```
   ## Memories matching "<query>"

   **[<memory_id>]** (<type>, confidence: 0.XX)
   <content>

   **[<memory_id>]** (<type>, confidence: 0.XX)
   <content>

   [If --expand and related found]:
   ### Related Memories
   **[<memory_id>]** via <relation_type>
   <content>
   ```

   **No results:**
   ```
   No memories found matching "<query>"

   Try:
   - Broader search terms
   - Different namespace (current: <ns>)
   - Removing type filter
   ```

   **Failure:**
   ```
   Failed to recall memories
   Error: <message>
   ```

4. Keep output concise but include memory IDs for reference.

## Graph Expansion

When using `--expand`, Theo follows relationship edges:
- `relates_to` - Conceptually related memories
- `supersedes` - Newer version of a memory
- `caused_by` - Causal relationships
- `contradicts` - Conflicting information

## Examples

```
/recall authentication
/recall coding preferences --type=preference
/recall project decisions --namespace=project:myapp --expand
/recall API patterns --type=pattern --limit=10
```
