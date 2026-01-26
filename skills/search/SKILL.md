---
name: search
description: Semantic search across indexed documents in Theo. Use when user wants to find relevant documents, code, or notes. Triggers on "search for", "find in docs", "look up", or explicit /search command.
---

# Semantic Search

Search indexed documents using semantic similarity.

## Arguments

```
/search <query> [--n=<count>] [--namespace=<ns>] [--budget=<tokens>]
```

- `query`: Search query (required)
- `--n=<count>`: Number of results (default: 5)
- `--namespace=<ns>`: Filter to specific namespace
- `--budget=<tokens>`: Limit results to token budget

## Instructions

1. Parse arguments from `$ARGUMENTS`:
   - Extract the query (everything before flags)
   - Check for `--n=X` flag
   - Check for `--namespace=X` flag
   - Check for `--budget=X` flag

2. Choose the appropriate search tool:

   **With --budget flag:**
   ```javascript
   await theo.search_with_budget({
     query: "<query>",
     max_tokens: <budget>
   });
   ```

   **With --namespace flag:**
   ```javascript
   await theo.search_with_filters({
     query: "<query>",
     filters: { namespace: "<namespace>" },
     n_results: <n>  // default: 5
   });
   ```

   **Basic search:**
   ```javascript
   await theo.search({
     query: "<query>",
     n_results: <n>  // default: 5
   });
   ```

3. Format results as a ranked list:

   ```
   ## Results for "<query>"

   1. **<source_file>** (score: 0.85)
      > First 100 chars of content...

   2. **<source_file>** (score: 0.72)
      > First 100 chars of content...

   ---
   Found <total> results | <tokens> tokens
   ```

   If using budget mode, also show:
   ```
   Budget: <used>/<max> tokens
   ```

4. If no results found:
   ```
   No results found for "<query>"
   ```

5. Keep output scannable - truncate long content snippets.

## Examples

```
/search authentication flow
/search database connection --n=10
/search API endpoints --namespace=code
/search error handling --budget=2000
```
