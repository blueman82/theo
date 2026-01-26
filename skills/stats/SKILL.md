---
name: stats
description: Show Theo system statistics including indexed documents and memory counts. Use when user wants to see storage usage, document counts, or system overview. Triggers on "theo stats", "show stats", "how many memories", or explicit /stats command.
---

# Theo Statistics

Show unified statistics for indexed documents and memories.

## Arguments

```
/stats                    # Full dashboard
/stats --namespace=<ns>   # Filter by namespace
```

## Instructions

1. Call both stat tools:

   ```javascript
   // Get document index stats
   const indexStats = await theo.get_index_stats({});

   // Get memory counts (all types)
   const memoryCount = await theo.memory_count({});
   ```

2. If `--namespace` provided, also get filtered counts:
   ```javascript
   const nsCount = await theo.memory_count({
     namespace: "<namespace>"
   });
   ```

3. Format as a unified dashboard:

   ```
   ## Theo Statistics

   ### Document Index
   | Metric | Value |
   |--------|-------|
   | Total chunks | <total_documents> |
   | Unique files | <unique_sources> |
   | Namespaces | <list or count> |

   ### Memories
   | Type | Count |
   |------|-------|
   | Total | <count> |
   | Preferences | <count> |
   | Decisions | <count> |
   | Patterns | <count> |
   | Session | <count> |

   ---
   Last updated: <timestamp>
   ```

4. If namespace filtered, show:
   ```
   ### Namespace: <namespace>
   Documents: <count>
   Memories: <count>
   ```

5. Keep output compact and scannable.

## Examples

```
/stats
/stats --namespace=project:theo
/stats --namespace=global
```
