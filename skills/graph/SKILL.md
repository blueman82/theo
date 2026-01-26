---
name: graph
description: Visualize memory graph relationships in Theo. Shows edges, related memories, and connection strengths. Triggers on "show graph", "memory connections", "related memories", or explicit /graph command.
---

# Memory Graph

Visualize the relationship graph around a memory.

## Arguments

```
/graph <memory_id>                # Show graph (default: mermaid)
/graph <memory_id> --depth=3      # Traverse more hops
/graph <memory_id> --json         # JSON output instead
/graph <memory_id> --direction=outgoing  # Only outgoing edges
```

- `memory_id`: Starting memory node (required)
- `--depth`: Max hops to traverse (default: 2)
- `--json`: Output as JSON instead of Mermaid
- `--direction`: "outgoing", "incoming", or "both" (default)
- `--type`: Filter edge types (e.g., "supersedes", "contradicts")

## Instructions

1. Parse arguments:
   - Extract memory_id (first argument, required)
   - Check for `--depth=X` flag
   - Check for `--json` flag
   - Check for `--direction=X` flag
   - Check for `--type=X` flag

2. Call memory_inspect_graph:
   ```javascript
   await theo.memory_inspect_graph({
     memory_id: "<id>",
     max_depth: 2,              // or from --depth
     direction: "both",         // or from --direction
     edge_types: null,          // or [type] from --type
     include_scores: true,
     decay_factor: 0.7,
     output_format: "mermaid"   // or "json" if --json
   });
   ```

3. Format output:

   **Mermaid format (default):**
   ```
   ## Memory Graph: <memory_id>

   ### Origin
   > <memory content snippet>

   ### Graph
   ```mermaid
   <mermaid diagram from tool>
   ```

   ### Statistics
   - Nodes: <count>
   - Edges: <count>
   - Max depth reached: <depth>

   ### Paths
   | Path | Relevance |
   |------|-----------|
   | A → B → C | 0.49 |
   | A → D | 0.70 |
   ```

   **JSON format (--json):**
   ```
   ## Memory Graph: <memory_id>

   ### Nodes
   - <id>: "<content>" (type: <type>)
   - ...

   ### Edges
   - <from> --[<relation>]--> <to>
   - ...
   ```

4. If no connections:
   ```
   ## Memory Graph: <memory_id>

   No connections found.

   This memory is not linked to other memories.
   Use `/remember` with related context to build connections.
   ```

## Edge Types

- `supersedes`: This memory replaces an older one
- `contradicts`: These memories conflict
- `related`: General semantic relationship
- `supports`: This memory backs up another

## Examples

```
/graph mem_abc123
/graph mem_abc123 --depth=3
/graph mem_abc123 --json
/graph mem_abc123 --direction=outgoing --type=supersedes
```
