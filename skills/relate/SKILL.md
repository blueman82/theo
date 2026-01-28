---
name: relate
description: Create or remove relationships between memories in Theo's knowledge graph. Use when user wants to link memories, mark supersession, note contradictions, or manage memory connections. Triggers on "relate memories", "link memories", "connect", or explicit /relate command.
---

# Relate Memories

Manage relationships between memories in Theo's knowledge graph.

## Arguments

```
/relate <source_id> <relation> <target_id> [--weight=<0-1>]
/relate remove <edge_id>
/relate remove <source_id> <target_id>
/relate show <memory_id>
```

- `source_id`: Source memory ID
- `relation`: Relationship type (relates_to, supersedes, caused_by, contradicts)
- `target_id`: Target memory ID
- `--weight=<0-1>`: Relationship strength (default: 0.5)
- `edge_id`: Edge ID to remove

## Instructions

### Mode 1: Create Relationship

If arguments are `<source_id> <relation> <target_id>`:

1. Validate relation type is one of: relates_to, supersedes, caused_by, contradicts
2. Call:
   ```javascript
   await theo.memory_relate({
     source_id: "<source_id>",
     target_id: "<target_id>",
     relation_type: "<relation>",
     weight: <weight>  // default: 0.5
   });
   ```
3. Report:
   ```
   Created relationship:
   [<source_id>] --<relation>--> [<target_id>]
   Edge ID: <edge_id>
   Weight: <weight>
   ```

### Mode 2: Remove Relationship

If arguments start with "remove":

1. Parse: either edge_id or source_id + target_id pair
2. Call:
   ```javascript
   await theo.memory_edge_forget({
     edge_id: "<edge_id>",           // if single ID
     // OR
     source_id: "<source_id>",       // if pair
     target_id: "<target_id>"
   });
   ```
3. Report:
   ```
   Deleted: <count> edge(s)
   IDs: <edge_id1>, ...
   ```

### Mode 3: Show Relationships

If arguments start with "show":

1. Call:
   ```javascript
   await theo.memory_inspect_graph({
     memory_id: "<memory_id>",
     max_depth: 1,
     output_format: "json"
   });
   ```
2. Report:
   ```
   Relationships for [<memory_id>]:

   Outgoing:
   --<relation>--> [<target_id>]: <preview>

   Incoming:
   [<source_id>] --<relation>--> (this)
   ```

## Relationship Types

| Type | Meaning | Use For |
|------|---------|---------|
| `relates_to` | Conceptually related | General connections |
| `supersedes` | Replaces older memory | Updated information |
| `caused_by` | Causal relationship | Cause and effect |
| `contradicts` | Conflicting info | Inconsistencies |

## Examples

```
/relate mem_abc123 supersedes mem_old456
/relate mem_123 relates_to mem_456 --weight=0.8
/relate mem_a contradicts mem_b
/relate remove edge_xyz789
/relate remove mem_123 mem_456
/relate show mem_abc123
```
