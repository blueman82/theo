# MCP Tools API Reference

This document provides detailed specifications for all MCP tools exposed by the Theo server.

## Overview

Theo exposes 26 tools organized into nine categories:

**Document Indexing Tools**
1. `index_file` - Index a single document
2. `index_directory` - Batch index multiple documents

**Search Tools**
3. `search` - Basic semantic search
4. `search_with_filters` - Search with metadata filtering
5. `search_with_budget` - Token-budget aware search

**Memory Tools (Core)**
6. `memory_store` - Store a new memory
7. `memory_recall` - Recall memories by semantic search
8. `memory_forget` - Delete memories
9. `memory_context` - Generate context for LLM injection

**TRY/LEARN Cycle Tools**
10. `memory_apply` - Record memory application (TRY phase)
11. `memory_outcome` - Record outcome and adjust confidence (LEARN phase)

**Graph Relationship Tools**
12. `memory_relate` - Create relationships between memories
13. `memory_edge_forget` - Delete edges/relationships
14. `memory_inspect_graph` - Visualize memory graph structure

**Memory Inspection Tools**
15. `memory_count` - Count memories with filters
16. `memory_list` - List memories with pagination
17. `memory_list_namespaces` - List all namespaces with counts
18. `validation_history` - Get validation event history

**Validation Analysis Tools**
19. `memory_analyze_health` - Analyze memory system health

**Maintenance Tools**
20. `memory_backfill_edges` - Backfill orphan memory edges

**Management Tools**
21. `delete_chunks` - Delete specific chunks by ID
22. `delete_file` - Delete all chunks from a source file
23. `clear_index` - Clear entire collection (requires confirmation)
24. `get_index_stats` - Get collection statistics

**Agent Trace Tools**
25. `trace_query` - Query AI attribution via git blame
26. `trace_list` - List recorded traces

---

## Document Indexing Tools

### Tool: index_file

Index a single document into the vector database.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute or relative path to the document"
    },
    "namespace": {
      "type": "string",
      "description": "Namespace for organizing documents",
      "default": "default"
    }
  },
  "required": ["file_path"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Path to document. Supports: .md, .markdown, .txt, .py, .pdf |
| `namespace` | string | No | "default" | Namespace for organizing documents |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "chunks_created": 12,
    "source_file": "/path/to/document.md",
    "namespace": "default",
    "chunk_ids": ["chunk_001", "chunk_002", "..."]
  }
}
```

#### Error Codes

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `File not found: {path}` | File doesn't exist | Check file path |
| `Unsupported file type: {ext}` | Unknown extension | Use supported format |
| `Failed to read file: {error}` | Permission or encoding issue | Check file permissions |

---

### Tool: index_directory

Batch index multiple documents from a directory with deduplication.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "dir_path": {
      "type": "string",
      "description": "Directory path to scan for documents"
    },
    "recursive": {
      "type": "boolean",
      "description": "Scan subdirectories recursively",
      "default": true
    },
    "namespace": {
      "type": "string",
      "description": "Namespace for organizing documents",
      "default": "default"
    },
    "async_mode": {
      "type": "boolean",
      "description": "Queue chunks for fast background processing",
      "default": true
    }
  },
  "required": ["dir_path"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dir_path` | string | Yes | - | Directory path to scan |
| `recursive` | boolean | No | true | Include subdirectories |
| `namespace` | string | No | "default" | Namespace for all indexed files |
| `async_mode` | boolean | No | true | If True, queue chunks for fast background processing. If False, wait for all embeddings synchronously. |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "files_indexed": 15,
    "total_chunks": 187,
    "failed_files": [
      {"file": "/path/to/bad.pdf", "error": "Failed to extract content"}
    ],
    "skipped_duplicates": 3
  }
}
```

#### Behavior

- Automatically detects supported file formats (.md, .pdf, .txt, .py)
- Skips files that have already been indexed (based on SHA-256 hash)
- Continues processing on individual file failures
- Returns comprehensive statistics and error details

---

## Search Tools

### Tool: search

Perform semantic search across indexed documents and memories.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "n_results": {
      "type": "integer",
      "description": "Maximum number of results to return",
      "default": 5,
      "minimum": 1,
      "maximum": 50
    }
  },
  "required": ["query"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `n_results` | integer | No | 5 | Maximum results (1-50) |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "chunk_042",
        "content": "## Authentication Configuration\n...",
        "score": 0.89,
        "namespace": "docs",
        "doc_type": "document",
        "metadata": {
          "source_file": "/path/to/config.md",
          "chunk_index": 3,
          "header_path": "Configuration > Authentication"
        }
      }
    ],
    "total": 5,
    "query": "authentication config",
    "total_tokens": 235
  }
}
```

#### Error Cases

| Error | Cause | Solution |
|-------|-------|----------|
| `Query cannot be empty` | Empty or whitespace query | Provide non-empty query |

---

### Tool: search_with_filters

Search with metadata filtering for scoped retrieval.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "filters": {
      "type": "object",
      "description": "Metadata filters to apply",
      "properties": {
        "namespace": {"type": "string"},
        "doc_type": {"type": "string"},
        "source_file": {"type": "string"}
      }
    },
    "n_results": {
      "type": "integer",
      "default": 5
    }
  },
  "required": ["query", "filters"]
}
```

#### Filter Examples

**Filter by namespace**:
```json
{"filters": {"namespace": "project:myapp"}}
```

**Filter by document type**:
```json
{"filters": {"doc_type": "memory"}}
```

**Filter by source file**:
```json
{"filters": {"source_file": "/path/to/specific.md"}}
```

**Combined filters**:
```json
{"filters": {"namespace": "docs", "doc_type": "document"}}
```

---

### Tool: search_with_budget

Search and return results within a token budget.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Maximum tokens in results",
      "default": 3000
    }
  },
  "required": ["query", "max_tokens"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `max_tokens` | integer | Yes | - | Token budget limit |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "results": [...],
    "tokens_used": 1987,
    "budget_remaining": 13,
    "truncated": true,
    "truncated_count": 3
  }
}
```

#### Behavior

1. Retrieves candidate results (up to 50)
2. Ranks by similarity score
3. Returns top results fitting within token budget
4. Prioritizes highest-scoring chunks

---

## Memory Tools

### Tool: memory_store

Store a new memory with semantic indexing.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "The memory content text"
    },
    "memory_type": {
      "type": "string",
      "enum": ["preference", "decision", "pattern", "session", "fact"],
      "description": "Type of memory",
      "default": "session"
    },
    "namespace": {
      "type": "string",
      "description": "Scope of the memory",
      "default": "global"
    },
    "importance": {
      "type": "number",
      "description": "Importance score 0.0-1.0",
      "default": 0.5
    },
    "metadata": {
      "type": "object",
      "description": "Optional additional metadata"
    }
  },
  "required": ["content"]
}
```

#### Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `preference` | User preferences or settings | "User prefers dark mode" |
| `decision` | Design or implementation decisions | "Chose FastAPI over Flask for performance" |
| `pattern` | Recognized patterns or behaviors | "Always use type hints in Python code" |
| `session` | Session-related information | "Currently working on authentication feature" |
| `fact` | Factual information | "Project uses PostgreSQL 15" |

#### Namespace Formats

| Format | Description |
|--------|-------------|
| `global` | Available across all projects |
| `default` | Default namespace |
| `project:{name}` | Scoped to specific project |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "id": "mem_abc123",
    "content_hash": "sha256_...",
    "namespace": "global",
    "memory_type": "preference",
    "duplicate": false,
    "confidence": 0.3
  }
}
```

#### Deduplication

If a memory with identical content already exists, the response includes:
```json
{
  "success": true,
  "data": {
    "id": "mem_existing",
    "duplicate": true
  }
}
```

---

### Tool: memory_recall

Recall memories using semantic search.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query text"
    },
    "n_results": {
      "type": "integer",
      "description": "Maximum number of results",
      "default": 5
    },
    "namespace": {
      "type": "string",
      "description": "Filter by namespace"
    },
    "memory_type": {
      "type": "string",
      "description": "Filter by memory type"
    },
    "min_importance": {
      "type": "number",
      "description": "Minimum importance score filter"
    },
    "min_confidence": {
      "type": "number",
      "description": "Minimum confidence score filter"
    }
  },
  "required": ["query"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_abc123",
        "content": "User prefers dark mode for their IDE",
        "memory_type": "preference",
        "namespace": "global",
        "confidence": 0.7,
        "importance": 0.8,
        "score": 0.92
      }
    ],
    "total": 1,
    "query": "dark mode preferences",
    "filters": {
      "namespace": null,
      "memory_type": null
    }
  }
}
```

---

### Tool: memory_forget

Delete memories by ID or semantic search.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "Specific memory ID to delete"
    },
    "query": {
      "type": "string",
      "description": "Search query to find memories to delete"
    },
    "input_value": {
      "type": "string",
      "description": "Smart parameter - auto-detects if ID or query"
    },
    "namespace": {
      "type": "string",
      "description": "Filter deletion to specific namespace"
    },
    "n_results": {
      "type": "integer",
      "description": "Number of search results to delete in query mode",
      "default": 5
    },
    "force": {
      "type": "boolean",
      "description": "If true, allow deletion of golden rules",
      "default": false
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "deleted_ids": ["mem_abc123"],
    "deleted_count": 1,
    "protected_ids": ["mem_golden456"]
  }
}
```

#### Golden Rule Protection

Golden rules (confidence >= 0.9 or type = golden_rule) are protected:
- Without `force=true`: Protected memories are skipped, listed in `protected_ids`
- With `force=true`: All matching memories are deleted

---

### Tool: memory_context

Fetch relevant memories and format for context injection.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Optional search query to filter relevant memories"
    },
    "namespace": {
      "type": "string",
      "description": "Project namespace"
    },
    "token_budget": {
      "type": "integer",
      "description": "Maximum tokens for context",
      "default": 4000
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "context": "## Relevant Memories\n\n### Preferences\n- User prefers dark mode\n...",
    "token_estimate": 350,
    "memory_count": 5,
    "golden_rule_count": 2
  }
}
```

#### Context Format

Generated context is formatted as markdown:
```markdown
## Relevant Memories

### Golden Rules (High Confidence)
- Always use type hints in Python code [confidence: 0.95]

### Preferences
- User prefers dark mode for their IDE [confidence: 0.7]

### Decisions
- Chose FastAPI over Flask for performance [confidence: 0.6]
```

---

## TRY/LEARN Cycle Tools

### Tool: memory_apply

Record that a memory is being applied (TRY phase of validation loop).

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory being applied"
    },
    "context": {
      "type": "string",
      "description": "Description of how/where the memory is being applied"
    },
    "session_id": {
      "type": "string",
      "description": "Optional session identifier"
    }
  },
  "required": ["memory_id", "context"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "memory_id": "mem_abc123",
    "event_id": 42
  }
}
```

---

### Tool: memory_outcome

Record the outcome of a memory application (LEARN phase) and adjust confidence.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory that was applied"
    },
    "success": {
      "type": "boolean",
      "description": "Whether the application was successful"
    },
    "error_msg": {
      "type": "string",
      "description": "Optional error message if failed"
    },
    "session_id": {
      "type": "string",
      "description": "Optional session identifier"
    }
  },
  "required": ["memory_id", "success"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "memory_id": "mem_abc123",
    "outcome_success": true,
    "old_confidence": 0.4,
    "new_confidence": 0.5,
    "promoted": false
  }
}
```

---

## Graph Relationship Tools

### Tool: memory_relate

Create a relationship (edge) between two memories.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "source_id": {
      "type": "string",
      "description": "ID of the source memory"
    },
    "target_id": {
      "type": "string",
      "description": "ID of the target memory"
    },
    "relation": {
      "type": "string",
      "enum": ["relates_to", "supersedes", "caused_by", "contradicts"],
      "description": "Type of relationship"
    },
    "weight": {
      "type": "number",
      "description": "Edge weight (default: 1.0)",
      "default": 1.0
    }
  },
  "required": ["source_id", "target_id", "relation"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "edge_id": 123
  }
}
```

---

### Tool: memory_edge_forget

Delete edges (relationships) between memories.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "edge_id": {
      "type": "integer",
      "description": "Specific edge ID to delete"
    },
    "memory_id": {
      "type": "string",
      "description": "Memory ID to delete all connected edges"
    },
    "source_id": {
      "type": "string",
      "description": "Source memory ID for pair deletion"
    },
    "target_id": {
      "type": "string",
      "description": "Target memory ID for pair deletion"
    },
    "relation": {
      "type": "string",
      "description": "Filter by relation type"
    },
    "direction": {
      "type": "string",
      "enum": ["outgoing", "incoming", "both"],
      "description": "For memory_id mode: edge direction",
      "default": "both"
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "deleted_ids": [1, 2, 3],
    "deleted_count": 3
  }
}
```

---

### Tool: memory_inspect_graph

Inspect the graph structure around a memory node.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory to start inspection from"
    },
    "max_depth": {
      "type": "integer",
      "description": "Maximum number of hops to traverse",
      "default": 2
    },
    "direction": {
      "type": "string",
      "enum": ["outgoing", "incoming", "both"],
      "default": "both"
    },
    "edge_types": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional list of edge types to include"
    },
    "include_scores": {
      "type": "boolean",
      "description": "Compute relevance scores for paths",
      "default": true
    },
    "decay_factor": {
      "type": "number",
      "description": "Factor by which relevance decays per hop",
      "default": 0.7
    },
    "output_format": {
      "type": "string",
      "enum": ["json", "mermaid"],
      "default": "json"
    }
  },
  "required": ["memory_id"]
}
```

#### Response Schema

```json
{
  "success": true,
  "origin_id": "mem_abc123",
  "nodes": [
    {"id": "mem_abc123", "content": "...", "depth": 0, "relevance": 1.0}
  ],
  "edges": [
    {"id": 1, "source": "mem_abc123", "target": "mem_def456", "relation": "relates_to"}
  ],
  "paths": [
    {"path": ["mem_abc123", "mem_def456"], "total_relevance": 0.7}
  ],
  "stats": {
    "total_nodes": 5,
    "total_edges": 4,
    "max_depth_reached": 2
  }
}
```

---

## Memory Inspection Tools

### Tool: memory_count

Count memories with optional filters.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "namespace": {
      "type": "string",
      "description": "Filter by namespace"
    },
    "memory_type": {
      "type": "string",
      "description": "Filter by memory type"
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "count": 42,
    "filters": {
      "namespace": "global",
      "memory_type": null
    }
  }
}
```

---

### Tool: memory_list

List memories with filtering and pagination.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "namespace": {
      "type": "string",
      "description": "Filter by namespace"
    },
    "memory_type": {
      "type": "string",
      "description": "Filter by memory type"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results (default: 100, max: 1000)",
      "default": 100
    },
    "offset": {
      "type": "integer",
      "description": "Number of results to skip for pagination",
      "default": 0
    },
    "order_by": {
      "type": "string",
      "description": "Field to sort by",
      "default": "created_at"
    },
    "descending": {
      "type": "boolean",
      "description": "Sort in descending order",
      "default": true
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_abc123",
        "content": "...",
        "memory_type": "pattern",
        "confidence": 0.7,
        "created_at": "2026-01-26T12:00:00Z"
      }
    ],
    "total": 100,
    "limit": 100,
    "offset": 0
  }
}
```

---

### Tool: validation_history

Get validation event history for a memory.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory to get history for"
    },
    "event_type": {
      "type": "string",
      "description": "Filter by event type (applied, succeeded, failed)"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of events to return",
      "default": 50
    }
  },
  "required": ["memory_id"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "events": [
      {
        "id": 1,
        "event_type": "applied",
        "context": "Applying to authentication flow",
        "timestamp": "2026-01-26T12:00:00Z"
      }
    ],
    "summary": {
      "total_applications": 10,
      "success_count": 8,
      "failure_count": 2,
      "success_rate": 0.8
    }
  }
}
```

---

## Validation Analysis Tools

### Tool: memory_detect_contradictions

Detect memories that contradict a given memory.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory to check for contradictions"
    },
    "similarity_threshold": {
      "type": "number",
      "description": "Minimum similarity for considering",
      "default": 0.7
    },
    "create_edges": {
      "type": "boolean",
      "description": "Whether to create CONTRADICTS edges",
      "default": true
    }
  },
  "required": ["memory_id"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "contradictions": [
      {
        "memory_id": "mem_def456",
        "content": "...",
        "similarity": 0.85,
        "contradiction_reason": "States opposite preference"
      }
    ],
    "edges_created": 1
  }
}
```

---

### Tool: memory_check_supersedes

Check if a memory should supersede another.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the (potentially newer) memory to check"
    },
    "create_edge": {
      "type": "boolean",
      "description": "Whether to create SUPERSEDES edge",
      "default": true
    }
  },
  "required": ["memory_id"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "superseded_id": "mem_old123",
    "edge_created": true
  }
}
```

---

### Tool: memory_analyze_health

Analyze the health of memories in the system.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "namespace": {
      "type": "string",
      "description": "Limit analysis to specific namespace"
    },
    "include_contradictions": {
      "type": "boolean",
      "description": "Check for contradictions",
      "default": true
    },
    "include_low_confidence": {
      "type": "boolean",
      "description": "Find low-confidence memories",
      "default": true
    },
    "include_stale": {
      "type": "boolean",
      "description": "Find stale memories",
      "default": true
    },
    "stale_days": {
      "type": "integer",
      "description": "Days without validation to consider stale",
      "default": 30
    }
  }
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "contradictions": [
      {"id1": "mem_a", "id2": "mem_b", "similarity": 0.9}
    ],
    "low_confidence": [
      {"id": "mem_c", "confidence": 0.1}
    ],
    "stale": [
      {"id": "mem_d", "last_validated": "2025-11-01T00:00:00Z"}
    ],
    "recommendations": [
      "Resolve 2 contradictions",
      "Review 5 low-confidence memories"
    ]
  }
}
```

---

## Management Tools

### Tool: delete_chunks

Delete specific chunks by their IDs.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of chunk IDs to delete"
    }
  },
  "required": ["ids"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "deleted_count": 3,
    "deleted_ids": ["chunk_001", "chunk_002", "chunk_003"]
  }
}
```

---

### Tool: delete_file

Delete all chunks from a specific source file.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "source_file": {
      "type": "string",
      "description": "Source file path to delete chunks for"
    }
  },
  "required": ["source_file"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "deleted_count": 15,
    "source_file": "/Users/harrison/docs/outdated.md"
  }
}
```

---

### Tool: clear_index

Delete all documents from the collection. **Requires explicit confirmation.**

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "confirm": {
      "type": "boolean",
      "description": "Safety gate - must be true to proceed"
    }
  },
  "required": ["confirm"]
}
```

#### Response Schema

```json
{
  "success": true,
  "data": {
    "deleted_count": 1523
  }
}
```

#### Safety Gate

Without `confirm=true`:
```json
{
  "success": false,
  "error": "Safety check: Set confirm=True to delete all documents"
}
```

---

### Tool: get_index_stats

Retrieve statistics about the current collection.

#### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

This tool takes no parameters.

#### Response Schema

```json
{
  "success": true,
  "data": {
    "total_documents": 487,
    "unique_sources": 23,
    "source_files": [
      "/Users/harrison/docs/README.md",
      "/Users/harrison/docs/API.md"
    ],
    "namespaces": ["default", "global", "project:myapp"],
    "memory_count": 45,
    "golden_rule_count": 5
  }
}
```

---

## Error Handling

All tools return a consistent error response format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Error Codes

| Error | Cause | Solution |
|-------|-------|----------|
| `Server not initialized` | Server startup incomplete | Wait for initialization |
| `File not found: {path}` | File doesn't exist | Check file path |
| `Unsupported file type: {ext}` | Unknown extension | Use .md, .txt, .py, .pdf |
| `Query cannot be empty` | Empty search query | Provide non-empty query |
| `Invalid memory type: {type}` | Unknown memory type | Use valid memory type |
| `Memory not found: {id}` | Memory ID doesn't exist | Check memory ID |
| `Safety check: Set confirm=True` | clear_index without confirmation | Set confirm=True |

---

## Best Practices

### Indexing

1. **Use namespaces**: Organize documents by project or scope
2. **Batch index directories**: More efficient than individual files
3. **Re-index updated files**: Hash-based dedup handles unchanged files

### Searching

1. **Be specific**: Detailed queries return better results
2. **Use filters**: Narrow scope for faster, more relevant results
3. **Set appropriate budgets**: Balance context vs. token usage

### Memory Management

1. **Use appropriate types**: Match memory type to content purpose
2. **Validate regularly**: Build confidence through feedback
3. **Namespace for projects**: Use `project:{name}` for project-specific memories
4. **Respect golden rules**: Protected memories are constitutional principles

### Maintenance

1. **Monitor with stats**: Use `get_index_stats` to track collection growth
2. **Clean up obsolete files**: Use `delete_file` for outdated documents
3. **Use clear_index sparingly**: Requires confirmation for good reason
