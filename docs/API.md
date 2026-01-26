# MCP Tools API Reference

This document provides detailed specifications for all MCP tools exposed by the Theo server.

## Overview

Theo exposes 25 tools organized into seven categories:

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
8. `memory_validate` - Validate and adjust memory confidence
9. `memory_forget` - Delete memories
10. `memory_context` - Generate context for LLM injection

**TRY/LEARN Cycle Tools**
11. `memory_apply` - Record memory application (TRY phase)
12. `memory_outcome` - Record outcome and adjust confidence (LEARN phase)

**Graph Relationship Tools**
13. `memory_relate` - Create relationships between memories
14. `memory_edge_forget` - Delete edges/relationships
15. `memory_inspect_graph` - Visualize memory graph structure

**Memory Inspection Tools**
16. `memory_count` - Count memories with filters
17. `memory_list` - List memories with pagination
18. `validation_history` - Get validation event history

**Validation Analysis Tools**
19. `memory_detect_contradictions` - Find contradicting memories
20. `memory_check_supersedes` - Check if memory supersedes another
21. `memory_analyze_health` - Analyze memory system health

**Management Tools**
22. `delete_chunks` - Delete specific chunks by ID
23. `delete_file` - Delete all chunks from a source file
24. `clear_index` - Clear entire collection (requires confirmation)
25. `get_index_stats` - Get collection statistics

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

### Tool: memory_validate

Validate a memory and adjust its confidence score.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "ID of the memory to validate"
    },
    "was_helpful": {
      "type": "boolean",
      "description": "Whether the memory was helpful"
    },
    "context": {
      "type": "string",
      "description": "Optional context describing the usage"
    }
  },
  "required": ["memory_id", "was_helpful"]
}
```

#### Confidence Adjustment

| Feedback | Adjustment | Resulting Range |
|----------|------------|-----------------|
| `was_helpful: true` | +0.1 | max 1.0 |
| `was_helpful: false` | -0.15 | min 0.0 |

#### Response Schema

```json
{
  "success": true,
  "data": {
    "memory_id": "mem_abc123",
    "old_confidence": 0.7,
    "new_confidence": 0.8,
    "promoted": false
  }
}
```

#### Golden Rule Promotion

When confidence reaches >= 0.9:
```json
{
  "success": true,
  "data": {
    "memory_id": "mem_abc123",
    "old_confidence": 0.85,
    "new_confidence": 0.95,
    "promoted": true,
    "new_type": "golden_rule"
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
