---
name: store
description: Store new memories in Theo with semantic indexing and auto-deduplication. Use when user wants to remember something, save a preference, record a decision, or note a pattern. Triggers on "remember this", "store memory", "save this", or explicit /store command.
---

# Store Memory

Store new memories in Theo's semantic memory system.

## Arguments

```
/store <content> [--type=<type>] [--importance=<0-1>] [--namespace=<ns>]
```

- `content`: The memory content to store (required)
- `--type=<type>`: Memory type (default: "fact")
  - `preference` - User preferences
  - `decision` - Decisions made
  - `pattern` - Coding/workflow patterns
  - `session` - Session learnings
  - `fact` - General facts
- `--importance=<0-1>`: Importance score (default: 0.5)
- `--namespace=<ns>`: Namespace for organization (default: "global")

## Instructions

1. Parse the arguments from `$ARGUMENTS`:
   - Extract content (everything not a flag)
   - Check for `--type=X` flag
   - Check for `--importance=X` flag
   - Check for `--namespace=X` flag

2. Call theo.memory_store:
   ```javascript
   await theo.memory_store({
     content: "<content>",
     memory_type: "<type>",      // default: "fact"
     importance: <importance>,    // default: 0.5
     namespace: "<namespace>"     // default: "global"
   });
   ```

3. Format the result:

   **Success (new memory):**
   ```
   Stored: <memory_id>
   Type: <type>
   Namespace: <namespace>
   Confidence: 0.30 (initial)
   ```

   **Success (duplicate detected):**
   ```
   Duplicate detected: <existing_memory_id>
   Content already exists in memory system.
   ```

   **Failure:**
   ```
   Failed to store memory
   Error: <message>
   ```

4. Keep output concise - no verbose explanations.

## Memory Types Guide

| Type | Use For | Example |
|------|---------|---------|
| `preference` | User likes/dislikes | "Prefers TypeScript over JavaScript" |
| `decision` | Choices made | "Using PostgreSQL for this project" |
| `pattern` | Recurring approaches | "Always run tests before committing" |
| `session` | Session learnings | "Fixed auth bug by checking token expiry" |
| `fact` | General knowledge | "API rate limit is 1000/hour" |

## Examples

```
/store Always use absolute imports in this project --type=pattern
/store User prefers dark mode --type=preference --importance=0.8
/store Decided to use Redis for caching --type=decision --namespace=project:myapp
/store The deploy script requires AWS credentials --type=fact
```
