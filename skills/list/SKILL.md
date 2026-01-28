---
name: list
description: List and browse memories in Theo with pagination and filtering. Use when user wants to see all memories, browse by type, or paginate through stored knowledge. Triggers on "list memories", "show all memories", "browse memories", or explicit /list command.
---

# List Memories

Browse and paginate through memories in Theo.

## Arguments

```
/list [--type=<type>] [--namespace=<ns>] [--sort=<field>] [--limit=<n>] [--offset=<n>]
```

- `--type=<type>`: Filter by memory type (preference, decision, pattern, session, fact)
- `--namespace=<ns>`: Filter by namespace
- `--sort=<field>`: Sort by field (created_at, confidence, importance) - default: created_at
- `--limit=<n>`: Results per page (default: 10)
- `--offset=<n>`: Skip first N results for pagination (default: 0)

## Instructions

1. Parse the arguments from `$ARGUMENTS`:
   - Check for `--type=X` flag
   - Check for `--namespace=X` flag
   - Check for `--sort=X` flag
   - Check for `--limit=X` flag
   - Check for `--offset=X` flag

2. Call theo.memory_list:
   ```javascript
   await theo.memory_list({
     memory_type: "<type>",      // optional
     namespace: "<namespace>",   // optional
     sort_by: "<field>",         // default: "created_at"
     sort_order: "desc",
     limit: <limit>,             // default: 10
     offset: <offset>            // default: 0
   });
   ```

3. Format the result:

   **Success:**
   ```
   ## Memories [<offset+1>-<offset+count> of <total>]

   **<memory_id>** | <type> | conf: <confidence> | <namespace>
   <content preview - first 100 chars>

   **<memory_id>** | <type> | conf: <confidence> | <namespace>
   <content preview>

   ---
   Page <page> of <total_pages>
   Next: /list --offset=<next_offset>
   ```

   **Empty:**
   ```
   No memories found.

   [If filters applied]:
   Filters: type=<type>, namespace=<namespace>
   Try removing filters or storing some memories first.
   ```

   **Failure:**
   ```
   Failed to list memories
   Error: <message>
   ```

4. Include pagination hints for easy navigation.

## Sort Options

| Field | Description |
|-------|-------------|
| `created_at` | Most recent first (default) |
| `confidence` | Highest confidence first |
| `importance` | Most important first |

## Examples

```
/list
/list --type=preference
/list --namespace=project:myapp --limit=20
/list --sort=confidence --limit=5
/list --offset=10 --limit=10
/list --type=decision --sort=importance
```
