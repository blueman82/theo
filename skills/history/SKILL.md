---
name: history
description: View validation history for a memory. Shows timeline of applications, outcomes, and confidence changes. Triggers on "validation history", "memory history", "show history for", or explicit /history command.
---

# Validation History

View the validation event history for a memory.

## Arguments

```
/history <memory_id>              # Show full history
/history <memory_id> --type=<t>   # Filter by event type
/history <memory_id> --limit=10   # Limit results
```

- `memory_id`: ID of the memory (required)
- `--type`: Filter to "applied", "succeeded", or "failed"
- `--limit`: Maximum events to show (default: 50)

## Instructions

1. Parse arguments:
   - Extract memory_id (first argument, required)
   - Check for `--type=X` flag
   - Check for `--limit=X` flag

2. Call validation_history:
   ```javascript
   await theo.validation_history({
     memory_id: "<id>",
     event_type: "<type>",    // null if not provided
     limit: 50                 // or from --limit
   });
   ```

3. Format as timeline:

   ```
   ## Validation History: <memory_id>

   ### Memory Content
   > <memory content snippet>

   ### Statistics
   | Metric | Value |
   |--------|-------|
   | Total applications | <count> |
   | Successes | <count> |
   | Failures | <count> |
   | Success rate | <X>% |
   | Current confidence | 0.XX |

   ### Timeline

   | Date | Event | Context | Result |
   |------|-------|---------|--------|
   | 2024-01-15 | applied | API design task | - |
   | 2024-01-15 | succeeded | - | conf +0.10 |
   | 2024-01-10 | applied | Code review | - |
   | 2024-01-10 | failed | Type errors | conf -0.15 |
   | ... | | | |

   ---
   Showing <count> of <total> events
   ```

4. If no history:
   ```
   ## Validation History: <memory_id>

   No validation events recorded.

   This memory has never been applied in practice.
   Use `/validate apply <id> <context>` to start tracking.
   ```

## Event Types

- `applied`: Memory was used in a task (TRY phase start)
- `succeeded`: Application worked (LEARN phase - success)
- `failed`: Application didn't work (LEARN phase - failure)

## Examples

```
/history mem_abc123
/history mem_abc123 --type=failed
/history mem_abc123 --limit=10
```
