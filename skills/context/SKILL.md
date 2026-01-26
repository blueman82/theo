---
name: context
description: Get formatted memory context for LLM injection. Use when needing relevant memories formatted as context block. Triggers on "get context", "memory context", "what do I know about", or explicit /context command.
---

# Memory Context

Fetch relevant memories formatted for LLM context injection.

## Arguments

```
/context <query>                 # Get context for topic
/context --namespace=<ns>        # Filter by namespace
/context --budget=<tokens>       # Set token budget (default: 4000)
```

- `query`: Topic or question to get context for
- `--namespace`: Filter to specific namespace
- `--budget`: Token budget for context (default: 4000)

## Instructions

1. Parse arguments:
   - Extract query (everything before flags)
   - Check for `--namespace=X` flag
   - Check for `--budget=X` flag

2. Call memory_context:
   ```javascript
   await theo.memory_context({
     query: "<query>",           // null if not provided
     namespace: "<namespace>",   // null if not provided
     token_budget: 4000          // or from --budget
   });
   ```

3. The tool returns pre-formatted markdown. Display it:

   ```
   ## Memory Context

   <formatted context from tool>

   ---
   Tokens: ~<token_estimate> / <budget>
   ```

4. If no relevant memories found:
   ```
   No relevant memories found for "<query>"
   ```

## Use Cases

- **Before a task**: Get all relevant context
- **For system prompts**: Inject personality/preferences
- **For code review**: Get coding standards memories
- **For decisions**: Get relevant past decisions

## Output Format

The tool returns markdown like:

```markdown
## Relevant Memories

### Preferences
- User prefers dark mode
- 2-space indentation

### Decisions
- Using FastAPI for backend
- PostgreSQL as database

### Patterns
- Always run tests before commit
```

## Examples

```
/context API design
/context --namespace=project:webapp
/context authentication --budget=2000
/context coding standards --namespace=global
```
