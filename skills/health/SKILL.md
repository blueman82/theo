---
name: health
description: Analyze memory system health in Theo. Finds contradictions, low-confidence, and stale memories. Triggers on "memory health", "check memories", "system health", or explicit /health command.
---

# Memory Health

Analyze the health of the memory system.

## Arguments

```
/health                          # Full health check
/health --namespace=<ns>         # Check specific namespace
/health --stale-days=60          # Custom stale threshold
/health --quick                  # Skip contradiction check (faster)
```

- `--namespace`: Limit to specific namespace
- `--stale-days`: Days without validation = stale (default: 30)
- `--quick`: Skip slow contradiction detection

## Instructions

1. Parse arguments:
   - Check for `--namespace=X` flag
   - Check for `--stale-days=X` flag
   - Check for `--quick` flag

2. Call memory_analyze_health:
   ```javascript
   await theo.memory_analyze_health({
     namespace: "<namespace>",           // null if not provided
     include_contradictions: true,       // false if --quick
     include_low_confidence: true,
     include_stale: true,
     stale_days: 30                       // or from --stale-days
   });
   ```

3. Format as health dashboard:

   ```
   ## Memory Health Report

   ### Summary
   | Metric | Status |
   |--------|--------|
   | Total memories | <count> |
   | Contradictions | <count> |
   | Low confidence | <count> |
   | Stale memories | <count> |
   | Health score | <X>/100 |

   ### Contradictions Found
   - <mem_a> vs <mem_b>: "<brief conflict>"
   - ...

   ### Low Confidence (< 0.5)
   - <id>: "<content snippet>" (conf: 0.3)
   - ...

   ### Stale Memories (> 30 days)
   - <id>: "<content snippet>" (last: 45 days ago)
   - ...

   ### Recommendations
   1. Resolve contradiction between X and Y
   2. Validate or remove low-confidence memory Z
   3. Review stale memories for relevance

   ---
   [Namespace: <namespace> | Checked: <timestamp>]
   ```

4. If everything healthy:
   ```
   ## Memory Health Report

   All systems healthy!

   - No contradictions
   - No low-confidence memories
   - No stale memories

   Total: <count> memories
   ```

## Health Score Calculation

- Start at 100
- -5 per contradiction pair
- -2 per low-confidence memory
- -1 per stale memory
- Minimum: 0

## Examples

```
/health
/health --namespace=project:webapp
/health --stale-days=60
/health --quick
```
