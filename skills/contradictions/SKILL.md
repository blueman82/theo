---
name: contradictions
description: "DEPRECATED: Use /health --include-contradictions instead. Contradiction detection is now part of memory_analyze_health."
---

# Detect Contradictions (DEPRECATED)

This skill has been consolidated into `/health`.

## Migration

Use `/health` with the `--include-contradictions` flag:

```
/health --include-contradictions
```

Or use the API directly:

```javascript
await theo.memory_analyze_health({
  include_contradictions: true,
  include_low_confidence: true,
  include_stale: true
});
```

## Why Changed?

The `memory_detect_contradictions` tool was removed in Theo v2.0 API consolidation:
- Contradiction detection is now part of `memory_analyze_health`
- Reduces API surface (13 → 10 memory tools)
- Health check provides comprehensive analysis in one call

## See Also

- `/health` - Comprehensive memory health analysis
- `/validate` - TRY-LEARN validation cycle
