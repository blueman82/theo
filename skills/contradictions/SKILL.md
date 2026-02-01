---
name: contradictions
description: Detect contradicting memories (redirects to /health --include-contradictions)
---

# Detect Contradictions

**This skill redirects to `/health` with contradiction detection enabled.**

## Instructions

When user invokes `/contradictions`, immediately call memory_analyze_health with contradiction detection:

```javascript
await theo.memory_analyze_health({
  include_contradictions: true,
  include_low_confidence: false,
  include_stale: false
});
```

Format output as:

```
## Contradiction Analysis

### Conflicts Found: <count>

1. **<source_id>** vs **<target_id>**
   - Source: "<content preview>"
   - Target: "<content preview>"
   - Similarity: 0.XX

2. ...

---
Note: Full health check available via `/health`
```

If no contradictions:
```
No contradictions found.

Tip: Run `/health` for full system analysis.
```
