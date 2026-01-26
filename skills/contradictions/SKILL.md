---
name: contradictions
description: Detect contradicting memories in Theo. Use when user wants to find conflicts, inconsistencies, or clean up memory. Triggers on "find contradictions", "check conflicts", "memory conflicts", or explicit /contradictions command.
---

# Detect Contradictions

Find memories that contradict each other.

## Arguments

```
/contradictions <memory_id>        # Check specific memory
/contradictions --scan             # Scan recent memories for conflicts
/contradictions --threshold=0.8    # Adjust similarity threshold
```

- `memory_id`: Specific memory to check (optional)
- `--scan`: Check multiple recent memories
- `--threshold`: Similarity threshold (default: 0.7)

## Instructions

### Mode 1: Check Specific Memory

If memory_id provided:

1. Call:
   ```javascript
   await theo.memory_detect_contradictions({
     memory_id: "<id>",
     similarity_threshold: 0.7,  // or from --threshold
     create_edges: true
   });
   ```

2. Format results:
   ```
   ## Contradictions for <memory_id>

   ### Memory Content
   > <original memory content>

   ### Conflicts Found: <count>

   1. **<conflicting_id>** (similarity: 0.85)
      > <conflicting content>
      Reason: <why they contradict>

   2. ...

   ---
   [If edges created]: Created CONTRADICTS edges in graph
   ```

3. If no contradictions:
   ```
   No contradictions found for <memory_id>
   ```

### Mode 2: Scan Recent Memories

If `--scan` flag:

1. First list recent memories:
   ```javascript
   const memories = await theo.memory_list({
     limit: 20,
     offset: 0
   });
   ```

2. For each memory, check contradictions:
   ```javascript
   for (const mem of memories) {
     await theo.memory_detect_contradictions({
       memory_id: mem.id,
       create_edges: true
     });
   }
   ```

3. Aggregate and report:
   ```
   ## Contradiction Scan Results

   Scanned: 20 memories
   Conflicts found: <count>

   ### Conflict Pairs
   1. <mem_a> vs <mem_b>
      - A: "<content a>"
      - B: "<content b>"

   2. ...

   ---
   Tip: Use `/forget <id>` to remove outdated memories
   ```

## Resolution Tips

When contradictions found, suggest:
- Keep the more recent memory
- Keep the one with higher confidence
- Use `/validate` to test which is correct
- Use `/forget` to remove the outdated one

## Examples

```
/contradictions mem_abc123
/contradictions --scan
/contradictions mem_xyz --threshold=0.8
```
