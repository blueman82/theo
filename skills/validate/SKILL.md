---
name: validate
description: Record memory application and outcomes for the TRY-LEARN validation cycle. Use when testing if a memory works in practice. Triggers on "validate memory", "test this advice", "record outcome", or explicit /validate command.
---

# Validate Memory

Record memory usage and outcomes to build confidence through the TRY-LEARN cycle.

## Arguments

```
/validate apply <memory_id> <context>     # Start TRY phase
/validate outcome <memory_id> <success>   # Record LEARN phase
/validate <memory_id>                     # Interactive mode
```

- `memory_id`: ID of the memory to validate
- `context`: Description of how memory is being applied
- `success`: "yes"/"no" or "true"/"false"

## Instructions

### Mode 1: Apply (TRY phase)

If arguments start with "apply":

1. Extract memory_id and context
2. Call:
   ```javascript
   await theo.memory_apply({
     memory_id: "<id>",
     context: "<context>"
   });
   ```
3. Report:
   ```
   TRY phase started for memory <id>
   Context: <context>

   When done, run: /validate outcome <id> yes|no
   ```

### Mode 2: Outcome (LEARN phase)

If arguments start with "outcome":

1. Extract memory_id and success (parse yes/no/true/false)
2. Call:
   ```javascript
   await theo.memory_outcome({
     memory_id: "<id>",
     success: true|false,
     error_msg: "<optional error if failed>"
   });
   ```
3. Report:
   ```
   LEARN phase recorded for memory <id>
   Outcome: SUCCESS|FAILURE
   New confidence: 0.XX (+0.10 | -0.15)

   [If promoted to golden rule]:
   Memory promoted to GOLDEN RULE status (confidence >= 0.9)
   ```

### Mode 3: Interactive

If only memory_id provided:

1. First, recall the memory to show its content:
   ```javascript
   await theo.memory_recall({
     query: "<memory_id>",
     n_results: 1
   });
   ```

2. Show the memory and ask:
   ```
   Memory: <content>
   Current confidence: 0.XX

   What would you like to do?
   1. Apply this memory (start TRY phase)
   2. Record outcome (complete LEARN phase)
   ```

3. Guide based on response.

## The TRY-LEARN Cycle

```
TRY: Apply memory to a task
  |
  v
OBSERVE: Did it work?
  |
  v
LEARN: Record outcome
  |
  v
CONFIDENCE: Adjusts automatically
  - Success: +0.10
  - Failure: -0.15
  - At 0.9+: Golden rule
```

## Examples

```
/validate apply mem_abc123 "Using for API design"
/validate outcome mem_abc123 yes
/validate outcome mem_abc123 no "Caused type errors"
/validate mem_abc123
```
