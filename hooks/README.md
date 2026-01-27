# Theo Hooks

Claude Code hooks for Theo integration.

## Files

| File | Purpose |
|------|---------|
| `theo-daemon.py` | Background daemon for non-blocking MLX embeddings |
| `theo_client.py` | Client library for communicating with daemon |
| `theo_worker.py` | Background worker for processing embedding queue |
| `theo-stop.py` | Stop hook - prompts for memory storage before session ends |
| `settings.example.json` | Example Claude Code MCP/hooks configuration |

## Setup

1. Copy hooks to your Claude Code hooks directory:
   ```bash
   cp hooks/*.py ~/.claude/hooks/
   ```

2. Copy and customize settings:
   ```bash
   cp hooks/settings.example.json ~/.claude/settings.json
   # Edit paths in settings.json
   ```

3. Update paths in `settings.json` to match your theo installation.

## MLX Threading Constraint

**CRITICAL**: The daemon runs MLX embeddings on the main thread by design. Do NOT use `asyncio.to_thread()` with MLX - Metal GPU operations are not thread-safe.

See `docs/architecture.md` for details.
