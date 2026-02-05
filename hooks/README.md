# Theo Hooks

Claude Code hooks for Theo integration - memory storage, context injection, and session management.

## Hook Files

### Core Daemon
| File | Purpose |
|------|---------|
| `theo-daemon.py` | Main background daemon for non-blocking MLX embeddings |
| `theo-daemon-ctl.py` | Daemon control (start/stop/status) |
| `theo_client.py` | Client library for daemon communication |
| `theo_worker.py` | Background worker for processing embedding queue |
| `theo_batcher.py` | Batch processing for embeddings |
| `theo_queue.py` | Queue management for async operations |

### Session Hooks
| File | Purpose |
|------|---------|
| `theo-stop.py` | Session end - prompts for memory storage |
| `theo_session_state.py` | Session state management |

### Context & Memory
| File | Purpose |
|------|---------|
| `theo-context.py` | Inject relevant memories into context |
| `theo-precontext.py` | Pre-context processing |
| `theo-capture.py` | Capture conversation for memory |
| `theo-compact.py` | Compact/summarize session on context limit |
| `theo-prompt.py` | Prompt enhancement with memories |

### Monitoring & Security
| File | Purpose |
|------|---------|
| `theo-monitor.py` | Health monitoring and alerts |
| `theo-notify.py` | Notification system |
| `theo-track.py` | Usage tracking |
| `theo-security.py` | Security checks |
| `theo-permissions.py` | Permission management |
| `theo-subagent.py` | Subagent coordination |

### Agent Trace
| File | Purpose |
|------|---------|
| `auto_commit.py` | Auto-commit + Agent Trace capture (agent-trace.dev spec) |

## Setup

1. Copy hooks to your Claude Code hooks directory:
   ```bash
   cp hooks/*.py ~/.claude/hooks/
   ```

2. Copy and customize settings:
   ```bash
   cp hooks/settings.example.json ~/.claude/settings.json
   # Edit paths in settings.json to match your installation
   ```

3. Start the daemon:
   ```bash
   ~/.claude/hooks/theo-daemon-ctl.py start
   ```

## Configuration

Edit `settings.example.json` and update:
- `/path/to/theo` - Your theo installation path
- Hook paths in the hooks section

## MLX Threading Constraint

**CRITICAL**: The daemon runs MLX embeddings on the main thread by design.
Do NOT use `asyncio.to_thread()` with MLX - Metal GPU operations are not thread-safe.

See `docs/architecture.md` for details.

## Hook Events

| Event | Hook | Purpose |
|-------|------|---------|
| `PrePromptSubmit` | `theo-daemon-ctl.py` | Ensure daemon running |
| `PreToolUse` | `theo-context.py` | Inject relevant context |
| `PostToolUse` | `theo-capture.py` | Capture for memory |
| `PostToolUse` | `auto_commit.py` | Auto-commit + Agent Trace (Write/Edit/MultiEdit) |
| `Stop` | `theo-stop.py` | Prompt memory storage |
| `Notification` | `theo-notify.py` | System notifications |
