#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "chromadb",
#     "httpx",
#     "pydantic",
#     "pydantic-settings",
# ]
# ///
"""Claude Code / Factory SubagentStop hook for tracking subagent (Task) results.

This hook runs when a subagent (Task tool call) finishes. It captures
the subagent's work for memory context, enabling better coordination
between subagents and the main agent.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SubagentStop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-subagent.py",
                            "timeout": 10
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "session_id": "abc123",
        "transcript_path": "/path/to/transcript.jsonl",
        "cwd": "/project/root",
        "permission_mode": "default",
        "hook_event_name": "SubagentStop",
        "stop_hook_active": false
    }

Output:
    - JSON with decision: "block" to continue, or nothing to allow stop
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from theo_client import DaemonClient


def read_hook_input() -> dict:
    """Read hook input from stdin."""
    try:
        if sys.stdin.isatty():
            return {}
        stdin_data = sys.stdin.read()
        if stdin_data:
            return json.loads(stdin_data)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def get_project_namespace() -> str:
    """Derive project namespace from current working directory."""
    cwd = str(Path.cwd())
    project_name = Path(cwd).name

    project_indicators = [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
    ]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"


def read_transcript_tail(transcript_path: str | None, lines: int = 30) -> str | None:
    """Read the last N lines of the transcript."""
    if not transcript_path:
        return None

    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None

        content = path.read_text()
        all_lines = content.strip().split("\n")
        return "\n".join(all_lines[-lines:])
    except Exception:
        return None


def extract_subagent_summary(transcript_tail: str) -> str | None:
    """Extract a summary of what the subagent accomplished.

    Parses the JSONL transcript to find the subagent's final text output.
    """
    if not transcript_tail:
        return None

    lines = transcript_tail.split("\n")
    text_outputs = []

    for line in lines:
        if not line.strip():
            continue

        # Parse JSONL line
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract text content from message
        message = entry.get("message", {})
        content = message.get("content", [])

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text and len(text) > 10:
                        text_outputs.append(text)

    if text_outputs:
        # Return last meaningful text output (the final response)
        return text_outputs[-1]

    return None


def get_daemon_client() -> DaemonClient:
    """Get a DaemonClient instance with default settings.

    Returns:
        Configured DaemonClient for IPC communication.
    """
    return DaemonClient(
        connect_timeout=2.0,
        request_timeout=5.0,
        auto_fallback=True,
    )


def record_subagent_activity(
    session_id: str,
    namespace: str,
    summary: str | None,
) -> None:
    """Record subagent activity for tracking."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-subagent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("a") as f:
            summary_preview = (
                (summary[:100] + "...") if summary and len(summary) > 100 else summary
            )
            f.write(
                f"{datetime.now(tz=timezone.utc).isoformat()} | SUBAGENT_STOP | session={session_id} | namespace={namespace} | summary={summary_preview}\n",
            )
    except Exception:
        pass


def main():
    """Main hook entry point.

    Tracks subagent completion for:
    1. Analytics on subagent usage
    2. Capturing subagent results for future reference
    3. Enabling memory sharing between subagents
    """
    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id") or hook_input.get(
            "sessionId",
            "unknown",
        )
        transcript_path = hook_input.get("transcript_path") or hook_input.get(
            "transcriptPath",
        )
        cwd = hook_input.get("cwd", str(Path.cwd()))
        stop_hook_active = hook_input.get("stop_hook_active", False)

        # Prevent infinite loops
        if stop_hook_active:
            return

        # Change to session's working directory
        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace()

        # Read recent transcript to understand what the subagent did
        transcript_tail = read_transcript_tail(transcript_path)
        summary = extract_subagent_summary(transcript_tail) if transcript_tail else None

        # Record the activity
        record_subagent_activity(session_id, namespace, summary)

        # ALWAYS store subagent activity - no exceptions
        with get_daemon_client() as client:
            if summary and len(summary) > 20:
                client.store(
                    content=f"Subagent completed: {summary}",
                    namespace=namespace,
                    memory_type="session",
                    importance=0.4,
                    metadata={
                        "source": "theo-subagent",
                        "session_id": session_id,
                        "is_subagent_result": True,
                    },
                )
            else:
                # FALLBACK: Always store at least a record
                client.store(
                    content=f"Subagent session {session_id} completed in {namespace}",
                    namespace=namespace,
                    memory_type="session",
                    importance=0.2,
                    metadata={
                        "source": "theo-subagent",
                        "session_id": session_id,
                        "is_subagent_result": True,
                        "fallback": True,
                    },
                )

        # Output memory reflection prompt for Claude
        reflection_prompt = """🤖 SUBAGENT COMPLETE - Memory Operations Available

Evaluate subagent results using the full memory toolkit:
- What subagent discovered (store if novel)
- Patterns during delegation (store for future)
- Insights for future tasks (store if actionable)

## Available Memory Tools (via theo MCP wrapper)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store with semantic indexing + auto-deduplication |
| `memory_recall` | Semantic search with graph expansion |
| `memory_validate` | Adjust confidence based on success/failure |
| `memory_apply` | Record memory usage (TRY phase) |
| `memory_outcome` | Record result (LEARN phase) |
| `memory_relate` | Create relationships (supersedes, contradicts) |

## Subagent Result Actions

**Store subagent insight:**
```javascript
await theo.memory_store_tool({
  content: "Subagent insight: ...",
  memory_type: "pattern",
  importance: 0.6,
  namespace: "project:name"
});
```

**If subagent validated a memory:**
```javascript
await theo.memory_validate({
  memory_id: "...",
  success: true,
  context: "Subagent confirmed this pattern"
});
```

**Link subagent discovery to existing memory:**
```javascript
await theo.memory_relate({
  source_id: "subagent_finding_id",
  target_id: "existing_memory_id",
  relation_type: "supports"  // or refines, contradicts
});
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]"""

        import json

        output = {"systemMessage": reflection_prompt}
        print(json.dumps(output))
        sys.stdout.flush()

    except BrokenPipeError:
        pass
    except Exception as e:
        # Log error but don't block
        try:
            log_path = (
                Path.home() / ".claude" / "hooks" / "logs" / "theo-subagent.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"{datetime.now(tz=timezone.utc).isoformat()} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
