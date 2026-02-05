#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "pydantic",
#     "pydantic-settings",
# ]
# ///
"""Claude Code / Factory Notification hook for memory system actions.

This hook runs when Claude Code displays notifications to the user.
It's informational only and cannot block the notification.

Notification types:
- idle_prompt: Claude has been idle for 60+ seconds
- permission_prompt: A permission dialog is being shown

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "Notification": [
                {
                    "matcher": "idle_prompt|permission_prompt",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-notify.py",
                            "timeout": 5
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
        "hook_event_name": "Notification",
        "message": "Claude needs your permission...",
        "notification_type": "permission_prompt"
    }

The hook processes notifications asynchronously and logs relevant events
for analytics and memory system maintenance recommendations.
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from theo_client import DaemonClient


def run_background(data_file: str) -> None:
    """Background worker - does actual notification processing."""
    try:
        data_path = Path(data_file)
        with data_path.open() as f:
            hook_input = json.load(f)
        data_path.unlink()
        _do_notification(hook_input)
    except Exception:
        pass


def read_hook_input() -> dict:
    """Read hook input from stdin.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable
    """
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
    """Derive project namespace from current working directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'
    """
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


def get_transcript_size(transcript_path: str | None) -> int:
    """Get the size of the transcript file in bytes.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Size in bytes, or 0 if unavailable
    """
    if not transcript_path:
        return 0

    try:
        path = Path(transcript_path).expanduser()
        if path.exists():
            return path.stat().st_size
    except Exception:
        pass

    return 0


def parse_permission_message(message: str) -> dict:
    """Parse permission prompt message to extract tool and action.

    Args:
        message: The permission prompt message

    Returns:
        Dictionary with parsed components: tool, action, details
    """
    result = {
        "tool": "unknown",
        "action": "unknown",
        "details": message,
    }

    # Common patterns in permission messages
    if "read" in message.lower():
        result["action"] = "read"
        # Extract file path if present
        for word in message.split():
            if "/" in word or "\\" in word:
                result["details"] = word
                break
    elif "write" in message.lower():
        result["action"] = "write"
        for word in message.split():
            if "/" in word or "\\" in word:
                result["details"] = word
                break
    elif "execute" in message.lower() or "run" in message.lower():
        result["action"] = "execute"
    elif "network" in message.lower():
        result["action"] = "network"

    # Extract tool name from common patterns
    if "bash" in message.lower():
        result["tool"] = "bash"
    elif "read" in message.lower() and "file" in message.lower():
        result["tool"] = "read"
    elif "write" in message.lower() and "file" in message.lower():
        result["tool"] = "write"
    elif "git" in message.lower():
        result["tool"] = "git"

    return result


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


def handle_idle_prompt(
    session_id: str,
    transcript_path: str | None,
    namespace: str,
    log_path: Path,
) -> None:
    """Handle idle_prompt notification.

    Args:
        session_id: Current session ID
        transcript_path: Path to transcript file
        namespace: Current namespace
        log_path: Path to log file
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    transcript_size = get_transcript_size(transcript_path)

    # Log idle event
    with log_path.open("a") as f:
        f.write(
            f"{timestamp} | IDLE | session={session_id} | "
            f"namespace={namespace} | transcript_size={transcript_size}\n",
        )

    # If transcript is large, log and store recommendation
    compaction_threshold = 5 * 1024 * 1024  # 5MB (matches main())
    if transcript_size > compaction_threshold:
        with log_path.open("a") as f:
            f.write(
                f"{timestamp} | AUTO_COMPACT | Transcript size {transcript_size} bytes "
                f"exceeds {compaction_threshold} bytes. Auto-compact triggered.\n",
            )

        # Store recommendation as memory for analytics
        try:
            with get_daemon_client() as client:
                client.store(
                    content=(
                        f"Session idle with large transcript ({transcript_size} bytes). "
                        f"Compaction recommended to improve performance."
                    ),
                    namespace=namespace,
                    memory_type="session",
                    importance=0.3,
                    metadata={
                        "source": "theo-notify",
                        "event": "idle_prompt",
                        "session_id": session_id,
                        "transcript_size": transcript_size,
                    },
                )
        except Exception as e:
            with log_path.open("a") as f:
                f.write(f"{timestamp} | ERROR | Failed to store memory: {e}\n")


def handle_permission_prompt(
    session_id: str,
    message: str,
    namespace: str,
    log_path: Path,
) -> None:
    """Handle permission_prompt notification.

    Args:
        session_id: Current session ID
        message: Permission prompt message
        namespace: Current namespace
        log_path: Path to log file
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    # Parse the permission message
    parsed = parse_permission_message(message)

    # Log permission request
    with log_path.open("a") as f:
        f.write(
            f"{timestamp} | PERMISSION | session={session_id} | "
            f"tool={parsed['tool']} | action={parsed['action']} | "
            f"details={parsed['details'][:100]}\n",
        )

    # Track permission patterns (store every 5th request to avoid spam)
    try:
        # Check if we should store this permission pattern
        # Use a simple counter based on log file line count
        with log_path.open() as f:
            permission_count = sum(1 for line in f if "| PERMISSION |" in line)

        # Store every 5th permission request for pattern analysis
        if permission_count % 5 == 0:
            with get_daemon_client() as client:
                client.store(
                    content=(
                        f"Permission requested: {parsed['action']} via {parsed['tool']}. "
                        f"Pattern: {parsed['details'][:100]}"
                    ),
                    namespace=namespace,
                    memory_type="session",
                    importance=0.2,
                    metadata={
                        "source": "theo-notify",
                        "event": "permission_prompt",
                        "session_id": session_id,
                        "tool": parsed["tool"],
                        "action": parsed["action"],
                    },
                )
    except Exception as e:
        with log_path.open("a") as f:
            f.write(f"{timestamp} | ERROR | Failed to store permission pattern: {e}\n")


def _do_notification(hook_input: dict) -> None:
    """Actual notification processing - runs in background.

    Args:
        hook_input: Hook input data from stdin
    """
    session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
    transcript_path = hook_input.get("transcript_path") or hook_input.get(
        "transcriptPath",
    )
    cwd = hook_input.get("cwd", str(Path.cwd()))
    notification_type = hook_input.get("notification_type", "unknown")
    message = hook_input.get("message", "")

    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-notify.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Change to session's working directory
        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace()

        # Route to appropriate handler based on notification type
        if notification_type == "idle_prompt":
            handle_idle_prompt(session_id, transcript_path, namespace, log_path)
        elif notification_type == "permission_prompt":
            handle_permission_prompt(session_id, message, namespace, log_path)
        else:
            # Unknown notification type - just log it
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            with log_path.open("a") as f:
                f.write(
                    f"{timestamp} | UNKNOWN | session={session_id} | "
                    f"type={notification_type} | message={message[:100]}\n",
                )

    except Exception as e:
        try:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            with log_path.open("a") as f:
                f.write(f"{timestamp} | ERROR | {e}\n")
        except Exception:
            pass


def main():
    """Main hook entry point - fire and forget.

    Notification hooks must not block, so we spawn a background process
    to do the actual work asynchronously.
    """
    # Handle background mode
    if len(sys.argv) > 2 and sys.argv[1] == "--background":
        run_background(sys.argv[2])
        return

    hook_input = read_hook_input()
    if not hook_input:
        return

    # For idle_prompt notifications, check if auto-compact needed
    notification_type = hook_input.get("notification_type", "")
    if notification_type == "idle_prompt":
        # Check transcript size for auto-compact
        transcript_path = hook_input.get("transcript_path")
        auto_compact_threshold = 5 * 1024 * 1024  # 5MB

        if transcript_path:
            try:
                transcript_size = Path(transcript_path).stat().st_size
                if transcript_size > auto_compact_threshold:
                    # Transcript is large - tell Claude to run /compact
                    size_mb = transcript_size / (1024 * 1024)
                    output = {
                        "additionalContext": (
                            f"⚠️ AUTO-COMPACT REQUIRED: Transcript is {size_mb:.1f}MB. "
                            f"Run /compact now to preserve context and improve performance."
                        )
                    }
                    print(json.dumps(output))
                    sys.stdout.flush()
                    # Skip the normal idle prompt - compact is more important
                    return
            except Exception:
                pass  # Continue with normal idle prompt if we can't check size

        # Normal idle prompt with memory tools
        reflection_prompt = """💭 IDLE - Memory Operations Available

While idle, evaluate pending learnings using the full memory toolkit:

## Available Memory Tools (via theo MCP wrapper)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store with semantic indexing + auto-deduplication |
| `memory_recall` | Semantic search with graph expansion |
| `memory_validate` | Adjust confidence based on success/failure |
| `memory_apply` | Record memory usage (TRY phase) |
| `memory_outcome` | Record result (LEARN phase) |
| `memory_relate` | Create relationships (supersedes, contradicts) |
| `memory_forget` | Delete with golden rule protection |

## Idle Time Actions

**Store new insights:**
```javascript
await theo.memory_store_tool({
  content: "...",
  memory_type: "pattern",  // preference, decision, golden_rule
  importance: 0.5,
  namespace: "project:name"
});
```

**Search for related memories before storing:**
```javascript
const related = await theo.memory_recall({ query: "...", n_results: 5 });
```

**Update memory that was just applied:**
```javascript
await theo.memory_outcome({
  memory_id: "...",
  success: true,
  outcome: "Applied successfully"
});
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]"""

        output = {"additionalContext": reflection_prompt}
        print(json.dumps(output))
        sys.stdout.flush()

    try:
        # Write hook input to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="theo-notify-")
        with os.fdopen(fd, "w") as f:
            json.dump(hook_input, f)

        # Spawn background process
        subprocess.Popen(
            [sys.executable, __file__, "--background", temp_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        # If background spawn fails, run synchronously (not ideal but works)
        _do_notification(hook_input)


if __name__ == "__main__":
    main()
