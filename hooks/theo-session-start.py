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
"""Claude Code SessionStart hook for recording session initiation.

Every session start MUST be recorded - no exceptions.

Usage:
    Configure in ~/.claude/settings.json:
    {
        "hooks": {
            "SessionStart": [
                {
                    "type": "command",
                    "command": "~/.claude/hooks/theo-session-start.py"
                }
            ]
        }
    }
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import DaemonClient for fast IPC
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from theo_client import DaemonClient, get_shared_client
except ImportError:
    DaemonClient = None  # type: ignore[misc, assignment]
    get_shared_client = None  # type: ignore[misc, assignment]


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


def main():
    """Main hook entry point."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-session-start.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
        cwd = hook_input.get("cwd", str(Path.cwd()))

        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace()
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        # Log the session start
        with log_path.open("a") as f:
            f.write(f"{timestamp} | SESSION_START | session={session_id} | namespace={namespace}\n")

        # ALWAYS store session start - no exceptions
        if get_shared_client is not None:
            try:
                client = get_shared_client()
                client.store(
                        content=f"Session started: {session_id} in {namespace} at {timestamp}",
                        namespace=namespace,
                        memory_type="session",
                        importance=0.2,
                        metadata={
                            "source": "theo-session-start",
                            "session_id": session_id,
                            "cwd": cwd,
                        },
                    )
            except Exception:
                pass  # Don't fail hook if storage fails

    except Exception as e:
        try:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            with log_path.open("a") as f:
                f.write(f"{timestamp} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
