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

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_client import DaemonClient as DaemonClientType

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


def register_session_with_daemon(
    session_id: str,
    transcript_path: str,
    project_path: str,
) -> None:
    """Register session with Theo daemon for Agent Trace.

    Sends the active session information to the daemon so that
    Agent Trace can track which Claude Code session is currently active.

    Args:
        session_id: The unique session identifier from Claude Code.
        transcript_path: Path to the session's JSONL transcript file.
        project_path: Path to the project/working directory.
    """
    if DaemonClient is None:
        return

    try:
        # Extract model from transcript if available
        model_id: str | None = None
        transcript = Path(transcript_path)
        if transcript.exists():
            # Read first few lines to find model
            with transcript.open() as f:
                for line in f:
                    if '"model"' in line:
                        try:
                            entry = json.loads(line)
                            model_id = entry.get("model")
                            if model_id:
                                break
                        except json.JSONDecodeError:
                            pass

        with DaemonClient() as client:
            client.send(
                "set_active_session",
                session_id=session_id,
                transcript_path=transcript_path,
                model_id=model_id,
                project_path=project_path,
            )
    except Exception:
        pass  # Don't fail hook if daemon not running


def main() -> None:
    """Main hook entry point."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-session-start.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
        cwd = hook_input.get("cwd", str(Path.cwd()))
        # Handle camelCase and snake_case for transcript path
        transcript_path = hook_input.get("transcript_path") or hook_input.get(
            "transcriptPath", ""
        )

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

        # Register session with daemon for Agent Trace
        register_session_with_daemon(
            session_id=session_id,
            transcript_path=transcript_path,
            project_path=cwd,
        )

    except Exception as e:
        try:
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            with log_path.open("a") as f:
                f.write(f"{timestamp} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
