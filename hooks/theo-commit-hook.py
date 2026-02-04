#!/usr/bin/env python3
"""Git post-commit hook for Agent Trace.

Records AI attribution by linking commits to Claude conversations.
Queries Theo daemon for active session and stores trace in SQLite.
Optionally writes git notes for portability.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Add hooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def get_commit_info() -> tuple[str, list[str]] | None:
    """Get current commit SHA and changed files."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        files = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().split("\n")

        return sha, [f for f in files if f]
    except subprocess.CalledProcessError:
        return None


def get_active_session() -> dict | None:
    """Query Theo daemon for active Claude session."""
    try:
        from theo_client import DaemonClient

        with DaemonClient() as client:
            response = client.send("get_active_session")
            if response.get("success") and response.get("data", {}).get("active"):
                return response["data"]["session"]
    except Exception:
        pass
    return None


def write_trace(
    commit_sha: str,
    conversation_url: str,
    model_id: str | None,
    session_id: str | None,
    files: list[str],
) -> bool:
    """Write trace to SQLite storage."""
    try:
        from theo.storage.sqlite_store import SQLiteStore

        store = SQLiteStore()
        store.add_trace(
            commit_sha=commit_sha,
            conversation_url=conversation_url,
            model_id=model_id,
            session_id=session_id,
            files=files,
        )
        return True
    except Exception as e:
        print(f"Failed to write trace: {e}", file=sys.stderr)
        return False


def write_git_note(commit_sha: str, conversation_url: str, model_id: str | None) -> bool:
    """Write git note with attribution info."""
    note_content = json.dumps({
        "agent_trace": {
            "conversation": conversation_url,
            "model": model_id,
        }
    })
    try:
        subprocess.run(
            ["git", "notes", "add", "-f", "-m", note_content, commit_sha],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    """Main entry point for post-commit hook."""
    # Check if tracing is enabled
    if os.environ.get("THEO_TRACE_ENABLED", "true").lower() == "false":
        return 0

    # Get commit info
    commit_info = get_commit_info()
    if commit_info is None:
        return 0

    commit_sha, files = commit_info

    # Get active session from daemon
    session = get_active_session()
    if session is None:
        # No active Claude session - human commit, skip
        return 0

    # Write trace to SQLite
    write_trace(
        commit_sha=commit_sha,
        conversation_url=session.get("transcript_path", ""),
        model_id=session.get("model_id"),
        session_id=session.get("session_id"),
        files=files,
    )

    # Optionally write git notes
    if os.environ.get("THEO_TRACE_GIT_NOTES", "true").lower() == "true":
        write_git_note(
            commit_sha=commit_sha,
            conversation_url=session.get("transcript_path", ""),
            model_id=session.get("model_id"),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
