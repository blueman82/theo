#!/usr/bin/env python3
"""Git post-commit hook for Agent Trace.

Records AI attribution by linking commits to Claude conversations.
Queries Theo daemon for active session and stores trace in SQLite.
Optionally writes git notes for portability.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Add hooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def parse_diff_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Parse git diff output to extract line ranges per file.

    Parses @@ -old,count +new_start,new_count @@ hunks.
    Returns dict mapping file paths to list of (start_line, end_line) tuples.
    """
    file_ranges: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        # Match file header: +++ b/path/to/file.py
        if line.startswith("+++ b/"):
            current_file = line[6:]
            file_ranges[current_file] = []
        # Match hunk header: @@ -old,count +new_start,new_count @@
        elif line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                if count > 0:  # Only additions
                    end = start + count - 1
                    file_ranges[current_file].append((start, end))

    return file_ranges


def get_commit_info() -> tuple[str, dict[str, list[tuple[int, int]]]] | None:
    """Get current commit SHA and file line ranges."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get diff with minimal context for line range extraction
        diff = subprocess.run(
            ["git", "diff", "HEAD~1..HEAD", "--unified=0"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

        file_ranges = parse_diff_ranges(diff)
        return sha, file_ranges
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
    file_ranges: dict[str, list[tuple[int, int]]],
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
            file_ranges=file_ranges,
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

    commit_sha, file_ranges = commit_info

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
        file_ranges=file_ranges,
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
