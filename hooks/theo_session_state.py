#!/usr/bin/env python3
"""Session state manager for cross-hook communication.

Provides shared state between hooks via a JSON file at /tmp/theo-session-{id}.json.
This enables hooks to work as a distributed state machine instead of isolated handlers.

Key capabilities:
- Track pending errors for error-fix correlation
- Track detected preferences for auto-storage
- Track shown memories for apply/outcome cycle
- Checkpoint tracking for PreCompact and Stop hooks
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Type aliases
MemoryStatus = Literal["pending", "applied", "ignored"]


def get_state_path(session_id: str) -> Path:
    """Get the path to the session state file."""
    return Path(f"/tmp/theo-session-{session_id}.json")


def create_empty_state(session_id: str) -> dict[str, Any]:
    """Create an empty session state structure."""
    return {
        "session_id": session_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "shown_memories": {},  # memory_id -> {timestamp, context, status}
        "pending_errors": [],  # [{id, tool, error, timestamp, context, resolved}]
        "detected_preferences": [],  # [{content, detected_at, stored, memory_id}]
        "error_fix_pairs": [],  # [{error_id, fix_tool, fix_input, pattern, stored}]
        "checkpoints": {
            "pre_compact_done": False,
            "stop_reviewed": False,
            "memory_stored": False,  # Track if any memory was stored this session
            "memory_stored_count": 0,  # Count of memories stored
            "last_stop_memory_count": 0,  # Memory count at last stop check
            "stop_block_count": 0,  # How many times stop was blocked (loop breaker)
        },
    }


def load_session_state(session_id: str) -> dict[str, Any]:
    """Load session state from file, creating if doesn't exist.

    Args:
        session_id: The session ID

    Returns:
        Session state dictionary
    """
    state_path = get_state_path(session_id)

    if state_path.exists():
        try:
            with state_path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupted file, start fresh
            pass

    return create_empty_state(session_id)


def save_session_state(session_id: str, state: dict[str, Any]) -> None:
    """Save session state to file.

    Args:
        session_id: The session ID
        state: The state dictionary to save
    """
    state_path = get_state_path(session_id)
    state["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    try:
        # Write atomically via temp file
        temp_path = state_path.with_suffix(".tmp")
        with temp_path.open("w") as f:
            json.dump(state, f, indent=2)
        temp_path.replace(state_path)
    except OSError:
        pass  # Don't fail hooks on state save errors


def add_pending_error(
    session_id: str,
    tool: str,
    error: str,
    context: str = "",
    method: str = "unknown",
) -> str:
    """Add a pending error to track for error-fix correlation.

    Args:
        session_id: The session ID
        tool: The tool that produced the error (e.g., "Bash", "Edit")
        error: The error message (will be truncated)
        context: What Claude was trying to do
        method: Detection method ("authoritative", "semantic", "claude")

    Returns:
        The error ID for later reference
    """
    state = load_session_state(session_id)
    error_id = str(uuid.uuid4())[:8]

    state["pending_errors"].append({
        "id": error_id,
        "tool": tool,
        "error": error[:500],  # Truncate long errors
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "context": context[:200],
        "method": method,
        "resolved": False,
    })

    # Keep only last 10 errors to prevent bloat
    state["pending_errors"] = state["pending_errors"][-10:]

    save_session_state(session_id, state)
    return error_id


def get_unresolved_errors(session_id: str) -> list[dict[str, Any]]:
    """Get all unresolved errors for the session.

    Args:
        session_id: The session ID

    Returns:
        List of unresolved error dictionaries
    """
    state = load_session_state(session_id)
    return [e for e in state["pending_errors"] if not e["resolved"]]


def resolve_error(
    session_id: str,
    error_id: str,
    fix_tool: str,
    fix_input: str,
) -> dict[str, Any] | None:
    """Mark an error as resolved and generate a fix pattern.

    Args:
        session_id: The session ID
        error_id: The ID of the error to resolve
        fix_tool: The tool used to fix (e.g., "Edit", "Bash")
        fix_input: Summary of the fix input

    Returns:
        The generated pattern dict, or None if error not found
    """
    state = load_session_state(session_id)

    for error in state["pending_errors"]:
        if error["id"] == error_id and not error["resolved"]:
            error["resolved"] = True

            # Generate pattern
            pattern = {
                "error_id": error_id,
                "fix_tool": fix_tool,
                "fix_input": fix_input[:300],
                "pattern": _generate_pattern_description(error, fix_tool, fix_input),
                "stored": False,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            state["error_fix_pairs"].append(pattern)
            save_session_state(session_id, state)
            return pattern

    return None


def _generate_pattern_description(
    error: dict[str, Any],
    fix_tool: str,
    fix_input: str,
) -> str:
    """Generate a human-readable pattern description.

    Args:
        error: The error dictionary
        fix_tool: The tool used to fix
        fix_input: Summary of the fix

    Returns:
        Pattern description string
    """
    error_summary = error["error"][:100]
    fix_summary = fix_input[:100]

    return (
        f"When {error['tool']} fails with '{error_summary}', "
        f"fix using {fix_tool}: {fix_summary}"
    )


def mark_pattern_stored(session_id: str, error_id: str, memory_id: str = "") -> None:
    """Mark an error-fix pattern as stored in memory.

    Args:
        session_id: The session ID
        error_id: The error ID
        memory_id: The stored memory ID (if available)
    """
    state = load_session_state(session_id)

    for pair in state["error_fix_pairs"]:
        if pair["error_id"] == error_id:
            pair["stored"] = True
            pair["memory_id"] = memory_id
            break

    save_session_state(session_id, state)


def add_detected_preference(session_id: str, content: str) -> None:
    """Add a detected preference for potential storage.

    Args:
        session_id: The session ID
        content: The preference content (RFC 2119 format)
    """
    state = load_session_state(session_id)

    # Check for duplicates
    for pref in state["detected_preferences"]:
        if pref["content"] == content:
            return  # Already exists

    state["detected_preferences"].append({
        "content": content,
        "detected_at": datetime.now(tz=timezone.utc).isoformat(),
        "stored": False,
        "memory_id": None,
    })

    save_session_state(session_id, state)


def mark_preference_stored(
    session_id: str,
    content: str,
    memory_id: str = "",
) -> None:
    """Mark a preference as stored in memory.

    Args:
        session_id: The session ID
        content: The preference content
        memory_id: The stored memory ID
    """
    state = load_session_state(session_id)

    for pref in state["detected_preferences"]:
        if pref["content"] == content:
            pref["stored"] = True
            pref["memory_id"] = memory_id
            break

    save_session_state(session_id, state)


def get_unstored_preferences(session_id: str) -> list[dict[str, Any]]:
    """Get all detected but unstored preferences.

    Args:
        session_id: The session ID

    Returns:
        List of unstored preference dictionaries
    """
    state = load_session_state(session_id)
    return [p for p in state["detected_preferences"] if not p["stored"]]


def add_shown_memory(
    session_id: str,
    memory_id: str,
    context: str = "",
) -> None:
    """Record that a memory was shown to Claude.

    Args:
        session_id: The session ID
        memory_id: The memory ID that was shown
        context: What Claude was doing when shown
    """
    state = load_session_state(session_id)

    state["shown_memories"][memory_id] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "context": context[:200],
        "status": "pending",
    }

    save_session_state(session_id, state)


def update_memory_status(
    session_id: str,
    memory_id: str,
    status: MemoryStatus,
) -> None:
    """Update the status of a shown memory.

    Args:
        session_id: The session ID
        memory_id: The memory ID
        status: New status ("pending", "applied", "ignored")
    """
    state = load_session_state(session_id)

    if memory_id in state["shown_memories"]:
        state["shown_memories"][memory_id]["status"] = status
        save_session_state(session_id, state)


def set_checkpoint(session_id: str, checkpoint: str, value: bool = True) -> None:
    """Set a checkpoint flag.

    Args:
        session_id: The session ID
        checkpoint: Checkpoint name ("pre_compact_done", "stop_reviewed", "memory_stored")
        value: Checkpoint value
    """
    state = load_session_state(session_id)
    state["checkpoints"][checkpoint] = value
    save_session_state(session_id, state)


def is_memory_stored(session_id: str) -> bool:
    """Check if any memory was stored in this session.

    Args:
        session_id: The session ID

    Returns:
        True if memory_stored checkpoint is set
    """
    state = load_session_state(session_id)
    return state.get("checkpoints", {}).get("memory_stored", False)


def increment_memory_count(session_id: str) -> int:
    """Increment the memory stored count.

    Args:
        session_id: The session ID

    Returns:
        New count value
    """
    state = load_session_state(session_id)
    checkpoints = state.get("checkpoints", {})
    count = checkpoints.get("memory_stored_count", 0) + 1
    checkpoints["memory_stored_count"] = count
    checkpoints["memory_stored"] = True  # Also set boolean for backward compat
    state["checkpoints"] = checkpoints
    save_session_state(session_id, state)
    return count


def has_new_memory_since_last_stop(session_id: str) -> bool:
    """Check if new memories were stored since last stop check.

    This enables per-response enforcement while preventing loops.

    Args:
        session_id: The session ID

    Returns:
        True if memory_stored_count > last_stop_memory_count
    """
    state = load_session_state(session_id)
    checkpoints = state.get("checkpoints", {})
    current_count = checkpoints.get("memory_stored_count", 0)
    last_stop_count = checkpoints.get("last_stop_memory_count", 0)
    return current_count > last_stop_count


def mark_stop_checked(session_id: str) -> None:
    """Mark that stop hook has checked, recording current memory count.

    Args:
        session_id: The session ID
    """
    state = load_session_state(session_id)
    checkpoints = state.get("checkpoints", {})
    current_count = checkpoints.get("memory_stored_count", 0)
    checkpoints["last_stop_memory_count"] = current_count
    checkpoints["stop_block_count"] = 0  # Reset block count on successful pass
    state["checkpoints"] = checkpoints
    save_session_state(session_id, state)


def increment_stop_block_count(session_id: str) -> int:
    """Increment the stop block count (for loop breaking).

    Args:
        session_id: The session ID

    Returns:
        New block count
    """
    state = load_session_state(session_id)
    checkpoints = state.get("checkpoints", {})
    count = checkpoints.get("stop_block_count", 0) + 1
    checkpoints["stop_block_count"] = count
    state["checkpoints"] = checkpoints
    save_session_state(session_id, state)
    return count


def get_stop_block_count(session_id: str) -> int:
    """Get current stop block count.

    Args:
        session_id: The session ID

    Returns:
        Current block count
    """
    state = load_session_state(session_id)
    return state.get("checkpoints", {}).get("stop_block_count", 0)


def get_session_summary(session_id: str) -> dict[str, Any]:
    """Get a summary of session state for Stop hook suggestions.

    Args:
        session_id: The session ID

    Returns:
        Summary dictionary with counts and top items
    """
    state = load_session_state(session_id)

    unresolved_errors = [e for e in state["pending_errors"] if not e["resolved"]]
    unstored_prefs = [p for p in state["detected_preferences"] if not p["stored"]]
    unstored_patterns = [p for p in state["error_fix_pairs"] if not p["stored"]]
    pending_memories = [
        m for m_id, m in state["shown_memories"].items() if m["status"] == "pending"
    ]

    return {
        "unresolved_errors": unresolved_errors[:5],
        "unstored_preferences": unstored_prefs[:5],
        "unstored_patterns": unstored_patterns[:5],
        "pending_memories": len(pending_memories),
        "checkpoints": state["checkpoints"],
        "has_learnings": bool(unresolved_errors or unstored_prefs or unstored_patterns),
    }


def cleanup_session(session_id: str) -> None:
    """Clean up session state file.

    Args:
        session_id: The session ID
    """
    state_path = get_state_path(session_id)
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        pass
