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
"""Claude Code / Factory PostToolUse hook for tracking tool operations.

This hook runs after tool calls complete and records activity in theo.
Tracks file operations, searches, web fetches, and subagent tasks.

ENHANCED: Now provides feedback to Claude via additionalContext output.
MIGRATED: Uses DaemonClient for IPC with automatic subprocess fallback.

Supported tools:
    Task, Bash, Glob, Grep, Read, Write, Edit, MultiEdit,
    WebFetch, WebSearch, mcp__*

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or
    ~/.factory/settings.json (Factory)::

        {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Task|Bash|Glob|Grep|Read|Write|Edit|...",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python /path/to/theo-track.py",
                                "timeout": 5
                            }
                        ]
                    }
                ]
            }
        }

Input (via stdin JSON):
    {
        "tool_name": "Write",
        "tool_input": {"file_path": "/path/to/file.py", "content": "..."},
        "tool_response": {"success": true, "filePath": "/path/to/file.py"},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

Output (to stdout JSON):
    {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "Note: file.py modified 4 times..."
        }
    }

The hook extracts relevant info and stores it in theo.
Failures are handled gracefully to avoid blocking the agent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Import DaemonClient for IPC communication
from theo_client import DaemonClient, get_shared_client

# Import session state and error detection for error-fix correlation
from theo_session_state import (
    add_pending_error,
    get_unresolved_errors,
    increment_memory_count,
    load_session_state,
    mark_pattern_stored,
    resolve_error,
    set_checkpoint,
)
from utils.error_detection import detect_error, extract_error_summary

# =============================================================================
# Constants
# =============================================================================

SESSION_ACTIVITY_FILE = Path("/tmp/claude-session-activity.json")

PROJECT_INDICATORS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    ".project",
    "pom.xml",
    "build.gradle",
)

FILE_TOOLS = frozenset({"Read", "Write", "Edit", "MultiEdit"})
SEARCH_TOOLS = frozenset({"Glob", "Grep"})
WEB_TOOLS = frozenset({"WebFetch", "WebSearch"})
MODIFICATION_ACTIONS = frozenset({"write", "edit", "multiedit"})
TEST_MARKERS = frozenset({"test_", "_test.", ".test.", ".spec."})

MOD_COUNT_THRESHOLD = 3
PATTERN_TRUNCATE_LENGTH = 50
URL_TRUNCATE_LENGTH = 100
COMMAND_TRUNCATE_LENGTH = 100
DESC_TRUNCATE_LENGTH = 50


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class HookInput:
    """Parsed hook input from Claude Code.

    Attributes:
        tool_name: Name of the tool that was invoked.
        tool_input: Input parameters passed to the tool.
        tool_response: Response returned by the tool.
        session_id: Current session identifier.
        cwd: Current working directory.
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_response: dict[str, Any]
    session_id: str | None
    cwd: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookInput:
        """Create HookInput from raw dictionary.

        Args:
            data: Raw hook input dictionary with camelCase or snake_case keys.

        Returns:
            Parsed HookInput instance.
        """
        import os

        return cls(
            tool_name=data.get("tool_name") or data.get("toolName", ""),
            tool_input=data.get("tool_input") or data.get("toolInput", {}),
            tool_response=data.get("tool_response") or data.get("toolResponse", {}),
            session_id=data.get("session_id") or data.get("sessionId"),
            cwd=data.get("cwd", os.getcwd()),
        )


# =============================================================================
# Input/Output Functions
# =============================================================================


def run_background(data_file: str) -> None:
    """Background worker that performs actual tracking work.

    Reads hook input from a temporary file, deletes it, then processes
    the tracking asynchronously.

    Args:
        data_file: Path to temporary JSON file containing hook input.
    """
    import json
    import os

    try:
        with open(data_file, encoding="utf-8") as f:
            hook_input = json.load(f)
        os.unlink(data_file)
        _do_tracking(hook_input)
    except Exception:
        pass


def read_hook_input() -> dict[str, Any]:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable.
    """
    import json

    if sys.stdin.isatty():
        return {}

    try:
        stdin_data = sys.stdin.read()
        if stdin_data:
            return json.loads(stdin_data)
    except (OSError, json.JSONDecodeError):
        pass

    return {}


# =============================================================================
# File Path Utilities
# =============================================================================


def get_file_type(file_path: str) -> str | None:
    """Get file type from extension.

    Args:
        file_path: Path to the file.

    Returns:
        File extension (e.g., '.py', '.ts') or None if no extension.
    """
    ext = Path(file_path).suffix.lower()
    return ext if ext else None


def extract_file_path(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Extract file path from tool input.

    Args:
        tool_name: Name of the tool (Read, Write, Edit, MultiEdit).
        tool_input: Tool input dictionary.

    Returns:
        File path or None if not found.
    """
    return tool_input.get("file_path")


def get_action(tool_name: str) -> str:
    """Convert tool name to action string.

    Args:
        tool_name: Name of the tool.

    Returns:
        Action string (read, write, edit, multiedit).
    """
    return tool_name.lower()


def find_project_root(file_path: str) -> str | None:
    """Find project root from file path.

    Walks up directory tree looking for project indicators like .git,
    pyproject.toml, package.json, etc.

    Args:
        file_path: Path to the file.

    Returns:
        Project root path or None if not found.
    """
    try:
        current = Path(file_path).resolve().parent
        while current != current.parent:
            for indicator in PROJECT_INDICATORS:
                if (current / indicator).exists():
                    return str(current)
            current = current.parent
    except Exception:
        pass

    return None


# Note: Using get_shared_client() from theo_client.py for connection reuse


# =============================================================================
# Session Activity Tracking
# =============================================================================


def load_session_activity() -> dict[str, int]:
    """Load session activity from temp file.

    Returns:
        Dictionary mapping file paths to modification counts.
    """
    import json

    if not SESSION_ACTIVITY_FILE.exists():
        return {}

    try:
        with open(SESSION_ACTIVITY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_session_activity(activity: dict[str, int]) -> None:
    """Save session activity to temp file.

    Args:
        activity: Dictionary mapping file paths to modification counts.
    """
    import json

    try:
        SESSION_ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_ACTIVITY_FILE, "w", encoding="utf-8") as f:
            json.dump(activity, f)
    except Exception:
        pass


def update_session_activity(file_path: str, action: str) -> int:
    """Update session activity for a file.

    Args:
        file_path: Path to the file that was modified.
        action: Action performed (write, edit, multiedit).

    Returns:
        Total number of modifications to this file in this session.
    """
    if action not in MODIFICATION_ACTIONS:
        return 0

    activity = load_session_activity()
    activity[file_path] = activity.get(file_path, 0) + 1
    save_session_activity(activity)
    return activity[file_path]


# =============================================================================
# Test File Discovery
# =============================================================================


def _get_python_test_patterns(path: Path) -> list[Path]:
    """Get test file patterns for Python files.

    Args:
        path: Source file path.

    Returns:
        List of potential test file paths.
    """
    patterns = [
        path.parent / f"test_{path.name}",
        path.parent / path.name.replace(".py", "_test.py"),
    ]

    path_str = str(path)
    for src_dir in ("/src/", "/lib/"):
        if src_dir in path_str:
            tests_path = Path(path_str.replace(src_dir, "/tests/"))
            patterns.extend([
                tests_path.parent / f"test_{tests_path.name}",
                tests_path.parent / tests_path.name.replace(".py", "_test.py"),
            ])

    return patterns


def _get_js_ts_test_patterns(path: Path) -> list[Path]:
    """Get test file patterns for JavaScript/TypeScript files.

    Args:
        path: Source file path.

    Returns:
        List of potential test file paths.
    """
    suffix = path.suffix
    patterns = [
        path.parent / path.name.replace(suffix, f".test{suffix}"),
        path.parent / path.name.replace(suffix, f".spec{suffix}"),
    ]

    path_str = str(path)
    if "/src/" in path_str:
        for test_dir in ("/tests/", "/__tests__/"):
            tests_path = Path(path_str.replace("/src/", test_dir))
            patterns.extend([
                tests_path.parent / tests_path.name.replace(suffix, f".test{suffix}"),
                tests_path.parent / tests_path.name.replace(suffix, f".spec{suffix}"),
            ])

    return patterns


def find_test_file(source_file: str) -> str | None:
    """Find corresponding test file for a source file.

    Checks common test patterns:
        - /src/ -> /tests/ or /test/
        - file.py -> test_file.py or file_test.py
        - file.ts -> file.test.ts or file.spec.ts

    Args:
        source_file: Path to the source file.

    Returns:
        Path to test file if it exists, None otherwise.
    """
    path = Path(source_file)

    if any(marker in path.name for marker in TEST_MARKERS):
        return None

    test_patterns: list[Path] = []

    if path.suffix == ".py":
        test_patterns.extend(_get_python_test_patterns(path))
    elif path.suffix in {".ts", ".js", ".tsx", ".jsx"}:
        test_patterns.extend(_get_js_ts_test_patterns(path))

    for test_path in test_patterns:
        if test_path.exists():
            return str(test_path)

    return None


# =============================================================================
# Lint/Test Command Suggestions
# =============================================================================


def get_lint_command(file_path: str, file_type: str | None) -> str | None:
    """Get suggested lint/test command for a file.

    Args:
        file_path: Path to the file.
        file_type: File extension (e.g., '.py', '.ts').

    Returns:
        Suggested command or None.
    """
    if file_type is None:
        return None
    commands = {
        ".py": f"pytest {file_path}",
        ".ts": "npm run lint",
        ".js": "npm run lint",
        ".tsx": "npm run lint",
        ".jsx": "npm run lint",
        ".rs": "cargo test",
        ".go": "go test",
    }
    return commands.get(file_type)


# =============================================================================
# Feedback Generation
# =============================================================================


def generate_feedback(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    session_activity: dict[str, int],
) -> str | None:
    """Generate feedback context for Claude.

    Args:
        tool_name: Name of the tool used.
        tool_input: Tool input parameters.
        tool_response: Tool response.
        session_activity: Current session activity counts.

    Returns:
        Feedback string or None if no feedback needed.
    """
    file_path = extract_file_path(tool_name, tool_input)
    if not file_path:
        return None

    action = get_action(tool_name)
    if action not in MODIFICATION_ACTIONS:
        return None

    file_type = get_file_type(file_path)
    feedback_items: list[str] = []

    mod_count = session_activity.get(file_path, 0)
    if mod_count > MOD_COUNT_THRESHOLD:
        feedback_items.append(
            f"Note: {Path(file_path).name} modified {mod_count} times "
            "this session. Consider consolidating changes.",
        )

    test_path = find_test_file(file_path)
    if test_path:
        feedback_items.append(f"Reminder: Test file at {test_path} may need updates.")

    lint_cmd = get_lint_command(file_path, file_type)
    if lint_cmd:
        feedback_items.append(f"Consider running: {lint_cmd}")

    return "\n".join(feedback_items) if feedback_items else None


# =============================================================================
# Error-Fix Correlation
# =============================================================================


def process_error_fix_correlation(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
    session_id: str | None,
) -> dict[str, Any] | None:
    """Process tool result for error-fix correlation.

    This is the heart of the learning system:
    1. Detect if this tool result is an error → record it
    2. Detect if this is a success after a pending error → correlate and auto-store

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters
        tool_response: Tool response (dict or str)
        session_id: Current session ID

    Returns:
        Hook output dict with additionalContext/systemMessage, or None
    """
    if not session_id:
        return None

    # Detect error using cascade approach
    detection = detect_error(tool_name, tool_response)

    # Case 1: Authoritative error detected
    if detection["is_error"]:
        error_msg = detection["error_message"] or extract_error_summary(tool_response)
        context = _get_tool_context(tool_name, tool_input)

        error_id = add_pending_error(
            session_id=session_id,
            tool=tool_name,
            error=error_msg,
            context=context,
            method=detection["detection_method"],
        )

        return {
            "additionalContext": (
                f"⚠️ ERROR DETECTED (#{error_id})\n"
                f"Tool: {tool_name}\n"
                f"Error: {error_msg[:150]}\n\n"
                "When you fix this, the pattern will be auto-captured for learning."
            ),
        }

    # Case 2: Request Claude classification for ambiguous output
    if detection["needs_classification"]:
        return {
            "systemMessage": detection["classification_prompt"],
        }

    # Case 3: Success - check if this resolves a pending error
    unresolved = get_unresolved_errors(session_id)
    if unresolved:
        # Check if this success is related to a recent error
        for error in reversed(unresolved[-3:]):  # Check last 3 errors
            if _is_likely_fix(error, tool_name, tool_input):
                fix_summary = _get_tool_context(tool_name, tool_input)
                pattern = resolve_error(
                    session_id=session_id,
                    error_id=error["id"],
                    fix_tool=tool_name,
                    fix_input=fix_summary,
                )

                if pattern:
                    # AUTO-STORE the pattern
                    stored = _store_fix_pattern(pattern, session_id)
                    if stored:
                        mark_pattern_stored(session_id, error["id"])

                    return {
                        "additionalContext": (
                            f"✅ FIX PATTERN CAPTURED\n"
                            f"Error: {error['error'][:80]}...\n"
                            f"Fix: {fix_summary[:80]}\n"
                            f"Pattern stored for future reference."
                        ),
                    }

    return None


def check_and_mark_memory_stored(
    tool_name: str,
    tool_input: dict[str, Any],
    session_id: str | None,
) -> None:
    """Check if this tool call is a memory_store and mark checkpoint.

    This enables the Stop hook to know memory was stored without
    waiting for transcript flush.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters
        session_id: Current session ID
    """
    if not session_id:
        return

    # Check for mcp-exec wrapper with theo + memory_store
    # Handle all possible tool name formats (Claude Code uses mcp__, Factory uses mcp-)
    mcp_exec_names = {
        "mcp__mcp-exec__execute_code_with_wrappers",  # Claude Code format
        "mcp-exec__execute_code_with_wrappers",
        "mcp-exec___execute_code_with_wrappers",      # Factory format (triple underscore)
        "execute_code_with_wrappers",
    }
    if tool_name in mcp_exec_names:
        wrappers = tool_input.get("wrappers", [])
        code = tool_input.get("code", "")
        if isinstance(wrappers, list) and "theo" in wrappers:
            if "memory_store" in code:
                # Use increment_memory_count for per-response tracking
                increment_memory_count(session_id)
                return

    # Check for direct memory_store tool calls
    if tool_name in [
        "mcp__theo__memory_store",
        "mcp__theo__memory_store_tool",
        "memory_store",
        "memory_store_tool",
    ]:
        increment_memory_count(session_id)


def _get_tool_context(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Get a concise description of what the tool was doing.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters

    Returns:
        Concise context string
    """
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return f"Running: {cmd[:100]}"
    elif tool_name in {"Edit", "MultiEdit"}:
        file_path = tool_input.get("file_path", "")
        return f"Editing: {file_path}"
    elif tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        return f"Writing: {file_path}"
    elif tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        return f"Reading: {file_path}"
    elif tool_name == "Task":
        desc = tool_input.get("description", "")
        return f"Task: {desc[:100]}"
    else:
        return f"{tool_name}: {str(tool_input)[:100]}"


def _is_likely_fix(
    error: dict[str, Any],
    tool_name: str,
    tool_input: dict[str, Any],
) -> bool:
    """Determine if this successful tool use likely fixes a pending error.

    Heuristics:
    - Same tool type (Bash error → Bash fix)
    - Same file (Edit error → Edit fix on same file)
    - Related command (npm error → npm fix)

    Args:
        error: The pending error record
        tool_name: Current tool name
        tool_input: Current tool input

    Returns:
        True if this likely fixes the error
    """
    error_tool = error.get("tool", "")

    # Same tool type is a strong signal
    if tool_name == error_tool:
        return True

    # Edit after Bash error (common pattern: run command, fix file)
    if error_tool == "Bash" and tool_name in {"Edit", "Write"}:
        return True

    # Bash after Edit (common pattern: edit file, re-run command)
    if error_tool in {"Edit", "Write"} and tool_name == "Bash":
        return True

    return False


def _store_fix_pattern(pattern: dict[str, Any], session_id: str) -> bool:
    """Store a fix pattern in the theo memory system.

    Args:
        pattern: The pattern dict from resolve_error
        session_id: Current session ID

    Returns:
        True if stored successfully
    """
    try:
        with get_shared_client() as client:
            result = client.store(
                content=pattern["pattern"],
                namespace=_get_project_namespace_from_cwd(),
                memory_type="pattern",
                importance=0.7,  # Fix patterns are valuable
                metadata={
                    "source": "error-fix-correlation",
                    "error_id": pattern["error_id"],
                    "fix_tool": pattern["fix_tool"],
                    "session_id": session_id,
                    "auto_captured": True,
                },
            )
            return result.get("success", False)
    except Exception:
        return False


def _get_project_namespace_from_cwd() -> str:
    """Get project namespace from current working directory.

    Returns:
        Namespace string like 'project:name' or 'global'
    """
    cwd = Path.cwd()
    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return f"project:{cwd.name}"
    return "global"


# =============================================================================
# Activity Tracking Functions
# =============================================================================


def _write_log(log_path: Path, message: str) -> None:
    """Write a timestamped message to the log file.

    Args:
        log_path: Path to the log file.
        message: Message to log.
    """
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {message}\n")


def track_file_activity(
    tool_name: str,
    tool_input: dict[str, Any],
    session_id: str | None,
    cwd: str,
    log_path: Path,
) -> None:
    """Track file-related tool activity using DaemonClient.

    Args:
        tool_name: Name of the file tool.
        tool_input: Tool input parameters.
        session_id: Current session ID.
        cwd: Current working directory.
        log_path: Path to the log file.
    """
    file_path = extract_file_path(tool_name, tool_input)
    if not file_path:
        return

    action = get_action(tool_name)
    file_type = get_file_type(file_path)
    project_root = find_project_root(file_path) or cwd
    project_name = Path(project_root).name

    _write_log(log_path, f"{action} | {file_path} | project={project_root}")

    # Use DaemonClient for IPC with automatic subprocess fallback
    with get_shared_client() as client:
        import json

        # Store file activity with full content
        content_parts = [f"File {action}: {file_path}"]
        if file_type:
            content_parts[0] += f" ({file_type})"

        # Include actual file content/changes from tool_input
        if tool_input:
            # For Write: include content
            if "content" in tool_input:
                content_parts.append(f"Content:\n{tool_input['content']}")
            # For Edit: include old_string and new_string
            if "old_string" in tool_input:
                content_parts.append(f"Old:\n{tool_input['old_string']}")
            if "new_string" in tool_input:
                content_parts.append(f"New:\n{tool_input['new_string']}")
            # For other tools: include full input
            if "content" not in tool_input and "old_string" not in tool_input:
                content_parts.append(f"Input:\n{json.dumps(tool_input, indent=2)}")

        full_content = "\n\n".join(content_parts)

        result = client.store(
            content=full_content,
            namespace=f"project:{project_name}",
            memory_type="observation",
            importance=0.3,  # Low importance for routine file operations
            metadata={
                "source": "theo-track",
                "action": action,
                "file_path": file_path,
                "file_type": file_type,
                "session_id": session_id,
                "project_root": project_root,
            },
        )

        if not result.get("success"):
            _write_log(log_path, f"WARN: {result.get('error', 'unknown error')}")


def track_search_activity(
    tool_name: str,
    tool_input: dict[str, Any],
    session_id: str | None,
    cwd: str,
    log_path: Path,
) -> None:
    """Track search tool activity (Glob, Grep).

    Args:
        tool_name: Name of the search tool.
        tool_input: Tool input parameters.
        session_id: Current session ID.
        cwd: Current working directory.
        log_path: Path to the log file.
    """
    pattern = tool_input.get("pattern", "") or str(tool_input.get("patterns", []))
    path = tool_input.get("path", cwd)
    truncated_pattern = pattern[:PATTERN_TRUNCATE_LENGTH]

    _write_log(log_path, f"{tool_name.lower()} | pattern={truncated_pattern} | path={path}")


def track_web_activity(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    session_id: str | None,
    log_path: Path,
) -> None:
    """Track web tool activity (WebFetch, WebSearch).

    Args:
        tool_name: Name of the web tool.
        tool_input: Tool input parameters.
        tool_response: Tool response.
        session_id: Current session ID.
        log_path: Path to the log file.
    """
    if tool_name == "WebFetch":
        url = tool_input.get("url", "")[:URL_TRUNCATE_LENGTH]
        _write_log(log_path, f"webfetch | url={url}")
    elif tool_name == "WebSearch":
        query = tool_input.get("query", "")[:URL_TRUNCATE_LENGTH]
        _write_log(log_path, f"websearch | query={query}")


def track_task_activity(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    session_id: str | None,
    log_path: Path,
) -> None:
    """Track Task (subagent) activity.

    Args:
        tool_input: Tool input parameters.
        tool_response: Tool response.
        session_id: Current session ID.
        log_path: Path to the log file.
    """
    description = tool_input.get("description", "")[:DESC_TRUNCATE_LENGTH]
    subagent_type = tool_input.get("subagent_type", "")

    _write_log(log_path, f"task | type={subagent_type} | desc={description}")


def track_bash_activity(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    session_id: str | None,
    log_path: Path,
) -> None:
    """Track Bash command activity.

    Args:
        tool_input: Tool input parameters.
        tool_response: Tool response.
        session_id: Current session ID.
        log_path: Path to the log file.
    """
    command = tool_input.get("command", "")[:COMMAND_TRUNCATE_LENGTH]
    _write_log(log_path, f"bash | cmd={command}")


def track_mcp_activity(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    session_id: str | None,
    log_path: Path,
) -> None:
    """Track MCP tool activity.

    Args:
        tool_name: Full MCP tool name (mcp__server__tool format).
        tool_input: Tool input parameters.
        tool_response: Tool response.
        session_id: Current session ID.
        log_path: Path to the log file.
    """
    parts = tool_name.split("__")
    server = parts[1] if len(parts) >= 2 else "unknown"
    mcp_tool = parts[2] if len(parts) >= 3 else "unknown"

    _write_log(log_path, f"mcp | server={server} | tool={mcp_tool}")


# =============================================================================
# Core Tracking Logic
# =============================================================================


def _do_tracking(hook_input: dict[str, Any]) -> None:
    """Perform actual tracking work (runs in background).

    Routes tool activity to appropriate tracking function based on tool type.

    Args:
        hook_input: Raw hook input dictionary.
    """
    from datetime import datetime

    parsed = HookInput.from_dict(hook_input)

    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-track.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if parsed.tool_name in FILE_TOOLS:
            track_file_activity(
                parsed.tool_name,
                parsed.tool_input,
                parsed.session_id,
                parsed.cwd,
                log_path,
            )
        elif parsed.tool_name in SEARCH_TOOLS:
            track_search_activity(
                parsed.tool_name,
                parsed.tool_input,
                parsed.session_id,
                parsed.cwd,
                log_path,
            )
        elif parsed.tool_name in WEB_TOOLS:
            track_web_activity(
                parsed.tool_name,
                parsed.tool_input,
                parsed.tool_response,
                parsed.session_id,
                log_path,
            )
        elif parsed.tool_name == "Task":
            track_task_activity(
                parsed.tool_input,
                parsed.tool_response,
                parsed.session_id,
                log_path,
            )
        elif parsed.tool_name == "Bash":
            track_bash_activity(
                parsed.tool_input,
                parsed.tool_response,
                parsed.session_id,
                log_path,
            )
        elif parsed.tool_name.startswith("mcp__"):
            track_mcp_activity(
                parsed.tool_name,
                parsed.tool_input,
                parsed.tool_response,
                parsed.session_id,
                log_path,
            )

        # ALWAYS store tool activity - no exceptions
        # Full content, formatted for readability
        try:
            import json
            import os
            os.chdir(parsed.cwd)
            namespace = _get_project_namespace_from_cwd()

            # Tool field config using NamedTuple for clarity
            from typing import NamedTuple, Literal

            class ToolField(NamedTuple):
                field: str                              # JSON field name
                label: str                              # Display label
                source: Literal["inp", "resp", "resp_str"]  # Where to get value

            F = ToolField  # Shorthand

            TOOL_FIELDS: dict[str, list[ToolField]] = {
                "Bash": [F("command", "Command", "inp"), F("stdout", "Output", "resp"), F("stderr", "Stderr", "resp")],
                "Read": [F("file_path", "File", "inp"), F("content", "Content", "resp_str")],
                "Write": [F("file_path", "File", "inp"), F("content", "Content", "inp")],
                "Edit": [F("file_path", "File", "inp"), F("old_string", "Old", "inp"), F("new_string", "New", "inp")],
                "MultiEdit": [F("file_path", "File", "inp")],
                "Glob": [F("pattern", "Pattern", "inp"), F("path", "Path", "inp")],
                "Grep": [F("pattern", "Pattern", "inp"), F("path", "Path", "inp")],
                "WebFetch": [F("url", "URL", "inp"), F("prompt", "Prompt", "inp"), F("content", "Content", "resp")],
                "Task": [F("description", "Description", "inp"), F("subagent_type", "Agent", "inp"), F("prompt", "Prompt", "inp"), F("result", "Result", "resp")],
                "TaskOutput": [F("task_id", "Task ID", "inp")],
                "TaskStop": [F("task_id", "Task ID", "inp")],
                "AskUserQuestion": [],
                "NotebookEdit": [F("notebook_path", "Notebook", "inp"), F("cell_type", "Cell type", "inp"), F("new_source", "Source", "inp")],
                "Skill": [F("skill", "Skill", "inp"), F("args", "Args", "inp")],
                "EnterPlanMode": [],
                "ExitPlanMode": [],
                "TaskCreate": [F("subject", "Subject", "inp"), F("description", "Description", "inp")],
                "TaskUpdate": [F("taskId", "Task ID", "inp"), F("status", "Status", "inp"), F("subject", "Subject", "inp")],
                "TaskGet": [F("taskId", "Task ID", "inp")],
                "TaskList": [],
                "mcp__mcp-exec__execute_code_with_wrappers": [F("code", "Code", "inp")],
                "mcp__mcp-exec__list_available_mcp_servers": [],
                "mcp__mcp-exec__get_mcp_tool_schema": [F("server", "Server", "inp"), F("tool", "Tool", "inp")],
            }

            def format_tool_content(tool: str, inp: dict, resp: dict) -> str:
                """Format tool activity for human-readable storage."""
                parts = [f"Tool: {tool}"]

                # Check if tool has defined fields
                if tool in TOOL_FIELDS:
                    for tf in TOOL_FIELDS[tool]:
                        if tf.source == "inp" and inp.get(tf.field):
                            val = inp[tf.field]
                            if isinstance(val, list):
                                val = ", ".join(str(v) for v in val)
                            parts.append(f"{tf.label}:\n{val}" if "\n" in str(val) or len(str(val)) > 50 else f"{tf.label}: {val}")
                        elif tf.source == "resp" and resp.get(tf.field):
                            parts.append(f"{tf.label}:\n{resp[tf.field]}")
                        elif tf.source == "resp_str" and isinstance(resp, str):
                            parts.append(f"{tf.label}:\n{resp}")
                        elif tf.source == "resp_str" and resp.get("content"):
                            parts.append(f"{tf.label}:\n{resp['content']}")

                    # Special handling for specific tools
                    if tool == "MultiEdit" and inp.get("edits"):
                        for i, edit in enumerate(inp["edits"]):
                            parts.append(f"Edit {i+1} Old:\n{edit.get('old_string', '')}")
                            parts.append(f"Edit {i+1} New:\n{edit.get('new_string', '')}")

                    if tool in ("Glob", "Grep"):
                        if resp.get("files"):
                            parts.append("Matches:\n" + "\n".join(resp["files"][:50]))
                        elif resp.get("matches"):
                            parts.append(f"Matches:\n{resp['matches']}")

                    if tool == "AskUserQuestion":
                        if inp.get("questions"):
                            for q in inp["questions"]:
                                parts.append(f"Question: {q.get('question', '')}")
                        if resp.get("answers"):
                            parts.append(f"Answers: {resp['answers']}")

                    if tool == "mcp__mcp-exec__execute_code_with_wrappers":
                        if inp.get("wrappers"):
                            parts.insert(1, f"Wrappers: {', '.join(inp['wrappers'])}")
                        if isinstance(resp, list):
                            for item in resp:
                                if isinstance(item, dict) and item.get("text"):
                                    parts.append(f"Result: {item['text']}")

                elif tool == "WebSearch":
                    if inp.get("query"):
                        parts.append(f"Query: {inp['query']}")
                    if isinstance(resp, list):
                        links = []
                        summary = ""
                        for item in resp:
                            if isinstance(item, dict) and item.get("content"):
                                for link in item.get("content", []):
                                    if isinstance(link, dict) and link.get("title"):
                                        links.append(f"- {link['title']}")
                            elif isinstance(item, str) and len(item) > 100:
                                summary = item
                        if links:
                            parts.append("Links:\n" + "\n".join(links[:8]))
                        if summary:
                            parts.append(f"Summary:\n{summary[:2000]}")

                elif tool.startswith("mcp__"):
                    # Other MCP tools - extract key fields
                    for key in ["content", "query", "memory_type", "namespace", "memory_id", "importance"]:
                        if inp.get(key):
                            parts.append(f"{key}: {inp[key]}")
                    if isinstance(resp, list) and resp and isinstance(resp[0], dict) and resp[0].get("text"):
                        parts.append(f"Result: {resp[0]['text']}")

                else:
                    # Unknown tool - extract common fields or dump JSON
                    common = ["file_path", "path", "pattern", "query", "content", "command", "url"]
                    found = False
                    for key in common:
                        if inp.get(key):
                            parts.append(f"{key}: {inp[key]}")
                            found = True
                    if not found and inp:
                        parts.append(f"Input:\n{json.dumps(inp, indent=2)}")
                    if resp:
                        parts.append(f"Response:\n{json.dumps(resp, indent=2)}")

                return "\n\n".join(parts)

            full_content = format_tool_content(
                parsed.tool_name,
                parsed.tool_input or {},
                parsed.tool_response or {},
            )

            with get_shared_client() as client:
                client.store(
                    content=full_content,
                    namespace=namespace,
                    memory_type="session",
                    importance=0.1,  # Very low - high volume
                    metadata={
                        "source": "theo-track",
                        "tool_name": parsed.tool_name,
                        "session_id": parsed.session_id,
                    },
                )
        except Exception:
            pass  # Don't fail if storage fails

    except Exception as e:
        try:
            timestamp = datetime.now().isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | ERROR: {e}\n")
        except Exception:
            pass


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main hook entry point.

    In foreground mode:
        - Reads stdin
        - Updates session activity
        - Generates feedback
        - Outputs to stdout
        - Spawns background worker for tracking

    In background mode:
        - Performs actual tracking work
    """
    if len(sys.argv) > 2 and sys.argv[1] == "--background":
        run_background(sys.argv[2])
        return

    hook_input = read_hook_input()
    if not hook_input:
        return

    _process_foreground(hook_input)


def _process_foreground(hook_input: dict[str, Any]) -> None:
    """Process hook input in foreground mode.

    Generates feedback synchronously, then spawns background worker.
    Now includes error-fix correlation for learning capture.

    Args:
        hook_input: Raw hook input dictionary.
    """
    import json
    import os
    import subprocess
    import tempfile

    parsed = HookInput.from_dict(hook_input)
    output_parts: list[str] = []
    system_message: str | None = None

    try:
        # 1. Update session activity for file operations
        file_path = extract_file_path(parsed.tool_name, parsed.tool_input)
        if file_path:
            action = get_action(parsed.tool_name)
            update_session_activity(file_path, action)

        # 2. Generate standard feedback (test reminders, lint suggestions)
        session_activity = load_session_activity()
        feedback = generate_feedback(
            parsed.tool_name,
            parsed.tool_input,
            parsed.tool_response,
            session_activity,
        )
        if feedback:
            output_parts.append(feedback)

        # 3. Error-fix correlation (the learning system)
        error_fix_output = process_error_fix_correlation(
            parsed.tool_name,
            parsed.tool_input,
            parsed.tool_response,
            parsed.session_id,
        )
        if error_fix_output:
            if "additionalContext" in error_fix_output:
                output_parts.append(error_fix_output["additionalContext"])
            if "systemMessage" in error_fix_output:
                system_message = error_fix_output["systemMessage"]

        # 4. Track memory storage for Stop hook (prevents infinite loop)
        check_and_mark_memory_stored(
            parsed.tool_name,
            parsed.tool_input,
            parsed.session_id,
        )

        # 5. Output combined feedback
        if output_parts or system_message:
            output: dict[str, Any] = {}
            if output_parts:
                output["additionalContext"] = "\n\n".join(output_parts)
            if system_message:
                output["systemMessage"] = system_message
            print(json.dumps(output))

    except Exception:
        pass

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="theo-track-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(hook_input, f)

        subprocess.Popen(
            [sys.executable, __file__, "--background", temp_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        _do_tracking(hook_input)


if __name__ == "__main__":
    main()
