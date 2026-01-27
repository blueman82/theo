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
"""Claude Code / Factory Stop hook for capturing session learnings.

This hook runs when the agent finishes responding. It can analyze the
conversation to identify important patterns, decisions, or issues that
should be remembered for future sessions.

Enhanced with intelligent continuation control to prevent stopping when work is incomplete.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-stop.py",
                            "timeout": 10
                        }
                    ]
                }
            ]
        }
    }

Environment Variables:
    THEO_STOP_CHECK_TODOS - Check for incomplete TODOs (default: true)
    THEO_STOP_CHECK_GIT - Check for uncommitted changes (default: false)
    THEO_STOP_CHECK_TESTS - Check if tests were run after code changes (default: false)
    THEO_STOP_ENFORCE_MEMORY - Enforce memory storage for significant sessions (default: true)
    THEO_PRUNE_ENABLED - Enable automatic transcript pruning (default: false)
    THEO_PRUNE_TRIGGER - Token count to trigger pruning (default: 140000)
    THEO_PRUNE_TARGET - Token count to prune down to (default: 50000)

Input (via stdin JSON):
    {
        "session_id": "abc123",
        "transcript_path": "/path/to/transcript.jsonl",
        "cwd": "/project/root",
        "permission_mode": "default",
        "hook_event_name": "Stop",
        "stop_hook_active": false
    }

Output:
    - JSON with decision: "block" to continue, or nothing to allow stop
    - If blocking, must provide "reason" for the agent to continue
"""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

# =============================================================================
# PRUNING CONFIGURATION
# =============================================================================

PRUNE_ENABLED = os.getenv("THEO_PRUNE_ENABLED", "false").lower() == "true"
PRUNE_TRIGGER = int(os.getenv("THEO_PRUNE_TRIGGER", "140000"))
PRUNE_TARGET = int(os.getenv("THEO_PRUNE_TARGET", "50000"))


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class Checkpoint(TypedDict):
    """Represents a checkpoint in the transcript (user message with token count)."""

    line_index: int
    token_count: int


class PruneResult(TypedDict):
    """Result of a prune operation."""

    success: bool
    lines_before: int
    lines_after: int
    tokens_before: int
    tokens_after: int
    error: str | None


# =============================================================================
# PRUNING FUNCTIONS
# =============================================================================


def get_current_token_count(transcript_path: str) -> int:
    """Extract the latest token count from transcript.

    Scans the JSONL transcript for assistant entries with usage data,
    returning the most recent cache_read_input_tokens value.

    Args:
        transcript_path: Path to the JSONL transcript file.

    Returns:
        Latest token count, or 0 if not found or on error.
    """
    last_tokens = 0
    try:
        with Path(transcript_path).open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Token count is at entry['message']['usage']['cache_read_input_tokens']
                    message = entry.get("message")
                    if not isinstance(message, dict):
                        continue
                    usage = message.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    tokens = usage.get("cache_read_input_tokens", 0)
                    if isinstance(tokens, int) and tokens > 0:
                        last_tokens = tokens
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return last_tokens


def find_checkpoints(transcript_path: str) -> list[Checkpoint]:
    """Build list of checkpoints (user messages) with their token counts.

    A checkpoint is a user message entry paired with the token count
    from the subsequent assistant response. This mirrors the Esc-Esc
    rewind behavior in Claude Code.

    Args:
        transcript_path: Path to the JSONL transcript file.

    Returns:
        List of checkpoints sorted by line index (ascending).
    """
    checkpoints: list[Checkpoint] = []
    entries: list[tuple[int, dict]] = []

    try:
        with Path(transcript_path).open() as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    entries.append((i, entry))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return checkpoints

    # Track the last user message line index
    last_user_line: int | None = None

    for line_idx, entry in entries:
        entry_type = entry.get("type")

        if entry_type == "user":
            last_user_line = line_idx

        elif entry_type == "assistant" and last_user_line is not None:
            # Extract token count from assistant response
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            tokens = usage.get("cache_read_input_tokens", 0)
            if isinstance(tokens, int) and tokens > 0:
                checkpoints.append({
                    "line_index": last_user_line,
                    "token_count": tokens,
                })
                last_user_line = None  # Reset to avoid double-counting

    return checkpoints


def find_prune_checkpoint(checkpoints: list[Checkpoint], target_tokens: int) -> int | None:
    """Find the checkpoint line index to prune to.

    Scans checkpoints in reverse to find the most recent one where
    token count was at or below the target.

    Args:
        checkpoints: List of checkpoints from find_checkpoints().
        target_tokens: Target token count to prune down to.

    Returns:
        Line index to keep from, or None if no suitable checkpoint found.
    """
    for checkpoint in reversed(checkpoints):
        if checkpoint["token_count"] <= target_tokens:
            return checkpoint["line_index"]

    # If all checkpoints exceed target, use the earliest one
    if checkpoints:
        return checkpoints[0]["line_index"]

    return None


def prune_transcript(transcript_path: str, keep_from_line: int) -> PruneResult:
    """Prune transcript, keeping entries from keep_from_line onwards.

    Performs atomic write: writes to temp file, then renames.
    Fixes parentUuid of first kept entry to null to maintain integrity.

    Args:
        transcript_path: Path to the JSONL transcript file.
        keep_from_line: Line index to keep from (0-based).

    Returns:
        PruneResult with success status and statistics.
    """
    result: PruneResult = {
        "success": False,
        "lines_before": 0,
        "lines_after": 0,
        "tokens_before": 0,
        "tokens_after": 0,
        "error": None,
    }

    path = Path(transcript_path)
    temp_path = path.with_suffix(".jsonl.tmp")

    try:
        # Get token count before pruning
        result["tokens_before"] = get_current_token_count(transcript_path)

        # Read all lines
        lines = path.read_text().splitlines(keepends=True)
        result["lines_before"] = len(lines)

        # Keep from checkpoint onwards
        kept_lines = lines[keep_from_line:]
        result["lines_after"] = len(kept_lines)

        if not kept_lines:
            result["error"] = "No lines to keep after pruning"
            return result

        # Fix parentUuid of first entry to maintain chain integrity
        first_entry = json.loads(kept_lines[0])
        first_entry["parentUuid"] = None
        kept_lines[0] = json.dumps(first_entry) + "\n"

        # Atomic write: temp file then rename
        temp_path.write_text("".join(kept_lines))
        temp_path.rename(path)

        # Get token count after pruning
        result["tokens_after"] = get_current_token_count(transcript_path)
        result["success"] = True

    except json.JSONDecodeError as e:
        result["error"] = f"JSON decode error: {e}"
    except OSError as e:
        result["error"] = f"File error: {e}"
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return result


def log_prune_event(session_id: str, result: PruneResult) -> None:
    """Log prune event for analytics.

    Args:
        session_id: Current session ID.
        result: PruneResult from prune_transcript().
    """
    try:
        log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-prune.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        status = "SUCCESS" if result["success"] else "FAILED"
        error_msg = f" | error={result['error']}" if result["error"] else ""

        log_line = (
            f"{timestamp} | {status} | session={session_id} | "
            f"lines={result['lines_before']}→{result['lines_after']} | "
            f"tokens={result['tokens_before']}→{result['tokens_after']}"
            f"{error_msg}\n"
        )

        with log_path.open("a") as f:
            f.write(log_line)

    except OSError:
        pass


def check_and_prune(transcript_path: str, session_id: str) -> None:
    """Check token count and prune if over threshold.

    This is the main entry point for pruning logic. Called from main()
    after memory checks pass.

    Args:
        transcript_path: Path to the JSONL transcript file.
        session_id: Current session ID for logging.
    """
    if not PRUNE_ENABLED:
        return

    current_tokens = get_current_token_count(transcript_path)

    if current_tokens < PRUNE_TRIGGER:
        return  # No pruning needed

    # Find checkpoints and determine prune point
    checkpoints = find_checkpoints(transcript_path)
    if not checkpoints:
        return  # No valid checkpoints found

    prune_line = find_prune_checkpoint(checkpoints, PRUNE_TARGET)
    if prune_line is None:
        return  # No suitable prune point

    # Perform the prune
    result = prune_transcript(transcript_path, prune_line)
    log_prune_event(session_id, result)


# =============================================================================
# DAEMON CLIENT IMPORTS
# =============================================================================

# Import shared client for fast IPC with theo-daemon
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from theo_client import get_shared_client
except ImportError:
    get_shared_client = None  # type: ignore[misc, assignment]

# Import session state for specific suggestions
try:
    from theo_session_state import get_session_summary, set_checkpoint
    SESSION_STATE_AVAILABLE = True
except ImportError:
    SESSION_STATE_AVAILABLE = False

    def get_session_summary(_session_id: str) -> dict[str, Any]:
        """Stub when theo_session_state unavailable."""
        return {}

    def set_checkpoint(_session_id: str, _key: str, _value: Any) -> None:
        """Stub when theo_session_state unavailable."""


def read_hook_input() -> dict[str, Any]:
    """Read hook input from stdin.

    Returns:
        Parsed JSON input or empty dict on error.
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


def parse_jsonl_entries(transcript: str) -> list[dict[str, Any]]:
    """Parse JSONL transcript into list of entries.

    Args:
        transcript: Raw JSONL content.

    Returns:
        List of parsed JSON entries, skipping invalid lines.
    """
    entries: list[dict[str, Any]] = []
    for line in transcript.strip().split("\n"):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


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


def read_transcript_tail(transcript_path: str | None, lines: int = 50) -> str | None:
    """Read the last N lines of the transcript."""
    if not transcript_path:
        return None

    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None

        content = path.read_text()
        # Get last N lines
        all_lines = content.strip().split("\n")
        return "\n".join(all_lines[-lines:])
    except Exception:
        return None


def read_full_transcript(transcript_path: str | None) -> str | None:
    """Read the full transcript for comprehensive analysis."""
    if not transcript_path:
        return None

    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None

        return path.read_text()
    except Exception:
        return None


def analyze_for_learnings(transcript_tail: str) -> list[dict[str, str]]:
    """Analyze transcript for potential learnings.

    Returns list of dicts with type and content for memories to store.
    This is a lightweight heuristic analysis - heavy analysis is done in SessionEnd.
    """
    learnings: list[dict[str, str]] = []

    # Look for error patterns that were resolved
    if "error" in transcript_tail.lower() and (
        "fixed" in transcript_tail.lower() or "resolved" in transcript_tail.lower()
    ):
        # There was an error that got fixed - potential pattern to remember
        pass  # Let SessionEnd capture the full context

    # Look for explicit memory storage requests
    if "remember" in transcript_tail.lower() or "note that" in transcript_tail.lower():
        # User asked to remember something - SessionEnd will capture
        pass

    return learnings


def _extract_tool_uses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool_use items from an assistant entry.

    Args:
        entry: A transcript entry dict.

    Returns:
        List of tool_use content items, empty if not an assistant entry.
    """
    if entry.get("type") != "assistant":
        return []
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict) and item.get("type") == "tool_use"]


def _is_memory_store_call(tool_use: dict[str, Any]) -> bool:
    """Check if a tool_use item is a memory_store call.

    Args:
        tool_use: A tool_use content item.

    Returns:
        True if this is a theo memory_store call.
    """
    tool_name = tool_use.get("name", "")
    params = tool_use.get("input", {})

    # Check 1: mcp-exec wrapper with theo
    if tool_name == "mcp__mcp-exec__execute_code_with_wrappers":
        wrappers = params.get("wrappers", [])
        code = params.get("code", "")
        if isinstance(wrappers, list) and "theo" in wrappers and "memory_store" in code:
            return True

    # Check 2: Direct memory_store tool call
    if tool_name in ("mcp__theo__memory_store", "memory_store"):
        return True

    return False


def check_memory_stored(transcript: str) -> bool:
    """Check if a memory was stored since the last user message.

    Only looks for memory stores AFTER the last user message, ensuring
    each turn requires a new memory store before stopping.

    Returns:
        True if memory store was made since last user message, False otherwise.
        MUST return False on parse errors to avoid false positives.
    """
    try:
        entries = parse_jsonl_entries(transcript)

        # Find the index of the last user message
        last_user_idx = -1
        for i, entry in enumerate(entries):
            if entry.get("type") == "user":
                last_user_idx = i

        if last_user_idx < 0:
            return False

        # Check entries AFTER the last user message
        for entry in entries[last_user_idx + 1:]:
            for tool_use in _extract_tool_uses(entry):
                if _is_memory_store_call(tool_use):
                    return True

        return False

    except Exception:
        return False


_CODE_MODIFYING_BASH_PATTERNS = (
    "git commit", "git add", "npm install", "pip install", "uv add",
    "yarn add", "make", "cargo build", "go build", "docker",
    "mkdir", "touch", "rm ", "mv ", "cp ", "chmod", "chown",
)

_READ_ONLY_BASH_PATTERNS = (
    "git status", "git log", "git diff", "ls", "pwd",
    "cat ", "head ", "tail ", "which", "echo", "python -c",
)


def session_had_significant_work(transcript: str) -> bool:
    """Check if the session had meaningful work worth remembering.

    Counts significant tool calls (Edit, Write, MultiEdit, code-modifying Bash).

    Returns:
        True if 2+ significant tool calls were made.
    """
    try:
        entries = parse_jsonl_entries(transcript)
        significant_count = 0

        for entry in entries:
            if entry.get("type") != "tool_use":
                continue

            tool_name = entry.get("tool_name") or entry.get("name", "")
            params = entry.get("params") or entry.get("input", {})

            if tool_name in ("Edit", "Write", "MultiEdit"):
                significant_count += 1
            elif tool_name == "Bash":
                command = params.get("command", "")
                if any(p in command for p in _READ_ONLY_BASH_PATTERNS):
                    continue
                if any(p in command for p in _CODE_MODIFYING_BASH_PATTERNS):
                    significant_count += 1

            if significant_count >= 2:
                return True

        return False

    except Exception:
        return False


def find_incomplete_todos(transcript: str) -> list[str]:
    """Parse transcript for TodoWrite tool calls and find incomplete items.

    Returns:
        List of incomplete TODO descriptions.
    """
    try:
        entries = parse_jsonl_entries(transcript)
        todo_tracker: dict[str, str] = {}

        for entry in entries:
            if entry.get("type") != "tool_use":
                continue
            tool_name = entry.get("tool_name") or entry.get("name", "")
            if tool_name != "TodoWrite":
                continue
            params = entry.get("params") or entry.get("input", {})
            todo_text = params.get("todo", "")
            if todo_text:
                todo_tracker[todo_text] = params.get("status", "pending")

        return [
            f"{text} (status: {status})"
            for text, status in todo_tracker.items()
            if status != "completed"
        ]

    except Exception:
        return []


def check_git_status(cwd: str) -> str | None:
    """Check for uncommitted changes using git status.

    Returns a summary string if there are changes, None otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse git status output
            lines = result.stdout.strip().split("\n")
            modified = []
            added = []
            deleted = []
            untracked = []

            for line in lines:
                if len(line) < 3:
                    continue
                status = line[:2]
                filename = line[3:]

                if status.startswith("M") or status.endswith("M"):
                    modified.append(filename)
                elif status.startswith("A"):
                    added.append(filename)
                elif status.startswith("D"):
                    deleted.append(filename)
                elif status.startswith("??"):
                    untracked.append(filename)

            parts = []
            if modified:
                parts.append(f"{len(modified)} modified")
            if added:
                parts.append(f"{len(added)} added")
            if deleted:
                parts.append(f"{len(deleted)} deleted")
            if untracked:
                parts.append(f"{len(untracked)} untracked")

            return ", ".join(parts) if parts else None

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


_COMMIT_KEYWORDS = ("commit", "create a commit", "git commit", "save these changes", "add and commit")


def _extract_user_text(entry: dict[str, Any]) -> str:
    """Extract text content from a user entry.

    Args:
        entry: A transcript entry dict.

    Returns:
        Concatenated text content, lowercase.
    """
    if entry.get("role") != "user":
        return ""
    content = entry.get("content", "")
    if isinstance(content, str):
        return content.lower()
    if isinstance(content, list):
        texts = [
            item.get("text", "").lower()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(texts)
    return ""


def should_have_committed(transcript: str) -> bool:
    """Check if user requested a commit based on transcript.

    Returns:
        True if user asked to commit changes.
    """
    try:
        for entry in parse_jsonl_entries(transcript):
            text = _extract_user_text(entry)
            if any(kw in text for kw in _COMMIT_KEYWORDS):
                return True
        return False
    except Exception:
        return False


_CODE_EXTENSIONS = frozenset((".py", ".ts", ".js", ".jsx", ".tsx", ".go", ".rs", ".java", ".cpp", ".c"))
_TEST_COMMANDS = ("pytest", "npm test", "jest", "go test", "cargo test", "python -m pytest", "python -m unittest")
_TEST_PATH_MARKERS = ("test_", "_test.", "tests/", "test/")


def code_modified_without_tests(transcript: str) -> bool:
    """Check if code was modified but no tests were run.

    Returns:
        True if code files were written/edited but no test commands were executed.
    """
    try:
        entries = parse_jsonl_entries(transcript)
        code_modified = False
        tests_run = False

        for entry in entries:
            if entry.get("type") != "tool_use":
                continue

            tool_name = entry.get("tool_name") or entry.get("name", "")
            params = entry.get("params") or entry.get("input", {})

            if tool_name in ("Write", "Edit"):
                file_path = params.get("file_path", "")
                is_test_file = any(m in file_path for m in _TEST_PATH_MARKERS)
                is_code_file = any(file_path.endswith(ext) for ext in _CODE_EXTENSIONS)
                if file_path and not is_test_file and is_code_file:
                    code_modified = True
            elif tool_name == "Bash":
                command = params.get("command", "")
                if any(cmd in command for cmd in _TEST_COMMANDS):
                    tests_run = True

        return code_modified and not tests_run

    except Exception:
        return False


def get_specific_suggestions(session_id: str) -> list[str]:
    """Get specific memory suggestions from session state.

    Looks at what was captured during the session and suggests
    specific things to store, rather than generic prompts.

    Args:
        session_id: Current session ID

    Returns:
        List of specific suggestion strings
    """
    if not SESSION_STATE_AVAILABLE:
        return []

    suggestions = []

    try:
        summary = get_session_summary(session_id)

        # Suggest storing unresolved errors as patterns
        for error in summary.get("unresolved_errors", [])[:3]:
            error_msg = error.get("error", "")[:60]
            tool = error.get("tool", "unknown")
            suggestions.append(f"• Error pattern: '{error_msg}...' from {tool}")

        # Suggest storing detected but unstored preferences
        for pref in summary.get("unstored_preferences", [])[:3]:
            content = pref.get("content", "")[:60]
            # Filter out internal tracking markers
            if content.startswith("__"):
                continue
            suggestions.append(f"• Preference: {content}")

        # Suggest storing fix patterns that weren't auto-stored
        for pair in summary.get("unstored_patterns", [])[:3]:
            pattern = pair.get("pattern", "")[:60]
            suggestions.append(f"• Fix pattern: {pattern}")

    except Exception:
        pass

    return suggestions


def check_completion(transcript: str, cwd: str, session_id: str = "") -> dict[str, str] | None:
    """Check if work is complete. Returns block decision or None.

    Performs multiple checks to determine if the agent should continue working:
    - Check 1: Incomplete TODOs (THEO_STOP_CHECK_TODOS, default: true)
    - Check 2: Uncommitted changes when user requested commit (THEO_STOP_CHECK_GIT, default: false)
    - Check 3: Code modified without running tests (THEO_STOP_CHECK_TESTS, default: false)
    - Check 4: Memory storage enforcement for significant sessions (THEO_STOP_ENFORCE_MEMORY, default: true)

    Returns:
        dict with "decision": "block" and "reason" if work is incomplete.
        None if stopping is allowed.
    """
    # Get configuration from environment
    check_todos = os.getenv("THEO_STOP_CHECK_TODOS", "true").lower() == "true"
    check_git = os.getenv("THEO_STOP_CHECK_GIT", "false").lower() == "true"
    check_tests = os.getenv("THEO_STOP_CHECK_TESTS", "false").lower() == "true"
    enforce_memory = os.getenv("THEO_STOP_ENFORCE_MEMORY", "true").lower() == "true"

    # Check 1: Incomplete TODOs
    if check_todos:
        incomplete_todos = find_incomplete_todos(transcript)
        if incomplete_todos:
            reason_lines = ["Incomplete tasks remain:"]
            reason_lines.extend(
                f"- {todo}" for todo in incomplete_todos[:5]
            )  # Limit to 5 items

            if len(incomplete_todos) > 5:
                reason_lines.append(f"... and {len(incomplete_todos) - 5} more")

            reason_lines.append("\nComplete these tasks before stopping.")

            return {
                "decision": "block",
                "reason": "\n".join(reason_lines),
            }

    # Check 2: Uncommitted changes (skip for hooks directory - golden rule)
    if check_git and "/.claude/hooks" not in cwd:
        uncommitted = check_git_status(cwd)
        if uncommitted and should_have_committed(transcript):
            return {
                "decision": "block",
                "reason": f"Uncommitted changes: {uncommitted}.\n\nYou indicated you wanted to commit these changes. Please commit before stopping.",
            }

    # Check 3: Tests not run after code modifications
    if check_tests:
        if code_modified_without_tests(transcript):
            return {
                "decision": "block",
                "reason": "Code was modified but tests were not run.\n\nRun tests to verify the changes before stopping.",
            }

    # Check 4: Memory storage enforcement (ALWAYS - no significant work check)
    if enforce_memory:
        if not check_memory_stored(transcript):
            # Get specific suggestions from session state
            suggestions = get_specific_suggestions(session_id) if session_id else []

            if suggestions:
                # Show specific suggestions from session activity
                suggestions_text = "\n".join(suggestions)
                return {
                    "decision": "block",
                    "reason": f"""🧠 MEMORY STORAGE REQUIRED

You MUST store at least one memory before stopping.

**Specific learnings from this session to consider:**
{suggestions_text}

## Quick Store Command

```javascript
await theo.memory_store({{
  content: "...",  // Use a suggestion above or describe what you learned
  memory_type: "pattern",  // or preference, decision
  importance: 0.6,
  namespace: "project:name"
}});
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]""",
                }
            else:
                # No specific suggestions - use generic prompt
                return {
                    "decision": "block",
                    "reason": """🧠 MEMORY STORAGE REQUIRED

You MUST store at least one memory before stopping.

## Available Memory Tools (via theo MCP wrapper)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store with semantic indexing + auto-deduplication |
| `memory_recall` | Semantic search with graph expansion |
| `memory_validate` | Adjust confidence based on success/failure |
| `memory_apply` | Record memory usage (TRY phase) |
| `memory_outcome` | Record result (LEARN phase) |
| `memory_relate` | Create relationships (supersedes, contradicts) |
| `memory_count` | Count memories with optional filters |
| `memory_list` | List memories with pagination |
| `memory_inspect_graph` | Visualize graph around a memory node |
| `memory_edge_forget` | Delete edges between memories |
| `memory_detect_contradictions` | Find contradicting memories |
| `memory_check_supersedes` | Check if memory supersedes another |
| `memory_analyze_health` | Analyze memory system health |

## Store Session Learning

```javascript
await theo.memory_store({
  content: "What happened: task requested, actions taken, outcome",
  memory_type: "session",  // or pattern, decision, preference
  importance: 0.5,
  namespace: "project:name"  // or "global"
});
```

## TRY-LEARN Cycle (if applying existing memory)

```javascript
// 1. Before applying advice
await theo.memory_apply({ memory_id: "...", context: "Applying to X" });

// 2. After seeing result
await theo.memory_outcome({ memory_id: "...", success: true, outcome: "Worked" });
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]""",
                }

    # Mark checkpoint that stop was reviewed
    if session_id and SESSION_STATE_AVAILABLE:
        try:
            set_checkpoint(session_id, "stop_reviewed", True)
        except Exception:
            pass

    return None  # Allow stopping


def call_theo(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call theo MCP tool directly via --call mode.

    Note:
        Uses process groups (start_new_session=True) to ensure all child
        processes are killed on timeout, preventing zombie processes.
    """
    proc = None
    try:
        theo_paths = [
            Path(__file__).parent.parent,
            Path.home() / "Documents" / "Github" / "theo",
            Path.home() / ".local" / "share" / "theo",
            Path("/opt/theo"),
        ]

        theo_dir = None
        for path in theo_paths:
            if (path / "src" / "theo" / "__main__.py").exists():
                theo_dir = path
                break

        if theo_dir is None:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "theo",
                "--call",
                tool_name,
                "--args",
                json.dumps(args),
            ]
        else:
            cmd = [
                "uv",
                "run",
                "--directory",
                str(theo_dir),
                "python",
                "-m",
                "theo",
                "--call",
                tool_name,
                "--args",
                json.dumps(args),
            ]

        # Use Popen with start_new_session=True to create a process group
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=theo_dir or Path.cwd(),
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # Kill the entire process group to prevent zombie children
            if proc.pid:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            proc.wait()
            return {"success": False, "error": "theo timed out"}

        if proc.returncode != 0:
            return {"success": False, "error": f"theo failed: {stderr}"}

        parsed = json.loads(stdout)
        if parsed is None:
            return {"success": False, "error": "theo returned null"}
        return parsed

    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON response: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": "uv or python not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            proc.wait()


def record_stop_event(session_id: str, namespace: str, blocked: bool = False) -> None:
    """Record that the agent stopped (for analytics/tracking)."""
    try:
        # This could be used to track session patterns
        # For now, we just log it
        log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-stop.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        status = "BLOCKED" if blocked else "STOP"

        with log_path.open("a") as f:
            f.write(
                f"{datetime.now(tz=timezone.utc).isoformat()} | {status} | session={session_id} | namespace={namespace}\n",
            )
    except Exception:
        pass


def main() -> None:
    """Main hook entry point.

    The Stop hook is primarily for:
    1. Recording that the agent stopped (analytics)
    2. Optionally blocking the stop if work is incomplete

    Heavy analysis is deferred to SessionEnd hook.
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

        # Prevent infinite loops - if stop hook already ran, don't block again
        if stop_hook_active:
            return

        # Change to session's working directory
        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace()

        # Read full transcript for comprehensive checks
        full_transcript = read_full_transcript(transcript_path)

        # Check if work is complete before allowing stop
        if full_transcript:
            completion_check = check_completion(full_transcript, cwd, session_id)

            if completion_check and completion_check.get("decision") == "block":
                # Block the stop and provide reason to continue
                record_stop_event(session_id, namespace, blocked=True)

                # Output JSON to block the stop
                output = {
                    "decision": "block",
                    "reason": completion_check["reason"],
                }
                print(json.dumps(output))
                sys.stdout.flush()
                return

        # Check and prune transcript if over token threshold (disabled by default)
        # NOTE: Pruning is experimental - it reduces file size but doesn't restore
        # tokens in the current session, and can break memory enforcement checks
        if transcript_path:
            check_and_prune(transcript_path, session_id)

        # Record the stop event
        record_stop_event(session_id, namespace, blocked=False)

        # Read recent transcript for learning analysis
        transcript_tail = read_transcript_tail(transcript_path, lines=30)

        if transcript_tail:
            # Light analysis for immediate learnings
            learnings = analyze_for_learnings(transcript_tail)

            # Store any immediate learnings using shared client (with auto-fallback)
            if learnings:
                if get_shared_client is not None:
                    client = get_shared_client()
                    for learning in learnings:
                        client.store(
                            content=learning["content"],
                            namespace=namespace,
                            memory_type=learning.get("type", "pattern"),
                            importance=0.6,
                            metadata={
                                "source": "theo-stop",
                                "session_id": session_id,
                            },
                        )
                else:
                    # Fallback to subprocess if shared client unavailable
                    for learning in learnings:
                        call_theo(
                            "memory_store",
                            {
                                "content": learning["content"],
                                "memory_type": learning.get("type", "pattern"),
                                "namespace": namespace,
                                "importance": 0.6,
                                "metadata": {
                                    "source": "theo-stop",
                                    "session_id": session_id,
                                },
                            },
                        )

        # Output memory reflection prompt for Claude
        reflection_prompt = """🛑 STOP - Memory Operations REQUIRED

You MUST evaluate learnings from this interaction before stopping:
- Patterns discovered or reinforced
- User preferences revealed (explicit or implicit)
- Technical decisions made and their rationale
- Bugs fixed that could recur

## Available Memory Tools (via theo MCP wrapper)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store with semantic indexing + auto-deduplication |
| `memory_recall` | Semantic search with graph expansion |
| `memory_validate` | Adjust confidence based on success/failure |
| `memory_apply` | Record memory usage (TRY phase) |
| `memory_outcome` | Record result (LEARN phase) |
| `memory_relate` | Create relationships (supersedes, contradicts, etc.) |
| `memory_forget` | Delete with golden rule protection |
| `memory_count` | Count memories with optional filters |
| `memory_list` | List memories with pagination |
| `memory_inspect_graph` | Visualize graph around a memory node |
| `memory_detect_contradictions` | Find contradicting memories |
| `memory_check_supersedes` | Check if memory supersedes another |
| `memory_analyze_health` | Analyze memory system health |

## Before Stopping

**Store new learnings:**
```javascript
await theo.memory_store({
  content: "...",
  memory_type: "pattern",  // preference, decision, golden_rule
  importance: 0.7,
  namespace: "project:name"
});
```

**If a memory was applied and worked:**
```javascript
await theo.memory_outcome({
  memory_id: "...",
  success: true,
  outcome: "Applied successfully in this session"
});
```

**If new memory supersedes old one:**
```javascript
await theo.memory_relate({
  source_id: "new_id",
  target_id: "old_id",
  relation_type: "supersedes"
});
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]"""

        output = {"systemMessage": reflection_prompt}
        print(json.dumps(output))
        sys.stdout.flush()

    except BrokenPipeError:
        pass
    except Exception as e:
        # Log error but don't block
        try:
            log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-stop.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"{datetime.now(tz=timezone.utc).isoformat()} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
