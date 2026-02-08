#!/usr/bin/env python3
"""Claude Code PostToolUse hook for autonomous memory recall.

This hook runs AFTER tool calls complete and reactively searches for relevant
memories based on tool OUTPUT. This is the missing "reactive" half of the
memory system - while PreToolUse hooks inject context before actions,
this hook recalls memories based on what actually happened.

Key capabilities:
1. Bash errors → search for similar error patterns and known fixes
2. Read files → search for file/module-specific memories (decisions, patterns)
3. Successful operations after errors → auto-validate surfaced memories

This hook uses DaemonClient for fast IPC (~10x faster than subprocess).

Usage:
    Configure in ~/.claude/settings.json:
    {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash|Read",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python ~/.claude/hooks/theo-recall.py",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "tool_name": "Bash",
        "tool_input": {"command": "uv run pytest tests/ -v"},
        "tool_response": {"stdout": "...", "stderr": "...", "exitCode": 1},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

Output (to stdout JSON):
    {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "# Recalled Memories\\n\\n## Known Fix..."
        }
    }
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from theo_client import get_shared_client
from theo_session_state import add_shown_memory, load_session_state


# =============================================================================
# Constants
# =============================================================================

# Rate limiting: minimum seconds between recalls for the same signal type
COOLDOWN_SECONDS = 10
COOLDOWN_FILE = Path("/tmp/theo-recall-cooldown.json")

# Maximum memories to surface per recall
MAX_RECALL_RESULTS = 3

# Minimum memory quality thresholds
MIN_CONFIDENCE = 0.4
MIN_IMPORTANCE = 0.5

# Error signal patterns for Bash output
ERROR_INDICATORS = (
    "error",
    "Error",
    "ERROR",
    "failed",
    "Failed",
    "FAILED",
    "traceback",
    "Traceback",
    "exception",
    "Exception",
    "ModuleNotFoundError",
    "ImportError",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "KeyError",
    "AttributeError",
    "FileNotFoundError",
    "PermissionError",
    "ConnectionError",
    "TimeoutError",
    "AssertionError",
    "FAIL",
    "ERRORS",
    "panic",
    "segfault",
    "command not found",
    "No such file or directory",
    "Permission denied",
)

# File paths that indicate test output
TEST_INDICATORS = ("test_", "_test.", ".test.", ".spec.", "pytest", "jest", "vitest")

# Tools where we skip Theo's own operations to avoid recursion
SKIP_TOOL_PATTERNS = ("mcp__theo__", "memory_store", "memory_recall", "memory_apply")


# =============================================================================
# Logging
# =============================================================================

LOG_PATH = Path.home() / ".claude" / "hooks" / "logs" / "theo-recall.log"


def _log(msg: str) -> None:
    """Write timestamped message to log file."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")
    except OSError:
        pass


# =============================================================================
# Input Handling
# =============================================================================


def read_hook_input() -> dict[str, Any]:
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


def get_project_namespace(cwd: str) -> str:
    """Derive project namespace from working directory."""
    project_name = Path(cwd).name
    indicators = (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")
    for indicator in indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"


# =============================================================================
# Rate Limiting
# =============================================================================


def _load_cooldowns() -> dict[str, float]:
    """Load cooldown timestamps from file."""
    try:
        if COOLDOWN_FILE.exists():
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_cooldowns(cooldowns: dict[str, float]) -> None:
    """Save cooldown timestamps to file."""
    try:
        # Prune old entries (>60s old)
        now = time.time()
        pruned = {k: v for k, v in cooldowns.items() if now - v < 60}
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(pruned, f)
    except OSError:
        pass


def _check_cooldown(signal_key: str) -> bool:
    """Check if enough time has passed since last recall for this signal.

    Returns True if recall is allowed, False if still in cooldown.
    """
    cooldowns = _load_cooldowns()
    last_time = cooldowns.get(signal_key, 0)
    now = time.time()

    if now - last_time < COOLDOWN_SECONDS:
        return False

    cooldowns[signal_key] = now
    _save_cooldowns(cooldowns)
    return True


# =============================================================================
# Signal Extraction
# =============================================================================


def extract_bash_error_signal(
    tool_input: dict[str, Any],
    tool_response: Any,
) -> str | None:
    """Extract a searchable error signal from Bash tool output.

    Looks for error indicators in stdout/stderr and extracts the most
    meaningful error message for semantic search.

    Returns:
        Query string for memory search, or None if no error detected.
    """
    # Handle both dict and string responses
    if isinstance(tool_response, dict):
        stdout = tool_response.get("stdout", "")
        stderr = tool_response.get("stderr", "")
        exit_code = tool_response.get("exitCode", tool_response.get("exit_code", 0))
    elif isinstance(tool_response, str):
        stdout = tool_response
        stderr = ""
        exit_code = None
    else:
        return None

    # Combine output for error detection
    output = f"{stdout}\n{stderr}".strip()
    if not output:
        return None

    # Check exit code first (most authoritative signal)
    if exit_code is not None and exit_code == 0:
        # Even on exit 0, check for test failures in output
        has_test_failure = any(
            marker in output.lower()
            for marker in ("failed", "error", "fail")
        )
        if not has_test_failure:
            return None

    # Check for error indicators
    has_error = any(indicator in output for indicator in ERROR_INDICATORS)
    if not has_error and (exit_code is None or exit_code == 0):
        return None

    # Extract the most meaningful error line(s)
    command = tool_input.get("command", "")
    error_lines = _extract_error_lines(output)

    if error_lines:
        # Combine command context with error for better semantic search
        signal = f"{command[:100]} {error_lines}"
        return signal[:500]

    # Fallback: use stderr or last lines of output
    if stderr.strip():
        return f"{command[:100]} {stderr[:400]}"

    return f"{command[:100]} {output[-400:]}"


def _extract_error_lines(output: str) -> str:
    """Extract the most meaningful error lines from command output.

    Prioritizes: traceback last lines, error messages, assertion failures.
    """
    lines = output.split("\n")
    error_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Python traceback - grab the last line (actual error)
        if stripped.startswith(("ModuleNotFoundError:", "ImportError:", "SyntaxError:",
                                "TypeError:", "ValueError:", "KeyError:",
                                "AttributeError:", "FileNotFoundError:",
                                "PermissionError:", "RuntimeError:", "OSError:",
                                "AssertionError:", "NameError:", "IndexError:")):
            error_lines.append(stripped[:200])
            continue

        # Pytest failure summary
        if stripped.startswith(("FAILED ", "ERROR ", "E ")):
            error_lines.append(stripped[:200])
            continue

        # Generic error patterns
        if re.match(r"^(error|Error|ERROR)\b", stripped):
            error_lines.append(stripped[:200])
            continue

        # npm/node errors
        if "ERR!" in stripped or stripped.startswith("npm error"):
            error_lines.append(stripped[:200])
            continue

        # Rust/cargo errors
        if stripped.startswith("error["):
            error_lines.append(stripped[:200])
            continue

    # Return unique error lines, max 3
    seen: set[str] = set()
    unique: list[str] = []
    for line in error_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
        if len(unique) >= 3:
            break

    return " | ".join(unique) if unique else ""


def extract_read_signal(tool_input: dict[str, Any]) -> str | None:
    """Extract a searchable signal from a Read tool operation.

    Uses the file path and module context for semantic search
    to find memories about the file being read.

    Returns:
        Query string for memory search, or None if not worth searching.
    """
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return None

    path = Path(file_path)

    # Skip common non-interesting files
    skip_patterns = (
        "__pycache__", "node_modules", ".git/", "package-lock.json",
        "yarn.lock", "uv.lock", ".pyc", ".pyo", ".so", ".dylib",
        ".log", ".tmp", ".cache",
    )
    if any(pattern in file_path for pattern in skip_patterns):
        return None

    # Build query from file context
    parts: list[str] = []

    # File name and extension
    parts.append(path.name)

    # Parent directory (module context)
    if path.parent.name and path.parent.name != ".":
        parts.append(path.parent.name)

    # Grandparent if meaningful (e.g., src/theo/storage/sqlite_store.py)
    if path.parent.parent.name and path.parent.parent.name not in (".", "src", "lib"):
        parts.append(path.parent.parent.name)

    return " ".join(parts)


# =============================================================================
# Memory Recall
# =============================================================================


def recall_memories(
    query: str,
    namespace: str,
) -> list[dict[str, Any]]:
    """Search for relevant memories using DaemonClient.

    Uses fast Unix socket IPC with automatic subprocess fallback.

    Returns:
        List of memory dictionaries sorted by relevance.
    """
    try:
        with get_shared_client() as client:
            result = client.fetch(
                namespace=namespace,
                query=query,
                n_results=MAX_RECALL_RESULTS + 2,  # Fetch extra for filtering
                include_related=True,
                max_depth=1,
            )

        if not result.get("success"):
            _log(f"fetch failed: {result.get('error')}")
            return []

        data = result.get("data", {})
        memories = data.get("memories", [])

        # Include graph-expanded memories
        expanded = data.get("expanded", [])
        memories.extend(expanded)

        # Filter by quality
        filtered = [
            m for m in memories
            if (m.get("confidence", 0) >= MIN_CONFIDENCE
                or m.get("importance", 0) >= MIN_IMPORTANCE
                or m.get("type") == "golden_rule")
        ]

        return filtered[:MAX_RECALL_RESULTS]

    except Exception as e:
        _log(f"recall error: {e}")
        return []


def recall_global_memories(query: str) -> list[dict[str, Any]]:
    """Search global namespace for relevant memories.

    Supplements project-specific search for cross-project patterns.
    """
    try:
        with get_shared_client() as client:
            result = client.fetch(
                namespace="global",
                query=query,
                n_results=2,
                include_related=False,
            )

        if not result.get("success"):
            return []

        memories = result.get("data", {}).get("memories", [])
        return [
            m for m in memories
            if (m.get("confidence", 0) >= MIN_CONFIDENCE
                or m.get("importance", 0) >= MIN_IMPORTANCE
                or m.get("type") == "golden_rule")
        ][:2]

    except Exception:
        return []


# =============================================================================
# Output Formatting
# =============================================================================


def format_recall_context(
    memories: list[dict[str, Any]],
    trigger: str,
    signal: str,
) -> str | None:
    """Format recalled memories as actionable context for Claude.

    Uses RFC 2119 language hierarchy for priority.

    Args:
        memories: List of memory dictionaries.
        trigger: What triggered the recall ("error", "file_context").
        signal: The signal that was searched (for context).

    Returns:
        Formatted context string or None if no memories to show.
    """
    if not memories:
        return None

    lines: list[str] = []

    if trigger == "error":
        lines.append("# Recalled Memories (error context)")
        lines.append(f"*Similar patterns found for: `{signal[:80]}...`*")
        lines.append("")
    elif trigger == "file_context":
        lines.append("# Recalled Memories (file context)")
        lines.append(f"*Related memories for: `{signal}`*")
        lines.append("")

    # Categorize by priority
    must_items: list[str] = []
    should_items: list[str] = []
    may_items: list[str] = []

    for mem in memories:
        content = mem.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."

        mem_type = mem.get("type", "")
        confidence = mem.get("confidence", 0)
        importance = mem.get("importance", 0)

        if mem_type == "golden_rule" or confidence >= 0.9:
            must_items.append(content)
        elif importance >= 0.8 or confidence >= 0.7:
            should_items.append(content)
        else:
            may_items.append(content)

    if must_items:
        lines.append("## MUST (Apply these)")
        for item in must_items:
            lines.append(f"- {item}")
        lines.append("")

    if should_items:
        lines.append("## SHOULD (Consider these)")
        for item in should_items:
            lines.append(f"- {item}")
        lines.append("")

    if may_items:
        lines.append("## MAY (Related context)")
        for item in may_items:
            lines.append(f"- {item}")
        lines.append("")

    # Only output if we have actual content
    if not (must_items or should_items or may_items):
        return None

    return "\n".join(lines)


# =============================================================================
# Memory Tracking
# =============================================================================


def track_surfaced_memories(
    memories: list[dict[str, Any]],
    session_id: str | None,
    context: str,
) -> None:
    """Record which memories were surfaced for later validation tracking.

    This enables the validation loop: memories that are surfaced and lead
    to successful outcomes can have their confidence increased.
    """
    if not session_id:
        return

    try:
        for mem in memories:
            mem_id = mem.get("id", "")
            if mem_id:
                add_shown_memory(session_id, mem_id, context)
    except Exception as e:
        _log(f"tracking error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main hook entry point."""
    _log("PostToolUse recall hook triggered")

    try:
        hook_input = read_hook_input()
        if not hook_input:
            _log("no input received")
            return

        tool_name = hook_input.get("tool_name") or hook_input.get("toolName", "")
        tool_input = hook_input.get("tool_input") or hook_input.get("toolInput", {})
        tool_response = hook_input.get("tool_response") or hook_input.get("toolResponse", {})
        session_id = hook_input.get("session_id") or hook_input.get("sessionId")
        cwd = hook_input.get("cwd", os.getcwd())

        _log(f"tool={tool_name}")

        # Skip Theo's own operations to avoid recursion
        if any(pattern in tool_name for pattern in SKIP_TOOL_PATTERNS):
            _log("skipping theo tool")
            return

        # Change to session working directory
        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace(cwd)
        query: str | None = None
        trigger: str = ""

        # -------------------------------------------------------------------
        # Signal extraction based on tool type
        # -------------------------------------------------------------------

        if tool_name == "Bash":
            query = extract_bash_error_signal(tool_input, tool_response)
            trigger = "error"
            if query:
                _log(f"bash error signal: {query[:100]}")

        elif tool_name == "Read":
            query = extract_read_signal(tool_input)
            trigger = "file_context"
            if query:
                _log(f"read signal: {query[:100]}")

        if not query:
            _log("no signal extracted")
            return

        # -------------------------------------------------------------------
        # Rate limiting
        # -------------------------------------------------------------------

        cooldown_key = f"{trigger}:{query[:50]}"
        if not _check_cooldown(cooldown_key):
            _log("cooldown active, skipping")
            return

        # -------------------------------------------------------------------
        # Memory recall
        # -------------------------------------------------------------------

        memories = recall_memories(query, namespace)
        _log(f"recalled {len(memories)} memories from {namespace}")

        # Also check global namespace if project returned few results
        if len(memories) < 2 and namespace != "global":
            global_memories = recall_global_memories(query)
            if global_memories:
                # Deduplicate by ID
                existing_ids = {m.get("id") for m in memories}
                for gm in global_memories:
                    if gm.get("id") not in existing_ids:
                        memories.append(gm)
                _log(f"added {len(global_memories)} global memories")

        if not memories:
            _log("no relevant memories found")
            return

        # -------------------------------------------------------------------
        # Track surfaced memories for validation
        # -------------------------------------------------------------------

        track_surfaced_memories(memories, session_id, f"{trigger}:{query[:100]}")

        # -------------------------------------------------------------------
        # Format and output
        # -------------------------------------------------------------------

        context = format_recall_context(memories, trigger, query)
        if context:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                },
            }
            print(json.dumps(output))
            _log(f"output context: {len(context)} chars")
        else:
            _log("no context to output after formatting")

    except BrokenPipeError:
        _log("broken pipe")
    except Exception as e:
        # Never block the agent
        _log(f"ERROR: {e}")


if __name__ == "__main__":
    main()
