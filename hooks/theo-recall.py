#!/usr/bin/env python3
"""Claude Code PostToolUse hook for autonomous memory recall.

Runs AFTER tool calls and reactively searches for relevant memories based on
tool output. Complements PreToolUse (proactive) with reactive recall:
  - Bash errors → search for similar error patterns and known fixes
  - Read files → search for file/module-specific memories
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
from theo_session_state import add_shown_memory

COOLDOWN_SECONDS = 10
COOLDOWN_FILE = Path("/tmp/theo-recall-cooldown.json")
MAX_RECALL_RESULTS = 3
MIN_CONFIDENCE = 0.4
MIN_IMPORTANCE = 0.5
LOG_PATH = Path.home() / ".claude" / "hooks" / "logs" / "theo-recall.log"

# Substring match — "error" already catches TypeError, ValueError, etc.
ERROR_INDICATORS = (
    "error", "Error", "ERROR", "failed", "Failed", "FAILED",
    "Traceback", "FAIL", "ERRORS", "panic", "segfault",
    "command not found", "No such file or directory", "Permission denied",
)

SKIP_TOOL_PATTERNS = ("mcp__theo__", "memory_store", "memory_recall", "memory_apply")

# Regex matching Python exceptions, pytest markers, generic errors, npm/cargo errors
_ERROR_LINE_RE = re.compile(
    r"^("
    r"\w+Error:|"       # Python exceptions (TypeError:, ValueError:, etc.)
    r"FAILED |ERROR |E |"  # Pytest failure markers
    r"error\b|Error\b|ERROR\b|"  # Generic error prefix
    r".*ERR!|npm error|"  # npm/node errors
    r"error\["           # Rust/cargo errors
    r")"
)

SKIP_FILE_PATTERNS = (
    "__pycache__", "node_modules", ".git/", "package-lock.json",
    "yarn.lock", "uv.lock", ".pyc", ".pyo", ".so", ".dylib",
    ".log", ".tmp", ".cache",
)


def _log(msg: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")
    except OSError:
        pass


def read_hook_input() -> dict[str, Any]:
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
    project_name = Path(cwd).name
    for indicator in (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"


def _check_cooldown(signal_key: str) -> bool:
    """Returns True if recall is allowed, False if still in cooldown."""
    try:
        cooldowns = json.loads(COOLDOWN_FILE.read_text()) if COOLDOWN_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        cooldowns = {}

    now = time.time()
    if now - cooldowns.get(signal_key, 0) < COOLDOWN_SECONDS:
        return False

    cooldowns[signal_key] = now
    # Prune entries older than 60s
    pruned = {k: v for k, v in cooldowns.items() if now - v < 60}
    try:
        COOLDOWN_FILE.write_text(json.dumps(pruned))
    except OSError:
        pass
    return True


def _extract_error_lines(output: str) -> str:
    """Extract the most meaningful error lines from command output."""
    seen: set[str] = set()
    unique: list[str] = []
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped and _ERROR_LINE_RE.match(stripped) and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped[:200])
            if len(unique) >= 3:
                break
    return " | ".join(unique)


def extract_bash_error_signal(
    tool_input: dict[str, Any],
    tool_response: Any,
) -> str | None:
    """Extract a searchable error signal from Bash tool output."""
    if isinstance(tool_response, dict):
        stdout = tool_response.get("stdout", "")
        stderr = tool_response.get("stderr", "")
        exit_code = tool_response.get("exitCode", tool_response.get("exit_code", 0))
    elif isinstance(tool_response, str):
        stdout, stderr, exit_code = tool_response, "", None
    else:
        return None

    output = f"{stdout}\n{stderr}".strip()
    if not output:
        return None

    # Exit 0 with no failure markers → skip
    if exit_code is not None and exit_code == 0:
        if not any(m in output.lower() for m in ("failed", "error", "fail")):
            return None

    has_error = any(ind in output for ind in ERROR_INDICATORS)
    if not has_error and (exit_code is None or exit_code == 0):
        return None

    command = tool_input.get("command", "")[:100]
    error_lines = _extract_error_lines(output)

    if error_lines:
        return f"{command} {error_lines}"[:500]
    if stderr.strip():
        return f"{command} {stderr[:400]}"
    return f"{command} {output[-400:]}"


def extract_read_signal(tool_input: dict[str, Any]) -> str | None:
    """Extract a searchable signal from a Read tool operation."""
    file_path = tool_input.get("file_path", "")
    if not file_path or any(p in file_path for p in SKIP_FILE_PATTERNS):
        return None

    path = Path(file_path)
    parts = [path.name]
    if path.parent.name and path.parent.name != ".":
        parts.append(path.parent.name)
    if path.parent.parent.name and path.parent.parent.name not in (".", "src", "lib"):
        parts.append(path.parent.parent.name)
    return " ".join(parts)


def _quality_filter(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter memories by confidence, importance, or golden rule status."""
    return [
        m for m in memories
        if (m.get("confidence", 0) >= MIN_CONFIDENCE
            or m.get("importance", 0) >= MIN_IMPORTANCE
            or m.get("type") == "golden_rule")
    ]


def recall_memories(
    query: str,
    namespace: str,
    n_results: int = MAX_RECALL_RESULTS,
    include_related: bool = True,
) -> list[dict[str, Any]]:
    """Search for relevant memories using DaemonClient."""
    try:
        with get_shared_client() as client:
            result = client.fetch(
                namespace=namespace,
                query=query,
                n_results=n_results + 2,
                include_related=include_related,
                max_depth=1 if include_related else 0,
            )
        if not result.get("success"):
            _log(f"fetch failed: {result.get('error')}")
            return []

        data = result.get("data", {})
        memories = data.get("memories", []) + data.get("expanded", [])
        return _quality_filter(memories)[:n_results]
    except Exception as e:
        _log(f"recall error: {e}")
        return []


def format_recall_context(
    memories: list[dict[str, Any]],
    trigger: str,
    signal: str,
) -> str | None:
    """Format recalled memories as actionable RFC 2119 context."""
    if not memories:
        return None

    headers = {
        "error": ("# Recalled Memories (error context)",
                  f"*Similar patterns found for: `{signal[:80]}...`*"),
        "file_context": ("# Recalled Memories (file context)",
                         f"*Related memories for: `{signal}`*"),
    }
    title, subtitle = headers.get(trigger, ("# Recalled Memories", ""))
    lines = [title, subtitle, ""]

    buckets: dict[str, list[str]] = {"MUST": [], "SHOULD": [], "MAY": []}
    for mem in memories:
        content = mem.get("content", "")[:300]
        if len(mem.get("content", "")) > 300:
            content += "..."
        confidence = mem.get("confidence", 0)
        importance = mem.get("importance", 0)

        if mem.get("type") == "golden_rule" or confidence >= 0.9:
            buckets["MUST"].append(content)
        elif importance >= 0.8 or confidence >= 0.7:
            buckets["SHOULD"].append(content)
        else:
            buckets["MAY"].append(content)

    labels = {"MUST": "Apply these", "SHOULD": "Consider these", "MAY": "Related context"}
    for level, label in labels.items():
        if buckets[level]:
            lines.append(f"## {level} ({label})")
            lines.extend(f"- {item}" for item in buckets[level])
            lines.append("")

    return "\n".join(lines) if any(buckets.values()) else None


def track_surfaced_memories(
    memories: list[dict[str, Any]],
    session_id: str | None,
    context: str,
) -> None:
    """Record surfaced memories in session state for validation tracking."""
    if not session_id:
        return
    try:
        for mem in memories:
            if mem_id := mem.get("id", ""):
                add_shown_memory(session_id, mem_id, context)
    except Exception as e:
        _log(f"tracking error: {e}")


def main() -> None:
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

        if any(p in tool_name for p in SKIP_TOOL_PATTERNS):
            _log("skipping theo tool")
            return

        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace(cwd)

        # Signal extraction
        if tool_name == "Bash":
            query = extract_bash_error_signal(tool_input, tool_response)
            trigger = "error"
        elif tool_name == "Read":
            query = extract_read_signal(tool_input)
            trigger = "file_context"
        else:
            query, trigger = None, ""

        if not query:
            _log("no signal extracted")
            return

        if not _check_cooldown(f"{trigger}:{query[:50]}"):
            _log("cooldown active, skipping")
            return

        # Recall from project namespace, supplement with global if sparse
        memories = recall_memories(query, namespace)
        _log(f"recalled {len(memories)} memories from {namespace}")

        if len(memories) < 2 and namespace != "global":
            existing_ids = {m.get("id") for m in memories}
            for gm in recall_memories(query, "global", n_results=2, include_related=False):
                if gm.get("id") not in existing_ids:
                    memories.append(gm)

        if not memories:
            _log("no relevant memories found")
            return

        track_surfaced_memories(memories, session_id, f"{trigger}:{query[:100]}")

        context = format_recall_context(memories, trigger, query)
        if context:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                },
            }))
            _log(f"output context: {len(context)} chars")

    except BrokenPipeError:
        _log("broken pipe")
    except Exception as e:
        _log(f"ERROR: {e}")


if __name__ == "__main__":
    main()
