#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Claude Code PostToolUseFailure hook for capturing failure patterns.

This hook runs when a tool invocation fails, capturing error context
for learning. Failed operations often reveal important patterns about
what doesn't work and why.

Usage:
    Configure in ~/.claude/settings.json:
    {
        "hooks": {
            "PostToolUseFailure": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-failure.py",
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
        "tool_input": {"command": "npm run build"},
        "error": "Command failed with exit code 1",
        "is_interrupt": false,
        "session_id": "abc123",
        "cwd": "/project/root"
    }
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
    """Derive project namespace from working directory.

    Args:
        cwd: Current working directory.

    Returns:
        Namespace string like 'project:name' or 'global'.
    """
    project_name = Path(cwd).name
    project_indicators = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"


def call_theo(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call theo MCP tool directly via --call mode.

    Args:
        tool_name: Name of the theo tool to call.
        args: Arguments to pass to the tool.

    Returns:
        Result dictionary from theo.
    """
    proc = None
    try:
        theo_paths = [
            Path(__file__).parent.parent,
            Path.home() / "Github" / "theo",
            Path.home() / ".local" / "share" / "theo",
            Path("/opt/theo"),
        ]

        theo_dir = None
        for path in theo_paths:
            if (path / "src" / "theo" / "__main__.py").exists():
                theo_dir = path
                break

        if theo_dir is None:
            cmd = ["uv", "run", "python", "-m", "theo", "--call", tool_name, "--args", json.dumps(args)]
        else:
            cmd = ["uv", "run", "--directory", str(theo_dir), "python", "-m", "theo", "--call", tool_name, "--args", json.dumps(args)]

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


def _should_store_failure(
    tool_name: str,
    error: str,
    is_interrupt: bool,
) -> tuple[bool, float]:
    """Determine if this failure should be stored.

    Args:
        tool_name: Name of the failed tool.
        error: Error message.
        is_interrupt: Whether user interrupted the operation.

    Returns:
        Tuple of (should_store, importance).
    """
    # Skip interrupts - user cancelled, not a real failure
    if is_interrupt:
        return False, 0.0

    # Skip empty or very short errors
    if not error or len(error) < 20:
        return False, 0.0

    # Bash failures often reveal environment issues - valuable
    if tool_name == "Bash" and len(error) > 50:
        return True, 0.5

    # Edit/Write failures may indicate permission or path issues
    if tool_name in ("Edit", "Write") and len(error) > 30:
        return True, 0.4

    # Task failures indicate subagent issues
    if tool_name == "Task" and len(error) > 50:
        return True, 0.5

    return False, 0.0


def _format_failure_content(
    tool_name: str,
    tool_input: dict[str, Any],
    error: str,
) -> str:
    """Format failure into storable memory content.

    Args:
        tool_name: Name of the failed tool.
        tool_input: Input parameters to the tool.
        error: Error message.

    Returns:
        Formatted content string.
    """
    parts = [f"Tool failure: {tool_name}"]

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")[:200]
        parts.append(f"Command: {cmd}")
    elif tool_name in ("Edit", "Write"):
        path = tool_input.get("file_path", "")
        parts.append(f"File: {path}")
    elif tool_name == "Task":
        desc = tool_input.get("description", "")
        agent = tool_input.get("subagent_type", "")
        parts.append(f"Task ({agent}): {desc}")

    # Truncate error to reasonable length
    parts.append(f"Error: {error[:500]}")

    return "\n".join(parts)


def main() -> None:
    """Main hook entry point."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-failure.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")

    log("PostToolUseFailure hook triggered")

    try:
        hook_input = read_hook_input()
        if not hook_input:
            log("no input received")
            return

        tool_name = hook_input.get("tool_name") or hook_input.get("toolName", "")
        tool_input = hook_input.get("tool_input") or hook_input.get("toolInput", {})
        error = hook_input.get("error", "")
        is_interrupt = hook_input.get("is_interrupt") or hook_input.get("isInterrupt", False)
        cwd = hook_input.get("cwd", os.getcwd())
        session_id = hook_input.get("session_id") or hook_input.get("sessionId")

        log(f"tool={tool_name} error_len={len(error)} interrupt={is_interrupt}")

        # Change to session's working directory
        if cwd:
            os.chdir(cwd)

        should_store, importance = _should_store_failure(tool_name, error, is_interrupt)
        if not should_store:
            log("failure not significant enough to store")
            return

        namespace = get_project_namespace(cwd)
        content = _format_failure_content(tool_name, tool_input, error)

        result = call_theo("memory_store", {
            "content": content,
            "memory_type": "pattern",
            "namespace": namespace,
            "importance": importance,
            "metadata": {
                "source": "theo-failure",
                "tool_name": tool_name,
                "session_id": session_id,
            },
        })

        if result.get("success"):
            log(f"stored failure pattern (importance={importance})")
        else:
            log(f"storage failed: {result.get('error', 'unknown')}")

    except BrokenPipeError:
        log("broken pipe")
    except Exception as e:
        log(f"ERROR: {e}")


if __name__ == "__main__":
    main()
