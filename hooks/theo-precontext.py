#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Claude Code / Factory PreToolUse hook for injecting relevant memory context.

This hook runs BEFORE tool calls and injects relevant memories as context
reminders. This helps prevent mistakes by surfacing relevant preferences,
patterns, and past decisions.

Enhanced with tool input modification to auto-correct commands based on
stored preferences (e.g., npm -> pnpm, pip -> uv).

Supported tools: Task, Bash, Glob, Grep, Read, Edit, Write, WebFetch, WebSearch, mcp__*

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Task|Bash|Glob|Grep|Read|Edit|Write|WebFetch|WebSearch|mcp__.*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-precontext.py",
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
        "tool_input": {"command": "npm install express"},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

Output (to stdout):
    1. If modification needed: JSON with updatedInput
    2. Otherwise: Relevant context/reminders based on the tool being used.
    Example: "Reminder: This project uses pnpm, not npm."

The hook extracts key terms from the tool input and searches theo
for relevant memories, outputting them as context.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def read_hook_input() -> dict:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable
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


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'
    """
    cwd = os.getcwd()
    project_name = Path(cwd).name

    project_indicators = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"

    return "global"


def extract_query(tool_name: str, tool_input: dict) -> str | None:
    """Extract search query from tool input - pass raw text to semantic search.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input dictionary

    Returns:
        Query string for theo semantic search, or None if nothing useful
    """
    if tool_name == "Bash":
        return tool_input.get("command", "")

    if tool_name in ("Write", "Edit", "MultiEdit", "Read"):
        file_path = tool_input.get("file_path", "")
        return f"{Path(file_path).name} {Path(file_path).suffix}"

    if tool_name == "Task":
        prompt = tool_input.get("prompt", "")
        description = tool_input.get("description", "")
        subagent_type = tool_input.get("subagent_type", "")
        return f"{subagent_type} {description} {prompt}"

    if tool_name == "Glob":
        patterns = tool_input.get("patterns", [])
        if isinstance(patterns, list):
            return " ".join(patterns)
        return tool_input.get("pattern", "")

    if tool_name == "Grep":
        return tool_input.get("pattern", "")

    if tool_name == "WebFetch":
        return tool_input.get("url", "")

    if tool_name == "WebSearch":
        return tool_input.get("query", "")

    if tool_name.startswith("mcp__"):
        # MCP tools - use server and tool name
        parts = tool_name.split("__")
        return " ".join(parts[1:])

    return None


def call_theo(tool_name: str, args: dict) -> dict:
    """Call theo MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (memory_recall, etc.)
        args: Dictionary of tool arguments

    Returns:
        Tool result as dictionary, or error dict on failure

    Note:
        Uses process groups (start_new_session=True) to ensure all child
        processes are killed on timeout, preventing zombie processes.
    """
    import os
    import signal

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
            cmd = [
                "uv", "run", "python", "-m", "theo",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]
        else:
            cmd = [
                "uv", "run",
                "--directory", str(theo_dir),
                "python", "-m", "theo",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]

        # Use Popen with start_new_session=True to create a process group
        # This ensures we can kill all child processes on timeout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=theo_dir or Path.cwd(),
            start_new_session=True,  # Creates new process group
        )

        try:
            stdout, stderr = proc.communicate(timeout=5)  # Graph expansion needs more time
        except subprocess.TimeoutExpired:
            # Kill the entire process group to prevent zombie children
            if proc.pid:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # Process already dead or no permission
            proc.wait()  # Reap the zombie
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
        # Ensure cleanup if proc exists and hasn't been waited on
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            proc.wait()


def apply_input_modifications(
    tool_name: str, tool_input: dict, memories: list[dict],
) -> tuple[dict, str] | None:
    """Apply modifications to tool input based on stored preferences.

    Args:
        tool_name: Name of the tool
        tool_input: Original tool input dictionary
        memories: List of memory dictionaries from theo

    Returns:
        Tuple of (updated_input, reason) if modification applied, None otherwise
    """
    if tool_name != "Bash":
        return None

    command = tool_input.get("command", "")
    if not command:
        return None

    updated_command = command
    reason = None

    # Check each memory for actionable preferences
    for mem in memories:
        content = mem.get("content", "").lower()
        original_content = mem.get("content", "")

        # Only apply modifications for high-confidence memories
        confidence = mem.get("confidence", 0)
        importance = mem.get("importance", 0)
        is_golden_rule = mem.get("type") == "golden_rule"

        if not (confidence >= 0.7 or importance >= 0.8 or is_golden_rule):
            continue

        # Package manager corrections (Node.js)
        if "must use pnpm" in content or "must not use npm" in content:
            if " npm " in command or command.startswith("npm "):
                updated_command = updated_command.replace("npm ", "pnpm ", 1)
                if command.startswith("npm "):
                    updated_command = "pnpm " + updated_command[4:]
                reason = f"Auto-corrected: npm → pnpm (per stored preference: {original_content[:80]})"
                break

        elif "must use yarn" in content:
            if " npm " in command or command.startswith("npm "):
                updated_command = updated_command.replace("npm ", "yarn ", 1)
                if command.startswith("npm "):
                    updated_command = "yarn " + updated_command[4:]
                reason = f"Auto-corrected: npm → yarn (per stored preference: {original_content[:80]})"
                break

        elif "must use bun" in content:
            if " npm " in command or command.startswith("npm "):
                updated_command = updated_command.replace("npm ", "bun ", 1)
                if command.startswith("npm "):
                    updated_command = "bun " + updated_command[4:]
                reason = f"Auto-corrected: npm → bun (per stored preference: {original_content[:80]})"
                break

        # Python package manager corrections
        elif "must use uv" in content:
            if command.startswith("pip install") or " pip install" in command:
                updated_command = updated_command.replace("pip install", "uv pip install")
                reason = f"Auto-corrected: pip → uv (per stored preference: {original_content[:80]})"
                break
            if command.startswith("pip ") or " pip " in command:
                # For other pip commands, just prefix with uv
                if command.startswith("pip "):
                    updated_command = "uv " + command
                else:
                    updated_command = updated_command.replace(" pip ", " uv pip ")
                reason = f"Auto-corrected: pip → uv (per stored preference: {original_content[:80]})"
                break

        elif "must use poetry" in content:
            if command.startswith("pip install"):
                # Extract package names
                packages = command.replace("pip install", "").strip()
                updated_command = f"poetry add {packages}"
                reason = f"Auto-corrected: pip install → poetry add (per stored preference: {original_content[:80]})"
                break
            if command.startswith("pip "):
                # Don't auto-modify other pip commands, just remind
                # (We'll let the reminder handle this case)
                _ = f"Poetry preferred (per stored preference: {original_content[:80]})"

        # Python interpreter corrections
        elif "must use python3" in content and command.startswith("python "):
            updated_command = "python3" + command[6:]
            reason = f"Auto-corrected: python → python3 (per stored preference: {original_content[:80]})"
            break

        # Git branch corrections
        elif "must not" in content and "push" in content and "main" in content:
            if "git push" in command and "main" in command:
                # Don't auto-modify but flag for user attention
                # This is a safety issue, so we'll just remind
                pass

        # Test runner corrections
        elif "must use pytest" in content:
            if command.startswith("python -m unittest") or "unittest" in command:
                # Extract test path if present
                parts = command.split()
                test_path = parts[-1] if len(parts) > 1 else ""
                updated_command = f"pytest {test_path}".strip()
                reason = f"Auto-corrected: unittest → pytest (per stored preference: {original_content[:80]})"
                break

        elif "must use vitest" in content:
            if "jest" in command:
                updated_command = updated_command.replace("jest", "vitest")
                reason = f"Auto-corrected: jest → vitest (per stored preference: {original_content[:80]})"
                break

    # Return modification if any change was made
    if updated_command != command and reason:
        return ({"command": updated_command}, reason)

    return None


def format_reminder(memories: list[dict], tool_name: str, tool_input: dict) -> str | None:
    """Format relevant memories as a reminder string.

    Args:
        memories: List of memory dictionaries from theo
        tool_name: Name of the tool being used
        tool_input: Tool input dictionary

    Returns:
        Formatted reminder string or None if no relevant memories
    """
    if not memories:
        return None

    # Filter to high-confidence or high-importance memories
    relevant = [
        m for m in memories
        if m.get("confidence", 0) >= 0.5 or m.get("importance", 0) >= 0.7
        or m.get("type") == "golden_rule"
    ]

    if not relevant:
        return None

    # Build reminder
    lines = ["# Memory Reminder"]

    # Add context about what triggered this
    if tool_name == "Bash":
        command = tool_input.get("command", "")[:50]
        lines.append(f"*Before running: `{command}...`*\n")
    elif tool_name == "Write":
        file_path = Path(tool_input.get("file_path", "")).name
        lines.append(f"*Before writing to: `{file_path}`*\n")

    # Add relevant memories (max 3)
    for mem in relevant[:3]:
        mem_type = mem.get("type", "memory").upper()
        content = mem.get("content", "")

        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."

        # Mark golden rules specially
        if mem.get("type") == "golden_rule":
            lines.append(f"**[GOLDEN RULE]** {content}")
        else:
            lines.append(f"- [{mem_type}] {content}")

    return "\n".join(lines)


def main():
    """Main hook entry point.

    Reads tool call data, searches for relevant memories, and either:
    1. Outputs JSON with updatedInput if modifications needed
    2. Outputs context reminders as text

    Failures are silent to avoid blocking the agent.
    """
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-precontext.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")

    log("PreToolUse hook triggered")

    # Supported tools for memory context injection
    SUPPORTED_TOOLS = {
        "Task", "Bash", "Glob", "Grep", "Read", "Edit", "MultiEdit",
        "Write", "WebFetch", "WebSearch",
    }

    try:
        # Read hook input from stdin
        hook_input = read_hook_input()

        tool_name = hook_input.get("tool_name") or hook_input.get("toolName", "")
        tool_input = hook_input.get("tool_input") or hook_input.get("toolInput", {})
        cwd = hook_input.get("cwd")

        log(f"tool={tool_name}")

        # Change to session's working directory if provided
        if cwd:
            os.chdir(cwd)

        # Check if tool is supported (including MCP tools)
        is_mcp = tool_name.startswith("mcp__")
        if tool_name not in SUPPORTED_TOOLS and not is_mcp:
            log("tool not supported, exiting")
            return

        # Extract query from tool input
        query = extract_query(tool_name, tool_input)
        if not query or not query.strip():
            log("no query extracted, exiting")
            return

        log(f"query: {query[:100]}")

        # Get project namespace
        namespace = get_project_namespace()
        log(f"namespace={namespace}")

        result = call_theo("memory_recall", {
            "query": query,
            "n_results": 5,
            "namespace": namespace,
            "include_related": True,
            "max_depth": 1,
        })

        if not result.get("success"):
            log(f"theo failed: {result.get('error')}")
            return

        memories = result.get("memories", [])

        # Include graph-expanded memories
        for expanded in result.get("expanded", []):
            memories.append(expanded)

        log(f"found {len(memories)} memories")

        # Also check global namespace if project namespace returned few results
        if len(memories) < 2 and namespace != "global":
            global_result = call_theo("memory_recall", {
                "query": query,
                "n_results": 3,
                "namespace": "global",
                "include_related": True,
                "max_depth": 1,
            })
            if global_result.get("success"):
                global_count = len(global_result.get("memories", []))
                memories.extend(global_result.get("memories", []))
                for expanded in global_result.get("expanded", []):
                    memories.append(expanded)
                log(f"added {global_count} global memories")

        # Try to apply input modifications first
        modification = apply_input_modifications(tool_name, tool_input, memories)
        if modification:
            updated_input, reason = modification
            # Output JSON response with updatedInput
            hook_output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": reason,
                    "updatedInput": updated_input,
                },
            }
            print(json.dumps(hook_output))
            log(f"modified input: {reason}")
            return

        # No modification needed, output reminder as usual
        reminder = format_reminder(memories, tool_name, tool_input)
        if reminder:
            print(reminder)
            log(f"output reminder: {len(reminder)} chars")
        else:
            log("no reminder to output")

        log("done")

    except BrokenPipeError:
        # Agent closed connection - this is fine
        log("broken pipe")
    except Exception as e:
        # Silently fail - don't block the agent
        log(f"ERROR: {e}")


if __name__ == "__main__":
    main()
