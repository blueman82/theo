#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Claude Code / Factory UserPromptSubmit hook for memory context injection.

This hook runs BEFORE the user's prompt is processed, allowing injection of
relevant memory context based on the prompt content. This helps Claude/Droid
make better decisions by surfacing relevant preferences and past decisions.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-prompt.py",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "session_id": "abc123",
        "transcript_path": "/path/to/transcript.jsonl",
        "cwd": "/project/root",
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Create a new React component for user settings"
    }

Output:
    - stdout: Additional context to inject (shown to Claude/Droid)
    - JSON with decision: "block" to reject prompt, or additionalContext
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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
    cwd = os.getcwd()
    project_name = Path(cwd).name

    project_indicators = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"




def call_theo(tool_name: str, args: dict) -> dict:
    """Call theo MCP tool directly via --call mode.

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
            cmd = ["uv", "run", "python", "-m", "theo", "--call", tool_name, "--args", json.dumps(args)]
        else:
            cmd = ["uv", "run", "--directory", str(theo_dir), "python", "-m", "theo", "--call", tool_name, "--args", json.dumps(args)]

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


def format_context(memories: list[dict]) -> str | None:
    """Format relevant memories as context for the prompt using RFC 2119 language."""
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

    lines = ["# Relevant Memories (RFC 2119)", ""]

    # Categorize by RFC level
    # MUST: golden_rule type first, then confidence >= 0.9
    # SHOULD: importance >= 0.8 OR confidence >= 0.7
    # MAY: everything else that passed the filter
    golden_rules = [m for m in relevant if m.get("type") == "golden_rule"]
    high_confidence = [m for m in relevant if m.get("type") != "golden_rule" and m.get("confidence", 0) >= 0.9]
    must = golden_rules + high_confidence  # Golden rules always first
    should = [m for m in relevant if m not in must and (m.get("importance", 0) >= 0.8 or m.get("confidence", 0) >= 0.7)]
    may = [m for m in relevant if m not in must and m not in should]

    if must:
        lines.append("## MUST (Required)")
        # Show all golden rules (up to 5), then up to 2 high-confidence
        shown = 0
        for mem in golden_rules[:5]:
            content = mem.get("content", "")[:300]
            lines.append(f"- {content}")
            shown += 1
        for mem in high_confidence[:max(0, 3 - shown)]:
            content = mem.get("content", "")[:300]
            lines.append(f"- {content}")
        lines.append("")

    if should:
        lines.append("## SHOULD (Recommended)")
        for mem in should[:3]:
            content = mem.get("content", "")[:250]
            lines.append(f"- {content}")
        lines.append("")

    if may:
        lines.append("## MAY (Optional)")
        for mem in may[:3]:
            content = mem.get("content", "")[:200]
            lines.append(f"- {content}")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main hook entry point."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-prompt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")

    log("UserPromptSubmit hook triggered")

    try:
        hook_input = read_hook_input()
        prompt = hook_input.get("prompt", "")
        cwd = hook_input.get("cwd", os.getcwd())

        if not prompt or len(prompt) < 5:
            log("prompt too short, exiting")
            return

        log(f"prompt length: {len(prompt)} chars")

        # Change to session's working directory
        if cwd:
            os.chdir(cwd)

        namespace = get_project_namespace()
        log(f"namespace={namespace}")

        # Use prompt directly as query - let semantic search do its job
        query = prompt

        # Search for relevant memories with graph expansion
        result = call_theo("memory_recall", {
            "query": query,
            "n_results": 10,
            "namespace": namespace,
            "include_related": True,
            "max_depth": 1,
        })

        memories = []
        if result.get("success"):
            data = result.get("data", {})
            memories = data.get("memories", [])
            # Include graph-expanded memories
            expanded = data.get("expanded", [])
            memories.extend(expanded)
            log(f"found {len(data.get('memories', []))} memories + {len(expanded)} expanded in {namespace}")
        else:
            log(f"theo failed: {result.get('error')}")

        # Also check global namespace with graph expansion
        if namespace != "global":
            global_result = call_theo("memory_recall", {
                "query": query,
                "n_results": 5,
                "namespace": "global",
                "include_related": True,
                "max_depth": 1,
            })
            if global_result.get("success"):
                global_data = global_result.get("data", {})
                global_mems = global_data.get("memories", [])
                global_expanded = global_data.get("expanded", [])
                memories.extend(global_mems)
                memories.extend(global_expanded)
                log(f"found {len(global_mems)} + {len(global_expanded)} expanded global memories")

        # ALWAYS fetch golden rules regardless of semantic match
        golden_result = call_theo("memory_list", {
            "memory_type": "golden_rule",
            "namespace": "global",
            "limit": 10,
        })
        if golden_result.get("success"):
            golden_data = golden_result.get("data", {})
            golden_rules = golden_data.get("memories", [])
            # Dedupe by ID
            existing_ids = {m.get("id") for m in memories}
            new_golden = [g for g in golden_rules if g.get("id") not in existing_ids]
            memories.extend(new_golden)
            log(f"added {len(new_golden)} golden rules (always included)")

        # Format and output context
        context = format_context(memories)
        if context:
            # Output as JSON for structured response
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                },
            }
            print(json.dumps(output))
            log(f"injected context: {len(context)} chars")
        else:
            log("no relevant context to inject")

        log("done")

    except BrokenPipeError:
        log("broken pipe")
    except Exception as e:
        # Silently fail - don't block the prompt
        log(f"ERROR: {e}")


if __name__ == "__main__":
    main()
