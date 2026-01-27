#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "pydantic",
#     "pydantic-settings",
# ]
# ///
"""Claude Code / Factory PermissionRequest hook for learned permission automation.

This hook runs when a tool requires user permission and automates approval/denial
based on stored memory patterns. It learns from user decisions and builds up
permission patterns over time.

PermissionRequest hook - fires before showing permission dialog to user.

Architecture:
    1. Read permission request from stdin (tool_name, tool_input, etc.)
    2. Query theo for past permission patterns using semantic search
    3. Apply decision rules based on memory content
    4. Track permission history for learning (auto-store patterns after 3+ approvals)

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json:

    .. code-block:: json

        {
            "hooks": {
                "PermissionRequest": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python /path/to/theo-permissions.py",
                                "timeout": 5
                            }
                        ]
                    }
                ]
            }
        }

Input (via stdin JSON):
    - tool_name: Name of the tool requiring permission (e.g., "Bash")
    - tool_input: Tool-specific input dictionary
    - session_id: Current session identifier
    - cwd: Current working directory
    - hook_event_name: Always "PermissionRequest"

Output (to stdout JSON):
    hookSpecificOutput containing decision with behavior ("allow" or "deny"),
    optional updatedInput, message, and interrupt flag.

Decision Rules (priority order):
    1. "always deny {tool} {pattern}" -> auto-deny
    2. "never allow {pattern}" -> auto-deny
    3. "always approve {tool} {pattern}" -> auto-approve
    4. "always allow {pattern}" -> auto-approve
    5. "ask for {tool}" -> show permission dialog (default)
    6. Default: no output (let user decide)

Learning:
    - After 3+ approvals of same pattern -> store "always approve" preference
    - After 2+ denials of same pattern -> store "always deny" preference
    - Patterns tracked in ~/.claude/hooks/logs/permission_history.json
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from theo_client import DaemonClient

if TYPE_CHECKING:
    from typing import Any

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LOG_DIR = Path.home() / ".claude" / "hooks" / "logs"
LOG_FILE = LOG_DIR / "theo-permissions.log"
HISTORY_FILE = LOG_DIR / "permission_history.json"

THEO_TIMEOUT_SECONDS = 4
CONTEXT_MAX_LENGTH = 200
HASH_PREFIX_LENGTH = 16

APPROVAL_THRESHOLD = 3
DENIAL_THRESHOLD = 2

APPROVAL_IMPORTANCE = 0.85
DENIAL_IMPORTANCE = 0.90

RELEVANCE_THRESHOLD = 0.5

PROJECT_INDICATORS = frozenset({
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
})

DENY_PATTERNS = (
    "always deny",
    "never allow",
    "must not allow",
    "must not approve",
    "do not allow",
    "do not approve",
)

APPROVE_PATTERNS = (
    "always approve",
    "always allow",
    "must allow",
    "must approve",
    "auto-approve",
    "auto approve",
)

ASK_PATTERNS = ("ask", "confirm", "prompt", "request permission")

PERMISSION_KEYWORDS = frozenset({
    "permission",
    "approve",
    "deny",
    "allow",
    "reject",
    "always approve",
    "always deny",
    "never allow",
    "auto-approve",
})


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class PermissionPattern:
    """Tracks approval/denial history for a permission pattern.

    Attributes:
        tool: Name of the tool (e.g., "Bash", "Write").
        context: Truncated context string describing the operation.
        approvals: Number of times this pattern was approved.
        denials: Number of times this pattern was denied.
        last_decision: Most recent decision ("approve" or "deny").
        last_timestamp: ISO format timestamp of last decision.
    """

    tool: str
    context: str
    approvals: int = 0
    denials: int = 0
    last_decision: str = ""
    last_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON storage.
        """
        return {
            "tool": self.tool,
            "context": self.context,
            "approvals": self.approvals,
            "denials": self.denials,
            "last_decision": self.last_decision,
            "last_timestamp": self.last_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionPattern:
        """Create instance from dictionary.

        Args:
            data: Dictionary with pattern data.

        Returns:
            New PermissionPattern instance.
        """
        return cls(
            tool=data.get("tool", ""),
            context=data.get("context", ""),
            approvals=data.get("approvals", 0),
            denials=data.get("denials", 0),
            last_decision=data.get("last_decision", ""),
            last_timestamp=data.get("last_timestamp", ""),
        )


@dataclass
class PermissionHistory:
    """Container for permission pattern history.

    Attributes:
        patterns: Mapping from pattern hash to PermissionPattern.
    """

    patterns: dict[str, PermissionPattern] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with serialized patterns.
        """
        return {
            "patterns": {
                key: pattern.to_dict() for key, pattern in self.patterns.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionHistory:
        """Create instance from dictionary.

        Args:
            data: Dictionary with history data.

        Returns:
            New PermissionHistory instance.
        """
        patterns = {}
        for key, value in data.get("patterns", {}).items():
            patterns[key] = PermissionPattern.from_dict(value)
        return cls(patterns=patterns)


@dataclass
class PermissionDecision:
    """Represents an automated permission decision.

    Attributes:
        behavior: Either "allow" or "deny".
        message: Optional explanation message.
        interrupt: Whether to interrupt the agent.
    """

    behavior: str
    message: str = ""
    interrupt: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output.

        Returns:
            Dictionary suitable for hook output.
        """
        result: dict[str, Any] = {
            "behavior": self.behavior,
            "interrupt": self.interrupt,
        }
        if self.message:
            result["message"] = self.message
        return result


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def create_logger() -> Callable[[str], None]:
    """Create a logging function that writes to the log file.

    Returns:
        Function that accepts a message string and writes it to the log.
    """
    from datetime import datetime

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        """Write timestamped log message to file.

        Args:
            msg: Message to log.
        """
        try:
            with LOG_FILE.open("a") as f:
                f.write(f"{datetime.now().isoformat()} | {msg}\n")
        except OSError:
            pass

    return log


# -----------------------------------------------------------------------------
# Input/Output
# -----------------------------------------------------------------------------


def read_hook_input() -> dict[str, Any]:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin. Returns empty dict
    if stdin is a TTY or if parsing fails.

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
    except (json.JSONDecodeError, OSError):
        pass

    return {}


def output_decision(decision: PermissionDecision) -> None:
    """Output permission decision as JSON to stdout.

    Args:
        decision: The decision to output.
    """
    import json

    reflection_prompt = """🔐 PERMISSION DECISION - Memory Check

A permission pattern was just applied. Consider storing:
- New trust patterns the user established
- Security boundaries they prefer
- Tool usage preferences revealed

Store via mcp-exec:
```javascript
await theo.memory_store_tool({
  content: "Permission pattern: ...",
  memory_type: "preference",  // or "pattern", "decision", "golden_rule"
  importance: 0.0-1.0,  // 0.3 minor, 0.5 useful, 0.7 important, 0.9+ critical/golden_rule
  namespace: "global"  // or "project:name"
});
```
Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]"""

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": decision.to_dict(),
        },
        "additionalContext": reflection_prompt,
    }
    print(json.dumps(output))


# -----------------------------------------------------------------------------
# Project Detection
# -----------------------------------------------------------------------------


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Checks for common project indicator files to determine if we're
    in a project directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'.
    """
    import os

    cwd = Path(os.getcwd())

    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return f"project:{cwd.name}"

    return "global"


# -----------------------------------------------------------------------------
# Recall Integration
# -----------------------------------------------------------------------------


def get_daemon_client() -> DaemonClient:
    """Get a DaemonClient instance with default settings.

    Returns:
        Configured DaemonClient for IPC communication.
    """
    return DaemonClient(
        connect_timeout=2.0,
        request_timeout=float(THEO_TIMEOUT_SECONDS),
        auto_fallback=True,
    )


# -----------------------------------------------------------------------------
# Context Extraction
# -----------------------------------------------------------------------------


def extract_permission_context(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract searchable context from permission request.

    Builds a context string optimized for semantic search based on
    the tool type and its input.

    Args:
        tool_name: Name of the tool requiring permission.
        tool_input: Tool input dictionary.

    Returns:
        Context string for semantic search.
    """
    context_parts = [tool_name]

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        context_parts.append(command[:CONTEXT_MAX_LENGTH])

    elif tool_name in ("Write", "Edit", "MultiEdit"):
        file_path = tool_input.get("file_path", "")
        context_parts.append(f"write {file_path}")

    elif tool_name == "Task":
        subagent_type = tool_input.get("subagent_type", "")
        description = tool_input.get("description", "")
        context_parts.extend([subagent_type, description[:100]])

    elif tool_name == "WebFetch":
        url = tool_input.get("url", "")
        context_parts.append(f"fetch {url}")

    elif tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        context_parts.extend(parts[1:])
        for key, value in tool_input.items():
            if isinstance(value, str) and len(value) < 100:
                context_parts.append(f"{key}={value}")

    return " ".join(context_parts)


# -----------------------------------------------------------------------------
# Pattern Hashing
# -----------------------------------------------------------------------------


def compute_pattern_hash(tool_name: str, context: str) -> str:
    """Compute a hash for the permission pattern.

    Used for tracking repeated approvals/denials to enable learning.

    Args:
        tool_name: Tool name.
        context: Permission context.

    Returns:
        SHA256 hash prefix (first 16 characters).
    """
    import hashlib

    pattern = f"{tool_name}:{context}"
    return hashlib.sha256(pattern.encode()).hexdigest()[:HASH_PREFIX_LENGTH]


# -----------------------------------------------------------------------------
# History Management
# -----------------------------------------------------------------------------


def load_permission_history() -> PermissionHistory:
    """Load permission history from JSON file.

    Creates the log directory if it doesn't exist.

    Returns:
        PermissionHistory instance with loaded or empty patterns.
    """
    import json

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_FILE.exists():
        return PermissionHistory()

    try:
        with HISTORY_FILE.open() as f:
            data = json.load(f)
            return PermissionHistory.from_dict(data)
    except (json.JSONDecodeError, OSError):
        return PermissionHistory()


def save_permission_history(history: PermissionHistory) -> None:
    """Save permission history to JSON file.

    Args:
        history: PermissionHistory instance to save.
    """
    import json

    try:
        with HISTORY_FILE.open("w") as f:
            json.dump(history.to_dict(), f, indent=2)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Learning
# -----------------------------------------------------------------------------


def _store_learned_pattern(
    tool_name: str,
    context: str,
    pattern_type: str,
    namespace: str,
    log_func: Callable[[str], None],
) -> None:
    """Store a learned permission pattern in theo.

    Args:
        tool_name: Tool name for the pattern.
        context: Context string (will be truncated).
        pattern_type: Either "approve" or "deny".
        namespace: Project namespace for storage.
        log_func: Logging function.
    """
    if pattern_type == "approve":
        content = f"You MUST always approve {tool_name} for: {context[:150]}"
        importance = APPROVAL_IMPORTANCE
        tag = "auto-approve"
    else:
        content = f"You MUST always deny {tool_name} for: {context[:150]}"
        importance = DENIAL_IMPORTANCE
        tag = "auto-deny"

    log_func(f"Learning: storing {tag} pattern for {tool_name}")

    with get_daemon_client() as client:
        result = client.store(
            content=content,
            namespace=namespace,
            memory_type="preference",
            importance=importance,
            metadata={"tags": ["permission", tag, tool_name.lower()]},
        )

    if result.get("success"):
        pattern_hash = compute_pattern_hash(tool_name, context)
        log_func(f"Stored {tag} preference: {pattern_hash}")


def record_permission_decision(
    tool_name: str,
    context: str,
    decision: str,
    log_func: Callable[[str], None],
) -> None:
    """Record a permission decision and potentially learn from it.

    If pattern occurs 3+ times with approval, stores "always approve".
    If pattern occurs 2+ times with denial, stores "always deny".

    Args:
        tool_name: Tool name.
        context: Permission context.
        decision: Either "approve" or "deny".
        log_func: Logging function.
    """
    from datetime import datetime

    pattern_hash = compute_pattern_hash(tool_name, context)
    history = load_permission_history()

    if pattern_hash not in history.patterns:
        history.patterns[pattern_hash] = PermissionPattern(
            tool=tool_name,
            context=context[:CONTEXT_MAX_LENGTH],
        )

    pattern = history.patterns[pattern_hash]

    if decision == "approve":
        pattern.approvals += 1
    else:
        pattern.denials += 1

    pattern.last_decision = decision
    pattern.last_timestamp = datetime.now().isoformat()

    save_permission_history(history)

    namespace = get_project_namespace()

    if pattern.approvals >= APPROVAL_THRESHOLD and pattern.denials == 0:
        _store_learned_pattern(tool_name, context, "approve", namespace, log_func)
    elif pattern.denials >= DENIAL_THRESHOLD and pattern.approvals == 0:
        _store_learned_pattern(tool_name, context, "deny", namespace, log_func)


# -----------------------------------------------------------------------------
# Memory Querying
# -----------------------------------------------------------------------------


def query_permission_memories(
    tool_name: str,
    context: str,
    namespace: str,
    log_func: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Query theo for relevant permission patterns.

    Searches both project and global namespaces for matching patterns.

    Args:
        tool_name: Tool name.
        context: Permission context.
        namespace: Project namespace.
        log_func: Logging function.

    Returns:
        List of relevant memory dictionaries.
    """
    query = f"permission {tool_name} {context}"
    log_func(f"Searching memories: query='{query[:100]}' namespace={namespace}")

    memories: list[dict[str, Any]] = []

    with get_daemon_client() as client:
        result = client.fetch(
            namespace=namespace,
            query=query,
            n_results=5,
            include_related=True,
            max_depth=1,
        )

        if result.get("success"):
            data = result.get("data", {})
            memories.extend(data.get("memories", []))
            for expanded in data.get("expanded", []):
                memories.append(expanded)

        if namespace != "global":
            global_result = client.fetch(
                namespace="global",
                query=query,
                n_results=3,
                include_related=True,
                max_depth=1,
            )

            if global_result.get("success"):
                global_data = global_result.get("data", {})
                memories.extend(global_data.get("memories", []))

    log_func(f"Found {len(memories)} relevant memories")
    return memories


# -----------------------------------------------------------------------------
# Memory Analysis
# -----------------------------------------------------------------------------


def _calculate_relevance(
    memory_content: str,
    tool_name: str,
    context_lower: str,
) -> float:
    """Calculate how relevant a memory is to the current permission request.

    Args:
        memory_content: Memory content (lowercase).
        tool_name: Tool name.
        context_lower: Permission context (lowercase).

    Returns:
        Relevance score between 0.0 and 1.0.
    """
    has_permission_keyword = any(kw in memory_content for kw in PERMISSION_KEYWORDS)
    if not has_permission_keyword:
        return 0.0

    score = 0.0

    if tool_name.lower() in memory_content:
        score += 0.5

    context_words = [w for w in context_lower.split() if len(w) > 3]
    if context_words:
        matching_words = sum(
            1 for word in context_words[:10] if word in memory_content
        )
        score += 0.3 * (matching_words / min(len(context_words), 10))

    if tool_name == "Bash":
        cmd_parts = context_lower.split()
        if len(cmd_parts) > 1:
            cmd_word = cmd_parts[1][:20]
            if cmd_word in memory_content:
                score += 0.2

    return min(score, 1.0)


def _check_pattern_match(
    content: str,
    patterns: tuple[str, ...],
) -> str | None:
    """Check if content matches any of the given patterns.

    Args:
        content: Lowercase content to check.
        patterns: Tuple of patterns to match.

    Returns:
        Matched pattern string, or None if no match.
    """
    for pattern in patterns:
        if pattern in content:
            return pattern
    return None


def analyze_permission_memories(
    memories: list[dict[str, Any]],
    tool_name: str,
    context: str,
    log_func: Callable[[str], None],
) -> PermissionDecision | None:
    """Analyze memories and determine if auto-decision should be made.

    Decision priority (highest to lowest):
        1. "always deny" / "never allow" / "MUST NOT" -> deny
        2. "always approve" / "always allow" / "MUST" -> allow
        3. "ask" / "confirm" -> let user decide (return None)

    Args:
        memories: List of memory dictionaries.
        tool_name: Tool name.
        context: Permission context.
        log_func: Logging function.

    Returns:
        PermissionDecision if auto-decision made, None to let user decide.
    """
    if not memories:
        return None

    sorted_memories = sorted(
        memories,
        key=lambda m: (m.get("importance", 0), m.get("confidence", 0)),
        reverse=True,
    )

    context_lower = context.lower()

    for mem in sorted_memories:
        content = mem.get("content", "").lower()
        mem_type = mem.get("type", "")
        importance = mem.get("importance", 0)

        log_func(
            f"Analyzing memory: type={mem_type} imp={importance:.2f} "
            f"content='{content[:100]}'",
        )

        relevance = _calculate_relevance(content, tool_name, context_lower)
        log_func(f"  Relevance score: {relevance:.2f}")

        if relevance < RELEVANCE_THRESHOLD:
            log_func("  Skipping - not relevant enough")
            continue

        matched_deny = _check_pattern_match(content, DENY_PATTERNS)
        if matched_deny:
            log_func(
                f"DENY triggered by pattern: '{matched_deny}' "
                f"(relevance={relevance:.2f})",
            )
            return PermissionDecision(
                behavior="deny",
                message=(
                    f"Denied by learned pattern: "
                    f"{mem.get('content', 'security policy')[:200]}"
                ),
            )

        matched_approve = _check_pattern_match(content, APPROVE_PATTERNS)
        if matched_approve:
            log_func(
                f"APPROVE triggered by pattern: '{matched_approve}' "
                f"(relevance={relevance:.2f})",
            )
            return PermissionDecision(behavior="allow")

        matched_ask = _check_pattern_match(content, ASK_PATTERNS)
        if matched_ask:
            log_func(f"ASK triggered - letting user decide (relevance={relevance:.2f})")
            return None

    log_func("No matching pattern found - defaulting to user decision")
    return None


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Main hook entry point.

    Reads permission request, queries memories, applies decision rules,
    and outputs decision or lets user decide.
    """
    import os

    log = create_logger()
    log("PermissionRequest hook triggered")

    try:
        hook_input = read_hook_input()
        if not hook_input:
            log("No hook input - exiting")
            return

        tool_name = hook_input.get("tool_name") or hook_input.get("toolName", "")
        tool_input = hook_input.get("tool_input") or hook_input.get("toolInput", {})
        cwd = hook_input.get("cwd")

        if not tool_name:
            log("No tool_name - exiting")
            return

        log(f"Permission request: tool={tool_name}")

        if cwd:
            os.chdir(cwd)

        context = extract_permission_context(tool_name, tool_input)
        log(f"Context: {context[:100]}")

        namespace = get_project_namespace()

        memories = query_permission_memories(tool_name, context, namespace, log)

        decision = analyze_permission_memories(memories, tool_name, context, log)

        if decision:
            output_decision(decision)
            log(f"Decision: {decision.behavior}")
        else:
            log("No auto-decision - letting user decide")

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
