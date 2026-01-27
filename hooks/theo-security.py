#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Claude Code / Factory PreToolUse hook for security validation.

This hook runs BEFORE tool calls and validates commands for security risks:
    - Blocks dangerous bash patterns (exit code 2)
    - Protects sensitive files (requires confirmation)
    - Warns about production paths (requires confirmation)
    - Checks theo memories for project-specific security rules

Supported tools: Bash, Write, Edit, MultiEdit

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or
    ~/.factory/settings.json (Factory)::

        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash|Write|Edit|MultiEdit",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python /path/to/theo/hooks/theo-security.py",
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
        "tool_input": {"command": "rm -rf /"},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

Output:
    - Exit code 0: Allow operation (no output or informational output)
    - Exit code 2: Block operation (prints error to stderr)
    - JSON with permissionDecision: "ask" for confirmation required

Security Checks:
    1. Dangerous bash patterns (blocks immediately)
    2. Sensitive file protection (requires confirmation)
    3. Production path warnings (requires confirmation)
    4. Memory-based security rules (blocks or allows based on preferences)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# =============================================================================
# Constants
# =============================================================================

# Exit codes
EXIT_SUCCESS = 0
EXIT_BLOCK = 2

# Subprocess timeout in seconds
THEO_TIMEOUT_SECONDS = 3

# Dangerous bash patterns that should be blocked immediately
# Each tuple contains (regex_pattern, human_readable_description)
DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"rm\s+-rf\s+/", "Recursive delete from root directory"),
    (r"rm\s+-rf\s+~", "Recursive delete of home directory"),
    (r"rm\s+-rf\s+\$HOME", "Recursive delete of home directory"),
    (r">\s*/etc/", "Write to /etc directory"),
    (r"chmod\s+777", "World-writable permissions"),
    (r"\|\s*bash", "Piped bash execution"),
    (r"curl.*\|\s*sh", "Remote code execution via curl"),
    (r"wget.*\|\s*sh", "Remote code execution via wget"),
    (r"dd\s+if=.*of=/dev/", "Raw device write access"),
    (r"mkfs\.", "Filesystem format command"),
    (r":.*\(\).*{.*}.*;.*:", "Fork bomb pattern"),
    (r":()\s*{\s*:\|:\&\s*};:", "Fork bomb pattern (compact)"),
    (r"eval.*\$\(", "Eval with command substitution"),
    (r"rm\s+-rf\s+\*", "Recursive delete with wildcard"),
    (r"rm\s+-rf\s+/\*", "Recursive delete from root with wildcard"),
)

# Sensitive files that require confirmation before modification
SENSITIVE_FILES: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    "credentials.json",
    "secrets.yaml",
    "secrets.json",
    "secrets.yml",
    ".aws/credentials",
    ".aws/config",
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".ssh/id_ecdsa",
    ".ssh/id_dsa",
    ".ssh/config",
    ".ssh/authorized_keys",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".dockercfg",
    ".docker/config.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".gitconfig",
    ".git-credentials",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "apikeys.json",
    "api_keys.json",
    "api-keys.json",
)

# Path patterns that indicate production environments
PRODUCTION_PATTERNS: tuple[str, ...] = (
    r"/prod/",
    r"/production/",
    r"/live/",
    r"-prod\.",
    r"-production\.",
    r"\.prod\.",
    r"prod-",
    r"production-",
)

# Hook event configuration
HOOK_EVENT_NAME = "PreToolUse"
PERMISSION_DECISION_ASK = "ask"

# Supported file operation tools
FILE_OPERATION_TOOLS = frozenset({"Write", "Edit", "MultiEdit"})


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of a security validation check.

    Attributes:
        action: The action to take ('block', 'ask', or 'allow').
        reason: Human-readable explanation of the validation result.
    """

    action: str
    reason: str

    @property
    def should_block(self) -> bool:
        """Check if this result indicates the operation should be blocked."""
        return self.action == "block"

    @property
    def should_ask(self) -> bool:
        """Check if this result indicates confirmation is required."""
        return self.action == "ask"


@dataclass(frozen=True, slots=True)
class HookInput:
    """Parsed input from Claude Code hook system.

    Attributes:
        tool_name: Name of the tool being invoked.
        tool_input: Dictionary of tool input parameters.
        cwd: Current working directory.
    """

    tool_name: str
    tool_input: dict[str, Any]
    cwd: str


# =============================================================================
# Input/Output Functions
# =============================================================================


def read_hook_input() -> HookInput:
    """Read and parse hook input from stdin.

    Claude Code passes hook data as JSON via stdin. This function reads
    that data and returns a structured HookInput object.

    Returns:
        HookInput with parsed data, or default empty values if unavailable.

    Raises:
        No exceptions are raised; errors result in default empty values.
    """
    import json

    default_cwd = str(Path.cwd())

    if sys.stdin.isatty():
        return HookInput(tool_name="", tool_input={}, cwd=default_cwd)

    try:
        stdin_data = sys.stdin.read()
        if not stdin_data:
            return HookInput(tool_name="", tool_input={}, cwd=default_cwd)

        data = json.loads(stdin_data)
        return HookInput(
            tool_name=data.get("tool_name") or data.get("toolName", ""),
            tool_input=data.get("tool_input") or data.get("toolInput", {}),
            cwd=data.get("cwd", default_cwd),
        )
    except (OSError, json.JSONDecodeError):
        return HookInput(tool_name="", tool_input={}, cwd=default_cwd)


def log_security_event(event_type: str, details: str, cwd: str) -> None:
    """Log security events to a dedicated log file.

    Logs are written to ~/.claude/hooks/logs/security.log with ISO timestamps.

    Args:
        event_type: Type of event (e.g., 'BLOCK', 'WARN', 'ALLOW', 'ERROR').
        details: Descriptive details about the security event.
        cwd: Current working directory where the event occurred.
    """
    import datetime

    try:
        log_dir = Path.home() / ".claude" / "hooks" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "security.log"

        timestamp = datetime.datetime.now().isoformat()

        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{event_type}] {cwd}: {details}\n")
    except OSError:
        # Don't fail if logging fails
        pass


def output_ask_permission(reason: str) -> None:
    """Output JSON to request user confirmation.

    Args:
        reason: Human-readable reason for requesting confirmation.
    """
    import json

    output = {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT_NAME,
            "permissionDecision": PERMISSION_DECISION_ASK,
            "permissionDecisionReason": reason,
        },
    }
    print(json.dumps(output))


# =============================================================================
# Pattern Checking Functions
# =============================================================================


def check_dangerous_patterns(command: str) -> tuple[str, str] | None:
    """Check if command contains dangerous patterns.

    Scans the command against known dangerous bash patterns that could
    cause system damage or security vulnerabilities.

    Args:
        command: Bash command to check.

    Returns:
        Tuple of (matched_pattern, description) if dangerous pattern found,
        None if the command appears safe.
    """
    import re

    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return (pattern, description)
    return None


def is_sensitive_file(file_path: str) -> bool:
    """Check if file path refers to a sensitive file.

    Sensitive files include credentials, secrets, SSH keys, and other
    security-critical files that require confirmation before modification.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file is considered sensitive, False otherwise.
    """
    path = Path(file_path)
    filename = path.name.lower()
    full_path = str(path).lower()

    for sensitive in SENSITIVE_FILES:
        if sensitive.startswith("*"):
            # Wildcard pattern (e.g., "*.key")
            extension = sensitive[1:]
            if filename.endswith(extension):
                return True
        elif "/" in sensitive:
            # Path pattern (e.g., ".ssh/id_rsa")
            if sensitive.lower() in full_path:
                return True
        else:
            # Exact filename match
            sensitive_lower = sensitive.lower()
            if filename == sensitive_lower:
                return True

    return False


def is_production_path(path: str) -> bool:
    """Check if path indicates a production environment.

    Args:
        path: File path or command string to check.

    Returns:
        True if the path matches production environment patterns.
    """
    import re

    path_lower = path.lower()
    return any(re.search(pattern, path_lower) for pattern in PRODUCTION_PATTERNS)


# =============================================================================
# Recall Integration
# =============================================================================


def _find_theo_directory() -> Path | None:
    """Find the theo installation directory.

    Searches common installation locations for the theo package.

    Returns:
        Path to theo directory if found, None otherwise.
    """
    theo_paths = (
        Path(__file__).parent.parent,
        Path.home() / "Github" / "theo",
        Path.home() / ".local" / "share" / "theo",
        Path("/opt/theo"),
    )

    for path in theo_paths:
        if (path / "src" / "theo" / "__main__.py").exists():
            return path
    return None


def call_theo(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call theo MCP tool directly via --call mode.

    Uses the theo CLI to invoke memory operations for checking
    project-specific security rules.

    Args:
        tool_name: Name of the theo tool (e.g., 'memory_recall').
        args: Dictionary of tool arguments.

    Returns:
        Tool result as dictionary. On failure, returns a dict with
        'success': False and 'error' describing the failure.

    Note:
        Uses process groups (start_new_session=True) to ensure all child
        processes are killed on timeout, preventing zombie processes.
    """
    import json
    import os
    import signal
    import subprocess

    theo_dir = _find_theo_directory()

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
        working_dir = Path.cwd()
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
        working_dir = theo_dir

    proc = None
    try:
        # Use Popen with start_new_session=True to create a process group
        # This ensures we can kill all child processes on timeout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            start_new_session=True,  # Creates new process group
        )

        try:
            stdout, stderr = proc.communicate(timeout=THEO_TIMEOUT_SECONDS)
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
    except OSError as e:
        return {"success": False, "error": str(e)}
    finally:
        # Ensure cleanup if proc exists and hasn't been waited on
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            proc.wait()


def check_memory_security_rules(
    command: str,
    file_path: str | None = None,
) -> ValidationResult | None:
    """Check theo memories for security rules.

    Searches stored memories for security rules that match the current
    operation. Rules can either block or explicitly allow operations.

    Looks for patterns like:
        - "security block {pattern}" - patterns to block
        - "security allow {pattern}" - exceptions to allow

    Args:
        command: Command or operation being performed.
        file_path: File path if applicable.

    Returns:
        ValidationResult with action and reason if a matching rule is found,
        None if no rules match.
    """
    query = f"{command} {file_path}" if file_path else command

    result = call_theo(
        "memory_recall",
        {
            "query": f"security block allow {query}",
            "n_results": 5,
            "namespace": "global",
            "include_related": False,
        },
    )

    if not result.get("success"):
        return None

    memories = result.get("memories", [])
    query_terms = query.lower().split()[:3]

    # Check for explicit blocks first
    for mem in memories:
        content = mem.get("content", "").lower()
        if "security" in content and "block" in content:
            if any(term in content for term in query_terms):
                return ValidationResult(
                    action="block",
                    reason=f"Memory rule: {mem.get('content', '')[:100]}",
                )

    # Check for explicit allows (overrides default behavior)
    for mem in memories:
        content = mem.get("content", "").lower()
        if "security" in content and "allow" in content:
            if any(term in content for term in query_terms):
                return ValidationResult(
                    action="allow",
                    reason="Allowed by memory rule",
                )

    return None


# =============================================================================
# Validation Functions
# =============================================================================


def validate_bash_command(command: str, cwd: str) -> ValidationResult | None:
    """Validate bash command for security issues.

    Checks the command against memory-based rules, dangerous patterns,
    and production path indicators.

    Args:
        command: Bash command to validate.
        cwd: Current working directory for logging.

    Returns:
        ValidationResult if security issue found, None if command is safe.
    """
    # Check memory-based rules first (can override defaults)
    memory_rule = check_memory_security_rules(command)
    if memory_rule is not None:
        if memory_rule.action == "allow":
            log_security_event("ALLOW", f"Command allowed by memory: {command}", cwd)
            return None
        if memory_rule.action == "block":
            log_security_event("BLOCK", f"Command blocked by memory: {command}", cwd)
            return memory_rule

    # Check dangerous patterns
    dangerous = check_dangerous_patterns(command)
    if dangerous is not None:
        pattern, description = dangerous
        log_security_event("BLOCK", f"Dangerous pattern '{pattern}': {command}", cwd)
        return ValidationResult(
            action="block",
            reason=f"Dangerous pattern detected: {description}",
        )

    # Check for production paths
    if is_production_path(command):
        log_security_event("WARN", f"Production path in command: {command}", cwd)
        return ValidationResult(
            action="ask",
            reason="Production environment detected in command",
        )

    return None


def validate_file_operation(
    file_path: str,
    operation: str,
    cwd: str,
) -> ValidationResult | None:
    """Validate file write/edit operation for security issues.

    Checks file operations against memory-based rules, sensitive file
    patterns, and production path indicators.

    Args:
        file_path: Path to file being modified.
        operation: Type of operation (Write, Edit, MultiEdit).
        cwd: Current working directory for logging.

    Returns:
        ValidationResult if security issue found, None if operation is safe.
    """
    # Check memory-based rules first
    memory_rule = check_memory_security_rules(operation, file_path)
    if memory_rule is not None:
        if memory_rule.action == "allow":
            log_security_event(
                "ALLOW",
                f"File operation allowed by memory: {file_path}",
                cwd,
            )
            return None
        if memory_rule.action == "block":
            log_security_event(
                "BLOCK",
                f"File operation blocked by memory: {file_path}",
                cwd,
            )
            return memory_rule

    # Check sensitive files
    if is_sensitive_file(file_path):
        log_security_event("WARN", f"Sensitive file operation: {file_path}", cwd)
        filename = Path(file_path).name
        return ValidationResult(
            action="ask",
            reason=f"Sensitive file: {filename}",
        )

    # Check production paths
    if is_production_path(file_path):
        log_security_event("WARN", f"Production path file operation: {file_path}", cwd)
        return ValidationResult(
            action="ask",
            reason="Production path detected",
        )

    return None


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main hook entry point.

    Validates tool calls for security issues:
        - Blocks dangerous commands (exit code 2)
        - Requires confirmation for sensitive operations (JSON output)
        - Logs security events

    Exit Codes:
        0: Operation allowed (or confirmation requested via JSON)
        2: Operation blocked due to security violation
    """
    try:
        hook_input = read_hook_input()

        # Change to session's working directory if provided
        if hook_input.cwd:
            try:
                import os

                os.chdir(hook_input.cwd)
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                pass  # Directory may not exist yet

        # Validate based on tool type
        validation_result: ValidationResult | None = None

        if hook_input.tool_name == "Bash":
            command = hook_input.tool_input.get("command", "")
            if command:
                validation_result = validate_bash_command(command, hook_input.cwd)

        elif hook_input.tool_name in FILE_OPERATION_TOOLS:
            file_path = hook_input.tool_input.get("file_path", "")
            if file_path:
                validation_result = validate_file_operation(
                    file_path,
                    hook_input.tool_name,
                    hook_input.cwd,
                )

        # Handle validation result
        if validation_result is not None:
            if validation_result.should_block:
                print(f"BLOCKED: {validation_result.reason}", file=sys.stderr)
                sys.exit(EXIT_BLOCK)

            if validation_result.should_ask:
                output_ask_permission(validation_result.reason)
                sys.exit(EXIT_SUCCESS)

        # No issues found - allow operation silently
        sys.exit(EXIT_SUCCESS)

    except BrokenPipeError:
        # Agent closed connection - exit cleanly
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        # Log error but don't block operation
        import os

        log_security_event("ERROR", f"Hook error: {e!s}", os.getcwd())
        sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
