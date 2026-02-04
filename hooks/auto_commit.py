#!/usr/bin/env python3
"""Auto-commit hook for Claude Code PostToolUse event.

Automatically commits file changes after Write, Edit, or MultiEdit operations.
Reads configuration from auto_commit_config.json to determine commit behavior,
file patterns, and notification settings.
"""
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load_config() -> dict:
    """Load auto-commit configuration from config file.

    Loads user configuration from ~/.claude/hooks/auto_commit_config.json
    and merges it with default values for any missing settings.

    Returns:
        Configuration dictionary with all auto-commit settings.
    """
    config_path = Path.home() / ".claude" / "hooks" / "auto_commit_config.json"
    default_config = {
        "auto_commit": {
            "enabled": True,
            "commit_message_format": "smart",
            "include_timestamp": True,
            "include_tool_name": True,
            "max_message_length": 200,
            "batch_commits": False,
            "batch_timeout_seconds": 5,
        },
        "git_settings": {
            "auto_init_repo": True,
            "default_branch": "main",
            "create_initial_commit": True,
            "auto_push": False,
        },
        "file_patterns": {
            "exclude": [
                "*.log",
                "*.tmp",
                ".env",
                ".env.local",
                "node_modules/*",
                ".git/*",
                "__pycache__/*",
                "*.pyc",
                ".DS_Store",
            ],
            "auto_commit_only": [],
        },
        "notifications": {
            "show_commit_hash": True,
            "show_diff_stats": True,
            "suppress_no_changes": True,
            "error_verbosity": "detailed",
        },
    }

    try:
        if config_path.exists():
            with config_path.open("r") as f:
                user_config = json.load(f)
                for key in default_config:
                    if key in user_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(user_config[key])
                        else:
                            default_config[key] = user_config[key]
                return default_config
    except (json.JSONDecodeError, OSError):
        pass

    return default_config


def matches_pattern(file_path: str, patterns: list[str]) -> bool:
    """Check if a file path matches any of the glob patterns.

    Args:
        file_path: Full path to the file to check.
        patterns: List of glob patterns to match against.

    Returns:
        True if file matches any pattern, False otherwise.
    """
    file_name = os.path.basename(file_path)
    for pattern in patterns:
        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_name, pattern):
            return True
    return False


def should_commit_file(file_path: str, config: dict) -> bool:
    """Determine if a file should be auto-committed based on config patterns.

    Checks the file against exclude patterns first, then verifies it matches
    any include-only patterns if specified.

    Args:
        file_path: Full path to the file.
        config: Configuration dictionary with file_patterns section.

    Returns:
        True if file should be committed, False if it should be skipped.
    """
    file_patterns = config.get("file_patterns", {})
    exclude = file_patterns.get("exclude", [])
    include_only = file_patterns.get("auto_commit_only", [])

    if matches_pattern(file_path, exclude):
        return False

    if include_only and not matches_pattern(file_path, include_only):
        return False

    return True


def get_git_root(work_dir: str) -> str | None:
    """Get the git repository root directory.

    Args:
        work_dir: Directory to start searching from.

    Returns:
        Path to git root directory, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=work_dir,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def init_git_repo(work_dir: str, config: dict) -> str | None:
    """Initialize a git repository if configured to do so.

    Args:
        work_dir: Directory to initialize as a git repository.
        config: Configuration dictionary with git_settings section.

    Returns:
        Path to initialized repo, or None if initialization failed/disabled.
    """
    git_settings = config.get("git_settings", {})
    if not git_settings.get("auto_init_repo", False):
        return None

    # Never init git in the hooks directory - it's not meant to be a repo
    hooks_dir = str(Path.home() / ".claude" / "hooks")
    if work_dir.startswith(hooks_dir):
        return None

    try:
        subprocess.run(
            ["git", "init"],
            capture_output=True,
            timeout=10,
            cwd=work_dir,
        )

        default_branch = git_settings.get("default_branch", "main")
        subprocess.run(
            ["git", "branch", "-M", default_branch],
            capture_output=True,
            timeout=5,
            cwd=work_dir,
        )

        if git_settings.get("create_initial_commit", True):
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "Initial commit"],
                capture_output=True,
                timeout=10,
                cwd=work_dir,
            )

        return work_dir
    except (subprocess.TimeoutExpired, OSError):
        return None


def generate_commit_message(file_path: str, tool_name: str, config: dict) -> str:
    """Generate a smart commit message based on configuration.

    Creates a commit message that includes the action type, file info,
    optional tool name, and optional timestamp based on config settings.

    Args:
        file_path: Path to the file being committed.
        tool_name: Name of the Claude tool that modified the file.
        config: Configuration dictionary with auto_commit section.

    Returns:
        Formatted commit message string.
    """
    auto_commit = config.get("auto_commit", {})
    message_format = auto_commit.get("commit_message_format", "smart")
    include_timestamp = auto_commit.get("include_timestamp", True)
    include_tool = auto_commit.get("include_tool_name", True)
    max_length = auto_commit.get("max_message_length", 200)

    file_name = os.path.basename(file_path)

    action_map = {
        "Write": "Create",
        "Edit": "Update",
        "MultiEdit": "Update",
    }
    action = action_map.get(tool_name, "Modify")

    if message_format == "smart":
        ext = os.path.splitext(file_name)[1].lower()
        file_type_hints = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React",
            ".tsx": "React",
            ".md": "documentation",
            ".json": "config",
            ".yaml": "config",
            ".yml": "config",
            ".css": "styles",
            ".scss": "styles",
            ".html": "template",
        }
        file_type = file_type_hints.get(ext, "")

        if file_type:
            message = f"{action} {file_type}: {file_name}"
        else:
            message = f"{action} {file_name}"
    else:
        message = f"{action} {file_name}"

    if include_tool and tool_name:
        message += f" via {tool_name}"

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        message += f" [{timestamp}]"

    if len(message) > max_length:
        message = message[: max_length - 3] + "..."

    return message


def commit_file(
    file_path: str,
    tool_name: str,
    config: dict,
    git_root: str,
) -> tuple[dict | None, str | None]:
    """Stage and commit a single file.

    Args:
        file_path: Path to the file to commit.
        tool_name: Name of the tool that modified the file.
        config: Configuration dictionary.
        git_root: Path to the git repository root.

    Returns:
        Tuple of (result_dict, error_string). On success, result_dict contains
        'hash', 'message', and 'stats'. On failure, error_string describes
        what went wrong.
    """
    try:
        result = subprocess.run(
            ["git", "add", file_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=git_root,
        )

        if result.returncode != 0:
            return None, f"Failed to stage: {result.stderr}"

        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=git_root,
        )

        if not status.stdout.strip():
            return None, "No changes to commit"

        message = generate_commit_message(file_path, tool_name, config)

        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=git_root,
        )

        if commit_result.returncode != 0:
            return None, f"Failed to commit: {commit_result.stderr}"

        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=git_root,
        )

        commit_hash = (
            hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
        )

        notifications = config.get("notifications", {})
        stats = ""
        if notifications.get("show_diff_stats", True):
            stats_result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1..HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=git_root,
            )
            if stats_result.returncode == 0:
                lines = stats_result.stdout.strip().split("\n")
                if lines:
                    stats = lines[-1].strip()

        return {
            "hash": commit_hash,
            "message": message,
            "stats": stats,
        }, None

    except subprocess.TimeoutExpired:
        return None, "Git operation timed out"
    except OSError as e:
        return None, str(e)


def _extract_model_from_transcript(session_id: str) -> str | None:
    """Extract model ID from Claude Code transcript.

    Reads the session transcript and extracts the model field,
    formatting it per models.dev convention (anthropic/<model>).
    """
    try:
        transcripts = (Path.home() / ".claude" / "projects").rglob(f"{session_id}.jsonl")
        if transcript := next(transcripts, None):
            with open(transcript) as f:
                if first_line := f.readline():
                    if model := json.loads(first_line).get("model"):
                        return f"anthropic/{model}"
    except Exception:
        pass
    return None


def parse_diff_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Parse git diff output to extract line ranges per file.

    Parses @@ -old,count +new_start,new_count @@ hunks from unified diff.

    Args:
        diff: Output from git diff --unified=0

    Returns:
        Dict mapping file paths to list of (start_line, end_line) tuples.
    """
    import re

    file_ranges: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        # Match file header: +++ b/path/to/file.py
        if line.startswith("+++ b/"):
            current_file = line[6:]
            file_ranges[current_file] = []
        # Match hunk header: @@ -old,count +new_start,new_count @@
        elif line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                if count > 0:  # Only additions
                    end = start + count - 1
                    file_ranges[current_file].append((start, end))

    return file_ranges


def capture_agent_trace(
    session_id: str | None,
    git_root: str,
) -> None:
    """Capture AI attribution trace after successful commit (spec v0.1).

    Records the commit in theo's trace storage with line-level attribution
    following the agent-trace.dev specification.

    Uses subprocess to call theo since it's not installed in system Python.

    Args:
        session_id: Claude Code session ID (from hook input)
        git_root: Path to git repository root
    """
    if not session_id:
        return  # No session = human commit, skip tracing

    try:
        # Get full commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=git_root,
        )
        if sha_result.returncode != 0:
            return
        commit_sha = sha_result.stdout.strip()

        # Get diff for line range extraction
        diff_result = subprocess.run(
            ["git", "diff", "HEAD~1..HEAD", "--unified=0"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=git_root,
        )
        if diff_result.returncode != 0:
            return
        file_ranges = parse_diff_ranges(diff_result.stdout)

        # Build transcript path from session_id
        transcript_path = f"session:{session_id}"

        # Extract model_id from transcript (auto-detected from session)
        # Transcripts at ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl
        model_id = _extract_model_from_transcript(session_id)

        trace_args = json.dumps({
            "commit_sha": commit_sha,
            "conversation_url": transcript_path,
            "session_id": session_id,
            "model_id": model_id,
            "file_ranges": {k: list(v) for k, v in file_ranges.items()},
        })

        # Call theo via uv run to store the trace
        # Uses the theo project directory for proper environment
        theo_dir = os.path.expanduser("~/Documents/Github/theo")
        subprocess.run(
            [
                "uv", "run", "python", "-c",
                f"""
import json
from theo.storage.sqlite_store import SQLiteStore
args = json.loads('''{trace_args}''')
store = SQLiteStore()
store.add_trace(
    commit_sha=args['commit_sha'],
    conversation_url=args['conversation_url'],
    model_id=args.get('model_id'),
    session_id=args['session_id'],
    file_ranges={{k: [tuple(r) for r in v] for k, v in args['file_ranges'].items()}},
)
store.close()
"""
            ],
            capture_output=True,
            timeout=10,
            cwd=theo_dir,
        )

    except Exception:
        # Trace capture is best-effort - never fail the commit
        pass


def auto_push(config: dict, git_root: str) -> bool | None:
    """Push to remote if configured.

    Args:
        config: Configuration dictionary with git_settings section.
        git_root: Path to the git repository root.

    Returns:
        True if push succeeded, False if failed, None if push not configured.
    """
    git_settings = config.get("git_settings", {})
    if not git_settings.get("auto_push", False):
        return None

    try:
        remote = git_settings.get("remote_name", "origin")
        result = subprocess.run(
            ["git", "push", remote],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=git_root,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main() -> None:
    """Main entry point for auto-commit hook.

    Reads tool input from stdin, validates against configuration,
    and commits the modified file if appropriate.
    """
    try:
        config = load_config()
        auto_commit_config = config.get("auto_commit", {})

        if not auto_commit_config.get("enabled", True):
            sys.exit(0)

        input_data = json.load(sys.stdin)

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        if tool_name not in ["Write", "Edit", "MultiEdit"]:
            sys.exit(0)

        file_path = tool_input.get("file_path", "")
        if not file_path:
            sys.exit(0)

        if not should_commit_file(file_path, config):
            notifications = config.get("notifications", {})
            if not notifications.get("suppress_no_changes", True):
                output = {
                    "suppressOutput": False,
                    "systemMessage": (
                        f"⏭️ Skipped auto-commit for {os.path.basename(file_path)} "
                        "(excluded by pattern)"
                    ),
                }
                print(json.dumps(output))
            sys.exit(0)

        work_dir = os.path.dirname(file_path) or os.getcwd()
        git_root = get_git_root(work_dir)

        if not git_root:
            git_root = init_git_repo(work_dir, config)
            if not git_root:
                sys.exit(0)

        result, error = commit_file(file_path, tool_name, config, git_root)

        if error:
            notifications = config.get("notifications", {})
            if notifications.get("error_verbosity", "detailed") != "silent":
                if "No changes to commit" in error:
                    if not notifications.get("suppress_no_changes", True):
                        output = {
                            "suppressOutput": False,
                            "systemMessage": (
                                f"ℹ️ No changes to commit for "
                                f"{os.path.basename(file_path)}"
                            ),
                        }
                        print(json.dumps(output))
                else:
                    output = {
                        "suppressOutput": False,
                        "systemMessage": f"⚠️ Auto-commit failed: {error}",
                    }
                    print(json.dumps(output))
            sys.exit(0)

        # Capture Agent Trace (AI attribution) - best effort, won't fail commit
        session_id = input_data.get("session_id")
        capture_agent_trace(session_id, git_root)

        notifications = config.get("notifications", {})
        message_parts = ["✅ Auto-committed"]

        if notifications.get("show_commit_hash", True):
            message_parts.append(f"[{result['hash']}]")

        message_parts.append(os.path.basename(file_path))

        if notifications.get("show_diff_stats", True) and result.get("stats"):
            message_parts.append(f"({result['stats']})")

        output = {
            "suppressOutput": False,
            "systemMessage": " ".join(message_parts),
        }
        print(json.dumps(output))

        auto_push(config, git_root)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        output = {
            "suppressOutput": False,
            "systemMessage": f"⚠️ Auto-commit error: {e}",
        }
        print(json.dumps(output))
        sys.exit(0)


if __name__ == "__main__":
    main()
