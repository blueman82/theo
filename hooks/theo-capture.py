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
"""Claude Code / Factory SessionEnd hook for capturing session memories.

This hook runs at the end of each session and summarizes the conversation
to store important decisions, preferences, and patterns.
It uses Ollama for local summarization and theo MCP for storage.

MIGRATED: Uses DaemonClient for IPC with automatic subprocess fallback.

SessionEnd reasons:
    clear, logout, prompt_input_exit, other

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or
    ~/.factory/settings.json (Factory):

    {
        "hooks": {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-capture.py",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "session_id": "abc123",
        "transcript_path": "~/.claude/projects/.../session.jsonl",
        "cwd": "/path/to/project",
        "hook_event_name": "SessionEnd",
        "reason": "exit"
    }

The hook reads the transcript, summarizes it with Ollama, and stores
relevant memories. Failures are handled gracefully.
"""

from __future__ import annotations

import fcntl
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Import DaemonClient for IPC communication
from theo_client import DaemonClient, get_shared_client

# =============================================================================
# Module-Level Constants
# =============================================================================

# Directory and file paths
LOG_DIR: Path = Path.home() / ".claude" / "hooks" / "logs"
LOCK_FILE: Path = LOG_DIR / "theo-capture.lock"
LOG_FILE: Path = LOG_DIR / "theo-capture.log"

# Transcript processing
MAX_TRANSCRIPT_CHARS: int = 12_000
TRANSCRIPT_BEGIN_RATIO: float = 0.3
TRUNCATION_SEPARATOR: str = "\n\n... [TRANSCRIPT MIDDLE TRUNCATED] ...\n\n"
TRUNCATION_SEPARATOR_LEN: int = 50
MIN_TRANSCRIPT_LENGTH: int = 100
MIN_MEMORY_CONTENT_LENGTH: int = 10

# Ollama configuration
OLLAMA_MODEL: str = "gemma3:12b"
OLLAMA_TIMEOUT_SECONDS: int = 60

# Deduplication
DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85

# Project indicators for namespace detection
PROJECT_INDICATORS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)

# Memory type to importance mapping (RFC 2119 MUST statements get higher importance)
IMPORTANCE_MAP: dict[str, float] = {
    "preference": 0.8,
    "decision": 0.8,
    "pattern": 0.7,
    "session": 0.4,
}

# Memory type normalization
MEMORY_TYPE_MAP: dict[str, str] = {
    "preference": "preference",
    "decision": "decision",
    "pattern": "pattern",
    "session": "session",
}

# DaemonClient timeout configuration
DAEMON_CONNECT_TIMEOUT: float = 2.0
DAEMON_REQUEST_TIMEOUT: float = 10.0

# Summarization prompt template
SUMMARIZATION_PROMPT: str = """Extract memories from this Claude Code session. Output ONLY a JSON array.

Rules for memories:
- Use RFC 2119 keywords: "MUST", "MUST NOT", "SHOULD", "REQUIRED"
- Transform preferences: "User prefers X" → "You MUST use X."
- Types: preference, decision, pattern

Output format (JSON array only, no explanation):
[{{"type": "decision", "content": "Use FastAPI for backend."}}]

If no memories found, output: []

Transcript:
{transcript}"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class HookInput:
    """Parsed hook input from Claude Code/Factory.

    Attributes:
        session_id: Unique identifier for the session.
        transcript_path: Path to the session transcript file.
        cwd: Current working directory when hook was triggered.
    """

    session_id: str
    transcript_path: str | None
    cwd: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookInput:
        """Create HookInput from raw dictionary.

        Args:
            data: Dictionary with hook input data.

        Returns:
            Parsed HookInput instance.
        """
        return cls(
            session_id=data.get("session_id") or data.get("sessionId", "unknown"),
            transcript_path=data.get("transcript_path") or data.get("transcriptPath"),
            cwd=data.get("cwd", str(Path.cwd())),
        )


@dataclass(frozen=True, slots=True)
class Memory:
    """A single extracted memory.

    Attributes:
        memory_type: Type of memory (preference, decision, pattern, session).
        content: The memory content text.
    """

    memory_type: str
    content: str

    @property
    def importance(self) -> float:
        """Get importance score based on memory type.

        Returns:
            Importance score between 0.0 and 1.0.
        """
        return IMPORTANCE_MAP.get(self.memory_type, 0.5)

    @property
    def normalized_type(self) -> str:
        """Get normalized memory type.

        Returns:
            Normalized memory type string.
        """
        return MEMORY_TYPE_MAP.get(self.memory_type, "session")

    def is_valid(self) -> bool:
        """Check if memory has valid content.

        Returns:
            True if content meets minimum length requirement.
        """
        return bool(self.content) and len(self.content) >= MIN_MEMORY_CONTENT_LENGTH

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> Memory | None:
        """Create Memory from dictionary or skip invalid input.

        Args:
            data: Dictionary with type and content, or malformed string.

        Returns:
            Memory instance or None if data is malformed.
        """
        if isinstance(data, str) or not isinstance(data, dict):
            return None

        return cls(
            memory_type=data.get("type", "session"),
            content=data.get("content", ""),
        )


@dataclass(frozen=True, slots=True)
class StoreResult:
    """Result of storing memories.

    Attributes:
        stored: Number of memories successfully stored.
        skipped: Number of memories skipped as duplicates.
    """

    stored: int
    skipped: int


# =============================================================================
# File Lock Management
# =============================================================================


class Lock:
    """File-based exclusive lock using flock.

    Ensures only one instance of the capture hook runs at a time.
    Uses a class-level file descriptor to maintain lock state.
    """

    _fd: int | None = None

    @classmethod
    def acquire(cls) -> bool:
        """Attempt to acquire an exclusive lock.

        Creates the lock file if it doesn't exist and attempts to acquire
        an exclusive, non-blocking lock.

        Returns:
            True if lock was acquired, False if another instance holds it.
        """
        if cls._fd is not None:
            return True

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            cls._fd = os.open(
                str(LOCK_FILE),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            )
            fcntl.flock(cls._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(cls._fd, str(os.getpid()).encode())
            return True
        except (BlockingIOError, OSError):
            if cls._fd is not None:
                os.close(cls._fd)
                cls._fd = None
            return False


# =============================================================================
# Logging Utilities
# =============================================================================


def _log_message(message: str) -> None:
    """Append a timestamped message to the log file.

    Args:
        message: Message to log.
    """
    from datetime import datetime, timezone

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(f"{datetime.now(tz=timezone.utc).isoformat()} | {message}\n")
    except OSError:
        pass


# =============================================================================
# Input/Output Functions
# =============================================================================


def read_hook_input() -> dict[str, Any]:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin. This function checks
    if stdin has data available and parses it as JSON.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable.
    """
    import json

    try:
        if sys.stdin.isatty():
            return {}

        stdin_data = sys.stdin.read()
        if stdin_data:
            return json.loads(stdin_data)
    except (OSError, json.JSONDecodeError):
        pass

    return {}


def read_transcript(transcript_path: str | None) -> str | None:
    """Read session transcript from provided path.

    Args:
        transcript_path: Path to the transcript file (from hook input).

    Returns:
        Transcript content or None if unavailable.
    """
    if not transcript_path:
        return None

    try:
        path = Path(transcript_path).expanduser()
        if path.exists():
            return path.read_text()
    except OSError:
        pass

    return None


# =============================================================================
# Namespace Detection
# =============================================================================


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Checks for common project indicators (.git, pyproject.toml, etc.)
    to determine if the current directory is a project root.

    Returns:
        Namespace string in format 'project:{name}' or 'global'.
    """
    cwd = Path.cwd()

    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return f"project:{cwd.name}"

    return "global"


# =============================================================================
# Transcript Processing
# =============================================================================


def smart_truncate_transcript(
    transcript: str,
    max_chars: int = MAX_TRANSCRIPT_CHARS,
) -> str:
    """Intelligently truncate transcript keeping beginning and end context.

    Preserves 30% from the beginning (setup/decisions) and 70% from
    the end (recent work) to maintain context for summarization.

    Args:
        transcript: Full transcript text.
        max_chars: Maximum characters to keep.

    Returns:
        Truncated transcript with beginning and end preserved.
    """
    if len(transcript) <= max_chars:
        return transcript

    begin_chars = int(max_chars * TRANSCRIPT_BEGIN_RATIO)
    end_chars = max_chars - begin_chars - TRUNCATION_SEPARATOR_LEN

    return transcript[:begin_chars] + TRUNCATION_SEPARATOR + transcript[-end_chars:]


# =============================================================================
# Ollama Summarization
# =============================================================================


def _extract_json_array(output: str) -> str:
    """Extract JSON array from potentially wrapped output.

    Handles markdown code blocks and finds the outermost JSON array.

    Args:
        output: Raw output from Ollama.

    Returns:
        Extracted JSON array string or "[]" if not found.
    """
    # Handle markdown code blocks
    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            output = output[start:end].strip()
    elif "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            output = output[start:end].strip()

    # Extract JSON array from text if not already clean
    if not output.startswith("["):
        start = output.find("[")
        if start != -1:
            depth = 0
            for i, char in enumerate(output[start:], start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        output = output[start : i + 1].strip()
                        break

    # Final validation
    if output.startswith("[") and output.endswith("]"):
        return output

    return "[]"


def summarize_with_ollama(
    transcript: str,
    model: str = OLLAMA_MODEL,
) -> str | None:
    """Summarize transcript using local Ollama model.

    Uses local Ollama model for summarization.

    Args:
        transcript: Session transcript text.
        model: Ollama model name.

    Returns:
        Summarized memories as JSON string or None on failure.
    """
    import subprocess

    transcript = smart_truncate_transcript(transcript, MAX_TRANSCRIPT_CHARS)
    prompt = SUMMARIZATION_PROMPT.format(transcript=transcript)

    try:
        # Try with --format json first
        result = subprocess.run(
            [
                "ollama",
                "run",
                model,
                "--think=medium",
                "--hidethinking",
                "--format",
                "json",
            ],
            check=False,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            # Fallback without --format flag
            result = subprocess.run(
                ["ollama", "run", model, "--think=medium", "--hidethinking"],
                check=False,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return None

        return _extract_json_array(result.stdout.strip())

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    except OSError:
        return None


# Note: Using get_shared_client() from theo_client.py for connection reuse


# =============================================================================
# Memory Deduplication
# =============================================================================


def check_duplicate(
    content: str,
    namespace: str,
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    client: DaemonClient | None = None,
) -> bool:
    """Check if a similar memory already exists using DaemonClient.

    Args:
        content: Memory content to check.
        namespace: Namespace to search in.
        threshold: Similarity threshold (0.0-1.0).
        client: Optional DaemonClient to reuse connection.

    Returns:
        True if a similar memory exists, False otherwise.
    """
    try:
        # Use provided client or create new one
        if client:
            result = client.fetch(
                namespace=namespace,
                query=content,
                n_results=1,
            )
        else:
            with get_shared_client() as new_client:
                result = new_client.fetch(
                    namespace=namespace,
                    query=content,
                    n_results=1,
                )

        if not result.get("success"):
            return False

        memories = result.get("data", {}).get("memories", [])
        if not memories:
            return False

        top_result = memories[0]
        similarity = top_result.get("score", 0)

        # Check for exact content match
        existing_content = top_result.get("content", "")
        if existing_content.strip().lower() == content.strip().lower():
            return True

        return similarity >= threshold

    except (KeyError, TypeError, IndexError):
        # On error, allow storage (fail open)
        return False


# =============================================================================
# Memory Storage
# =============================================================================


def store_memories(memories: list[Any], namespace: str) -> StoreResult:
    """Store extracted memories in theo with deduplication using DaemonClient.

    Args:
        memories: List of memory dicts with type and content.
        namespace: Namespace for storage.

    Returns:
        StoreResult with counts of stored and skipped memories.
    """
    stored = 0
    skipped = 0

    # Use a single client for all operations to reuse connection
    with get_shared_client() as client:
        for raw_memory in memories:
            memory = Memory.from_dict(raw_memory)
            if memory is None or not memory.is_valid():
                continue

            if check_duplicate(memory.content, namespace, client=client):
                skipped += 1
                continue

            result = client.store(
                content=memory.content,
                namespace=namespace,
                memory_type=memory.normalized_type,
                importance=memory.importance,
                metadata={"source": "theo-capture"},
            )

            if result.get("success"):
                stored += 1

    return StoreResult(stored=stored, skipped=skipped)


# =============================================================================
# Core Capture Logic
# =============================================================================


def _do_capture(hook_input: dict[str, Any]) -> None:
    """Execute the actual memory capture work.

    Reads the transcript, summarizes it with Ollama, and stores
    extracted memories. Runs in background process.

    Args:
        hook_input: Raw hook input dictionary.
    """
    import json

    parsed = HookInput.from_dict(hook_input)

    _log_message(
        f"SessionEnd fired | session={parsed.session_id} | "
        f"transcript={parsed.transcript_path}",
    )

    transcript = read_transcript(parsed.transcript_path)
    if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
        _log_message(
            f"No transcript or too short | len={len(transcript) if transcript else 0}",
        )
        return

    os.chdir(parsed.cwd)
    namespace = get_project_namespace()

    _log_message(
        f"Processing transcript | len={len(transcript)} | namespace={namespace}",
    )

    summary_json = summarize_with_ollama(transcript)
    if not summary_json:
        _log_message("Ollama summarization failed")
        return

    try:
        memories = json.loads(summary_json)
        if not isinstance(memories, list):
            return
    except json.JSONDecodeError:
        _log_message(f"JSON parse failed | raw={summary_json[:200]}")
        return

    if not memories:
        # FALLBACK: Always store at least a session record
        _log_message("No memories extracted - storing session record fallback")
        fallback_memory = {
            "content": f"Session {parsed.session_id} in {namespace} - transcript length {len(transcript)} chars",
            "type": "session",
            "importance": 0.3,
        }
        memories = [fallback_memory]

    result = store_memories(memories, namespace)

    _log_message(
        f"Stored {result.stored} memories, skipped {result.skipped} duplicates "
        f"from session {parsed.session_id}",
    )


def run_background(data_file: str) -> None:
    """Background worker entry point.

    Reads hook input from temp file and executes capture.

    Args:
        data_file: Path to temporary file containing hook input JSON.
    """
    import json

    if not Lock.acquire():
        return  # Another instance running

    try:
        data_path = Path(data_file)
        with data_path.open() as f:
            hook_input = json.load(f)
        data_path.unlink(missing_ok=True)
        _do_capture(hook_input)
    except (OSError, json.JSONDecodeError):
        pass


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main hook entry point - fire and forget.

    In foreground mode: reads hook input and spawns background process.
    In background mode: executes the actual capture work.
    """
    import json
    import subprocess
    import tempfile

    # Handle background mode
    if len(sys.argv) > 2 and sys.argv[1] == "--background":
        run_background(sys.argv[2])
        return

    hook_input = read_hook_input()
    if not hook_input:
        return

    # Output memory reflection prompt for Claude (SessionEnd)
    reflection_prompt = """📝 SESSION ENDING - Memory Operations REQUIRED

You MUST evaluate session learnings before ending:
- Patterns that emerged (MUST store if repeatable)
- User preferences (MUST store explicit, SHOULD store implicit)
- Technical decisions (MUST include rationale)
- What worked vs. caused friction (SHOULD store)

## Available Memory Tools (via theo MCP wrapper)

| Tool | Purpose |
|------|---------|
| `memory_store` | Store with semantic indexing + auto-deduplication |
| `memory_recall` | Semantic search with graph expansion |
| `memory_context` | Format memories for context injection |
| `memory_validate` | Adjust confidence based on success/failure |
| `memory_apply` | Record memory usage (TRY phase) |
| `memory_outcome` | Record result (LEARN phase) |
| `memory_relate` | Create relationships (supersedes, contradicts, etc.) |
| `memory_forget` | Delete with golden rule protection |

## Common Operations

**Store a new memory:**
```javascript
await theo.memory_store_tool({
  content: "Session learning: ...",
  memory_type: "pattern",  // preference, decision, golden_rule, pattern
  importance: 0.7,  // 0.3 minor, 0.5 useful, 0.7 important, 0.9+ critical
  namespace: "project:name"  // or "global"
});
```

**Update existing memory confidence (reinforce or weaken):**
```javascript
await theo.memory_validate({
  memory_id: "...",
  success: true,  // false to decrease confidence
  context: "Applied successfully in X scenario"
});
```

**Create memory relationships:**
```javascript
await theo.memory_relate({
  source_id: "new_memory_id",
  target_id: "old_memory_id",
  relation_type: "supersedes"  // contradicts, supports, refines
});
```

Use mcp__mcp-exec__execute_code_with_wrappers with wrappers: ["theo"]"""

    output = {"additionalContext": reflection_prompt}
    print(json.dumps(output))
    sys.stdout.flush()

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="theo-capture-")
        with os.fdopen(fd, "w") as f:
            json.dump(hook_input, f)

        subprocess.Popen(
            [sys.executable, __file__, "--background", temp_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        _do_capture(hook_input)


if __name__ == "__main__":
    main()
