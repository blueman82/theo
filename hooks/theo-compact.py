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
"""Claude Code / Factory PreCompact hook for preserving important context.

This hook runs BEFORE a compact operation (context window compression).
It extracts and stores important information that might be lost during
compaction, ensuring critical decisions and context are preserved.

MIGRATED: Uses DaemonClient for IPC with automatic subprocess fallback.
          Uses client.curate() for Ollama summarization when daemon is available,
          falling back to local Ollama subprocess when daemon is unavailable.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or
    ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "PreCompact": [
                {
                    "matcher": "auto|manual",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/theo/hooks/theo-compact.py",
                            "timeout": 15
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
        "hook_event_name": "PreCompact",
        "trigger": "auto",
        "custom_instructions": ""
    }

Output:
    - Stores important context before it's compressed away
    - Does not block compaction
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Import DaemonClient for IPC communication
from theo_client import DaemonClient, get_shared_client

# Import session state for cross-hook coordination
try:
    from theo_session_state import (
        get_session_summary,
        get_unstored_preferences,
        get_unresolved_errors,
        load_session_state,
        set_checkpoint,
    )
    SESSION_STATE_AVAILABLE = True
except ImportError:
    SESSION_STATE_AVAILABLE = False

# Module-level constants
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TIMEOUT_SECONDS = 30

# DaemonClient timeout configuration
DAEMON_CONNECT_TIMEOUT = 2.0
DAEMON_REQUEST_TIMEOUT = 5.0
MAX_TRANSCRIPT_LENGTH = 12000
MIN_TRANSCRIPT_LENGTH = 500
MAX_CONTEXT_ITEMS = 5
MIN_CONTENT_LENGTH = 10
DEFAULT_IMPORTANCE = 0.6
DEFAULT_CONFIDENCE = 0.6

PROJECT_INDICATORS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)

DECISION_PHRASES = (
    "i'll use",
    "i will use",
    "let's use",
    "we should use",
    "i've decided",
    "the approach is",
    "implementing with",
)

ERROR_RESOLUTION_PHRASES = (
    "fixed by",
    "resolved by",
    "the issue was",
    "the problem was",
    "solution:",
    "fix:",
)

USER_CORRECTION_PHRASES = (
    "no,",
    "don't",
    "instead",
    "actually",
    "wrong",
    "that's not",
    "use X instead",
    "prefer",
)



@dataclass
class HookInput:
    """Structured representation of hook input data.

    Attributes:
        session_id: Unique identifier for the current session.
        transcript_path: Path to the JSONL transcript file.
        cwd: Current working directory for the session.
        trigger: Type of compact trigger ('auto' or 'manual').
        custom_instructions: Any custom instructions provided.
    """

    session_id: str = "unknown"
    transcript_path: str | None = None
    cwd: str | None = None
    trigger: str = "unknown"
    custom_instructions: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookInput:
        """Create HookInput from a dictionary.

        Args:
            data: Dictionary containing hook input fields.

        Returns:
            HookInput instance populated from the dictionary.
        """
        return cls(
            session_id=data.get("session_id") or data.get("sessionId", "unknown"),
            transcript_path=data.get("transcript_path") or data.get("transcriptPath"),
            cwd=data.get("cwd"),
            trigger=data.get("trigger", "unknown"),
            custom_instructions=data.get("custom_instructions", ""),
        )


@dataclass
class ContextItem:
    """A single context item to be preserved.

    Attributes:
        type: Category of the context (decision, preference, pattern, session).
        content: The actual content to preserve.
    """

    type: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with type and content keys.
        """
        return {"type": self.type, "content": self.content}


@dataclass
class ExtractionResult:
    """Result of context extraction from transcript.

    Attributes:
        decisions: List of decisions made during the session.
        errors_resolved: List of errors that were resolved.
        user_corrections: List of user corrections or preferences.
    """

    decisions: list[str] = field(default_factory=list)
    errors_resolved: list[str] = field(default_factory=list)
    user_corrections: list[str] = field(default_factory=list)

    def to_context_items(self) -> list[ContextItem]:
        """Convert extraction results to context items.

        Returns:
            List of ContextItem objects for storage.
        """
        items: list[ContextItem] = []

        if self.decisions:
            items.append(
                ContextItem(
                    type="decision",
                    content=f"Session decisions before compact: "
                    f"{' | '.join(self.decisions[-3:])}",
                ),
            )

        if self.errors_resolved:
            items.append(
                ContextItem(
                    type="pattern",
                    content=f"Errors resolved in session: "
                    f"{' | '.join(self.errors_resolved[-3:])}",
                ),
            )

        if self.user_corrections:
            items.append(
                ContextItem(
                    type="preference",
                    content=f"User corrections noted: "
                    f"{' | '.join(self.user_corrections[-3:])}",
                ),
            )

        return items


def read_hook_input() -> HookInput:
    """Read and parse hook input from stdin.

    Returns:
        HookInput instance with parsed data or defaults if parsing fails.
    """
    import json

    if sys.stdin.isatty():
        return HookInput()

    try:
        stdin_data = sys.stdin.read()
        if stdin_data:
            return HookInput.from_dict(json.loads(stdin_data))
    except (json.JSONDecodeError, OSError):
        pass

    return HookInput()


def get_project_namespace(cwd: Path) -> str:
    """Derive project namespace from working directory.

    Checks for common project indicators to determine if the directory
    is a project root, and returns an appropriate namespace.

    Args:
        cwd: Current working directory path.

    Returns:
        Namespace string in format 'project:<name>' or 'global'.
    """
    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return f"project:{cwd.name}"

    return "global"


def read_transcript(transcript_path: str | None) -> str | None:
    """Read the full transcript from the given path.

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        Transcript content as string, or None if reading fails.
    """
    if not transcript_path:
        return None

    path = Path(transcript_path).expanduser()
    if not path.exists():
        return None

    try:
        return path.read_text()
    except OSError:
        return None


def _extract_text_from_content(content: Any) -> str:
    """Extract text from message content (string or list format).

    Args:
        content: Message content, either as string or list of dicts.

    Returns:
        Extracted text content as string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(c.get("text", "")) for c in content if isinstance(c, dict)
        )
    return ""


def _process_assistant_message(content: str, result: ExtractionResult) -> None:
    """Process an assistant message for decisions and error resolutions.

    Args:
        content: The message content to analyze.
        result: ExtractionResult to update with findings.
    """
    lower_content = content.lower()

    if any(phrase in lower_content for phrase in DECISION_PHRASES):
        result.decisions.append(content[:200])

    if any(phrase in lower_content for phrase in ERROR_RESOLUTION_PHRASES):
        result.errors_resolved.append(content[:200])


def _process_user_message(content: str, result: ExtractionResult) -> None:
    """Process a user message for corrections and preferences.

    Args:
        content: The message content to analyze.
        result: ExtractionResult to update with findings.
    """
    if not isinstance(content, str):
        return

    lower_content = content.lower()
    if any(phrase in lower_content for phrase in USER_CORRECTION_PHRASES):
        result.user_corrections.append(content[:200])


def extract_key_context(transcript: str) -> list[ContextItem]:
    """Extract key context items from transcript before compaction.

    Analyzes the transcript to find:
    - Decisions made during the session
    - Errors encountered and how they were resolved
    - User corrections and preferences expressed

    Args:
        transcript: The full transcript content as string.

    Returns:
        List of ContextItem objects representing important context.
    """
    import json

    if not transcript:
        return []

    result = ExtractionResult()
    lines = transcript.strip().split("\n")

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = entry.get("role")
        content = _extract_text_from_content(entry.get("content", ""))

        if role == "assistant":
            _process_assistant_message(content, result)
        elif role == "user":
            _process_user_message(entry.get("content", ""), result)

    return result.to_context_items()


def _extract_json_from_output(output: str) -> str:
    """Extract JSON array from Ollama output.

    Handles various output formats including markdown code blocks
    and raw JSON.

    Args:
        output: Raw output from Ollama.

    Returns:
        Extracted JSON string.
    """
    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        return output[start:end].strip()

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        return output[start:end].strip()

    if "[" in output:
        start = output.find("[")
        depth = 0
        for i, char in enumerate(output[start:], start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return output[start : i + 1].strip()

    return output


def _summarize_with_local_ollama(transcript: str) -> str | None:
    """Use local Ollama subprocess to extract key context.

    This is the fallback when daemon is not available.

    Args:
        transcript: The transcript content to analyze.

    Returns:
        JSON string of extracted context items, or None if extraction fails.
    """
    import subprocess

    # Truncate to last portion for speed
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        transcript = "...(earlier context)...\n" + transcript[-MAX_TRANSCRIPT_LENGTH:]

    prompt = f"""This conversation is about to be compacted (compressed). \
Extract ONLY the most critical information that should be remembered.

Focus on:
1. Technical decisions made (e.g., "Using FastAPI for backend")
2. User preferences expressed (e.g., "User prefers TypeScript")
3. Errors that were resolved and how
4. Important context that would be lost

Format as JSON array (max 5 items):
[
  {{"type": "decision", "content": "Brief decision..."}},
  {{"type": "preference", "content": "User preference..."}},
  {{"type": "pattern", "content": "Error pattern and fix..."}}
]

Return empty array [] if nothing critical to preserve.

Transcript:
{transcript}

JSON:"""

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, "--think=medium", "--hidethinking"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT_SECONDS,
            check=False,
        )

        if result.returncode != 0:
            return None

        return _extract_json_from_output(result.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _summarize_with_daemon_curate(
    transcript: str,
    project_name: str,
    project_root: str,
) -> str | None:
    """Use DaemonClient.curate() for Ollama summarization via daemon.

    This is the preferred method when daemon is available, as it leverages
    the daemon's Ollama integration and connection pooling.

    Args:
        transcript: The transcript content to analyze.
        project_name: Name of the current project.
        project_root: Root path of the project.

    Returns:
        JSON string of extracted context items, or None if extraction fails.
    """

    # Truncate to last portion for speed
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        transcript = "...(earlier context)...\n" + transcript[-MAX_TRANSCRIPT_LENGTH:]

    # Create pseudo-memories from transcript for curation
    # The daemon's curate command expects a list of memory dicts
    memories = [
        {
            "content": transcript,
            "type": "session",
            "metadata": {"source": "theo-compact", "context": "pre-compaction"},
        },
    ]

    try:
        # Use auto_fallback=False to detect daemon availability
        client = DaemonClient(
            connect_timeout=DAEMON_CONNECT_TIMEOUT,
            request_timeout=OLLAMA_TIMEOUT_SECONDS,  # Use longer timeout for LLM
            auto_fallback=False,
        )

        result = client.curate(
            memories=memories,
            project_name=project_name,
            project_root=project_root,
            model=OLLAMA_MODEL,
        )
        client.close()

        if not result.get("success"):
            return None

        # The curate response should contain the curated context
        curated = result.get("data", {}).get("curated", "")
        if not curated:
            return None

        # The daemon's curate response may be raw text or JSON
        # Try to extract JSON array if present
        return _extract_json_from_output(curated)

    except Exception:
        return None


def summarize_with_ollama(
    transcript: str,
    project_name: str = "unknown",
    project_root: str = "",
) -> str | None:
    """Use Ollama to extract key context before compaction.

    Tries daemon's curate command first (via client.curate()), falling back
    to local Ollama subprocess if daemon is unavailable.

    Args:
        transcript: The transcript content to analyze.
        project_name: Name of the current project (for daemon curate).
        project_root: Root path of the project (for daemon curate).

    Returns:
        JSON string of extracted context items, or None if extraction fails.
    """
    if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
        return None

    # Try daemon's curate command first
    result = _summarize_with_daemon_curate(transcript, project_name, project_root)
    if result:
        return result

    # Fall back to local Ollama subprocess
    return _summarize_with_local_ollama(transcript)


# Note: Using get_shared_client() from theo_client.py for connection reuse


def _get_log_path() -> Path:
    """Get the path to the log file, creating parent directories if needed.

    Returns:
        Path to the log file.
    """
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "theo-compact.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _write_log(log_path: Path, message: str) -> None:
    """Write a timestamped message to the log file.

    Args:
        log_path: Path to the log file.
        message: Message to write.
    """
    from datetime import datetime

    with log_path.open("a") as f:
        f.write(f"{datetime.now().isoformat()} | {message}\n")


def _parse_ollama_response(summary_json: str | None) -> list[ContextItem]:
    """Parse Ollama response into ContextItem list.

    Args:
        summary_json: JSON string from Ollama, or None.

    Returns:
        List of ContextItem objects.
    """
    import json

    if not summary_json:
        return []

    try:
        items = json.loads(summary_json)
        if not isinstance(items, list):
            return []
        return [
            ContextItem(type=item.get("type", "session"), content=item.get("content", ""))
            for item in items
            if isinstance(item, dict) and item.get("content")
        ]
    except json.JSONDecodeError:
        return []


def _store_context_items(
    context_items: list[ContextItem],
    namespace: str,
    trigger: str,
    session_id: str,
) -> int:
    """Store context items in theo using DaemonClient.

    Args:
        context_items: Items to store.
        namespace: Project namespace for storage.
        trigger: Compact trigger type.
        session_id: Current session identifier.

    Returns:
        Number of successfully stored items.
    """
    stored = 0

    # Use a single client for all operations to reuse connection
    with get_shared_client() as client:
        for item in context_items[:MAX_CONTEXT_ITEMS]:
            if not item.content or len(item.content) < MIN_CONTENT_LENGTH:
                continue

            content = f"[Pre-compact preservation] {item.content}"

            result = client.store(
                content=content,
                namespace=namespace,
                memory_type=item.type,
                importance=DEFAULT_IMPORTANCE,
                metadata={
                    "source": "theo-compact",
                    "trigger": trigger,
                    "session_id": session_id,
                },
            )

            if result.get("success"):
                stored += 1

    return stored


def extract_from_session_state(session_id: str, namespace: str) -> list[ContextItem]:
    """Extract learnings from session state before compaction.

    This captures information that might be lost during context compression:
    - Unresolved errors (store as "watch out for" patterns)
    - Unstored preferences (make sure they're persisted)
    - Pending error-fix patterns

    Args:
        session_id: Current session ID
        namespace: Project namespace

    Returns:
        List of ContextItem to store
    """
    if not SESSION_STATE_AVAILABLE:
        return []

    items: list[ContextItem] = []

    try:
        summary = get_session_summary(session_id)

        # Extract unresolved errors as warning patterns
        for error in summary.get("unresolved_errors", [])[:3]:
            content = f"Watch out for: {error.get('error', '')[:150]} when using {error.get('tool', 'unknown')}"
            items.append(ContextItem(type="pattern", content=content))

        # Extract unstored preferences
        for pref in summary.get("unstored_preferences", [])[:3]:
            content = pref.get("content", "")
            if content:
                items.append(ContextItem(type="preference", content=content))

        # Extract unstored fix patterns
        for pattern in summary.get("unstored_patterns", [])[:3]:
            content = pattern.get("pattern", "")
            if content:
                items.append(ContextItem(type="pattern", content=content))

    except Exception:
        pass  # Don't fail compaction if session state read fails

    return items


def main() -> None:
    """Main hook entry point.

    Preserves important context before compaction by:
    1. Extracting key decisions, preferences, and patterns
    2. Storing them in theo for future reference
    3. Logging the compaction event
    """
    import os

    log_path = _get_log_path()

    try:
        hook_input = read_hook_input()

        # Change to session's working directory
        cwd = Path(hook_input.cwd) if hook_input.cwd else Path.cwd()
        os.chdir(cwd)

        namespace = get_project_namespace(cwd)

        # Log the compaction event
        _write_log(
            log_path,
            f"PRE_COMPACT | trigger={hook_input.trigger} | "
            f"session={hook_input.session_id} | namespace={namespace}",
        )

        # Read transcript
        transcript = read_transcript(hook_input.transcript_path)

        if not transcript or len(transcript) < MIN_TRANSCRIPT_LENGTH:
            _write_log(log_path, "SKIP | Transcript too short")
            return

        # Extract project name from namespace (e.g., "project:myproject" -> "myproject")
        project_name = namespace.split(":")[-1] if ":" in namespace else cwd.name

        # 1. EXTRACT FROM SESSION STATE (new!)
        # Get learnings captured by other hooks during this session
        session_state_items = extract_from_session_state(
            hook_input.session_id,
            namespace,
        )
        if session_state_items:
            _write_log(log_path, f"SESSION_STATE | Found {len(session_state_items)} items")

        # 2. EXTRACT FROM TRANSCRIPT
        # Skip Ollama if we already have session state items (faster)
        transcript_items: list[ContextItem] = []
        if not session_state_items:
            _write_log(log_path, "OLLAMA | Starting summarization...")
            # Try Ollama summarization first (uses client.curate() when daemon available)
            summary_json = summarize_with_ollama(
                transcript,
                project_name=project_name,
                project_root=str(cwd),
            )
            _write_log(log_path, "OLLAMA | Done")
            transcript_items = _parse_ollama_response(summary_json)

            # Fallback to heuristic extraction
            if not transcript_items:
                transcript_items = extract_key_context(transcript)

        # 3. COMBINE ALL ITEMS (session state takes priority)
        all_items = session_state_items + transcript_items

        if not all_items:
            _write_log(log_path, "SKIP | No critical context found")
            return

        # Store the context items
        _write_log(log_path, f"STORING | {len(all_items)} items...")
        stored = _store_context_items(
            all_items,
            namespace,
            hook_input.trigger,
            hook_input.session_id,
        )

        _write_log(log_path, f"STORED | {stored} items preserved before compact")

        # 4. MARK CHECKPOINT
        # Let other hooks know we've done our extraction
        if SESSION_STATE_AVAILABLE:
            try:
                set_checkpoint(hook_input.session_id, "pre_compact_done", True)
            except Exception:
                pass

        # Hook completes silently - no additionalContext output
        # The memory preservation is already done above; injecting context
        # into the summary was causing "Failed to generate conversation summary"
        # errors due to the complex markdown/emoji content.

    except BrokenPipeError:
        pass
    except Exception as e:
        try:
            _write_log(log_path, f"ERROR | {e}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
