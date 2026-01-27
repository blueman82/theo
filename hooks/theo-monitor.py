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
"""SessionEnd hook for memory system monitoring.

Fast health check with background deep analysis that survives hook cancellation.
Uses double-fork daemon pattern for reliable background execution.

Environment Variables:
    THEO_MONITOR_ENABLED: Set to "true" to enable monitoring (default: "false").
    THEO_MONITOR_DEEP: Set to "true" for Opus analysis (default: "true").

Example:
    export THEO_MONITOR_ENABLED=true
    python theo-monitor.py

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo.monitoring import MonitorResult

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Directory and file paths
LOG_DIR: Path = Path.home() / ".claude" / "hooks" / "logs"
LOG_FILE: Path = LOG_DIR / "theo-monitor.log"
LOCK_FILE: Path = LOG_DIR / "theo-monitor.lock"
COOLDOWN_FILE: Path = LOG_DIR / "theo-monitor.cooldown"

# Timing constants
COOLDOWN_SECONDS: int = 60

# Environment variable names
ENV_MONITOR_ENABLED: str = "THEO_MONITOR_ENABLED"
ENV_MONITOR_DEEP: str = "THEO_MONITOR_DEEP"

# Environment variable values
ENV_TRUE: str = "true"

# Project indicator files for namespace detection
PROJECT_INDICATORS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)

# Issue severity levels
SEVERITY_CRITICAL: str = "critical"
SEVERITY_WARNING: str = "warning"

# Report formatting
REPORT_SEPARATOR: str = "=" * 60
REPORT_HEADER: str = "[theo-monitor] Memory System Health Report"
ISSUES_PER_CATEGORY: int = 3

# Unicode markers for issue display
MARKER_CRITICAL: str = "\u274c"
MARKER_WARNING: str = "\u26a0\ufe0f"

# Memory storage defaults
DEFAULT_IMPORTANCE_CRITICAL: float = 0.7
DEFAULT_IMPORTANCE_WARNING: float = 0.5
DEFAULT_CONFIDENCE: float = 0.8

# Recall source paths (in priority order)
THEO_PATHS: tuple[Path, ...] = (
    Path.home() / "Documents" / "Github" / "theo" / "src",
    Path(__file__).parent.parent / "src",
    Path.home() / ".local" / "share" / "theo" / "src",
)


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IssueData:
    """Serializable representation of a monitor issue.

    Attributes:
        category: The category of the issue (e.g., "memory", "storage").
        severity: The severity level ("critical" or "warning").
        memory_id: The ID of the related memory, if applicable.
        description: Human-readable description of the issue.
        recommendation: Suggested action to resolve the issue.
    """

    category: str
    severity: str
    memory_id: str | None
    description: str
    recommendation: str

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all issue fields.
        """
        return {
            "category": self.category,
            "severity": self.severity,
            "memory_id": self.memory_id,
            "description": self.description,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> IssueData:
        """Create IssueData from dictionary.

        Args:
            data: Dictionary containing issue fields.

        Returns:
            New IssueData instance.
        """
        return cls(
            category=str(data.get("category", "")),
            severity=str(data.get("severity", "")),
            memory_id=data.get("memory_id"),
            description=str(data.get("description", "")),
            recommendation=str(data.get("recommendation", "")),
        )


@dataclass(frozen=True, slots=True)
class IssuesPayload:
    """Payload for daemon process containing issues and metadata.

    Attributes:
        issues: List of issues to analyze.
        namespace: The namespace context for memory storage.
        critical_count: Number of critical issues.
        warning_count: Number of warning issues.
    """

    issues: tuple[IssueData, ...]
    namespace: str
    critical_count: int
    warning_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all payload fields.
        """
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "namespace": self.namespace,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IssuesPayload:
        """Create IssuesPayload from dictionary.

        Args:
            data: Dictionary containing payload fields.

        Returns:
            New IssuesPayload instance.
        """
        return cls(
            issues=tuple(
                IssueData.from_dict(issue) for issue in data.get("issues", [])
            ),
            namespace=str(data.get("namespace", "global")),
            critical_count=int(data.get("critical_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
        )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    """Append timestamped message to log file.

    Creates the log directory if it does not exist. Messages are formatted
    with ISO 8601 timestamps.

    Args:
        msg: The message to log.
    """
    from datetime import datetime

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    with LOG_FILE.open("a") as f:
        f.write(f"{timestamp} | {msg}\n")


# -----------------------------------------------------------------------------
# Lock Management
# -----------------------------------------------------------------------------


class Lock:
    """File-based exclusive lock using flock.

    Provides process-level mutual exclusion to prevent concurrent execution.
    Uses non-blocking flock to immediately fail if lock is held.

    Attributes:
        _fd: File descriptor for the lock file, or None if not acquired.
    """

    _fd: int | None = None

    @classmethod
    def acquire(cls) -> bool:
        """Try to acquire the exclusive lock.

        Creates the lock file if it does not exist. Uses non-blocking
        flock to prevent waiting. Writes the current PID to the lock file.

        Returns:
            True if the lock was acquired successfully, False otherwise.
        """
        import fcntl
        import os

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


# -----------------------------------------------------------------------------
# Cooldown Management
# -----------------------------------------------------------------------------


def check_cooldown() -> bool:
    """Check if the cooldown period has passed.

    Prevents the hook from running more than once per COOLDOWN_SECONDS.
    Updates the cooldown timestamp if allowed to run.

    Returns:
        True if the cooldown has passed and execution is allowed,
        False if still in cooldown period.
    """
    from datetime import datetime

    try:
        if COOLDOWN_FILE.exists():
            last_run = float(COOLDOWN_FILE.read_text().strip())
            elapsed = datetime.now().timestamp() - last_run
            if elapsed < COOLDOWN_SECONDS:
                return False

        COOLDOWN_FILE.write_text(str(datetime.now().timestamp()))
        return True
    except (ValueError, OSError):
        return True


# -----------------------------------------------------------------------------
# Namespace Detection
# -----------------------------------------------------------------------------


def get_namespace() -> str:
    """Get project namespace from current working directory.

    Detects project root by looking for common project indicator files.
    Returns a project-specific namespace if found, otherwise "global".

    Returns:
        Namespace string in format "project:<name>" or "global".
    """
    cwd = Path.cwd()
    for indicator in PROJECT_INDICATORS:
        if (cwd / indicator).exists():
            return f"project:{cwd.name}"
    return "global"


# -----------------------------------------------------------------------------
# Recall Path Setup
# -----------------------------------------------------------------------------


def setup_theo_path() -> bool:
    """Add theo module to sys.path.

    Searches THEO_PATHS in order and adds the first existing path
    to the beginning of sys.path.

    Returns:
        True if theo was found and added to path, False otherwise.
    """
    import sys

    for path in THEO_PATHS:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return True
    return False


# -----------------------------------------------------------------------------
# Daemon Process
# -----------------------------------------------------------------------------


def spawn_daemon(issues_file: str) -> None:
    """Spawn detached daemon process for Opus analysis.

    Creates a new session and fully detaches the child process so it
    survives parent termination. Inherits the current environment.

    Args:
        issues_file: Path to the JSON file containing issues data.
    """
    import os
    import subprocess
    import sys

    subprocess.Popen(
        [sys.executable, __file__, issues_file],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=os.environ.copy(),
    )


# -----------------------------------------------------------------------------
# Health Checks
# -----------------------------------------------------------------------------


async def run_haiku_check() -> MonitorResult | None:
    """Run fast Haiku health check on the memory system.

    Creates a temporary HybridStore connection and runs the monitor's
    haiku_check method for quick issue detection.

    Returns:
        MonitorResult containing detected issues, or None on error.
    """
    from theo.config import RecallSettings
    from theo.monitoring import Monitor
    from theo.storage.hybrid import HybridStore

    settings = RecallSettings()
    store = await HybridStore.create(
        sqlite_path=settings.get_sqlite_path(),
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
    )

    try:
        monitor = Monitor(store, settings, use_claude_cli=True)
        return await monitor.haiku_check(namespace=None)
    finally:
        store.close()


async def run_opus_analysis(issues_file: str) -> None:
    """Run deep Opus analysis on issues and store results as memory.

    Reads issues from a JSON file, performs deep analysis using Claude,
    and stores the analysis results as a new memory entry.

    Args:
        issues_file: Path to the JSON file containing issues data.
            The file will be deleted after processing.
    """
    import json
    import logging

    theo_logger = logging.getLogger("theo")
    theo_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | theo.%(name)s | %(message)s"),
    )
    theo_logger.addHandler(handler)

    from theo.config import RecallSettings
    from theo.monitoring import MonitorIssue
    from theo.storage.hybrid import HybridStore

    log("daemon: starting opus analysis")

    try:
        with Path(issues_file).open() as f:
            data = json.load(f)

        payload = IssuesPayload.from_dict(data)

        issues = [
            MonitorIssue(
                category=issue.category,
                severity=issue.severity,
                memory_id=issue.memory_id,
                description=issue.description,
                recommendation=issue.recommendation,
            )
            for issue in payload.issues
        ]

        settings = RecallSettings()
        store = await HybridStore.create(
            sqlite_path=settings.get_sqlite_path(),
            ollama_host=settings.ollama_host,
            ollama_model=settings.ollama_model,
        )

        try:
            await _perform_opus_analysis(store, settings, issues, payload)
        finally:
            store.close()

    except Exception as e:
        log(f"daemon: error - {e}")
    finally:
        Path(issues_file).unlink(missing_ok=True)


async def _perform_opus_analysis(
    store,
    settings,
    issues: list,
    payload: IssuesPayload,
) -> None:
    """Perform Opus analysis and store results.

    Args:
        store: The HybridStore instance.
        settings: RecallSettings instance.
        issues: List of MonitorIssue objects.
        payload: The original IssuesPayload with metadata.
    """
    import os
    import shutil

    claude_path = shutil.which("claude")
    if not claude_path:
        log("daemon: ERROR - claude CLI not in PATH")
        log(f"daemon: PATH = {os.environ.get('PATH', 'NOT SET')}")
        return

    log(f"daemon: claude found at {claude_path}")

    from theo.monitoring import Monitor

    monitor = Monitor(store, settings, use_claude_cli=True)
    log(f"daemon: calling opus with {len(issues)} issues")

    try:
        analysis = await monitor.opus_analyze(issues)
    except Exception as e:
        log(f"daemon: opus_analyze exception: {type(e).__name__}: {e}")
        analysis = None

    if not analysis:
        log("daemon: opus analysis returned None")
        return

    log("daemon: opus analysis complete, storing memory")
    await _store_analysis_memory(store, analysis, payload)
    log("daemon: memory stored successfully")


async def _store_analysis_memory(
    store,
    analysis: dict,
    payload: IssuesPayload,
) -> None:
    """Store analysis results as a memory entry.

    Args:
        store: The HybridStore instance.
        analysis: Analysis results from Opus.
        payload: The original IssuesPayload with metadata.
    """
    content_lines = [
        f"Memory system monitoring found {len(payload.issues)} issues:",
        f"Critical: {payload.critical_count}, Warnings: {payload.warning_count}",
        "",
        f"Analysis: {analysis.get('summary', '')}",
        "",
        "Recommendations:",
    ]

    for rec in analysis.get("recommendations", []):
        priority = rec.get("priority", "medium")
        action = rec.get("action", "")
        content_lines.append(f"- [{priority}] {action}")

    content = "\n".join(content_lines)

    importance = (
        DEFAULT_IMPORTANCE_CRITICAL
        if payload.critical_count > 0
        else DEFAULT_IMPORTANCE_WARNING
    )

    await store.add_memory(
        content=content,
        memory_type="session",
        namespace=payload.namespace,
        importance=importance,
        metadata={
            "source": "theo-monitor",
            "issues_found": len(payload.issues),
            "critical_count": payload.critical_count,
            "warning_count": payload.warning_count,
        },
    )


# -----------------------------------------------------------------------------
# Report Output
# -----------------------------------------------------------------------------


def output_report(result: MonitorResult) -> None:
    """Output issues summary to stderr.

    Formats issues grouped by category with severity markers.
    Shows up to ISSUES_PER_CATEGORY issues per category.

    Args:
        result: MonitorResult containing issues to display.
    """
    import sys

    issues = result.issues
    critical = sum(1 for i in issues if i.severity == SEVERITY_CRITICAL)
    warnings = sum(1 for i in issues if i.severity == SEVERITY_WARNING)

    lines = [
        "",
        REPORT_SEPARATOR,
        REPORT_HEADER,
        REPORT_SEPARATOR,
        f"Found {len(issues)} issues: {critical} CRITICAL, {warnings} WARNINGS",
        "",
    ]

    by_category: dict[str, list] = {}
    for issue in issues:
        by_category.setdefault(issue.category, []).append(issue)

    for category, category_issues in by_category.items():
        lines.append(f"{category.upper()} ({len(category_issues)}):")

        for issue in category_issues[:ISSUES_PER_CATEGORY]:
            marker = (
                MARKER_CRITICAL
                if issue.severity == SEVERITY_CRITICAL
                else MARKER_WARNING
            )
            lines.append(f"  {marker} {issue.description}")
            lines.append(f"     -> {issue.recommendation}")

        remaining = len(category_issues) - ISSUES_PER_CATEGORY
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")
        lines.append("")

    lines.extend([REPORT_SEPARATOR, ""])
    print("\n".join(lines), file=sys.stderr)


# -----------------------------------------------------------------------------
# Issue Payload Creation
# -----------------------------------------------------------------------------


def create_issues_payload(result: MonitorResult) -> IssuesPayload:
    """Create IssuesPayload from MonitorResult.

    Args:
        result: MonitorResult containing issues.

    Returns:
        IssuesPayload ready for serialization.
    """
    issues = tuple(
        IssueData(
            category=i.category,
            severity=i.severity,
            memory_id=i.memory_id,
            description=i.description,
            recommendation=i.recommendation,
        )
        for i in result.issues
    )

    critical_count = sum(1 for i in result.issues if i.severity == SEVERITY_CRITICAL)
    warning_count = sum(1 for i in result.issues if i.severity == SEVERITY_WARNING)

    return IssuesPayload(
        issues=issues,
        namespace=get_namespace(),
        critical_count=critical_count,
        warning_count=warning_count,
    )


# -----------------------------------------------------------------------------
# Main Entry Points
# -----------------------------------------------------------------------------


def _run_hook_mode() -> None:
    """Run in normal hook mode with health check and optional daemon spawn."""
    import asyncio
    import json
    import os
    import tempfile

    if not Lock.acquire():
        return

    if not check_cooldown():
        log("cooldown active, skipping")
        return

    log("hook triggered")

    if os.environ.get(ENV_MONITOR_ENABLED, "").lower() != ENV_TRUE:
        log("disabled, exiting")
        return

    if not setup_theo_path():
        log("theo not found")
        return

    log("starting haiku check")

    try:
        result = asyncio.run(run_haiku_check())
        if not result:
            log("haiku check failed")
            return

        log(f"haiku done: {len(result.issues)} issues")

        if not result.issues:
            log("no issues")
            return

        output_report(result)

        if os.environ.get(ENV_MONITOR_DEEP, ENV_TRUE).lower() != ENV_TRUE:
            return

        payload = create_issues_payload(result)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(payload.to_dict(), f)
            issues_file = f.name

        log("spawning daemon for opus analysis")
        spawn_daemon(issues_file)
        log("daemon spawned, hook complete")

    except ImportError as e:
        log(f"import error: {e}")
    except Exception as e:
        log(f"error: {e}")


def _run_daemon_mode(issues_file: str) -> None:
    """Run in daemon mode to perform Opus analysis.

    Args:
        issues_file: Path to the JSON file containing issues data.
    """
    import asyncio

    asyncio.run(run_opus_analysis(issues_file))


def main() -> None:
    """Main entry point for the theo monitor.

    Determines execution mode based on command-line arguments:
    - If called with a .json file argument, runs in daemon mode.
    - Otherwise, runs in hook mode with health check.
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        _run_daemon_mode(sys.argv[1])
    else:
        _run_hook_mode()


if __name__ == "__main__":
    main()
