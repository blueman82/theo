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
"""Recall Daemon Control - CLI for managing the theo daemon lifecycle.

This script provides a command-line interface for managing the theo-daemon
service. It supports start, stop, restart, and status operations, and can
integrate with macOS launchd for automatic startup.

Architecture:
    The daemon is a persistent Unix socket server that provides fast IPC
    for Claude Code hooks. This control script manages its lifecycle:

    - start: Spawns daemon in background, writes PID file
    - stop: Reads PID file, sends SIGTERM, waits for graceful exit
    - restart: Performs stop + start with proper sequencing
    - status: Checks PID file and process state, reports status
    - launchd-load: Loads the launchd plist for auto-start
    - launchd-unload: Unloads the launchd plist

PID File Management:
    PID file is stored at /tmp/theo.pid
    Socket file is at /tmp/theo.sock

Usage:
    theo-daemon-ctl.py start    # Start daemon in background
    theo-daemon-ctl.py stop     # Stop running daemon
    theo-daemon-ctl.py restart  # Restart daemon
    theo-daemon-ctl.py status   # Check daemon status
    theo-daemon-ctl.py launchd-load    # Load launchd plist
    theo-daemon-ctl.py launchd-unload  # Unload launchd plist

Exit Codes:
    0 - Success
    1 - Daemon not running (for stop/restart when already stopped)
    2 - Failed to start daemon
    3 - Failed to stop daemon
    4 - Invalid command

Integration with launchd:
    For automatic startup on macOS, use the launchd commands:
        theo-daemon-ctl.py launchd-load

    The launchd plist is located at:
        ~/Library/LaunchAgents/com.theo.daemon.plist
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

PID_FILE = Path("/tmp/theo.pid")
SOCKET_PATH = Path("/tmp/theo.sock")
LOG_DIR = Path.home() / ".claude" / "hooks" / "logs"

DAEMON_SCRIPT = Path(__file__).parent / "theo-daemon.py"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.theo.daemon.plist"

UV_PATHS = (
    Path.home() / ".local" / "bin" / "uv",
    Path("/opt/homebrew/bin/uv"),
    Path("/usr/local/bin/uv"),
)

# Timeouts
STOP_TIMEOUT_SECONDS = 5.0
STOP_POLL_INTERVAL = 0.1
START_WAIT_SECONDS = 1.0


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    """Status information about the daemon.

    Attributes:
        running: Whether the daemon process is currently running.
        pid: Process ID if running, None otherwise.
        socket_exists: Whether the Unix socket file exists.
        pid_file_exists: Whether the PID file exists.
    """

    running: bool
    pid: int | None
    socket_exists: bool
    pid_file_exists: bool

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary with status fields.
        """
        return {
            "running": self.running,
            "pid": self.pid,
            "socket_exists": self.socket_exists,
            "pid_file_exists": self.pid_file_exists,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def find_uv_executable() -> str:
    """Find the uv executable path.

    Searches common installation locations for the uv package manager.

    Returns:
        Path to uv executable, or 'uv' for PATH lookup.
    """
    for uv_path in UV_PATHS:
        if uv_path.exists():
            return str(uv_path)
    return "uv"


def read_pid_file() -> int | None:
    """Read PID from the PID file.

    Returns:
        Process ID as integer, or None if file doesn't exist or is invalid.
    """
    if not PID_FILE.exists():
        return None

    try:
        content = PID_FILE.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def remove_socket_file() -> None:
    """Remove the socket file if it exists."""
    try:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
    except OSError:
        pass


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Uses kill(pid, 0) to check process existence without sending a signal.

    Args:
        pid: Process ID to check.

    Returns:
        True if process exists, False otherwise.
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_daemon_status() -> DaemonStatus:
    """Get comprehensive status of the daemon.

    Checks PID file, process state, and socket file existence.

    Returns:
        DaemonStatus with current state information.
    """
    pid = read_pid_file()
    pid_file_exists = PID_FILE.exists()
    socket_exists = SOCKET_PATH.exists()

    if pid is not None and is_process_running(pid):
        running = True
    else:
        running = False
        # Clean up stale files if process not running
        if pid is not None:
            remove_pid_file()

    return DaemonStatus(
        running=running,
        pid=pid if running else None,
        socket_exists=socket_exists,
        pid_file_exists=pid_file_exists,
    )


# =============================================================================
# Daemon Control Functions
# =============================================================================


def start_daemon() -> bool:
    """Start the daemon in the background.

    Spawns the daemon script as a background process using uv run.
    Waits briefly to verify successful startup.

    Returns:
        True if daemon started successfully, False otherwise.
    """
    status = get_daemon_status()
    if status.running:
        print(f"Daemon already running (PID {status.pid})")
        return True

    # Clean up any stale files
    remove_pid_file()
    remove_socket_file()

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Find uv executable
    uv_exe = find_uv_executable()

    # Build command
    cmd = [
        uv_exe, "run", "python", str(DAEMON_SCRIPT),
    ]

    try:
        # Start daemon in background
        # Redirect stdout/stderr to log files
        stdout_log = LOG_DIR / "daemon-stdout.log"
        stderr_log = LOG_DIR / "daemon-stderr.log"

        with open(stdout_log, "a") as stdout_file, open(stderr_log, "a") as stderr_file:
            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(Path.home()),
            )

        # Wait briefly for daemon to initialize
        time.sleep(START_WAIT_SECONDS)

        # Verify startup
        status = get_daemon_status()
        if status.running:
            print(f"Daemon started (PID {status.pid})")
            print(f"Socket: {SOCKET_PATH}")
            print(f"Logs: {LOG_DIR}")
            return True
        # Check if process died immediately
        if process.poll() is not None:
            print(f"Daemon failed to start (exit code {process.returncode})")
            print(f"Check logs at: {stderr_log}")
            return False
        # Process still running but no PID file yet
        # Wait a bit more
        time.sleep(0.5)
        status = get_daemon_status()
        if status.running:
            print(f"Daemon started (PID {status.pid})")
            return True
        print("Daemon may have started but PID file not found")
        return False

    except FileNotFoundError as e:
        print(f"Failed to start daemon: {e}")
        print("Make sure 'uv' is installed and in PATH")
        return False
    except OSError as e:
        print(f"Failed to start daemon: {e}")
        return False


def stop_daemon() -> bool:
    """Stop the running daemon.

    Sends SIGTERM and waits for graceful shutdown. Falls back to
    SIGKILL if process doesn't exit within timeout.

    Returns:
        True if daemon stopped successfully, False if not running.
    """
    status = get_daemon_status()
    if not status.running or status.pid is None:
        print("Daemon not running")
        # Clean up any stale files
        remove_pid_file()
        remove_socket_file()
        return False

    pid = status.pid
    print(f"Stopping daemon (PID {pid})...")

    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit
        elapsed = 0.0
        while elapsed < STOP_TIMEOUT_SECONDS:
            if not is_process_running(pid):
                print("Daemon stopped gracefully")
                remove_pid_file()
                remove_socket_file()
                return True
            time.sleep(STOP_POLL_INTERVAL)
            elapsed += STOP_POLL_INTERVAL

        # Process didn't exit, force kill
        print("Daemon did not stop gracefully, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)

        if not is_process_running(pid):
            print("Daemon killed")
            remove_pid_file()
            remove_socket_file()
            return True

        print("Failed to stop daemon")
        return False

    except OSError as e:
        if e.errno == 3:  # No such process
            print("Daemon already stopped")
            remove_pid_file()
            remove_socket_file()
            return True
        print(f"Error stopping daemon: {e}")
        return False


def restart_daemon() -> bool:
    """Restart the daemon.

    Performs stop (if running) followed by start.

    Returns:
        True if restart successful, False otherwise.
    """
    status = get_daemon_status()
    if status.running:
        if not stop_daemon():
            print("Warning: Could not stop daemon cleanly")
        # Brief pause between stop and start
        time.sleep(0.5)

    return start_daemon()


def print_status() -> None:
    """Print daemon status information."""
    status = get_daemon_status()

    if status.running:
        print(f"Daemon: running (PID {status.pid})")
    else:
        print("Daemon: stopped")

    print(f"Socket: {'exists' if status.socket_exists else 'not found'} ({SOCKET_PATH})")
    print(f"PID file: {'exists' if status.pid_file_exists else 'not found'} ({PID_FILE})")
    print(f"Log directory: {LOG_DIR}")

    # Check if launchd plist exists
    if LAUNCHD_PLIST.exists():
        print(f"Launchd plist: installed ({LAUNCHD_PLIST})")
    else:
        print("Launchd plist: not installed")


# =============================================================================
# Launchd Integration
# =============================================================================


def load_launchd_plist() -> bool:
    """Load the launchd plist for automatic startup.

    Returns:
        True if loaded successfully, False otherwise.
    """
    if not LAUNCHD_PLIST.exists():
        print(f"Launchd plist not found at: {LAUNCHD_PLIST}")
        print("Please create the plist file first.")
        return False

    try:
        result = subprocess.run(
            ["launchctl", "load", str(LAUNCHD_PLIST)],
            check=False, capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Loaded launchd plist: {LAUNCHD_PLIST}")
            print("Daemon will now start automatically at login")
            return True
        if "already loaded" in result.stderr.lower():
            print("Launchd plist already loaded")
            return True
        print(f"Failed to load launchd plist: {result.stderr}")
        return False

    except FileNotFoundError:
        print("launchctl not found - are you on macOS?")
        return False
    except OSError as e:
        print(f"Error loading launchd plist: {e}")
        return False


def unload_launchd_plist() -> bool:
    """Unload the launchd plist.

    Returns:
        True if unloaded successfully, False otherwise.
    """
    if not LAUNCHD_PLIST.exists():
        print(f"Launchd plist not found at: {LAUNCHD_PLIST}")
        return True  # Already not loaded

    try:
        result = subprocess.run(
            ["launchctl", "unload", str(LAUNCHD_PLIST)],
            check=False, capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Unloaded launchd plist: {LAUNCHD_PLIST}")
            print("Daemon will no longer start automatically")
            return True
        if "could not find specified service" in result.stderr.lower():
            print("Launchd plist was not loaded")
            return True
        print(f"Failed to unload launchd plist: {result.stderr}")
        return False

    except FileNotFoundError:
        print("launchctl not found - are you on macOS?")
        return False
    except OSError as e:
        print(f"Error unloading launchd plist: {e}")
        return False


# =============================================================================
# CLI Argument Parsing
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="theo-daemon-ctl",
        description="Control the theo daemon lifecycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start          Start the daemon
  %(prog)s stop           Stop the daemon
  %(prog)s restart        Restart the daemon
  %(prog)s status         Show daemon status
  %(prog)s launchd-load   Enable auto-start at login
  %(prog)s launchd-unload Disable auto-start at login

Files:
  PID file:  /tmp/theo.pid
  Socket:    /tmp/theo.sock
  Logs:      ~/.claude/hooks/logs/
  Plist:     ~/Library/LaunchAgents/com.theo.daemon.plist
""",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Daemon control commands",
        metavar="COMMAND",
    )

    # Start command
    subparsers.add_parser(
        "start",
        help="Start the daemon in background",
        description="Start the theo daemon as a background process.",
    )

    # Stop command
    subparsers.add_parser(
        "stop",
        help="Stop the running daemon",
        description="Stop the theo daemon gracefully (SIGTERM).",
    )

    # Restart command
    subparsers.add_parser(
        "restart",
        help="Restart the daemon",
        description="Stop the daemon (if running) and start it again.",
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Show daemon status",
        description="Display current daemon status information.",
    )

    # Launchd commands
    subparsers.add_parser(
        "launchd-load",
        help="Load launchd plist for auto-start",
        description="Enable automatic daemon startup at user login via launchd.",
    )

    subparsers.add_parser(
        "launchd-unload",
        help="Unload launchd plist",
        description="Disable automatic daemon startup at user login.",
    )

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the daemon control script.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 4  # Invalid command

    if args.command == "start":
        return 0 if start_daemon() else 2

    if args.command == "stop":
        return 0 if stop_daemon() else 1

    if args.command == "restart":
        return 0 if restart_daemon() else 2

    if args.command == "status":
        print_status()
        status = get_daemon_status()
        return 0 if status.running else 1

    if args.command == "launchd-load":
        return 0 if load_launchd_plist() else 2

    if args.command == "launchd-unload":
        return 0 if unload_launchd_plist() else 2

    parser.print_help()
    return 4


if __name__ == "__main__":
    sys.exit(main())
