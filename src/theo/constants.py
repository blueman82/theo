"""Theo configuration constants.

Split into two categories:
1. HARDCODED: Implementation details that don't change
2. CONFIGURABLE: User settings read from environment (.env)
"""

import os
from pathlib import Path

# =============================================================================
# HARDCODED CONSTANTS (implementation details)
# =============================================================================

# Socket/IPC paths
SOCKET_PATH = Path("/tmp/theo.sock")
PID_FILE = Path("/tmp/theo.pid")

# Buffer sizes (technical implementation details)
RECV_BUFFER = 1024 * 1024  # 1MB

# Validation algorithm thresholds (core design decisions)
GOLDEN_RULE_THRESHOLD = 0.9
SUCCESS_ADJUSTMENT = 0.1
FAILURE_MULTIPLIER = 1.5
LOW_CONFIDENCE_THRESHOLD = 0.15
INITIAL_CONFIDENCE = 0.3
DEMOTION_FAILURE_THRESHOLD = 3


# =============================================================================
# CONFIGURABLE SETTINGS (from .env)
# =============================================================================

# Load .env from theo project root if not already loaded
_theo_root = Path(__file__).parent.parent.parent
_env_file = _theo_root / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Read .env manually if dotenv not available
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value


def _require(key: str) -> str:
    """Get required environment variable or raise error.

    Args:
        key: Environment variable name to retrieve.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    value = os.environ.get(key)
    if value is None:
        raise RuntimeError(f"Required environment variable {key} not set. Check .env file.")
    return value


# =============================================================================
# Connection Settings
# =============================================================================


def get_connect_timeout() -> float:
    """Get socket connection timeout in seconds.

    Returns:
        Connection timeout from THEO_CONNECT_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_CONNECT_TIMEOUT_SECONDS"))


def get_request_timeout() -> float:
    """Get request/response timeout in seconds.

    Returns:
        Request timeout from THEO_REQUEST_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_REQUEST_TIMEOUT_SECONDS"))


def get_idle_timeout() -> float:
    """Get connection idle timeout in seconds.

    Connections idle longer than this are considered stale and reconnected.

    Returns:
        Idle timeout from THEO_CONNECTION_IDLE_TIMEOUT_SECONDS.
    """
    return float(_require("THEO_CONNECTION_IDLE_TIMEOUT_SECONDS"))


# =============================================================================
# Daemon Settings
# =============================================================================


def get_max_memory_mb() -> int:
    """Get maximum memory threshold for daemon watchdog.

    Returns:
        Max memory in MB from THEO_MAX_MEMORY_MB.
    """
    return int(_require("THEO_MAX_MEMORY_MB"))


def get_cache_ttl_seconds() -> int:
    """Get cache entry time-to-live in seconds.

    Returns:
        Cache TTL from THEO_CACHE_TTL_SECONDS.
    """
    return int(_require("THEO_CACHE_TTL_SECONDS"))


def get_warmup_interval_seconds() -> int:
    """Get daemon warmup interval in seconds.

    Returns:
        Warmup interval from THEO_WARMUP_INTERVAL_SECONDS.
    """
    return int(_require("THEO_WARMUP_INTERVAL_SECONDS"))


def get_ollama_timeout() -> int:
    """Get Ollama API request timeout in seconds.

    Returns:
        Ollama timeout from THEO_OLLAMA_TIMEOUT.
    """
    return int(_require("THEO_OLLAMA_TIMEOUT"))


def get_timeout_seconds() -> int:
    """Get general operation timeout in seconds.

    Returns:
        Timeout from THEO_TIMEOUT_SECONDS.
    """
    return int(_require("THEO_TIMEOUT_SECONDS"))


# =============================================================================
# Classification Queue Settings
# =============================================================================


def get_classification_batch_size() -> int:
    """Get batch size for memory classification processing.

    Returns:
        Batch size from THEO_CLASSIFICATION_BATCH_SIZE.
    """
    return int(_require("THEO_CLASSIFICATION_BATCH_SIZE"))


def get_classification_interval_seconds() -> int:
    """Get interval between classification batch runs.

    Returns:
        Interval in seconds from THEO_CLASSIFICATION_INTERVAL_SECONDS.
    """
    return int(_require("THEO_CLASSIFICATION_INTERVAL_SECONDS"))


def get_classification_max_queue_size() -> int:
    """Get maximum classification queue size before blocking.

    Returns:
        Max queue size from THEO_CLASSIFICATION_MAX_QUEUE_SIZE.
    """
    return int(_require("THEO_CLASSIFICATION_MAX_QUEUE_SIZE"))


# =============================================================================
# Memory Management Settings
# =============================================================================


def get_gc_interval_seconds() -> int:
    """Get interval between garbage collection runs.

    Returns:
        GC interval from THEO_GC_INTERVAL_SECONDS.
    """
    return int(_require("THEO_GC_INTERVAL_SECONDS"))


def get_memory_watchdog_interval_seconds() -> int:
    """Get interval between memory watchdog checks.

    Returns:
        Watchdog interval from THEO_MEMORY_WATCHDOG_INTERVAL_SECONDS.
    """
    return int(_require("THEO_MEMORY_WATCHDOG_INTERVAL_SECONDS"))


def get_queue_cleanup_interval_seconds() -> int:
    """Get interval between queue cleanup runs.

    Returns:
        Cleanup interval from THEO_QUEUE_CLEANUP_INTERVAL_SECONDS.
    """
    return int(_require("THEO_QUEUE_CLEANUP_INTERVAL_SECONDS"))


def get_queue_cleanup_age_hours() -> int:
    """Get age threshold for removing completed queue entries.

    Returns:
        Age threshold in hours from THEO_QUEUE_CLEANUP_AGE_HOURS.
    """
    return int(_require("THEO_QUEUE_CLEANUP_AGE_HOURS"))


def get_max_cache_entries() -> int:
    """Get maximum number of entries in namespace cache.

    Returns:
        Max entries from THEO_MAX_CACHE_ENTRIES.
    """
    return int(_require("THEO_MAX_CACHE_ENTRIES"))
