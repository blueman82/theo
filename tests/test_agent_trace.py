"""Tests for Agent Trace feature.

Tests cover:
- Trace storage operations (add_trace, get_trace, list_traces_for_conversation)
- Trace upsert behavior
- Daemon get_active_session command
- Commit hook script importability
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from theo.storage.sqlite_store import SQLiteStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_sqlite(tmp_path: Path) -> Path:
    """Create temporary SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
def store(temp_sqlite: Path) -> SQLiteStore:
    """Provide SQLiteStore instance with temporary database.

    Yields:
        SQLiteStore: Store with sqlite-vec loaded
    """
    from theo.storage.sqlite_store import SQLiteStore

    s = SQLiteStore(temp_sqlite)
    yield s
    s.close()


# ============================================================================
# Trace Storage Tests
# ============================================================================


class TestTraceStorage:
    """Tests for trace storage in SQLiteStore."""

    def test_add_trace(self, store: SQLiteStore) -> None:
        """Test adding a trace record."""
        store.add_trace(
            commit_sha="abc123def456",
            conversation_url="/path/to/transcript.jsonl",
            model_id="anthropic/claude-opus-4-5-20251101",
            session_id="test-session-123",
            files=["src/foo.py", "src/bar.py"],
        )

        trace = store.get_trace("abc123def456")
        assert trace is not None
        assert trace.commit_sha == "abc123def456"
        assert trace.conversation_url == "/path/to/transcript.jsonl"
        assert trace.model_id == "anthropic/claude-opus-4-5-20251101"
        assert trace.session_id == "test-session-123"
        assert trace.files == ["src/foo.py", "src/bar.py"]

    def test_add_trace_minimal(self, store: SQLiteStore) -> None:
        """Test adding trace with minimal required fields."""
        store.add_trace(
            commit_sha="minimal123",
            conversation_url="/path/to/conv.jsonl",
        )

        trace = store.get_trace("minimal123")
        assert trace is not None
        assert trace.commit_sha == "minimal123"
        assert trace.conversation_url == "/path/to/conv.jsonl"
        assert trace.model_id is None
        assert trace.session_id is None
        assert trace.files == []

    def test_get_trace_not_found(self, store: SQLiteStore) -> None:
        """Test getting non-existent trace returns None."""
        trace = store.get_trace("nonexistent")
        assert trace is None

    def test_list_traces_by_conversation(self, store: SQLiteStore) -> None:
        """Test listing traces for a conversation."""
        # Add multiple traces for same conversation
        store.add_trace(
            commit_sha="commit1",
            conversation_url="/path/to/conv1.jsonl",
            files=["a.py"],
        )
        store.add_trace(
            commit_sha="commit2",
            conversation_url="/path/to/conv1.jsonl",
            files=["b.py"],
        )
        store.add_trace(
            commit_sha="commit3",
            conversation_url="/path/to/conv2.jsonl",
            files=["c.py"],
        )

        traces = store.list_traces_for_conversation("/path/to/conv1.jsonl")
        assert len(traces) == 2
        assert {t.commit_sha for t in traces} == {"commit1", "commit2"}

    def test_list_traces_empty_conversation(self, store: SQLiteStore) -> None:
        """Test listing traces for conversation with no traces."""
        traces = store.list_traces_for_conversation("/nonexistent/path.jsonl")
        assert traces == []

    def test_trace_upsert(self, store: SQLiteStore) -> None:
        """Test that add_trace updates existing record."""
        store.add_trace(
            commit_sha="abc123",
            conversation_url="/old/path.jsonl",
            files=["old.py"],
        )
        store.add_trace(
            commit_sha="abc123",
            conversation_url="/new/path.jsonl",
            files=["new.py"],
        )

        trace = store.get_trace("abc123")
        assert trace is not None
        assert trace.conversation_url == "/new/path.jsonl"
        assert trace.files == ["new.py"]

    def test_trace_created_at(self, store: SQLiteStore) -> None:
        """Test that trace records have created_at timestamp."""
        store.add_trace(
            commit_sha="timestamp123",
            conversation_url="/path/to/conv.jsonl",
        )

        trace = store.get_trace("timestamp123")
        assert trace is not None
        assert trace.created_at > 0


# ============================================================================
# Daemon Active Session Tests
# ============================================================================


class TestDaemonActiveSession:
    """Tests for daemon get_active_session command."""

    @pytest.mark.asyncio
    async def test_get_active_session_no_session(self) -> None:
        """Test get_active_session when no session is active."""
        from theo.daemon.server import DaemonServer
        from unittest.mock import MagicMock

        # Create server with mock provider
        mock_provider = MagicMock()
        mock_provider.health_check.return_value = True
        server = DaemonServer(embedding_provider=mock_provider)

        # Call handler directly
        result = await server._handle_get_active_session({})

        assert result == {"active": False, "session": None}

    @pytest.mark.asyncio
    async def test_get_active_session_with_session(self) -> None:
        """Test get_active_session when a session is active."""
        from theo.daemon.server import DaemonServer
        from unittest.mock import MagicMock

        # Create server with mock provider
        mock_provider = MagicMock()
        mock_provider.health_check.return_value = True
        server = DaemonServer(embedding_provider=mock_provider)

        # Set active session
        await server._handle_set_active_session({
            "session_id": "test-session",
            "transcript_path": "/path/to/transcript.jsonl",
            "model_id": "claude-opus-4-5-20251101",
            "project_path": "/Users/test/project",
        })

        # Get active session
        result = await server._handle_get_active_session({})

        assert result["active"] is True
        assert result["session"] is not None
        assert result["session"]["session_id"] == "test-session"
        assert result["session"]["transcript_path"] == "/path/to/transcript.jsonl"
        assert result["session"]["model_id"] == "claude-opus-4-5-20251101"
        assert result["session"]["project_path"] == "/Users/test/project"
        assert "started_at" in result["session"]

    @pytest.mark.asyncio
    async def test_set_active_session(self) -> None:
        """Test set_active_session sets session info."""
        from theo.daemon.server import DaemonServer
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.health_check.return_value = True
        server = DaemonServer(embedding_provider=mock_provider)

        result = await server._handle_set_active_session({
            "session_id": "new-session",
            "transcript_path": "/new/path.jsonl",
        })

        assert result == {"status": "active_session_set"}
        assert server._active_session is not None
        assert server._active_session["session_id"] == "new-session"


# ============================================================================
# Commit Hook Tests
# ============================================================================


class TestCommitHook:
    """Tests for commit hook script."""

    def test_hook_importable(self) -> None:
        """Test that hook script can be imported without syntax errors."""
        import importlib.util

        hook_path = Path(__file__).parent.parent / "hooks" / "theo-commit-hook.py"
        assert hook_path.exists(), f"Hook file not found: {hook_path}"

        # Load module from file path (handles hyphenated filename)
        spec = importlib.util.spec_from_file_location("theo_commit_hook", hook_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Verify key functions exist
        assert hasattr(module, "get_commit_info")
        assert hasattr(module, "get_active_session")
        assert hasattr(module, "write_trace")
        assert hasattr(module, "write_git_note")
        assert hasattr(module, "main")

    def test_hook_main_returns_zero_when_disabled(self) -> None:
        """Test hook returns 0 when tracing is disabled."""
        import importlib.util
        import os

        hook_path = Path(__file__).parent.parent / "hooks" / "theo-commit-hook.py"
        original_env = os.environ.get("THEO_TRACE_ENABLED")

        try:
            os.environ["THEO_TRACE_ENABLED"] = "false"

            # Load module from file path
            spec = importlib.util.spec_from_file_location("theo_commit_hook", hook_path)
            assert spec is not None
            assert spec.loader is not None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            result = module.main()
            assert result == 0
        finally:
            if original_env is None:
                os.environ.pop("THEO_TRACE_ENABLED", None)
            else:
                os.environ["THEO_TRACE_ENABLED"] = original_env


# ============================================================================
# Protocol Tests
# ============================================================================


class TestDaemonProtocol:
    """Tests for daemon protocol with Agent Trace commands."""

    def test_daemon_command_enum_has_session_commands(self) -> None:
        """Test that DaemonCommand enum has session tracking commands."""
        from theo.daemon.protocol import DaemonCommand

        assert hasattr(DaemonCommand, "SET_ACTIVE_SESSION")
        assert hasattr(DaemonCommand, "GET_ACTIVE_SESSION")
        assert DaemonCommand.SET_ACTIVE_SESSION.value == "set_active_session"
        assert DaemonCommand.GET_ACTIVE_SESSION.value == "get_active_session"


# ============================================================================
# TraceRecord Dataclass Tests
# ============================================================================


class TestTraceRecord:
    """Tests for TraceRecord dataclass."""

    def test_trace_record_fields(self) -> None:
        """Test TraceRecord has correct fields."""
        from theo.storage.sqlite_store import TraceRecord

        trace = TraceRecord(
            commit_sha="abc123",
            conversation_url="/path/to/conv.jsonl",
            model_id="claude-opus-4-5-20251101",
            session_id="session-123",
            files=["a.py", "b.py"],
            created_at=1234567890.0,
        )

        assert trace.commit_sha == "abc123"
        assert trace.conversation_url == "/path/to/conv.jsonl"
        assert trace.model_id == "claude-opus-4-5-20251101"
        assert trace.session_id == "session-123"
        assert trace.files == ["a.py", "b.py"]
        assert trace.created_at == 1234567890.0

    def test_trace_record_optional_fields(self) -> None:
        """Test TraceRecord with optional fields as None."""
        from theo.storage.sqlite_store import TraceRecord

        trace = TraceRecord(
            commit_sha="abc123",
            conversation_url="/path/to/conv.jsonl",
            model_id=None,
            session_id=None,
            files=[],
            created_at=1234567890.0,
        )

        assert trace.model_id is None
        assert trace.session_id is None
        assert trace.files == []
