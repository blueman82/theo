"""Tests for Agent Trace feature.

Tests cover:
- Trace storage operations (add_trace, get_trace, list_traces_for_conversation)
- Trace upsert behavior
- TraceRecord dataclass (agent-trace.dev spec compliant)
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
            file_ranges={"src/foo.py": [(10, 25)], "src/bar.py": [(1, 10)]},
        )

        trace = store.get_trace("abc123def456")
        assert trace is not None
        assert trace.commit_sha == "abc123def456"
        assert trace.conversation_url == "/path/to/transcript.jsonl"
        assert trace.model_id == "anthropic/claude-opus-4-5-20251101"
        assert trace.session_id == "test-session-123"
        assert set(trace.files) == {"src/foo.py", "src/bar.py"}

    def test_add_trace_minimal(self, store: SQLiteStore) -> None:
        """Test adding trace with minimal required fields."""
        store.add_trace(
            commit_sha="minimal123",
            conversation_url="/path/to/conv.jsonl",
        )

        trace = store.get_trace("minimal123")
        assert trace is not None
        assert trace.commit_sha == "minimal123"
        # With no file_ranges, conversation_url won't be in files_json
        assert trace.conversation_url == ""
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
            file_ranges={"a.py": [(1, 10)]},
        )
        store.add_trace(
            commit_sha="commit2",
            conversation_url="/path/to/conv1.jsonl",
            file_ranges={"b.py": [(1, 10)]},
        )
        store.add_trace(
            commit_sha="commit3",
            conversation_url="/path/to/conv2.jsonl",
            file_ranges={"c.py": [(1, 10)]},
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
            file_ranges={"old.py": [(1, 10)]},
        )
        store.add_trace(
            commit_sha="abc123",
            conversation_url="/new/path.jsonl",
            file_ranges={"new.py": [(1, 10)]},
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

    def test_add_trace_spec_compliant(self, store: SQLiteStore) -> None:
        """Test adding a spec-compliant trace record."""
        import json
        import uuid

        trace_id = store.add_trace(
            commit_sha="abc123def456spec",
            conversation_url="/path/to/transcript.jsonl",
            model_id="claude-opus-4",
            session_id="test-session",
            file_ranges={
                "src/foo.py": [(10, 25), (30, 35)],
                "src/bar.py": [(1, 5)],
            },
        )

        # Verify UUID format
        uuid.UUID(trace_id)  # Raises if invalid

        trace = store.get_trace("abc123def456spec")
        assert trace is not None

        # Verify spec fields
        assert trace.version == "0.1"
        assert "T" in trace.timestamp  # RFC 3339 has T separator

        # Verify files_json structure
        files = json.loads(trace.files_json)
        assert len(files) == 2

        # Find src/foo.py file object
        foo_file = next((f for f in files if f["path"] == "src/foo.py"), None)
        assert foo_file is not None
        assert len(foo_file["conversations"]) == 1
        assert len(foo_file["conversations"][0]["ranges"]) == 2
        assert foo_file["conversations"][0]["ranges"][0]["start_line"] == 10
        assert foo_file["conversations"][0]["ranges"][0]["end_line"] == 25
        assert foo_file["conversations"][0]["ranges"][1]["start_line"] == 30
        assert foo_file["conversations"][0]["ranges"][1]["end_line"] == 35

        # Verify vcs_json
        vcs = json.loads(trace.vcs_json) if trace.vcs_json else None
        assert vcs is not None
        assert vcs["type"] == "git"
        assert vcs["revision"] == "abc123def456spec"

        # Verify tool_json
        tool = json.loads(trace.tool_json) if trace.tool_json else None
        assert tool is not None
        assert tool["name"] == "claude-code"


# ============================================================================
# ============================================================================
# Protocol Tests
# ============================================================================


# ============================================================================
# TraceRecord Dataclass Tests
# ============================================================================


class TestTraceRecord:
    """Tests for TraceRecord dataclass (agent-trace.dev spec compliant)."""

    def test_trace_record_fields(self) -> None:
        """Test TraceRecord has correct spec fields."""
        import json

        from theo.storage.sqlite_store import TraceRecord

        # Spec-compliant files_json with conversations containing ranges
        files_spec = [
            {
                "path": "a.py",
                "conversations": [
                    {
                        "ranges": [{"start_line": 1, "end_line": 10}],
                        "url": "/path/to/conv.jsonl",
                        "contributor": {"type": "ai", "model_id": "claude-opus-4-5-20251101"},
                    }
                ],
            },
            {
                "path": "b.py",
                "conversations": [
                    {
                        "ranges": [{"start_line": 1, "end_line": 5}],
                        "url": "/path/to/conv.jsonl",
                        "contributor": {"type": "ai", "model_id": "claude-opus-4-5-20251101"},
                    }
                ],
            },
        ]

        trace = TraceRecord(
            id="abc123uuid",
            version="0.1",
            timestamp="2024-01-15T10:30:00+00:00",
            files_json=json.dumps(files_spec),
            vcs_json=json.dumps({"type": "git", "revision": "abc123"}),
            tool_json=json.dumps({"name": "claude-code", "version": "1.0"}),
            metadata_json=None,
            commit_sha="abc123",
            session_id="session-123",
        )

        assert trace.id == "abc123uuid"
        assert trace.version == "0.1"
        assert trace.timestamp == "2024-01-15T10:30:00+00:00"
        assert trace.commit_sha == "abc123"
        assert trace.session_id == "session-123"
        # Test backward compatibility properties
        assert set(trace.files) == {"a.py", "b.py"}
        assert trace.conversation_url == "/path/to/conv.jsonl"
        assert trace.model_id == "claude-opus-4-5-20251101"
        assert trace.created_at > 0  # Parsed from RFC 3339

    def test_trace_record_optional_fields(self) -> None:
        """Test TraceRecord with optional fields as None."""
        import json

        from theo.storage.sqlite_store import TraceRecord

        trace = TraceRecord(
            id="minimal123uuid",
            version="0.1",
            timestamp="2024-01-15T10:30:00+00:00",
            files_json=json.dumps([]),
            vcs_json=None,
            tool_json=None,
            metadata_json=None,
            commit_sha="abc123",
            session_id=None,
        )

        assert trace.model_id is None
        assert trace.session_id is None
        assert trace.files == []
        assert trace.conversation_url == ""
