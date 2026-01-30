"""Tests for transcription storage integration.

Tests TranscriptionStorage with mocked HybridStore.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestTranscriptionStorage:
    """Tests for TranscriptionStorage class."""

    def test_transcription_storage_init(self) -> None:
        """Verify _store is set correctly."""
        from theo.transcription.storage import TranscriptionStorage

        mock_store = MagicMock()

        storage = TranscriptionStorage(mock_store)

        assert storage._store is mock_store

    @pytest.mark.asyncio
    async def test_save_session(self) -> None:
        """Mock add_memory and verify it's called with MemoryDocument."""
        from theo.transcription.storage import TranscriptionStorage
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession
        from theo.types import MemoryType

        mock_store = MagicMock()
        mock_store.add_memory = AsyncMock(return_value="stored-id-123")

        storage = TranscriptionStorage(mock_store)

        # Create a session with content
        session = TranscriptionSession(
            id="session-id",
            source="microphone",
        )
        session.add_segment(
            TranscriptionSegment(
                text="Test content",
                start_time=0.0,
                end_time=5.0,
                is_final=True,
            )
        )
        session.end_time = session.start_time + timedelta(seconds=5)

        result = await storage.save_session(session, namespace="default")

        # Verify add_memory was called
        mock_store.add_memory.assert_called_once()
        call_kwargs = mock_store.add_memory.call_args.kwargs

        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["memory_type"] == MemoryType.DOCUMENT.value
        assert call_kwargs["namespace"] == "default"
        assert call_kwargs["memory_id"] == "session-id"

        # Verify metadata
        assert call_kwargs["metadata"]["source"] == "microphone"
        assert call_kwargs["metadata"]["segment_count"] == 1

        # Verify return value
        assert result == "stored-id-123"

    @pytest.mark.asyncio
    async def test_save_session_returns_id(self) -> None:
        """Mock add_memory returns doc with id='test-id', verify save_session returns it."""
        from theo.transcription.storage import TranscriptionStorage
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        mock_store = MagicMock()
        mock_store.add_memory = AsyncMock(return_value="test-id")

        storage = TranscriptionStorage(mock_store)

        session = TranscriptionSession()
        session.add_segment(
            TranscriptionSegment(
                text="Content",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        )

        result = await storage.save_session(session)

        assert result == "test-id"

    @pytest.mark.asyncio
    async def test_save_session_custom_namespace(self) -> None:
        """Verify custom namespace is passed to storage."""
        from theo.transcription.storage import TranscriptionStorage
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        mock_store = MagicMock()
        mock_store.add_memory = AsyncMock(return_value="stored-id")

        storage = TranscriptionStorage(mock_store)

        session = TranscriptionSession()
        session.add_segment(
            TranscriptionSegment(
                text="Content",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        )

        await storage.save_session(session, namespace="global")

        call_kwargs = mock_store.add_memory.call_args.kwargs
        assert call_kwargs["namespace"] == "global"

    @pytest.mark.asyncio
    async def test_save_session_sets_namespace_in_metadata(self) -> None:
        """Verify namespace is set in session metadata before conversion."""
        from theo.transcription.storage import TranscriptionStorage
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        mock_store = MagicMock()
        mock_store.add_memory = AsyncMock(return_value="stored-id")

        storage = TranscriptionStorage(mock_store)

        session = TranscriptionSession(metadata={})
        session.add_segment(
            TranscriptionSegment(
                text="Content",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        )

        await storage.save_session(session, namespace="global")

        # Session metadata should have namespace set
        assert session.metadata["namespace"] == "global"
