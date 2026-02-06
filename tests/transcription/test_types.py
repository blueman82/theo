"""Tests for transcription type definitions.

Tests the dataclass types: AudioChunk, TranscriptionSegment, TranscriptionSession.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_transcription_segment_creation(self) -> None:
        """Create segment and verify all fields are set correctly."""
        from theo.transcription.types import TranscriptionSegment

        segment = TranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=1.5,
            is_final=True,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 1.5
        assert segment.is_final is True
        assert segment.confidence == 0.95

    def test_transcription_segment_default_confidence(self) -> None:
        """Verify default confidence value is 0.8."""
        from theo.transcription.types import TranscriptionSegment

        segment = TranscriptionSegment(
            text="Test",
            start_time=0.0,
            end_time=1.0,
            is_final=False,
        )

        assert segment.confidence == 0.8

    def test_transcription_segment_frozen(self) -> None:
        """Verify immutability raises FrozenInstanceError."""
        from theo.transcription.types import TranscriptionSegment

        segment = TranscriptionSegment(
            text="Frozen",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
        )

        with pytest.raises(AttributeError):
            segment.text = "Changed"  # type: ignore[misc]


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_audio_chunk_creation(self) -> None:
        """Create chunk with np.zeros and verify fields."""
        from theo.transcription.types import AudioChunk

        audio_data = np.zeros(16000, dtype=np.float32)
        chunk = AudioChunk(
            data=audio_data,
            timestamp=0.5,
            sample_rate=16000,
        )

        assert np.array_equal(chunk.data, audio_data)
        assert chunk.timestamp == 0.5
        assert chunk.sample_rate == 16000

    def test_audio_chunk_default_sample_rate(self) -> None:
        """Verify default sample_rate is 16000."""
        from theo.transcription.types import AudioChunk

        chunk = AudioChunk(
            data=np.zeros(1000, dtype=np.float32),
            timestamp=0.0,
        )

        assert chunk.sample_rate == 16000

    def test_audio_chunk_frozen(self) -> None:
        """Verify immutability raises AttributeError."""
        from theo.transcription.types import AudioChunk

        chunk = AudioChunk(
            data=np.zeros(1000, dtype=np.float32),
            timestamp=0.0,
        )

        with pytest.raises(AttributeError):
            chunk.timestamp = 1.0  # type: ignore[misc]


class TestTranscriptionSession:
    """Tests for TranscriptionSession dataclass."""

    def test_transcription_session_creation(self) -> None:
        """Create session and verify default fields."""
        from theo.transcription.types import TranscriptionSession

        session = TranscriptionSession()

        assert isinstance(session.id, str)
        assert len(session.id) == 36  # UUID format
        assert session.segments == []
        assert isinstance(session.start_time, datetime)
        assert session.end_time is None
        assert session.source == "microphone"
        assert session.metadata == {}

    def test_transcription_session_add_segment(self) -> None:
        """Add segments and verify list contents."""
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        session = TranscriptionSession()

        segment1 = TranscriptionSegment(
            text="First",
            start_time=0.0,
            end_time=1.0,
            is_final=True,
        )
        segment2 = TranscriptionSegment(
            text="Second",
            start_time=1.0,
            end_time=2.0,
            is_final=True,
        )

        session.add_segment(segment1)
        session.add_segment(segment2)

        assert len(session.segments) == 2
        assert session.segments[0].text == "First"
        assert session.segments[1].text == "Second"

    def test_transcription_session_get_full_text(self) -> None:
        """Only final segments should be included, joined with space."""
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        session = TranscriptionSession()

        # Add final segment
        session.add_segment(
            TranscriptionSegment(
                text="Hello",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        )
        # Add partial/interim segment (should be excluded)
        session.add_segment(
            TranscriptionSegment(
                text="wor",
                start_time=1.0,
                end_time=1.5,
                is_final=False,
            )
        )
        # Add another final segment
        session.add_segment(
            TranscriptionSegment(
                text="world",
                start_time=1.0,
                end_time=2.0,
                is_final=True,
            )
        )

        full_text = session.get_full_text()

        assert full_text == "Hello world"

    def test_transcription_session_get_full_text_empty(self) -> None:
        """Empty session should return empty string."""
        from theo.transcription.types import TranscriptionSession

        session = TranscriptionSession()

        assert session.get_full_text() == ""

    def test_transcription_session_duration(self) -> None:
        """Calculate duration from start_time and end_time."""
        from theo.transcription.types import TranscriptionSession

        start = datetime.now()
        end = start + timedelta(seconds=10.5)

        session = TranscriptionSession(
            start_time=start,
            end_time=end,
        )

        duration = session.duration_seconds()

        assert abs(duration - 10.5) < 0.01

    def test_transcription_session_duration_ongoing(self) -> None:
        """Ongoing session calculates duration from now."""
        from theo.transcription.types import TranscriptionSession

        session = TranscriptionSession(
            start_time=datetime.now() - timedelta(seconds=5),
        )

        duration = session.duration_seconds()

        # Should be approximately 5 seconds
        assert 4.9 < duration < 6.0

    def test_transcription_session_to_memory_document(self) -> None:
        """Verify MemoryDocument has correct fields."""
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession
        from theo.types import MemoryType

        session = TranscriptionSession(
            id="test-session-id",
            source="microphone",
            metadata={"namespace": "default"},
        )
        session.add_segment(
            TranscriptionSegment(
                text="Test transcription content",
                start_time=0.0,
                end_time=5.0,
                is_final=True,
            )
        )
        session.end_time = session.start_time + timedelta(seconds=5)

        doc = session.to_memory_document()

        assert doc.id == "test-session-id"
        assert doc.content == "Test transcription content"
        assert doc.memory_type == MemoryType.SESSION
        assert doc.namespace == "default"
        assert doc.metadata is not None
        assert doc.metadata["source"] == "microphone"
        assert doc.metadata["segment_count"] == 1
        assert abs(doc.metadata["duration"] - 5.0) < 0.01

    def test_transcription_session_to_memory_document_default_namespace(self) -> None:
        """Verify default namespace when not in metadata."""
        from theo.transcription.types import TranscriptionSegment, TranscriptionSession

        session = TranscriptionSession()
        session.add_segment(
            TranscriptionSegment(
                text="Content",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        )

        doc = session.to_memory_document()

        assert doc.namespace == "default"
