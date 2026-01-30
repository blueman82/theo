"""Transcription type definitions for Theo.

Core types for voice transcription with audio chunks, segments, and sessions.
Integrates with Theo's memory system via to_memory_document() method.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from theo.types import MemoryDocument, MemoryType


@dataclass(frozen=True)
class AudioChunk:
    """Immutable audio chunk for processing.

    Attributes:
        data: Mono audio samples as numpy array
        timestamp: Seconds since recording start
        sample_rate: Audio sample rate (default: 16000 Hz for Whisper)
    """

    data: np.ndarray
    timestamp: float
    sample_rate: int = 16000


@dataclass(frozen=True)
class TranscriptionSegment:
    """Immutable transcription segment.

    Attributes:
        text: Transcribed text content
        start_time: Start time in seconds
        end_time: End time in seconds
        is_final: True if committed, False if partial/interim
        confidence: Transcription confidence score
    """

    text: str
    start_time: float
    end_time: float
    is_final: bool
    confidence: float = 0.8


@dataclass
class TranscriptionSession:
    """Mutable transcription session tracking segments.

    Attributes:
        id: Unique session identifier
        segments: List of transcription segments
        start_time: Session start datetime
        end_time: Session end datetime (None if ongoing)
        source: Audio source type
        metadata: Additional session metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    segments: list[TranscriptionSegment] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    source: Literal["microphone", "file"] = "microphone"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_segment(self, segment: TranscriptionSegment) -> None:
        """Add a transcription segment to the session."""
        self.segments.append(segment)

    def get_full_text(self) -> str:
        """Get concatenated text from all final segments."""
        return " ".join(s.text for s in self.segments if s.is_final)

    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    def to_memory_document(self) -> MemoryDocument:
        """Convert session to a MemoryDocument for storage.

        Returns:
            MemoryDocument with transcription content and metadata
        """
        namespace = self.metadata.get("namespace", "default")
        return MemoryDocument.from_memory(
            memory_id=self.id,
            content=self.get_full_text(),
            memory_type=MemoryType.DOCUMENT,
            namespace=namespace,
            metadata={
                "source": self.source,
                "duration": self.duration_seconds(),
                "segment_count": len(self.segments),
            },
        )
