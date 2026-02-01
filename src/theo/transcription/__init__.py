"""Transcription module for Theo.

Provides voice transcription types and utilities for converting
audio to text and integrating with Theo's memory system.
"""

from theo.transcription import audio_storage, tts
from theo.transcription.types import (
    AudioChunk,
    TranscriptionRecord,
    TranscriptionSegment,
    TranscriptionSession,
)

__all__ = [
    "AudioChunk",
    "TranscriptionRecord",
    "TranscriptionSegment",
    "TranscriptionSession",
    "audio_storage",
    "tts",
]
