"""Transcription module for Theo.

Provides voice transcription types and utilities for converting
audio to text and integrating with Theo's memory system.
"""

from theo.transcription.types import (
    AudioChunk,
    TranscriptionSegment,
    TranscriptionSession,
)

__all__ = [
    "AudioChunk",
    "TranscriptionSegment",
    "TranscriptionSession",
]
