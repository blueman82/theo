"""Transcription storage integration for Theo.

Saves transcription sessions to Theo's memory system via HybridStore.
Also provides SQLite storage for transcriptions with audio file management.
"""

from pathlib import Path

import numpy as np

from theo.storage.hybrid import HybridStore
from theo.transcription import audio_storage
from theo.transcription.types import TranscriptionRecord, TranscriptionSession


class TranscriptionStorage:
    """Storage adapter for saving transcription sessions to Theo.

    Wraps HybridStore to provide simple session persistence.
    The TranscriptionSession.to_memory_document() factory handles
    content formatting and metadata extraction.

    Attributes:
        _store: HybridStore instance for memory operations
    """

    def __init__(self, hybrid_store: HybridStore) -> None:
        """Initialize with a HybridStore instance.

        Args:
            hybrid_store: Injected HybridStore dependency
        """
        self._store = hybrid_store

    async def save_session(self, session: TranscriptionSession, namespace: str = "default") -> str:
        """Save a transcription session to Theo's memory.

        Sets the namespace in session metadata, converts to MemoryDocument,
        and stores via HybridStore.add_memory().

        Args:
            session: TranscriptionSession to save
            namespace: Storage namespace (default: "default")

        Returns:
            The ID of the stored memory
        """
        session.metadata["namespace"] = namespace
        doc = session.to_memory_document()
        return await self._store.add_memory(
            content=doc.content,
            memory_type=doc.memory_type.value,
            namespace=doc.namespace,
            importance=doc.importance,
            confidence=doc.confidence,
            metadata=doc.metadata,
            memory_id=doc.id,
        )

    def save_session_to_db(
        self,
        session: TranscriptionSession,
        audio_data: np.ndarray | None = None,
        namespace: str = "default",
        model_used: str | None = None,
        language: str | None = None,
        audio_dir: Path | None = None,
    ) -> str:
        """Save transcription session to SQLite with optional audio file.

        Saves audio to ~/.theo/audio/{session_id}.wav if audio_data provided,
        then stores transcription text and segments to SQLite.

        Args:
            session: TranscriptionSession with segments
            audio_data: Optional audio samples as numpy array (mono, 16kHz)
            namespace: Storage namespace
            model_used: Whisper model identifier
            language: Language code
            audio_dir: Override audio directory

        Returns:
            The transcription ID
        """
        sqlite_store = self._store._sqlite

        # Save audio file if data provided
        audio_path: str | None = None
        if audio_data is not None and len(audio_data) > 0:
            path = audio_storage.save_audio(session.id, audio_data, audio_dir=audio_dir)
            audio_path = str(path)

        # Build segments list for SQLite
        segments = [
            {
                "text": seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "confidence": seg.confidence,
            }
            for seg in session.segments
            if seg.is_final
        ]

        # Save to SQLite
        return sqlite_store.save_transcription(
            transcription_id=session.id,
            full_text=session.get_full_text(),
            duration_seconds=session.duration_seconds(),
            model_used=model_used,
            language=language,
            namespace=namespace,
            segments=segments,
            audio_path=audio_path,
        )

    def get_transcription(self, transcription_id: str) -> TranscriptionRecord | None:
        """Get transcription by ID.

        Args:
            transcription_id: Transcription ID

        Returns:
            TranscriptionRecord or None if not found
        """
        sqlite_store = self._store._sqlite
        data = sqlite_store.get_transcription(transcription_id)
        if data is None:
            return None
        return TranscriptionRecord.from_dict(data)

    def list_transcriptions(
        self,
        namespace: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TranscriptionRecord]:
        """List transcriptions with optional namespace filter.

        Args:
            namespace: Optional namespace filter
            limit: Maximum results
            offset: Skip count

        Returns:
            List of TranscriptionRecord objects
        """
        sqlite_store = self._store._sqlite
        rows = sqlite_store.list_transcriptions(namespace, limit, offset)
        return [TranscriptionRecord.from_dict(row) for row in rows]

    def delete_transcription(self, transcription_id: str) -> bool:
        """Delete transcription and its audio file.

        Args:
            transcription_id: Transcription ID

        Returns:
            True if deleted, False if not found
        """
        sqlite_store = self._store._sqlite

        # Get transcription to find audio path
        data = sqlite_store.get_transcription(transcription_id)
        if data is None:
            return False

        # Delete audio file if exists
        if data.get("audio_path"):
            audio_storage.delete_audio(Path(data["audio_path"]))

        # Delete from SQLite
        return sqlite_store.delete_transcription(transcription_id)
