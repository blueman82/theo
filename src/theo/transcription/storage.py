"""Transcription storage integration for Theo.

Saves transcription sessions to Theo's memory system via HybridStore.
"""

from theo.storage.hybrid import HybridStore
from theo.transcription.types import TranscriptionSession


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
