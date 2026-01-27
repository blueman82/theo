"""Document type compatibility layer.

This module provides backward compatibility by re-exporting the Document
type from the storage module. New code should prefer using MemoryDocument
from theo.types.memory for the unified type system.

The Document type in storage/types.py is a dataclass used directly by
SQLiteStore. MemoryDocument is the higher-level Pydantic model that
can convert to/from the storage format.
"""

from theo.storage.types import Document

__all__ = ["Document"]
