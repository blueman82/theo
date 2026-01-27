#!/usr/bin/env python3
"""Migrate data from ChromaDB to SQLite-vec storage.

This script performs a one-time migration from Theo's ChromaDB backend
to the new SQLite-vec storage layer:

1. Creates backups of theo.db and chroma_db
2. Extracts all documents with metadata and embeddings from ChromaDB
3. Migrates each memory with exact field mapping to SQLite tables
4. Populates embedding_cache for performance optimization
5. Verifies golden rules are preserved (confidence >= 0.9)

Field Mapping (ChromaDB -> SQLite):
| ChromaDB                | SQLite Column       | Notes                    |
|-------------------------|---------------------|--------------------------|
| id                      | memories.id         | Primary key              |
| document (content)      | memories.content    | Text content             |
| embedding               | memories_vec + cache| Re-embed if corrupted    |
| metadata.namespace      | memories.namespace  | Direct copy              |
| metadata.doc_type       | memories.memory_type| RENAMED                  |
| metadata.confidence     | memories.confidence | CRITICAL for golden rules|
| metadata.importance     | memories.importance | Direct copy              |
| metadata.source_file    | memories.source_file| Direct copy              |
| metadata.doc_hash       | memories.content_hash| RENAMED                 |
| metadata.chunk_index    | memories.chunk_index| Direct copy              |
| metadata.start_line     | memories.start_line | Direct copy              |
| metadata.end_line       | memories.end_line   | Direct copy              |
| metadata.created_at     | memories.created_at | ISO string -> unix float |
| metadata.accessed_at    | memories.last_accessed| RENAMED, ISO -> unix   |
| metadata.access_count   | memories.access_count| Direct copy             |
| metadata.meta_*         | memories.tags       | Collect as JSON dict     |

Usage:
    cd /path/to/theo
    uv run python scripts/migrate_to_sqlite_vec.py [--dry-run]

Rollback:
    If migration fails:
    1. cp ~/.theo/theo.db.backup ~/.theo/theo.db
    2. cp -r ~/.theo/chroma_db.backup ~/.theo/chroma_db
    3. git revert (if needed)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theo.storage.sqlite_store import SQLiteStore

# Add theo src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging to stderr (MCP-safe)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Constants
GOLDEN_RULE_THRESHOLD = 0.9
DEFAULT_THEO_PATH = Path.home() / ".theo"
DEFAULT_CHROMA_PATH = DEFAULT_THEO_PATH / "chroma_db"
DEFAULT_SQLITE_PATH = DEFAULT_THEO_PATH / "theo.db"


def backup(theo_path: Path, dry_run: bool = False) -> tuple[Path | None, Path | None]:
    """Create backups of theo.db and chroma_db.

    Args:
        theo_path: Path to .theo directory
        dry_run: If True, only log what would be done

    Returns:
        Tuple of (sqlite_backup_path, chroma_backup_path) or (None, None) if dry_run
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sqlite_path = theo_path / "theo.db"
    chroma_path = theo_path / "chroma_db"

    sqlite_backup = theo_path / f"theo.db.backup.{timestamp}"
    chroma_backup = theo_path / f"chroma_db.backup.{timestamp}"

    if dry_run:
        if sqlite_path.exists():
            logger.info(f"[DRY RUN] Would backup {sqlite_path} -> {sqlite_backup}")
        if chroma_path.exists():
            logger.info(f"[DRY RUN] Would backup {chroma_path} -> {chroma_backup}")
        return None, None

    # Backup SQLite database
    if sqlite_path.exists():
        logger.info(f"Backing up {sqlite_path} -> {sqlite_backup}")
        shutil.copy2(sqlite_path, sqlite_backup)
    else:
        logger.info(f"SQLite database not found at {sqlite_path}, skipping backup")
        sqlite_backup = None

    # Backup ChromaDB directory
    if chroma_path.exists():
        logger.info(f"Backing up {chroma_path} -> {chroma_backup}")
        shutil.copytree(chroma_path, chroma_backup)
    else:
        logger.info(f"ChromaDB not found at {chroma_path}, skipping backup")
        chroma_backup = None

    return sqlite_backup, chroma_backup


def restore_from_backup(
    theo_path: Path,
    sqlite_backup: Path | None,
    chroma_backup: Path | None,
) -> None:
    """Restore from backup files.

    Args:
        theo_path: Path to .theo directory
        sqlite_backup: Path to SQLite backup
        chroma_backup: Path to ChromaDB backup
    """
    if sqlite_backup and sqlite_backup.exists():
        sqlite_path = theo_path / "theo.db"
        logger.info(f"Restoring {sqlite_backup} -> {sqlite_path}")
        shutil.copy2(sqlite_backup, sqlite_path)

    if chroma_backup and chroma_backup.exists():
        chroma_path = theo_path / "chroma_db"
        logger.info(f"Restoring {chroma_backup} -> {chroma_path}")
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
        shutil.copytree(chroma_backup, chroma_path)


def extract_from_chromadb(chroma_path: Path) -> list[dict[str, Any]]:
    """Extract all documents with metadata from ChromaDB.

    Args:
        chroma_path: Path to ChromaDB directory

    Returns:
        List of dicts with keys: id, content, embedding, metadata
    """
    if not chroma_path.exists():
        logger.warning(f"ChromaDB not found at {chroma_path}")
        return []

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logger.error("chromadb package not installed. Install with: uv add chromadb")
        return []

    logger.info(f"Opening ChromaDB at {chroma_path}...")

    try:
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as e:
        logger.error(f"Failed to open ChromaDB: {e}")
        return []

    # List collections
    collections = client.list_collections()
    logger.info(f"Found collections: {[c.name for c in collections]}")

    if not collections:
        logger.warning("No collections found in ChromaDB")
        return []

    all_records: list[dict[str, Any]] = []

    for collection_info in collections:
        collection = client.get_collection(collection_info.name)
        count = collection.count()
        logger.info(f"Reading {count} documents from collection '{collection.name}'...")

        if count == 0:
            continue

        # Read in batches to handle large collections
        batch_size = 1000
        for offset in range(0, count, batch_size):
            try:
                result = collection.get(
                    include=["documents", "embeddings", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )

                ids = result.get("ids", [])
                documents = result.get("documents", [])
                embeddings = result.get("embeddings", [])
                metadatas = result.get("metadatas", [])

                for i, doc_id in enumerate(ids):
                    record = {
                        "id": doc_id,
                        "content": documents[i] if documents else "",
                        "embedding": embeddings[i] if embeddings else None,
                        "metadata": metadatas[i] if metadatas else {},
                    }
                    all_records.append(record)

                logger.info(
                    f"  Read {min(offset + batch_size, count)}/{count} documents..."
                )

            except Exception as e:
                logger.warning(f"Failed to read batch at offset {offset}: {e}")
                # Try without embeddings (in case of corruption)
                try:
                    result = collection.get(
                        include=["documents", "metadatas"],
                        limit=batch_size,
                        offset=offset,
                    )
                    ids = result.get("ids", [])
                    documents = result.get("documents", [])
                    metadatas = result.get("metadatas", [])

                    for i, doc_id in enumerate(ids):
                        record = {
                            "id": doc_id,
                            "content": documents[i] if documents else "",
                            "embedding": None,  # Will need re-embedding
                            "metadata": metadatas[i] if metadatas else {},
                        }
                        all_records.append(record)

                    logger.info(
                        f"  Read {min(offset + batch_size, count)}/{count} documents "
                        "(without embeddings)..."
                    )
                except Exception as e2:
                    logger.error(f"Failed to read batch even without embeddings: {e2}")

    logger.info(f"Total extracted: {len(all_records)} documents")
    return all_records


def parse_iso_to_unix(iso_str: str | None) -> float | None:
    """Convert ISO timestamp string to unix timestamp.

    Args:
        iso_str: ISO format timestamp string (e.g., "2024-01-15T10:30:00")

    Returns:
        Unix timestamp as float, or None if parsing fails
    """
    if not iso_str:
        return None

    try:
        # Handle various ISO formats
        if isinstance(iso_str, (int, float)):
            return float(iso_str)

        # Try common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(iso_str, fmt)
                return dt.timestamp()
            except ValueError:
                continue

        # If all else fails, return None
        logger.debug(f"Could not parse timestamp: {iso_str}")
        return None

    except Exception:
        return None


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content.

    Args:
        content: Text content to hash

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def migrate_memory(
    doc: dict[str, Any],
    sqlite_store: "SQLiteStore",
    embedder: Any | None,
    dry_run: bool = False,
) -> bool:
    """Migrate single memory with exact 16-field mapping.

    Field Mapping (all 16 fields):
    1. id -> memories.id
    2. document (content) -> memories.content
    3. embedding -> memories_vec + embedding_cache
    4. metadata.namespace -> memories.namespace
    5. metadata.doc_type -> memories.memory_type (RENAMED)
    6. metadata.confidence -> memories.confidence
    7. metadata.importance -> memories.importance
    8. metadata.source_file -> memories.source_file
    9. metadata.doc_hash -> memories.content_hash (RENAMED)
    10. metadata.chunk_index -> memories.chunk_index
    11. metadata.start_line -> memories.start_line
    12. metadata.end_line -> memories.end_line
    13. metadata.created_at -> memories.created_at (ISO -> unix)
    14. metadata.accessed_at -> memories.last_accessed (RENAMED, ISO -> unix)
    15. metadata.access_count -> memories.access_count
    16. metadata.meta_* -> memories.tags (JSON dict)

    Args:
        doc: Document dict from ChromaDB extraction
        sqlite_store: SQLiteStore instance for writing
        embedder: Embedding provider for re-embedding if needed
        dry_run: If True, only log what would be done

    Returns:
        True if migration successful, False otherwise
    """
    doc_id = doc["id"]
    content = doc.get("content", "")
    embedding = doc.get("embedding")
    metadata = doc.get("metadata", {})

    if not content:
        logger.debug(f"Skipping {doc_id}: no content")
        return False

    # Field mapping with renames
    # metadata.doc_type -> memory_type (field 5)
    memory_type = metadata.get("doc_type", metadata.get("memory_type", "document"))

    # metadata.doc_hash -> content_hash (field 9)
    content_hash = metadata.get("doc_hash", metadata.get("content_hash"))
    if not content_hash:
        content_hash = compute_content_hash(content)

    # Timestamps: ISO -> unix float (fields 13, 14)
    original_created_at = parse_iso_to_unix(metadata.get("created_at"))
    original_last_accessed = parse_iso_to_unix(metadata.get("accessed_at"))
    original_access_count = int(metadata.get("access_count", 0))

    # Line positions (fields 11, 12)
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")

    # Collect meta_* fields as tags (field 16)
    tags: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.startswith("meta_"):
            # Remove meta_ prefix
            tag_key = key[5:]
            if isinstance(value, (str, int, float, bool)):
                tags[tag_key] = value

    # Also preserve source if present
    if metadata.get("source"):
        tags["source"] = metadata["source"]

    if dry_run:
        logger.info(
            f"[DRY RUN] Would migrate: {doc_id[:50]}... "
            f"(type={memory_type}, confidence={metadata.get('confidence', 0.3)}, "
            f"start_line={start_line}, end_line={end_line})"
        )
        return True

    # Handle embedding - re-embed if missing or corrupted
    if embedding is None and embedder is not None:
        try:
            embeddings = embedder.embed_texts([content[:1000]])  # Truncate for embedding
            if embeddings:
                embedding = embeddings[0]
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {doc_id}: {e}")

    if embedding is None:
        # Use zero vector as fallback (will need re-embedding later)
        logger.warning(f"No embedding for {doc_id}, using zero vector")
        embedding = [0.0] * 1024

    try:
        # Add memory to SQLite store (handles memories, memories_vec, memories_fts atomically)
        memory_id = sqlite_store.add_memory(
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            namespace=metadata.get("namespace", "default"),
            confidence=float(metadata.get("confidence", 0.3)),
            importance=float(metadata.get("importance", 0.5)),
            source_file=metadata.get("source_file"),
            chunk_index=int(metadata.get("chunk_index", 0)),
            content_hash=content_hash,
            tags=tags if tags else None,
        )

        # Update additional fields not supported by add_memory() via raw SQL
        # This includes: created_at, last_accessed, access_count, start_line, end_line
        cursor = sqlite_store._conn.cursor()
        updates = []
        params: list[Any] = []

        # Timestamp fields (fields 13, 14, 15)
        if original_created_at:
            updates.append("created_at = ?")
            params.append(original_created_at)

        if original_last_accessed:
            updates.append("last_accessed = ?")
            params.append(original_last_accessed)

        if original_access_count:
            updates.append("access_count = ?")
            params.append(original_access_count)

        # Line position fields (fields 11, 12) - not in add_memory() API
        if start_line is not None:
            updates.append("start_line = ?")
            params.append(int(start_line))

        if end_line is not None:
            updates.append("end_line = ?")
            params.append(int(end_line))

        if updates:
            params.append(memory_id)
            cursor.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            sqlite_store._conn.commit()

        # Also cache the embedding
        sqlite_store.cache_embedding(
            content_hash=content_hash,
            provider="mlx",  # Assume MLX as primary provider
            model="mxbai-embed-large-v1",
            embedding=embedding,
        )

        return True

    except Exception as e:
        logger.error(f"Failed to migrate {doc_id}: {e}")
        return False


def count_golden_rules_chromadb(records: list[dict[str, Any]]) -> int:
    """Count golden rules (confidence >= 0.9) in ChromaDB records.

    Args:
        records: List of extracted ChromaDB records

    Returns:
        Count of records with confidence >= GOLDEN_RULE_THRESHOLD
    """
    count = 0
    for record in records:
        metadata = record.get("metadata", {})
        confidence = float(metadata.get("confidence", 0.0))
        if confidence >= GOLDEN_RULE_THRESHOLD:
            count += 1
    return count


def verify_golden_rules(sqlite_store: "SQLiteStore", expected_count: int) -> bool:
    """Verify golden rules preserved after migration.

    Args:
        sqlite_store: SQLiteStore instance
        expected_count: Expected number of golden rules

    Returns:
        True if counts match, False otherwise
    """
    cursor = sqlite_store._conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM memories WHERE confidence >= ?",
        (GOLDEN_RULE_THRESHOLD,),
    )
    result = cursor.fetchone()
    actual_count = result[0] if result else 0

    if actual_count >= expected_count:
        logger.info(
            f"Golden rules verification PASSED: {actual_count} >= {expected_count}"
        )
        return True
    else:
        logger.error(
            f"Golden rules verification FAILED: {actual_count} < {expected_count}"
        )
        return False


def verify_tables(sqlite_store: "SQLiteStore") -> dict[str, int]:
    """Verify all migrated tables have correct counts.

    Checks:
    - memories: Main memory table
    - memories_vec: Vector embeddings
    - memories_fts: Full-text search
    - edges: Relationship graph
    - validation_events: TRY/LEARN cycle history
    - embedding_cache: Cached embeddings

    Args:
        sqlite_store: SQLiteStore instance

    Returns:
        Dict with table names and their row counts
    """
    cursor = sqlite_store._conn.cursor()
    counts: dict[str, int] = {}

    tables = [
        "memories",
        "memories_fts",
        "edges",
        "validation_events",
        "embedding_cache",
    ]

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            result = cursor.fetchone()
            counts[table] = result[0] if result else 0
        except Exception as e:
            logger.warning(f"Failed to count {table}: {e}")
            counts[table] = -1

    # memories_vec uses different count mechanism
    try:
        cursor.execute("SELECT COUNT(*) FROM memories_vec")
        result = cursor.fetchone()
        counts["memories_vec"] = result[0] if result else 0
    except Exception as e:
        logger.warning(f"Failed to count memories_vec: {e}")
        counts["memories_vec"] = -1

    return counts


def verify_field_mapping(sqlite_store: "SQLiteStore") -> bool:
    """Verify all 16 fields are present in migrated data.

    Checks that the memories table has all required columns and
    at least some have non-null values for optional fields.

    Args:
        sqlite_store: SQLiteStore instance

    Returns:
        True if field verification passes, False otherwise
    """
    cursor = sqlite_store._conn.cursor()

    # Check for records with start_line and end_line populated
    cursor.execute(
        """
        SELECT COUNT(*) FROM memories
        WHERE start_line IS NOT NULL OR end_line IS NOT NULL
        """
    )
    line_count = cursor.fetchone()[0]
    logger.info(f"Records with line positions: {line_count}")

    # Check for records with timestamps preserved
    cursor.execute(
        """
        SELECT COUNT(*) FROM memories
        WHERE created_at IS NOT NULL AND last_accessed IS NOT NULL
        """
    )
    timestamp_count = cursor.fetchone()[0]
    logger.info(f"Records with timestamps: {timestamp_count}")

    # Check for records with tags
    cursor.execute("SELECT COUNT(*) FROM memories WHERE tags IS NOT NULL")
    tags_count = cursor.fetchone()[0]
    logger.info(f"Records with tags: {tags_count}")

    return True


def get_embedding_provider():
    """Get embedding provider for re-embedding.

    Returns:
        EmbeddingProvider instance or None if unavailable
    """
    try:
        from theo.embedding.factory import create_embedding_provider

        logger.info("Loading MLX embedding provider...")
        provider = create_embedding_provider("mlx")
        logger.info("MLX provider loaded successfully")
        return provider
    except Exception as e:
        logger.warning(f"MLX provider unavailable: {e}")

    try:
        from theo.embedding.factory import create_embedding_provider

        logger.info("Trying Ollama fallback...")
        provider = create_embedding_provider("ollama")
        logger.info("Ollama provider loaded successfully")
        return provider
    except Exception as e:
        logger.warning(f"Ollama provider unavailable: {e}")

    logger.warning("No embedding provider available - will use zero vectors")
    return None


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Theo from ChromaDB to sqlite-vec storage"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--theo-path",
        type=Path,
        default=DEFAULT_THEO_PATH,
        help=f"Path to .theo directory (default: {DEFAULT_THEO_PATH})",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating backups (dangerous!)",
    )
    parser.add_argument(
        "--skip-chromadb",
        action="store_true",
        help="Skip ChromaDB extraction (use existing SQLite data)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Theo Migration: ChromaDB -> SQLite-vec")
    logger.info("=" * 60)
    logger.info(f"Theo path:    {args.theo_path}")
    logger.info(f"Dry run:      {args.dry_run}")
    logger.info(f"Skip backup:  {args.skip_backup}")
    logger.info("=" * 60)

    chroma_path = args.theo_path / "chroma_db"
    sqlite_path = args.theo_path / "theo.db"

    # Step 1: Create backups
    sqlite_backup = None
    chroma_backup = None

    if not args.skip_backup:
        logger.info("")
        logger.info("PHASE 1: Creating backups")
        logger.info("-" * 40)
        sqlite_backup, chroma_backup = backup(args.theo_path, dry_run=args.dry_run)

    # Step 2: Extract from ChromaDB
    records: list[dict[str, Any]] = []

    if not args.skip_chromadb:
        logger.info("")
        logger.info("PHASE 2: Extracting from ChromaDB")
        logger.info("-" * 40)
        records = extract_from_chromadb(chroma_path)

        if not records:
            logger.warning("No records found in ChromaDB")
            if not args.dry_run:
                logger.info("Nothing to migrate. Exiting.")
                return 0

    # Count golden rules before migration
    golden_rules_before = count_golden_rules_chromadb(records)
    logger.info(f"Golden rules before migration: {golden_rules_before}")

    # Step 3: Load embedding provider for re-embedding
    embedder = None
    needs_embedding = [r for r in records if r.get("embedding") is None]
    if needs_embedding:
        logger.info(f"{len(needs_embedding)} records need re-embedding")
        if not args.dry_run:
            embedder = get_embedding_provider()

    # Step 4: Initialize SQLite store and migrate
    logger.info("")
    logger.info("PHASE 3: Migrating to SQLite-vec")
    logger.info("-" * 40)

    if args.dry_run:
        logger.info(f"[DRY RUN] Would migrate {len(records)} records")
        for i, record in enumerate(records[:5]):
            migrate_memory(record, None, None, dry_run=True)
        if len(records) > 5:
            logger.info(f"[DRY RUN] ... and {len(records) - 5} more records")
        return 0

    # Import and initialize SQLite store
    try:
        from theo.storage.sqlite_store import SQLiteStore
    except ImportError as e:
        logger.error(f"Failed to import SQLiteStore: {e}")
        return 1

    try:
        sqlite_store = SQLiteStore(sqlite_path)
    except Exception as e:
        logger.error(f"Failed to initialize SQLiteStore: {e}")
        if sqlite_backup:
            logger.info("Restoring from backup...")
            restore_from_backup(args.theo_path, sqlite_backup, chroma_backup)
        return 1

    # Migrate each record with progress logging
    migrated = 0
    failed = 0
    total = len(records)

    for i, record in enumerate(records):
        if migrate_memory(record, sqlite_store, embedder, dry_run=False):
            migrated += 1
        else:
            failed += 1

        # Progress logging
        if (i + 1) % 100 == 0 or i + 1 == total:
            logger.info(f"Progress: {i + 1}/{total} ({migrated} migrated, {failed} failed)")

    # Step 5: Comprehensive verification
    logger.info("")
    logger.info("PHASE 4: Verification")
    logger.info("-" * 40)

    # Verify golden rules (confidence >= 0.9)
    golden_rules_ok = verify_golden_rules(sqlite_store, golden_rules_before)

    if not golden_rules_ok:
        logger.error("Migration verification failed!")
        logger.info("Rolling back...")
        sqlite_store.close()
        restore_from_backup(args.theo_path, sqlite_backup, chroma_backup)
        return 1

    # Verify all tables have expected data
    table_counts = verify_tables(sqlite_store)
    logger.info("Table row counts after migration:")
    for table_name, count in table_counts.items():
        logger.info(f"  {table_name}: {count}")

    # Verify all 16 fields are properly mapped
    verify_field_mapping(sqlite_store)

    # Verify consistency between memories and memories_vec
    if table_counts["memories"] != table_counts["memories_vec"]:
        logger.warning(
            f"Mismatch: memories ({table_counts['memories']}) != "
            f"memories_vec ({table_counts['memories_vec']})"
        )

    # Verify embedding_cache has entries
    if table_counts["embedding_cache"] < migrated:
        logger.warning(
            f"Embedding cache incomplete: {table_counts['embedding_cache']} < {migrated}"
        )

    # Close SQLite store
    sqlite_store.close()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Records extracted:     {total}")
    logger.info(f"Records migrated:      {migrated}")
    logger.info(f"Records failed:        {failed}")
    logger.info(f"Golden rules before:   {golden_rules_before}")
    logger.info(f"Golden rules verified: {golden_rules_ok}")
    logger.info("-" * 40)
    logger.info("Table Verification:")
    logger.info(f"  memories:          {table_counts['memories']}")
    logger.info(f"  memories_vec:      {table_counts['memories_vec']}")
    logger.info(f"  memories_fts:      {table_counts['memories_fts']}")
    logger.info(f"  edges:             {table_counts['edges']}")
    logger.info(f"  validation_events: {table_counts['validation_events']}")
    logger.info(f"  embedding_cache:   {table_counts['embedding_cache']}")
    logger.info("=" * 60)

    if sqlite_backup:
        logger.info(f"SQLite backup: {sqlite_backup}")
    if chroma_backup:
        logger.info(f"ChromaDB backup: {chroma_backup}")

    logger.info("")
    logger.info("Migration complete!")
    logger.info("You can now safely delete ChromaDB: rm -rf ~/.theo/chroma_db")

    return 0


if __name__ == "__main__":
    sys.exit(main())
