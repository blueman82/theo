#!/usr/bin/env python3
"""Direct SQLite-to-SQLite migration from ChromaDB to sqlite-vec.

This script bypasses the chromadb Python library entirely, reading directly
from ChromaDB's internal SQLite tables and writing to theo's sqlite-vec tables.

Why this exists:
- chromadb library segfaults on large collections (70k+ documents)
- Direct SQLite access is reliable and fast

What it does:
1. Reads metadata and content from ChromaDB's embedding_metadata table
2. Re-generates embeddings using MLX (HNSW binary format is complex)
3. Writes to theo.db (memories, memories_vec, memories_fts tables)

Usage:
    uv run python scripts/migrate_direct_sqlite.py [--dry-run] [--batch-size 100]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add theo src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Paths
THEO_PATH = Path.home() / ".theo"
CHROMA_SQLITE = THEO_PATH / "chroma_db" / "chroma.sqlite3"
THEO_DB = THEO_PATH / "theo.db"


def extract_from_chroma_sqlite(chroma_path: Path) -> list[dict[str, Any]]:
    """Extract all documents from ChromaDB's SQLite directly.

    ChromaDB stores:
    - embeddings table: id, embedding_id (our doc ID), segment_id, seq_id
    - embedding_metadata table: id (FK to embeddings), key, string/int/float/bool_value
    - Document content is stored with key='chroma:document'

    Returns:
        List of dicts with: id, content, metadata (dict of all other fields)
    """
    if not chroma_path.exists():
        raise FileNotFoundError(f"ChromaDB SQLite not found: {chroma_path}")

    conn = sqlite3.connect(str(chroma_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get embedding count
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total = cursor.fetchone()[0]
    logger.info(f"Found {total} embeddings in ChromaDB")

    # Build mapping from internal ID to embedding_id (our document ID)
    logger.info("Reading embeddings table...")
    cursor.execute("SELECT id, embedding_id FROM embeddings")
    id_map = {row["id"]: row["embedding_id"] for row in cursor.fetchall()}

    # Read all metadata
    logger.info("Reading embedding_metadata table...")
    cursor.execute("""
        SELECT id, key, string_value, int_value, float_value, bool_value
        FROM embedding_metadata
    """)

    # Group metadata by document
    docs: dict[str, dict[str, Any]] = {}
    row_count = 0

    for row in cursor:
        row_count += 1
        internal_id = row["id"]
        doc_id = id_map.get(internal_id)
        if not doc_id:
            continue

        if doc_id not in docs:
            docs[doc_id] = {"id": doc_id, "content": "", "metadata": {}}

        key = row["key"]

        # Get the non-null value
        value = None
        if row["string_value"] is not None:
            value = row["string_value"]
        elif row["float_value"] is not None:
            value = row["float_value"]
        elif row["int_value"] is not None:
            value = row["int_value"]
        elif row["bool_value"] is not None:
            value = bool(row["bool_value"])

        # Document content has special key
        if key == "chroma:document":
            docs[doc_id]["content"] = value or ""
        else:
            docs[doc_id]["metadata"][key] = value

    conn.close()

    logger.info(f"Processed {row_count} metadata rows into {len(docs)} documents")

    # Filter out documents without content
    valid_docs = [d for d in docs.values() if d.get("content")]
    logger.info(f"Documents with content: {len(valid_docs)}")

    return valid_docs


def get_embedding_provider():
    """Get MLX embedding provider."""
    try:
        from theo.embedding.factory import create_embedding_provider
        logger.info("Loading MLX embedding provider...")
        provider = create_embedding_provider("mlx")
        logger.info("MLX provider loaded")
        return provider
    except Exception as e:
        logger.warning(f"MLX failed: {e}, trying Ollama...")
        try:
            from theo.embedding.factory import create_embedding_provider
            provider = create_embedding_provider("ollama")
            logger.info("Ollama provider loaded")
            return provider
        except Exception as e2:
            logger.error(f"No embedding provider available: {e2}")
            return None


def parse_timestamp(ts: Any) -> float | None:
    """Parse various timestamp formats to unix float."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue
    return None


def migrate_to_sqlite_vec(
    docs: list[dict[str, Any]],
    theo_db_path: Path,
    embedder: Any,
    batch_size: int = 100,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate documents to sqlite-vec tables.

    Returns:
        Tuple of (migrated_count, failed_count)
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {len(docs)} documents")
        return len(docs), 0

    # Import SQLiteStore
    from theo.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(theo_db_path)

    migrated = 0
    failed = 0

    # Process in batches for embedding generation
    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start:batch_start + batch_size]

        # Generate embeddings for batch
        texts = [d["content"][:2000] for d in batch]  # Truncate for embedding

        try:
            if embedder:
                embeddings = embedder.embed_texts(texts)
            else:
                embeddings = [[0.0] * 1024 for _ in texts]
        except Exception as e:
            logger.warning(f"Embedding batch failed: {e}, using zero vectors")
            embeddings = [[0.0] * 1024 for _ in texts]

        # Insert each document
        for i, doc in enumerate(batch):
            try:
                meta = doc.get("metadata", {})

                # Map ChromaDB metadata to sqlite-vec fields
                memory_type = meta.get("doc_type", meta.get("memory_type", "document"))
                namespace = meta.get("namespace", "default")
                confidence = float(meta.get("confidence", 0.3))
                importance = float(meta.get("importance", meta.get("meta_importance", 0.5)))
                source_file = meta.get("source_file")
                chunk_index = int(meta.get("chunk_index", 0))
                content_hash = meta.get("doc_hash") or hashlib.sha256(doc["content"].encode()).hexdigest()

                # Collect meta_* fields as tags
                tags = {}
                for k, v in meta.items():
                    if k.startswith("meta_") and isinstance(v, (str, int, float, bool)):
                        tags[k[5:]] = v  # Remove meta_ prefix

                # Add memory
                memory_id = store.add_memory(
                    content=doc["content"],
                    embedding=embeddings[i],
                    memory_type=memory_type,
                    namespace=namespace,
                    confidence=confidence,
                    importance=importance,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                    tags=tags if tags else None,
                )

                # Update additional fields via raw SQL
                cursor = store._conn.cursor()
                updates = []
                params: list[Any] = []

                # Timestamps
                created_at = parse_timestamp(meta.get("created_at"))
                if created_at:
                    updates.append("created_at = ?")
                    params.append(created_at)

                last_accessed = parse_timestamp(meta.get("accessed_at"))
                if last_accessed:
                    updates.append("last_accessed = ?")
                    params.append(last_accessed)

                access_count = meta.get("access_count")
                if access_count:
                    updates.append("access_count = ?")
                    params.append(int(access_count))

                # Line positions
                start_line = meta.get("start_line")
                if start_line is not None:
                    updates.append("start_line = ?")
                    params.append(int(start_line))

                end_line = meta.get("end_line")
                if end_line is not None:
                    updates.append("end_line = ?")
                    params.append(int(end_line))

                if updates:
                    params.append(memory_id)
                    cursor.execute(
                        f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
                    store._conn.commit()

                migrated += 1

            except Exception as e:
                logger.warning(f"Failed to migrate {doc['id'][:50]}: {e}")
                failed += 1

        logger.info(f"Progress: {batch_start + len(batch)}/{len(docs)} ({migrated} migrated, {failed} failed)")

    store.close()
    return migrated, failed


def verify_migration(theo_db_path: Path) -> dict[str, int]:
    """Verify migration by checking table counts."""
    conn = sqlite3.connect(str(theo_db_path))
    cursor = conn.cursor()

    counts = {}
    for table in ["memories", "memories_fts", "edges", "validation_events"]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        except Exception:
            counts[table] = -1

    conn.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description="Direct SQLite-to-SQLite migration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--chroma-path", type=Path, default=CHROMA_SQLITE)
    parser.add_argument("--theo-db", type=Path, default=THEO_DB)
    parser.add_argument("--skip-existing", action="store_true", help="Skip if memories already exist")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Direct SQLite Migration: ChromaDB -> sqlite-vec")
    logger.info("=" * 60)
    logger.info(f"Source: {args.chroma_path}")
    logger.info(f"Target: {args.theo_db}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Check if migration already done
    if args.skip_existing and args.theo_db.exists():
        counts = verify_migration(args.theo_db)
        if counts.get("memories", 0) > 1000:
            logger.info(f"Skipping: {counts['memories']} memories already exist")
            return 0

    # Phase 1: Extract from ChromaDB
    logger.info("")
    logger.info("PHASE 1: Extracting from ChromaDB SQLite")
    logger.info("-" * 40)
    docs = extract_from_chroma_sqlite(args.chroma_path)

    if not docs:
        logger.error("No documents found!")
        return 1

    # Count golden rules
    golden_rules = sum(1 for d in docs if float(d.get("metadata", {}).get("confidence", 0)) >= 0.9)
    logger.info(f"Golden rules (confidence >= 0.9): {golden_rules}")

    # Phase 2: Load embedder
    logger.info("")
    logger.info("PHASE 2: Loading embedding provider")
    logger.info("-" * 40)
    embedder = None if args.dry_run else get_embedding_provider()

    # Phase 3: Migrate
    logger.info("")
    logger.info("PHASE 3: Migrating to sqlite-vec")
    logger.info("-" * 40)
    migrated, failed = migrate_to_sqlite_vec(
        docs, args.theo_db, embedder, args.batch_size, args.dry_run
    )

    # Phase 4: Verify
    if not args.dry_run:
        logger.info("")
        logger.info("PHASE 4: Verification")
        logger.info("-" * 40)
        counts = verify_migration(args.theo_db)
        for table, count in counts.items():
            logger.info(f"  {table}: {count}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Documents extracted: {len(docs)}")
    logger.info(f"Golden rules:        {golden_rules}")
    logger.info(f"Migrated:            {migrated}")
    logger.info(f"Failed:              {failed}")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
