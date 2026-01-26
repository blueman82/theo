#!/usr/bin/env python3
"""Migrate data from DocVec and Recall into Theo.

This script performs a clean API-based migration that:
1. Reads DocVec chunks from ~/.docvec/chroma_db/ (collection: "documents")
2. Reads Recall memories from ~/.recall/recall.db (SQLite)
3. Generates fresh embeddings using Theo's MLX provider
4. Writes everything to Theo's ~/.theo/chroma_db/

Benefits over file copy:
- Avoids copying 351GB of bloated/corrupted HNSW index
- Creates fresh, properly-sized index
- Handles schema transformation
- Deduplicates content by hash

Usage:
    cd /path/to/theo
    uv run python scripts/migrate_data.py [--dry-run]
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add theo src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb
from chromadb.config import Settings

# Embedding provider (lazy loaded)
_embedding_provider = None


def get_embedding_provider():
    """Lazy load Theo's embedding provider."""
    global _embedding_provider

    if _embedding_provider is None:
        log("Loading Theo embedding provider (MLX)...")
        try:
            from theo.embedding import create_embedding_provider
            _embedding_provider = create_embedding_provider("mlx")
            log("  MLX provider loaded successfully")
        except Exception as e:
            log(f"  Warning: Failed to load MLX provider: {e}")
            log("  Trying Ollama fallback...")
            try:
                from theo.embedding import create_embedding_provider
                _embedding_provider = create_embedding_provider("ollama")
                log("  Ollama provider loaded successfully")
            except Exception as e2:
                log(f"  Warning: Ollama also failed: {e2}")
                _embedding_provider = None

    return _embedding_provider


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Theo's provider."""
    provider = get_embedding_provider()

    if provider is None:
        log("  Warning: No embedding provider available, using zero vectors")
        return [[0.0] * 1024 for _ in texts]

    try:
        embeddings = provider.embed_texts(texts)
        return embeddings
    except Exception as e:
        log(f"  Warning: Embedding generation failed: {e}")
        return [[0.0] * 1024 for _ in texts]


def log(msg: str) -> None:
    """Log to stderr (safe for MCP context)."""
    print(msg, file=sys.stderr)


def get_content_hash(content: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def read_docvec_data(db_path: Path) -> dict[str, Any]:
    """Read all data from DocVec's ChromaDB.

    Returns:
        Dict with keys: ids, documents, embeddings, metadatas
    """
    log(f"Reading DocVec data from {db_path}...")

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Get the documents collection
    try:
        collection = client.get_collection("documents")
    except Exception as e:
        log(f"  Warning: Could not get 'documents' collection: {e}")
        return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    count = collection.count()
    log(f"  Found {count} documents in 'documents' collection")

    if count == 0:
        return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    # Get all data including embeddings
    # ChromaDB limits batch size, so we may need to paginate
    batch_size = 1000
    all_data = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    for offset in range(0, count, batch_size):
        log(f"  Reading batch {offset}-{min(offset + batch_size, count)}...")
        try:
            # Try to read with embeddings first
            result = collection.get(
                include=["documents", "embeddings", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            all_data["ids"].extend(result["ids"])
            all_data["documents"].extend(result["documents"] or [])
            all_data["embeddings"].extend(result["embeddings"] or [])
            all_data["metadatas"].extend(result["metadatas"] or [])
        except Exception as e:
            log(f"  Warning: Failed to read embeddings for batch {offset}: {e}")
            log(f"  Trying without embeddings...")
            # Fall back to reading without embeddings
            result = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            all_data["ids"].extend(result["ids"])
            all_data["documents"].extend(result["documents"] or [])
            # Add None for each embedding we couldn't read
            all_data["embeddings"].extend([None] * len(result["ids"]))
            all_data["metadatas"].extend(result["metadatas"] or [])

    log(f"  Read {len(all_data['ids'])} documents with embeddings")
    return all_data


def read_recall_sqlite(db_path: Path) -> dict[str, Any]:
    """Read memories and edges from Recall's SQLite database.

    Returns:
        Dict with keys: memories (list of dicts), edges (list of dicts)
    """
    log(f"Reading Recall data from {db_path}...")

    if not db_path.exists():
        log(f"  Warning: SQLite database not found at {db_path}")
        return {"memories": [], "edges": []}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Read memories
    cursor.execute("""
        SELECT id, content, content_hash, type, namespace,
               importance, confidence, created_at, accessed_at,
               access_count, metadata
        FROM memories
    """)
    memories = [dict(row) for row in cursor.fetchall()]
    log(f"  Found {len(memories)} memories")

    # Read edges (relationships)
    cursor.execute("""
        SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
        FROM edges
    """)
    edges = [dict(row) for row in cursor.fetchall()]
    log(f"  Found {len(edges)} edges")

    conn.close()
    return {"memories": memories, "edges": edges}


def transform_docvec_to_theo(docvec_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform DocVec data to Theo format.

    Returns:
        List of dicts with keys: id, content, embedding, metadata
    """
    log("Transforming DocVec data to Theo format...")

    # Deduplicate by ID first (in case of retry duplicates)
    seen_ids: set[str] = set()
    unique_indices = []
    for i, doc_id in enumerate(docvec_data["ids"]):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_indices.append(i)

    log(f"  Deduplicating: {len(docvec_data['ids'])} -> {len(unique_indices)} unique documents")

    transformed = []
    embeddings_list = docvec_data.get("embeddings", [])
    for i in unique_indices:
        doc_id = docvec_data["ids"][i]
        content = docvec_data["documents"][i] if i < len(docvec_data["documents"]) else ""
        embedding = embeddings_list[i] if i < len(embeddings_list) else None
        metadata = docvec_data["metadatas"][i] if i < len(docvec_data["metadatas"]) else {}

        # Transform metadata to Theo format
        theo_metadata = {
            "namespace": "default",
            "doc_type": "document",
            "confidence": 1.0,  # Documents have full confidence
            "chunk_index": metadata.get("chunk_index", 0),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            "source": "docvec",  # Track migration source
        }

        # Preserve DocVec-specific fields
        if metadata.get("source_file"):
            theo_metadata["source_file"] = metadata["source_file"]
        if metadata.get("doc_hash"):
            theo_metadata["doc_hash"] = metadata["doc_hash"]
        else:
            theo_metadata["doc_hash"] = get_content_hash(content)

        # Preserve additional metadata
        for key in ["start_line", "end_line", "header_path", "header_title", "header_level", "type"]:
            if key in metadata:
                theo_metadata[key] = metadata[key]

        transformed.append({
            "id": f"docvec_{doc_id}",  # Prefix to avoid ID collisions
            "content": content,
            "embedding": embedding,
            "metadata": theo_metadata,
        })

    log(f"  Transformed {len(transformed)} DocVec documents")
    return transformed


def transform_recall_to_theo(recall_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transform Recall memories to Theo format.

    Note: Recall memories have embeddings in the shared ChromaDB at ~/.docvec/chroma_db/,
    but that HNSW index is corrupted (608GB bloat). Embeddings cannot be read.
    All items will need re-embedding via Theo's MLX backend after migration.

    Returns:
        Tuple of (memories list, edges list) in Theo format
    """
    log("Transforming Recall data to Theo format...")

    memories = []
    for mem in recall_data["memories"]:
        theo_metadata = {
            "namespace": mem.get("namespace", "global"),
            "doc_type": "memory",
            "confidence": mem.get("confidence", 0.3),
            "chunk_index": 0,
            "created_at": datetime.fromtimestamp(mem.get("created_at", 0)).isoformat() if mem.get("created_at") else datetime.now().isoformat(),
            "source": "recall",
            "doc_hash": mem.get("content_hash") or get_content_hash(mem.get("content", "")),
            "memory_type": mem.get("type", "general"),
            "importance": mem.get("importance", 0.5),
            "accessed_at": datetime.fromtimestamp(mem.get("accessed_at", 0)).isoformat() if mem.get("accessed_at") else None,
            "access_count": mem.get("access_count", 0),
        }

        # Parse and include any additional metadata
        if mem.get("metadata"):
            try:
                extra_meta = json.loads(mem["metadata"]) if isinstance(mem["metadata"], str) else mem["metadata"]
                for k, v in extra_meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        theo_metadata[f"meta_{k}"] = v
            except (json.JSONDecodeError, TypeError):
                pass

        memories.append({
            "id": f"recall_{mem['id']}",
            "content": mem.get("content", ""),
            "embedding": None,  # Will need to be re-generated or fetched from ChromaDB
            "metadata": theo_metadata,
        })

    # Transform edges - store as JSON for now (Theo doesn't have native edge support yet)
    edges = []
    for edge in recall_data["edges"]:
        edges.append({
            "id": edge["id"],
            "source_id": f"recall_{edge['source_id']}",
            "target_id": f"recall_{edge['target_id']}",
            "edge_type": edge.get("edge_type", "related"),
            "weight": edge.get("weight", 1.0),
            "created_at": edge.get("created_at"),
            "metadata": edge.get("metadata"),
        })

    log(f"  Transformed {len(memories)} memories and {len(edges)} edges")
    return memories, edges


def deduplicate_by_hash(
    docvec_items: list[dict[str, Any]],
    recall_items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate items by content hash.

    When duplicates exist, prefer Recall items (they have confidence scores).

    Returns:
        Tuple of (deduplicated list, duplicate count)
    """
    log("Deduplicating by content hash...")

    seen_hashes: dict[str, dict[str, Any]] = {}
    duplicates = 0

    # First add Recall items (preferred source)
    for item in recall_items:
        doc_hash = item["metadata"].get("doc_hash")
        if doc_hash:
            seen_hashes[doc_hash] = item

    # Then add DocVec items, skipping duplicates
    for item in docvec_items:
        doc_hash = item["metadata"].get("doc_hash")
        if doc_hash and doc_hash in seen_hashes:
            duplicates += 1
            continue
        if doc_hash:
            seen_hashes[doc_hash] = item
        else:
            # No hash, just add it
            seen_hashes[item["id"]] = item

    log(f"  Found {duplicates} duplicates (preferring Recall versions)")
    return list(seen_hashes.values()), duplicates


def write_to_theo(
    items: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    db_path: Path,
    dry_run: bool = False
) -> int:
    """Write migrated data to Theo's ChromaDB.

    Returns:
        Number of items written
    """
    log(f"Writing to Theo at {db_path}...")

    if dry_run:
        log(f"  [DRY RUN] Would write {len(items)} items")
        return len(items)

    # Separate items with embeddings from those without
    with_embeddings = [i for i in items if i["embedding"] is not None]
    without_embeddings = [i for i in items if i["embedding"] is None]

    log(f"  {len(with_embeddings)} items with embeddings")
    log(f"  {len(without_embeddings)} items without embeddings (will generate)")

    # Generate embeddings for items that don't have them
    if without_embeddings:
        log(f"  Generating embeddings for {len(without_embeddings)} items...")
        texts = [i["content"] for i in without_embeddings]

        # Generate in batches with progress
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = generate_embeddings_batch(batch_texts)

            for j, emb in enumerate(batch_embeddings):
                without_embeddings[i + j]["embedding"] = emb

            log(f"    Generated {min(i + batch_size, len(texts))}/{len(texts)} embeddings...")

        # Merge back
        with_embeddings.extend(without_embeddings)

    if not with_embeddings:
        log("  Warning: No items to write")
        return 0

    # Initialize Theo's ChromaDB
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )

    # Write in batches
    batch_size = 1000
    written = 0

    for i in range(0, len(with_embeddings), batch_size):
        batch = with_embeddings[i:i + batch_size]

        ids = [item["id"] for item in batch]
        documents = [item["content"] for item in batch]
        embeddings = [item["embedding"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        written += len(batch)
        log(f"  Written {written}/{len(with_embeddings)} items...")

    # Store edges in Theo's SQLite database
    if edges:
        # Import here to avoid dependency issues with uv script
        sqlite_path = db_path.parent / "theo.db"
        sqlite_conn = sqlite3.connect(str(sqlite_path))
        cursor = sqlite_conn.cursor()

        # Create edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'relates_to',
                weight REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                metadata TEXT,
                UNIQUE(source_id, target_id, edge_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

        # Insert edges
        for edge in edges:
            cursor.execute(
                """
                INSERT OR REPLACE INTO edges
                (source_id, target_id, edge_type, weight, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    edge["source_id"],
                    edge["target_id"],
                    edge.get("edge_type", "relates_to"),
                    edge.get("weight", 1.0),
                    edge.get("created_at") or time.time(),
                    json.dumps(edge.get("metadata")) if edge.get("metadata") else None,
                ),
            )

        sqlite_conn.commit()
        sqlite_conn.close()
        log(f"  Saved {len(edges)} edges to {sqlite_path}")

    # Report on items needing re-embedding
    if without_embeddings:
        needs_embedding_file = db_path.parent / "needs_embedding.json"
        with open(needs_embedding_file, "w") as f:
            json.dump([{"id": i["id"], "content": i["content"][:100]} for i in without_embeddings], f, indent=2)
        log(f"  Saved {len(without_embeddings)} items needing embedding to {needs_embedding_file}")

    log(f"  Total written: {written}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Migrate data from DocVec/Recall to Theo")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    parser.add_argument("--docvec-path", type=Path, default=Path.home() / ".docvec" / "chroma_db")
    parser.add_argument("--recall-path", type=Path, default=Path.home() / ".recall" / "recall.db")
    parser.add_argument("--theo-path", type=Path, default=Path.home() / ".theo" / "chroma_db")
    args = parser.parse_args()

    log("=" * 60)
    log("Theo Data Migration")
    log("=" * 60)
    log(f"DocVec source: {args.docvec_path}")
    log(f"Recall source: {args.recall_path}")
    log(f"Theo target:   {args.theo_path}")
    log(f"Dry run:       {args.dry_run}")
    log("=" * 60)

    # Step 1: Read source data
    docvec_data = read_docvec_data(args.docvec_path)
    recall_data = read_recall_sqlite(args.recall_path)

    # Step 2: Transform to Theo format
    docvec_items = transform_docvec_to_theo(docvec_data)
    recall_items, edges = transform_recall_to_theo(recall_data)

    # Step 3: Deduplicate
    all_items, dup_count = deduplicate_by_hash(docvec_items, recall_items)

    # Step 4: Write to Theo
    written = write_to_theo(all_items, edges, args.theo_path, dry_run=args.dry_run)

    # Summary
    log("")
    log("=" * 60)
    log("Migration Summary")
    log("=" * 60)
    log(f"DocVec documents read:    {len(docvec_data['ids'])}")
    log(f"Recall memories read:     {len(recall_data['memories'])}")
    log(f"Recall edges read:        {len(recall_data['edges'])}")
    log(f"Duplicates removed:       {dup_count}")
    log(f"Items written to Theo:    {written}")
    log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
