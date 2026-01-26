#!/usr/bin/env python3
"""Migrate high-value memories from old recall.db to theo."""

import sqlite3
import json
from pathlib import Path

RECALL_DB = Path.home() / ".recall" / "recall.db"
OUTPUT_FILE = Path(__file__).parent / "memories_to_migrate.json"

def extract_memories():
    """Extract high-value memories from recall.db."""
    conn = sqlite3.connect(RECALL_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get high-value memories (patterns, decisions, preferences with importance >= 0.7)
    cursor.execute("""
        SELECT id, content, type, namespace, importance, confidence, created_at
        FROM memories
        WHERE type IN ('pattern', 'decision', 'preference', 'golden_rule')
        AND importance >= 0.7
        ORDER BY importance DESC, created_at DESC
    """)

    memories = []
    for row in cursor.fetchall():
        memories.append({
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "namespace": row["namespace"],
            "importance": row["importance"],
            "confidence": row["confidence"],
        })

    conn.close()
    return memories


def main():
    print(f"Reading from: {RECALL_DB}")
    memories = extract_memories()
    print(f"Found {len(memories)} high-value memories")

    # Group by namespace
    by_namespace = {}
    for m in memories:
        ns = m["namespace"]
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(m)

    print("\nBy namespace:")
    for ns, mems in sorted(by_namespace.items(), key=lambda x: -len(x[1])):
        print(f"  {ns}: {len(mems)}")

    # Save to JSON for migration
    with open(OUTPUT_FILE, "w") as f:
        json.dump(memories, f, indent=2)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Print sample
    print("\n=== SAMPLE (first 5) ===")
    for m in memories[:5]:
        preview = m["content"][:200].replace("\n", " ")
        print(f"[{m['namespace']}/{m['type']}] {preview}...")


if __name__ == "__main__":
    main()
