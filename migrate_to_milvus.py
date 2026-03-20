#!/usr/bin/env python3
"""
Reset the files.json manifest so that documents get re-ingested through Milvus
on next server startup. Does NOT call the server or embed anything — it simply
resets manifest state. The server's _recover_false_done_documents() and startup
logic handle the actual re-ingestion.

Usage:
    python migrate_to_milvus.py [--rag-dir ./rag_storage] [--include-errors]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reset manifest for Milvus migration")
    parser.add_argument(
        "--rag-dir",
        default="./rag_storage",
        help="Path to RAG working directory (default: ./rag_storage)",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Also reset documents in 'error' state for re-ingestion",
    )
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    files_json = rag_dir / "files.json"
    if not files_json.exists():
        print(f"ERROR: {files_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(files_json) as f:
        records = json.load(f)

    reset_done = []
    reset_error = []
    skipped = []

    for record in records:
        rid = record["id"]
        status = record.get("status", "")
        modality = record.get("modality", "")
        name = record.get("original_name", record.get("name", rid))

        # Skip archive parents — children are handled individually
        if modality == "archive":
            skipped.append((rid, name, "archive parent"))
            continue

        if status == "done" and record.get("queryable"):
            record["status"] = "pending"
            record["queryable"] = False
            record["engine_doc_id"] = None
            record["ingested_at"] = None
            record["error"] = None
            reset_done.append((rid, name))
        elif status == "error" and args.include_errors:
            record["status"] = "pending"
            record["queryable"] = False
            record["engine_doc_id"] = None
            record["ingested_at"] = None
            record["error"] = None
            reset_error.append((rid, name))
        elif status == "error":
            skipped.append((rid, name, f"error (use --include-errors)"))

    # Also clear LightRAG's kv_store_doc_status so it doesn't skip re-ingestion
    kv_doc_status = rag_dir / "kv_store_doc_status.json"
    kv_cleared = False
    if kv_doc_status.exists():
        with open(kv_doc_status) as f:
            kv_data = json.load(f)
        if kv_data:
            # Back up before clearing
            backup = kv_doc_status.with_suffix(".json.bak")
            with open(backup, "w") as f:
                json.dump(kv_data, f, indent=2)
            # Clear all entries so LightRAG re-processes everything
            with open(kv_doc_status, "w") as f:
                json.dump({}, f)
            kv_cleared = True

    # Write updated manifest
    with open(files_json, "w") as f:
        json.dump(records, f, indent=2)

    # Print summary
    print(f"=== Milvus Migration: Manifest Reset ===\n")
    if reset_done:
        print(f"Reset {len(reset_done)} done documents to pending:")
        for rid, name in reset_done:
            print(f"  [{rid}] {name}")
    if reset_error:
        print(f"\nReset {len(reset_error)} error documents to pending:")
        for rid, name in reset_error:
            print(f"  [{rid}] {name}")
    if skipped:
        print(f"\nSkipped {len(skipped)} documents:")
        for rid, name, reason in skipped:
            print(f"  [{rid}] {name} — {reason}")
    if kv_cleared:
        print(f"\nCleared kv_store_doc_status.json (backup: kv_store_doc_status.json.bak)")
    print(f"\nTotal reset: {len(reset_done) + len(reset_error)} documents")
    print("Run 'docker compose up -d' to start re-ingestion through Milvus.")


if __name__ == "__main__":
    main()
