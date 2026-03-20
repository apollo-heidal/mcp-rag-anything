---
name: LightRAG cascade retry behavior
description: LightRAG re-processes ALL failed/processing docs on every new insert — cascade prevention implemented in ingest_file()
type: project
---

LightRAG's insert pipeline resets ALL documents with FAILED/PROCESSING status to PENDING and re-processes them alongside any new document. This causes O(n²) error cascades and GPU contention.

**Why:** This is LightRAG's built-in consistency mechanism in `_prepare_docs_status()` (~line 1599 of lightrag.py). It can't be disabled via configuration.

**How to apply:** `DocumentEngineAdapter.ingest_file()` in server.py implements cascade prevention: before calling `process_document_complete()`, it temporarily marks all FAILED/PROCESSING docs as PROCESSED using `doc_status.get_docs_by_status()` + `upsert()`, then restores them in a `finally` block. This prevents the cascade from picking up unrelated failed docs. Also, LightRAG has a 60s worker timeout on embedding functions and 360s on LLM extraction — retry waits must stay under these limits.
