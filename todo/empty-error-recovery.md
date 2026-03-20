# Empty error string not caught by recovery (0a1fa3ed)

## Problem
Document `0a1fa3ed` (Interconnected Episode 1) has status `"error"` in files.json but the error string is empty (`""`). The startup recovery function `_recover_false_done_documents()` checks for keyword matches in the error string (`"embedding"`, `"lightrag"`, `"500"`, etc.) — an empty string matches none of them, so this file is silently skipped.

## Root cause
The original whisper transcription failed (`whisper_full_with_state: failed to encode` on 76-min audio) but the error message wasn't captured properly, leaving an empty string.

## Fix
In `_recover_false_done_documents()`, add a case for empty/missing error strings on error-status files — treat them as potentially recoverable (or at least log them for manual review). Something like:

```python
if not err or err.strip() == "":
    log.warning("Recovery: %s has empty error — flagging for manual review", record["id"])
    continue  # or attempt recovery
```

Now that long-audio chunking is implemented (Change 4), retrying this file should succeed.
