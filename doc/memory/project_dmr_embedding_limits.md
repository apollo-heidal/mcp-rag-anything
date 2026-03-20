---
name: DMR embedding n_batch=512 limit
description: Docker Model Runner has a physical batch size of 512 tokens for embeddings — texts must be truncated to ~1200 chars to stay under this limit
type: project
---

DMR's llama.cpp backend has a physical batch size of 512 tokens. Any single embedding input exceeding ~512 tokens causes a 500 error with message: "input (N tokens) is too large to process. increase the physical batch size (current batch size: 512)".

**Why:** DMR uses llama.cpp which has a fixed n_batch=512 parameter that cannot be changed via the API. The mxbai-embed-large model tokenizes at roughly 2.3-2.7 chars/token depending on content.

**How to apply:** `_EMBED_MAX_CHARS` in server.py is set to 1200 to stay safely under 512 tokens. If embedding 500s return, check the response body for the "input too large" message and reduce this limit further. Also, LLM and embedding models share the Metal GPU in DMR — concurrent requests cause 500s even when individual texts are within limits.
