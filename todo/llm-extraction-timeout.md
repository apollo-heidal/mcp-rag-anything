# LLM extraction timeout (c9cf3f7c)

## Problem
Document `c9cf3f7c` fails with: `LLM func: Worker execution timeout after 360s` on chunk `7f9b4412b261b102d89ff892b50183d1`. The qwen3.5-2b-vlm model takes >360s on complex/large chunks.

## Root cause
LightRAG's internal worker timeout is 360s for LLM extraction. This is hardcoded in lightrag.py and not configurable via API. The chunk in question is likely too large or complex for the small 2B model.

## Fix options
1. **Increase LightRAG timeout** — patch `lightrag.py` to raise the 360s limit (risks blocking the pipeline longer)
2. **Use a faster/larger model** — switch to a model that handles extraction faster (e.g. qwen-7b or an API-based model)
3. **Pre-split large chunks** — break oversized chunks before they reach LLM extraction so each stays under the timeout
4. **Skip and mark partial** — catch the timeout, mark the doc as partially ingested (missing that chunk), and continue
