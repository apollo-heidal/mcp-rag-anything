# RAG-Anywhere

RAG-Anywhere is a distributed runtime for making any digital file queryable from any LLM system in minutes. It treats RAG as infrastructure rather than an app-specific feature: a user, agent, desktop tool, web app, or remote peer connects to a trusted RAG endpoint, uploads or routes content through signed flows, and queries a unified knowledge plane through MCP.

In its finished form, RAG-Anywhere acts as the control plane, auth plane, routing plane, and execution plane for multimodal retrieval. It ingests documents, images, audio, video, and future file types through modality-specific engines; exposes MCP as the canonical agent interface; keeps HTTP for browser UX, uploads, and operational endpoints; and federates results across local and remote queryable peers without requiring raw source material to leave the node that owns it.

The project is local-first in deployment shape and distributed-first in architecture. A single node works as a complete personal or team RAG runtime. Multiple nodes form a semi-trusted network of queryable peers, where access is explicit, transport is authenticated, and shared value comes from the output of retrieval systems rather than unrestricted replication of private source corpora.

The long-term model is:

- RAG any file: every important digital artifact becomes queryable through the right modality engine.
- RAG anywhere: any trusted agent or application can attach to a RAG-Anywhere runtime through MCP and signed upload or UI flows.
- RAG all at once: many peers can participate in one routed retrieval fabric, with provenance, auth, and policy preserved across the network.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Docker Model Runner enabled in Docker Desktop. See Docker's official setup docs: <https://docs.docker.com/ai/model-runner/get-started/>

## Quickstart

Required prestart step:

```bash
mkdir -p model_assets
curl -L --fail -o model_assets/mmproj-F16.gguf \
  https://hf.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf
docker model package \
  --from hf.co/unsloth/Qwen3.5-2B-GGUF \
  --mmproj "$PWD/model_assets/mmproj-F16.gguf" \
  docker.io/local/qwen3.5-2b-vlm:latest
```

Then start the app:

```bash
docker compose up --build
```

Package the local vision model first, then start the stack. Docker Compose builds the container and starts the server. The declared Docker Model Runner models are provisioned by Docker when needed.

## Claude Desktop config

```json
{
  "mcpServers": {
    "rag-anywhere": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Start the server first with `docker compose up --build`.

## Tools

| Tool | Description |
|------|-------------|
| `query` | Retrieve context through the routed knowledge plane |
| `create_ui_session_link` | Create a short-lived signed browser link to the library UI |
| `create_upload_link` | Create a one-time upload URL for agent-driven file upload over HTTP |

### `query`
```json
{
  "query": "what does X do?",
  "mode": "mix",
  "target": "auto",
  "collection_ids": ["default"],
  "top_k": 8
}
```

### `create_ui_session_link`
```json
{ "requested_for": "alice", "ttl_seconds": 900 }
```

### `create_upload_link`
```json
{
  "requested_for": "alice",
  "filename": "/absolute/path/to/file.pdf",
  "collection_id": "default",
  "ttl_seconds": 900
}
```

Retrieval modes: `local` · `global` · `hybrid` · `naive` · `mix` (default)

Targets: `auto` (default) · `document` · `video` · `all`

## Architecture

```
Agent / App / User / Trusted Peer
              │
              │ MCP for query and session bootstrap
              │ HTTP for browser UX, uploads, health, and operations
              ▼
      RAG-Anywhere Runtime
              │
              ├──────────────► Control plane
              │                 identity, trust, session grants, upload grants
              │
              ├──────────────► Routing plane
              │                 modality routing, query federation, provenance
              │
              ├──────────────► Local engines
              │                 documents, images, audio, video, future modalities
              │
              └──────────────► Queryable peers
                                trusted remote runtimes that share retrieval output
```

RAG-Anywhere presents one agent-facing RAG surface: MCP. Agents query the runtime, request signed UI links, and request one-time upload links. HTTP remains as an operational and user-facing companion surface, not as a second competing RAG API. Users open a signed library UI, upload content, inspect status, and manage a local knowledge node. Over time, the same runtime shape extends to trusted peer federation so that a node can route queries across local engines and approved remote peers while preserving access policy and provenance.

The current codebase runs as a single Python process with local modality engines and a local manifest, but the architectural identity is broader: this process is the single-node form of a queryable peer runtime. It owns ingestion, indexing, signed access grants, query routing, and result federation. Local storage remains peer-local by default. Remote interaction is modeled as controlled query access, not raw filesystem sharing.

## Media routing

RAG-Anywhere treats file support as a runtime capability, not a special case:

- Documents route into a document graph and multimodal parsing engine.
- Images contribute metadata and semantics through a dedicated vision path.
- Audio routes through local speech recognition and becomes timestamped knowledge.
- Video routes through a native video engine that fuses visual descriptions, segment audio, timestamps, and graph extraction.
- Future file classes fit into the same model: attach a modality engine, expose it through the routed runtime, and preserve provenance in the result set.

The runtime fails closed when a required modality path cannot produce the metadata needed to make a file meaningfully queryable. The goal is not to accept any file extension and pretend success; the goal is to make any important digital artifact genuinely retrievable.

The web UI at `/ui` is accessed through a signed session link and shows the detected modality, the selected ingest path, and the engine used for each file.

## Agent upload flow

Agents do not pass local filesystem paths into MCP and hope the server can see them. Instead:

- MCP remains the control surface for query and session bootstrap.
- `create_ui_session_link` returns a short-lived browser session for human-in-the-loop library access.
- `create_upload_link` returns a one-time HTTP upload URL that any shell-capable agent can use to push bytes to the runtime.
- The same model generalizes to remote peers: MCP establishes intent and trust, signed HTTP flows move user-facing sessions and upload payloads, and the runtime performs ingestion locally where the data is meant to live.

This design makes the runtime easy to attach to desktop agents, web agents, local shells, and future remote peers without coupling RAG to a specific client filesystem or UI model.

## Distributed Model

RAG-Anywhere is built around the idea of a queryable peer:

- A peer owns its local files, indexes, policies, and modality engines.
- A peer exposes retrieval output through trusted interfaces rather than blindly exporting source material.
- A peer can participate in a semi-trusted network where access is explicit and federated query is policy-aware.
- A peer can serve one user, one team, or a wider subnet of connected runtimes.

In the long run, the runtime is compatible with a private mesh or VPN trust layer, such as a Headscale-style WireGuard control plane, while still issuing application-layer session grants for browser and upload flows. Network trust and application trust complement each other: a device can belong to the network, but individual sessions, uploads, and agent actions still receive scoped, short-lived authorization.

Useful runtime commands:

```bash
docker compose up --build
docker compose logs -f
docker compose down
```

`ffmpeg` and `ffprobe` are required inside the app container for native video ingest. `/api/health` reports whether both tools are available.

## Models

| Role | Model | Size | License |
|------|-------|------|---------|
| LLM synthesis | `hf.co/unsloth/Qwen3.5-2B-GGUF` | ~1.4 GB | Apache 2.0 |
| Vision metadata | `docker.io/local/qwen3.5-2b-vlm:latest` | depends on packaged GGUF + `mmproj` | depends on packaged source |
| Embeddings (remote) | `hf.co/unsloth/Qwen3-Embedding-0.6B` | ~400 MB | Apache 2.0 |
| Embeddings (local) | `all-MiniLM-L6-v2` via SentenceTransformers | ~80 MB | Apache 2.0 |

Local embeddings (`all-MiniLM-L6-v2`) are used by default in `server.py`. Any OpenAI-compatible endpoint works for text or vision if it accepts standard multimodal `chat/completions` requests.

### Local vision packaging

Use a multimodal artifact that includes both the Qwen GGUF weights and the matching `mmproj` projector file. A text-only GGUF will not accept image input, even if the model family is multimodal.

Example:

```bash
docker model package \
  --gguf /absolute/path/to/model.gguf \
  --mmproj /absolute/path/to/mmproj.gguf \
  docker.io/local/qwen3.5-2b-vlm:latest
docker model list --openai
```

The packaged model reference must match `VISION_MODEL`.

## Configuration

Copy `.env.example` to `.env` and edit for local (non-Docker) runs. Docker Compose injects its own values and takes priority.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | `http://localhost:12434/engines/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `hf.co/unsloth/Qwen3.5-2B-GGUF` | LLM for RAG synthesis |
| `VISION_API_BASE` | `http://localhost:12434/engines/v1` | OpenAI-compatible multimodal API base URL for document images |
| `VISION_MODEL` | `docker.io/local/qwen3.5-2b-vlm:latest` | Vision model used for image metadata extraction |
| `VISION_API_KEY` | `docker-model-runner` | Optional override for the vision client auth header |
| `VISION_REQUIRED` | `true` | Fail document ingest if required image metadata extraction does not succeed |
| `EMBEDDING_MODEL` | `hf.co/unsloth/Qwen3-Embedding-0.6B` | Remote embedding model (unused when local embeddings are active) |
| `ST_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local SentenceTransformers embedding model |
| `EMBEDDING_DIM` | unset | Embedding dimension override; when unset, `server.py` auto-detects it from `ST_MODEL` |
| `OPENAI_API_KEY` | `docker-model-runner` | Required by client lib; value ignored by Model Runner |
| `APP_BASE_URL` | `http://localhost:8000` | External base URL used when generating signed UI and upload links |
| `RAG_SESSION_SECRET` | `change-me-for-production` | HMAC secret used to sign short-lived UI and upload grants |
| `UI_SESSION_TTL_SECONDS` | `900` | Default TTL for signed browser UI session links |
| `UPLOAD_LINK_TTL_SECONDS` | `900` | Default TTL for one-time upload links |
| `RAG_WORKING_DIR` | `~/.rag_storage` | Where LightRAG stores its graph and vectors |
| `DEFAULT_COLLECTION_ID` | `default` | Collection assigned to uploads when none is provided |
| `VIDEO_ENGINE_ENABLED` | `true` | Enables the in-process `VideoEngineAdapter` |
| `VIDEO_SEGMENT_LENGTH` | `30` | Segment length in seconds for video splitting and indexing |
| `AUDIO_TRANSCRIBE_MODEL` | `openai/whisper-small` | Local ASR model used for audio uploads and video segment transcription |
