# rag-anywhere

An MCP server that gives agents a persistent RAG knowledge base. Point it at files or folders, then query them through a unified interface that routes documents to [RAG-Anything](https://github.com/HKUDS/RAG-Anything) / [LightRAG](https://github.com/HKUDS/LightRAG), routes audio through local transcription into the document graph, and routes video through an in-process `VideoEngineAdapter` backed by vendored VideoRAG code.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Docker Model Runner enabled in Docker Desktop. See Docker's official setup docs: <https://docs.docker.com/ai/model-runner/get-started/>

## Quickstart

Required prestart step:

```bash
mkdir -p model_assets
curl -L --fail -o model_assets/mmproj-F16.gguf \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf
docker model package \
  --gguf "$HOME/.docker/models/bundles/sha256/bf0404c6cadfc3a56220bfcd6daed69533348212e68471675238f2a6b706de07/model/model.gguf" \
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
| `ingest` | Add documents, audio, video, files or folders to the knowledge base |
| `query` | Retrieve context using routed / federated RAG |

### `ingest`
```json
{ "paths": ["/absolute/path/to/file-or-folder"], "recursive": true }
```

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

Retrieval modes: `local` · `global` · `hybrid` · `naive` · `mix` (default)

Targets: `auto` (default) · `document` · `video` · `all`

## Architecture

```
Claude / MCP Client
      │ streamable-http (or stdio)
      ▼
  server.py  (Python + FastMCP)
      │ routes by modality
      ├──────────────► RAGAnything / LightRAG  (documents + transcripts)
      │
      └──────────────► VideoEngineAdapter  (in-process native video indexing)
                              │
                              ▼
                    vendored VideoRAG + local vision / ASR models
```

`server.py` is a single Python process that serves as both the MCP server and a web UI for uploading files. It classifies uploads into document, audio, or video, records routing metadata in the manifest, and federates a single query across the document and video engines. The `VideoEngineAdapter` runs in the same process as the rest of the app, with per-video workspaces under `RAG_WORKING_DIR`. Embeddings are computed locally via SentenceTransformers. Config is loaded from environment variables, with `.env` as a fallback for local (non-Docker) runs.

## Media routing

- Documents are ingested directly into `RAGAnything`.
- Document image metadata is extracted through a dedicated vision model path. If image metadata extraction fails and `VISION_REQUIRED=true`, document ingest fails instead of silently degrading.
- Audio files are transcribed with a local ASR model, converted into timestamped transcript artifacts, then ingested into the document graph.
- Video files are indexed by the in-process `VideoEngineAdapter`, which uses vendored VideoRAG code plus the local vision model to caption sampled frames and local ASR to transcribe segment audio.
- Native video ingest probes source audio with `ffprobe` and extracts segment audio with `ffmpeg` before ASR. If a lecture-style video has an audio stream but segment extraction or transcription still fails, ingest ends as `error` instead of degrading silently.
- Video ingest fails closed if native video indexing cannot complete. There is no transcript-only success path for video files.

The web UI at `/ui` shows the detected modality, the selected ingest path, and the engine used for each file.

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
| `RAG_WORKING_DIR` | `~/.rag_storage` | Where LightRAG stores its graph and vectors |
| `DEFAULT_COLLECTION_ID` | `default` | Collection assigned to uploads when none is provided |
| `VIDEO_ENGINE_ENABLED` | `true` | Enables the in-process `VideoEngineAdapter` |
| `VIDEO_SEGMENT_LENGTH` | `30` | Segment length in seconds for video splitting and indexing |
| `AUDIO_TRANSCRIBE_MODEL` | `openai/whisper-small` | Local ASR model used for audio uploads and video segment transcription |
