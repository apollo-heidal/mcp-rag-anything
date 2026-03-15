# rag-anywhere

An MCP server that gives agents a persistent RAG knowledge base. Point it at files or folders, then query them with graph-aware retrieval powered by [RAG-Anything](https://github.com/HKUDS/RAG-Anything) / [LightRAG](https://github.com/HKUDS/LightRAG).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Docker Model Runner enabled in Docker Desktop. See Docker's official setup docs: <https://docs.docker.com/ai/model-runner/get-started/>

## Quickstart

```bash
docker compose up --build
```

That's it. Docker Compose builds the container and starts the server. The declared Docker Model Runner models are provisioned by Docker when needed.

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
| `ingest` | Add files or folders to the knowledge base |
| `query` | Retrieve context using graph-aware RAG |

### `ingest`
```json
{ "paths": ["/absolute/path/to/file-or-folder"], "recursive": true }
```

### `query`
```json
{ "query": "what does X do?", "mode": "mix" }
```

Retrieval modes: `local` · `global` · `hybrid` · `naive` · `mix` (default)

## Architecture

```
Claude / MCP Client
      │ streamable-http (or stdio)
      ▼
  server.py  (Python + FastMCP)
      │ uses
      ▼
  RAGAnything / LightRAG  (Python libs)
      │ OpenAI-compat HTTP
      ▼
  Docker Model Runner
      │
      ▼
  LightRAG storage (Docker volume)
```

`server.py` is a single Python process that serves as both the MCP server and a web UI for uploading documents. Embeddings are computed locally via SentenceTransformers. Config is loaded from environment variables, with `.env` as a fallback for local (non-Docker) runs.

Useful runtime commands:

```bash
docker compose up --build
docker compose logs -f
docker compose down
```

## Models

| Role | Model | Size | License |
|------|-------|------|---------|
| LLM synthesis | `hf.co/unsloth/Qwen3.5-2B-GGUF` | ~1.4 GB | Apache 2.0 |
| Embeddings (remote) | `hf.co/unsloth/Qwen3-Embedding-0.6B` | ~400 MB | Apache 2.0 |
| Embeddings (local) | `all-MiniLM-L6-v2` via SentenceTransformers | ~80 MB | Apache 2.0 |

Local embeddings (`all-MiniLM-L6-v2`) are used by default in `server.py`. Any OpenAI-compatible endpoint works — set `LLM_API_BASE` in `.env` to point elsewhere.

## Configuration

Copy `.env.example` to `.env` and edit for local (non-Docker) runs. Docker Compose injects its own values and takes priority.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | `http://localhost:12434/engines/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `hf.co/unsloth/Qwen3.5-2B-GGUF` | LLM for RAG synthesis |
| `EMBEDDING_MODEL` | `hf.co/unsloth/Qwen3-Embedding-0.6B` | Remote embedding model (unused when local embeddings are active) |
| `ST_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local SentenceTransformers embedding model |
| `EMBEDDING_DIM` | unset | Embedding dimension override; when unset, `server.py` auto-detects it from `ST_MODEL` |
| `OPENAI_API_KEY` | `docker-model-runner` | Required by client lib; value ignored by Model Runner |
| `RAG_WORKING_DIR` | `~/.rag_storage` | Where LightRAG stores its graph and vectors |
