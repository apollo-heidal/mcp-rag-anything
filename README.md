# rag-anywhere

An MCP server that gives Claude a persistent RAG knowledge base. Point it at files or folders, then query them with graph-aware retrieval powered by [RAG-Anything](https://github.com/HKUDS/RAG-Anything) / [LightRAG](https://github.com/HKUDS/LightRAG).

## Quickstart

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **Model Runner** enabled (Settings → Features in development).

```bash
./rag.sh start
```

That's it. This pulls the models, builds the container, and starts the server.

## Claude Desktop config

```json
{
  "mcpServers": {
    "rag-anywhere": {
      "command": "docker",
      "args": ["compose", "-f", "/absolute/path/to/mcp-rag-anywhere/docker-compose.yml", "run", "--rm", "mcp-rag"]
    }
  }
}
```

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
Claude (MCP client)
      │ stdio JSON-RPC
      ▼
  server.ts  (Bun)
      │ spawn
      ▼
  rag_bridge.py  (Python + RAG-Anything)
      │ HTTP OpenAI-compat API
      ▼
  Docker Model Runner (host)
  ai/qwen3.5:2b-instruct    ← LLM synthesis
  ai/qwen3-embedding:0.6b   ← embeddings
      │
      ▼
  LightRAG storage (Docker volume)
```

`server.ts` is a thin MCP adapter. All RAG logic lives in `rag_bridge.py`, which reads a single JSON line from stdin and writes a JSON result to stdout. The Python bridge loads config from environment variables, with `.env` as a fallback for local (non-Docker) runs.

## Models

| Role | Model | Size | License |
|------|-------|------|---------|
| LLM synthesis | `ai/qwen3.5:2b-instruct-Q4_K_M` | ~1.4 GB | Apache 2.0 |
| Embeddings | `ai/qwen3-embedding:0.6b-Q4_K_M` | ~400 MB | Apache 2.0 |

Any OpenAI-compatible endpoint works — set `LLM_API_BASE` in `.env` to point elsewhere.

## Configuration

Copy `.env.example` to `.env` and edit for local (non-Docker) runs. Docker Compose injects its own values and takes priority.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | `http://localhost:12434/engines/llama.cpp/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `ai/qwen3.5:2b-instruct-Q4_K_M` | LLM for RAG synthesis |
| `EMBEDDING_MODEL` | `ai/qwen3-embedding:0.6b-Q4_K_M` | Embedding model |
| `OPENAI_API_KEY` | `docker-model-runner` | Required by client lib; value ignored by Model Runner |
| `RAG_WORKING_DIR` | `~/.rag_storage` | Where LightRAG stores its graph and vectors |
