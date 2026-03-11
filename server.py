"""
RAG-Anywhere server — single Python process, RAGAnything initialized once.
"""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import aiofiles
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

load_dotenv(Path(__file__).parent / ".env", override=False)

_working_dir = os.environ.get("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
_uploads_dir = Path(_working_dir) / "uploads"
_uploads_dir.mkdir(parents=True, exist_ok=True)
_ingest_tasks: dict[str, asyncio.Task] = {}
_mineru_backend = os.environ.get("MINERU_BACKEND", "hybrid-http-client")


@asynccontextmanager
async def _lifespan(app):
    # Reset any files left stuck as "ingesting" / "uploading" from a previous run.
    records = await _manifest.load()
    for r in records:
        if r["status"] in ("ingesting", "uploading"):
            await _manifest.update_status(
                r["id"], status="error", error="Server restarted during ingestion"
            )
    yield


mcp = FastMCP("rag-anywhere", host="0.0.0.0", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# Singleton RAGAnything instance
# ---------------------------------------------------------------------------
_rag = None
_rag_lock = asyncio.Lock()


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


async def _get_rag():
    global _rag
    if _rag is not None:
        return _rag
    async with _rag_lock:
        if _rag is not None:
            return _rag

        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig
        from sentence_transformers import SentenceTransformer

        working_dir = _get_env("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
        output_dir = _get_env("RAG_OUTPUT_DIR", str(Path(working_dir) / "output"))
        api_base = _get_env("LLM_API_BASE", "https://api.openai.com/v1")
        llm_model = _get_env("LLM_MODEL", "gpt-4o-mini")
        embedding_model = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = _get_env("OPENAI_API_KEY", "")

        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        parse_method = _get_env("PARSE_METHOD", "txt")
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser_output_dir=output_dir,
            parse_method=parse_method,
        )

        async def llm_func(prompt, system_prompt=None, **kwargs):
            return await openai_complete_if_cache(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                base_url=api_base,
                **kwargs,
            )

        _st_model_name = _get_env("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _st_model = await asyncio.to_thread(SentenceTransformer, _st_model_name)

        async def _embed(texts: list[str]):
            return await asyncio.to_thread(
                lambda: _st_model.encode(texts, show_progress_bar=False)
            )

        embedding_dim = int(
            _get_env("EMBEDDING_DIM", str(_st_model.get_sentence_embedding_dimension()))
        )
        embed_func = EmbeddingFunc(
            embedding_dim=embedding_dim, max_token_size=8192, func=_embed
        )

        _rag = RAGAnything(
            config=config, llm_model_func=llm_func, embedding_func=embed_func
        )
        return _rag


# ---------------------------------------------------------------------------
# File manifest
# ---------------------------------------------------------------------------


class FileManifest:
    def __init__(self, working_dir: str):
        self._path = Path(working_dir) / "files.json"
        self._lock = asyncio.Lock()

    async def load(self) -> list[dict]:
        async with self._lock:
            if not self._path.exists():
                return []
            async with aiofiles.open(self._path) as f:
                return json.loads(await f.read())

    async def _save(self, records: list[dict]) -> None:
        async with aiofiles.open(self._path, "w") as f:
            await f.write(json.dumps(records, indent=2))

    async def add(self, record: dict) -> None:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            records.append(record)
            await self._save(records)

    async def update_status(self, file_id: str, **fields) -> None:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            for r in records:
                if r["id"] == file_id:
                    r.update(fields)
            await self._save(records)

    async def get(self, file_id: str) -> dict | None:
        for r in await self.load():
            if r["id"] == file_id:
                return r
        return None

    async def remove(self, file_id: str) -> bool:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            new = [r for r in records if r["id"] != file_id]
            if len(new) == len(records):
                return False
            await self._save(new)
            return True


_manifest = FileManifest(_working_dir)


# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------


async def _ingest_background(file_id: str, file_path: Path) -> None:
    log = logging.getLogger("ingest")
    log.info("Starting ingestion: %s (%s)", file_path.name, file_id)
    await _manifest.update_status(file_id, status="ingesting")
    try:
        rag = await _get_rag()
        await rag.process_document_complete(
            file_path=str(file_path), backend=_mineru_backend
        )
        log.info("Ingestion done: %s (%s)", file_path.name, file_id)
        await _manifest.update_status(
            file_id,
            status="done",
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        log.exception("Ingestion failed: %s (%s)", file_path.name, file_id)
        await _manifest.update_status(file_id, status="error", error=str(e))
    finally:
        _ingest_tasks.pop(file_id, None)


# ---------------------------------------------------------------------------
# UI HTML
# ---------------------------------------------------------------------------

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG-Anywhere — Document Manager</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2e3147;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }
  .drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 3rem 2rem; text-align: center; cursor: pointer;
    transition: border-color 0.2s, background 0.2s; margin-bottom: 2rem;
  }
  .drop-zone:hover, .drop-zone.drag-over { border-color: var(--accent); background: rgba(99,102,241,0.06); }
  .drop-zone p { color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; }
  .drop-btn {
    display: inline-block; margin-top: 1rem; padding: 0.5rem 1.25rem;
    background: var(--accent); color: #fff; border: none; border-radius: var(--radius);
    cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: background 0.2s;
  }
  .drop-btn:hover { background: var(--accent-hover); }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 0.75rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.05em; padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border); }
  td { padding: 0.875rem 1rem; font-size: 0.875rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface); }
  .badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.2rem 0.6rem; border-radius: 99px; font-size: 0.75rem; font-weight: 500;
  }
  .badge-done { background: rgba(34,197,94,0.15); color: var(--success); }
  .badge-ingesting, .badge-uploading { background: rgba(245,158,11,0.15); color: var(--warning); }
  .badge-error { background: rgba(239,68,68,0.15); color: var(--error); }
  .spinner { width: 10px; height: 10px; border: 2px solid currentColor;
    border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .del-btn { background: none; border: none; color: var(--muted); cursor: pointer;
    font-size: 1rem; padding: 0.25rem 0.5rem; border-radius: 4px; transition: color 0.2s; }
  .del-btn:hover { color: var(--error); }
  .empty { text-align: center; color: var(--muted); padding: 3rem; font-size: 0.875rem; }
  .table-wrap { background: var(--surface); border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden; }
</style>
</head>
<body>
<h1>RAG-Anywhere</h1>
<p class="subtitle">Upload documents to ingest them into the knowledge graph.</p>

<div class="drop-zone" id="dropZone">
  <div>&#128196; Drop files here</div>
  <p>PDF, DOCX, XLSX, PPTX, images, and more</p>
  <label class="drop-btn">Browse files<input type="file" id="fileInput" multiple hidden></label>
</div>

<div class="table-wrap">
  <table>
    <thead><tr>
      <th>Name</th><th>Size</th><th>Uploaded</th><th>Status</th><th></th>
    </tr></thead>
    <tbody id="tbody"><tr><td colspan="5" class="empty">No files yet.</td></tr></tbody>
  </table>
</div>

<script>
const tbody = document.getElementById('tbody');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const polls = {};

function fmt_size(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}
function fmt_date(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}
function badge(status) {
  const spin = `<div class="spinner"></div>`;
  if (status === 'done') return `<span class="badge badge-done">&#10003; Done</span>`;
  if (status === 'error') return `<span class="badge badge-error">&#10007; Error</span>`;
  if (status === 'ingesting') return `<span class="badge badge-ingesting">${spin} Ingesting</span>`;
  return `<span class="badge badge-uploading">${spin} Uploading</span>`;
}

function render(files) {
  if (!files.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No files yet.</td></tr>';
    return;
  }
  tbody.innerHTML = files.map(f => `
    <tr id="row-${f.id}">
      <td title="${f.original_name}">${f.original_name}</td>
      <td>${fmt_size(f.size_bytes)}</td>
      <td>${fmt_date(f.uploaded_at)}</td>
      <td id="status-${f.id}">${badge(f.status)}</td>
      <td><button class="del-btn" onclick="del('${f.id}')" title="Remove">&#128465;</button></td>
    </tr>`).join('');
}

async function load() {
  const res = await fetch('/api/files');
  const files = await res.json();
  render(files);
  files.forEach(f => { if (f.status === 'uploading' || f.status === 'ingesting') startPoll(f.id); });
}

function startPoll(id) {
  if (polls[id]) return;
  polls[id] = setInterval(async () => {
    const res = await fetch(`/api/files/${id}/status`);
    if (!res.ok) { clearInterval(polls[id]); delete polls[id]; return; }
    const data = await res.json();
    const el = document.getElementById(`status-${id}`);
    if (el) el.innerHTML = badge(data.status);
    if (data.status === 'done' || data.status === 'error') {
      clearInterval(polls[id]); delete polls[id];
    }
  }, 2000);
}

async function upload(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/upload', { method: 'POST', body: fd });
  const data = await res.json();
  await load();
  startPoll(data.id);
}

async function del(id) {
  await fetch(`/api/files/${id}`, { method: 'DELETE' });
  clearInterval(polls[id]); delete polls[id];
  await load();
}

fileInput.addEventListener('change', () => { [...fileInput.files].forEach(upload); fileInput.value=''; });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  [...e.dataTransfer.files].forEach(upload);
});

load();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@mcp.custom_route("/ui", methods=["GET"])
async def ui_index(request: Request) -> Response:
    return HTMLResponse(_UI_HTML)


@mcp.custom_route("/api/files", methods=["GET"])
async def api_list_files(request: Request) -> Response:
    return JSONResponse(await _manifest.load())


@mcp.custom_route("/api/upload", methods=["POST"])
async def api_upload(request: Request) -> Response:
    form = await request.form()
    upload = form["file"]
    file_id = uuid4().hex[:8]
    safe_name = f"{file_id}_{re.sub(r'[^a-zA-Z0-9._-]', '_', upload.filename)}"
    dest = _uploads_dir / safe_name
    async with aiofiles.open(dest, "wb") as f:
        await f.write(await upload.read())
    record = {
        "id": file_id,
        "name": safe_name,
        "original_name": upload.filename,
        "size_bytes": dest.stat().st_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "uploading",
        "ingested_at": None,
        "error": None,
    }
    await _manifest.add(record)
    _ingest_tasks[file_id] = asyncio.create_task(_ingest_background(file_id, dest))
    return JSONResponse({"id": file_id, "status": "uploading"}, status_code=202)


@mcp.custom_route("/api/files/{file_id}/status", methods=["GET"])
async def api_file_status(request: Request) -> Response:
    record = await _manifest.get(request.path_params["file_id"])
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(
        {k: record[k] for k in ("id", "status", "error", "ingested_at")}
    )


@mcp.custom_route("/api/files/{file_id}", methods=["DELETE"])
async def api_delete_file(request: Request) -> Response:
    removed = await _manifest.remove(request.path_params["file_id"])
    return JSONResponse({"ok": removed}, status_code=200 if removed else 404)


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health(request: Request) -> Response:
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def ingest(paths: list[str], recursive: bool = True) -> dict:
    """Ingest documents or folders into the RAG knowledge graph."""
    rag = await _get_rag()
    added = []
    errors = []

    for p in paths:
        path = Path(p)
        try:
            if path.is_dir():
                await rag.process_folder_complete(
                    folder_path=str(path),
                    recursive=recursive,
                    backend=_mineru_backend,
                )
                added.append(str(path))
            elif path.is_file():
                await rag.process_document_complete(
                    file_path=str(path), backend=_mineru_backend
                )
                added.append(str(path))
            else:
                errors.append(f"Path not found: {p}")
        except Exception as e:
            errors.append(f"{p}: {e}")

    return {"added": added, "errors": errors}


@mcp.tool()
async def query(query: str, mode: str = "mix") -> dict:
    """Query the RAG knowledge graph."""
    rag = await _get_rag()
    result = await rag.query(query, mode=mode)
    return {"result": result}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
