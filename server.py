"""
RAG-Anywhere server with modality-aware routing and federated query support.
"""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import aiofiles
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

load_dotenv(Path(__file__).parent / ".env", override=False)

_DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".md",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rtf",
    ".tex",
    ".tif",
    ".tiff",
    ".tsv",
    ".txt",
    ".xls",
    ".xlsx",
}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

_working_dir = os.environ.get("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
_uploads_dir = Path(_working_dir) / "uploads"
_uploads_dir.mkdir(parents=True, exist_ok=True)
_ingest_tasks: dict[str, asyncio.Task] = {}
_mineru_backend = os.environ.get("MINERU_BACKEND", "hybrid-http-client")
_default_collection_id = os.environ.get("DEFAULT_COLLECTION_ID", "default")
_video_engine_enabled = os.environ.get("VIDEO_ENGINE_ENABLED", "true").lower() == "true"
_video_segment_length = int(os.environ.get("VIDEO_SEGMENT_LENGTH", "30"))

_rag = None
_rag_lock = asyncio.Lock()
_llm_func = None
_llm_lock = asyncio.Lock()
_vision_func = None
_vision_lock = asyncio.Lock()
_vision_probe_lock = asyncio.Lock()
_asr_pipeline = None
_asr_lock = asyncio.Lock()
_st_model = None
_st_lock = asyncio.Lock()
_vision_probe_state = {
    "checked_at": None,
    "ok": None,
    "error": None,
    "model": None,
    "response_preview": None,
}
_VISION_PROBE_TTL_SECONDS = 60
_VISION_PROBE_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X6Z0AAAAASUVORK5CYII="
)


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _DOCUMENT_EXTENSIONS:
        return "document"
    return "document"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_to_str(path: Path | None) -> str | None:
    return str(path) if path else None


def _media_tool_status() -> dict:
    return {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }


def _vision_required() -> bool:
    return _get_env("VISION_REQUIRED", "true").lower() == "true"


def _infer_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    raise ValueError("Unsupported image format for vision request")


def _coerce_image_data_url(image_data: str) -> str:
    if image_data.startswith("data:image/"):
        return image_data
    image_bytes = base64.b64decode(image_data, validate=True)
    mime_type = _infer_image_mime(image_bytes)
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(part for part in text_parts if part)
    return str(content or "")


async def _call_openai_chat(
    *,
    api_base: str,
    model: str,
    api_key: str,
    messages: list[dict],
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> str:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
    }
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Vision model returned no choices")
    message = choices[0].get("message") or {}
    content = _normalize_message_content(message.get("content"))
    if not content.strip():
        raise RuntimeError("Vision model returned empty content")
    return content


async def _build_llm_func():
    from lightrag.llm.openai import openai_complete_if_cache

    api_base = _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1")
    llm_model = _get_env("LLM_MODEL", "hf.co/unsloth/Qwen3.5-2B-GGUF")
    api_key = _get_env("OPENAI_API_KEY", "docker-model-runner")

    async def llm_func(prompt, system_prompt=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    return llm_func


async def _get_llm_func():
    global _llm_func
    if _llm_func is not None:
        return _llm_func
    async with _llm_lock:
        if _llm_func is None:
            _llm_func = await _build_llm_func()
    return _llm_func


async def _get_st_model():
    global _st_model
    if _st_model is not None:
        return _st_model
    async with _st_lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer

            st_model_name = _get_env("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            _st_model = await asyncio.to_thread(SentenceTransformer, st_model_name)
    return _st_model


async def _build_vision_func():
    api_base = _get_env(
        "VISION_API_BASE", _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1")
    )
    vision_model = _get_env("VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest")
    api_key = _get_env("VISION_API_KEY", _get_env("OPENAI_API_KEY", "docker-model-runner"))

    async def vision_func(prompt, image_data=None, system_prompt=None, **kwargs):
        if not image_data:
            raise ValueError("Vision requests require image_data")

        image_url = _coerce_image_data_url(image_data)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 1024
        try:
            content = await _call_openai_chat(
                api_base=api_base,
                model=vision_model,
                api_key=api_key,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                presence_penalty=kwargs.get("presence_penalty"),
                frequency_penalty=kwargs.get("frequency_penalty"),
            )
            _vision_probe_state.update(
                {
                    "checked_at": _iso_now(),
                    "ok": True,
                    "error": None,
                    "model": vision_model,
                    "response_preview": content[:200],
                }
            )
            return content
        except Exception as exc:
            _vision_probe_state.update(
                {
                    "checked_at": _iso_now(),
                    "ok": False,
                    "error": str(exc),
                    "model": vision_model,
                    "response_preview": None,
                }
            )
            raise

    return vision_func


async def _get_vision_func():
    global _vision_func
    if _vision_func is not None:
        return _vision_func
    async with _vision_lock:
        if _vision_func is None:
            _vision_func = await _build_vision_func()
    return _vision_func


async def _probe_vision_model(force: bool = False) -> dict:
    vision_model = _get_env("VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest")
    now = datetime.now(timezone.utc)
    checked_at = _vision_probe_state.get("checked_at")
    if (
        not force
        and checked_at
        and _vision_probe_state.get("model") == vision_model
        and (now - datetime.fromisoformat(checked_at)).total_seconds() < _VISION_PROBE_TTL_SECONDS
    ):
        return dict(_vision_probe_state)

    async with _vision_probe_lock:
        checked_at = _vision_probe_state.get("checked_at")
        if (
            not force
            and checked_at
            and _vision_probe_state.get("model") == vision_model
            and (now - datetime.fromisoformat(checked_at)).total_seconds() < _VISION_PROBE_TTL_SECONDS
        ):
            return dict(_vision_probe_state)

        vision_func = await _get_vision_func()
        try:
            content = await vision_func(
                "Reply with exactly OK.",
                image_data=_VISION_PROBE_IMAGE,
                system_prompt="You are a health check.",
                max_completion_tokens=8,
            )
            _vision_probe_state.update(
                {
                    "checked_at": _iso_now(),
                    "ok": True,
                    "error": None,
                    "model": vision_model,
                    "response_preview": content[:200],
                }
            )
        except Exception as exc:
            _vision_probe_state.update(
                {
                    "checked_at": _iso_now(),
                    "ok": False,
                    "error": str(exc),
                    "model": vision_model,
                    "response_preview": None,
                }
            )
        return dict(_vision_probe_state)


class StrictImageModalProcessor:
    _FALLBACK_SUMMARY_PREFIX = "Image content:"

    def __init__(self, processor):
        self._processor = processor
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.attempted = 0
        self.succeeded = 0
        self.failed = 0

    def metrics(self) -> dict:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }

    def __getattr__(self, name):
        return getattr(self._processor, name)

    async def generate_description_only(self, *args, **kwargs):
        self.attempted += 1
        modal_content = args[0] if args else kwargs.get("modal_content")
        try:
            enhanced_caption, entity_info = await self._processor.generate_description_only(
                *args, **kwargs
            )
        except Exception:
            self.failed += 1
            raise

        summary = ""
        if isinstance(entity_info, dict):
            summary = str(entity_info.get("summary", ""))
        if summary.startswith(self._FALLBACK_SUMMARY_PREFIX) or enhanced_caption == str(
            modal_content
        ):
            self.failed += 1
            raise RuntimeError("Image metadata extraction fell back to placeholder content")

        self.succeeded += 1
        return enhanced_caption, entity_info


class StrictRAGAnything:
    def __init__(self, rag):
        self._rag = rag

    def __getattr__(self, name):
        return getattr(self._rag, name)

    async def _ensure_lightrag_initialized(self):
        await self._rag._ensure_lightrag_initialized()
        image_processor = self._rag.modal_processors.get("image")
        if image_processor and not isinstance(image_processor, StrictImageModalProcessor):
            self._rag.modal_processors["image"] = StrictImageModalProcessor(image_processor)

    def reset_image_processing_state(self) -> None:
        image_processor = self._rag.modal_processors.get("image")
        if isinstance(image_processor, StrictImageModalProcessor):
            image_processor.reset_metrics()

    def image_processing_summary(self) -> dict:
        image_processor = self._rag.modal_processors.get("image")
        if isinstance(image_processor, StrictImageModalProcessor):
            return image_processor.metrics()
        return {"attempted": 0, "succeeded": 0, "failed": 0}

    async def process_document_complete(self, *args, **kwargs):
        await self._ensure_lightrag_initialized()
        self.reset_image_processing_state()
        await self._rag.process_document_complete(*args, **kwargs)
        summary = self.image_processing_summary()
        if _vision_required() and summary["attempted"] and (
            summary["failed"] or summary["succeeded"] != summary["attempted"]
        ):
            raise RuntimeError(
                "Image metadata extraction failed for one or more document images"
            )


async def _get_rag():
    global _rag
    if _rag is not None:
        return _rag
    async with _rag_lock:
        if _rag is not None:
            return _rag

        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig
        working_dir = _get_env("RAG_WORKING_DIR", str(Path.home() / ".rag_storage"))
        output_dir = _get_env("RAG_OUTPUT_DIR", str(Path(working_dir) / "output"))

        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        parse_method = _get_env("PARSE_METHOD", "txt")
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser_output_dir=output_dir,
            parse_method=parse_method,
        )

        llm_func = await _get_llm_func()
        vision_func = await _get_vision_func()

        st_model = await _get_st_model()

        async def _embed(texts: list[str]):
            return await asyncio.to_thread(
                lambda: st_model.encode(texts, show_progress_bar=False)
            )

        embedding_dim = int(
            _get_env(
                "EMBEDDING_DIM", str(st_model.get_sentence_embedding_dimension())
            )
        )
        embed_func = EmbeddingFunc(
            embedding_dim=embedding_dim, max_token_size=8192, func=_embed
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embed_func,
        )
        _rag = StrictRAGAnything(rag)
        return _rag


async def _get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is not None:
        return _asr_pipeline
    async with _asr_lock:
        if _asr_pipeline is not None:
            return _asr_pipeline

        import torch
        from transformers import pipeline

        model_name = _get_env("AUDIO_TRANSCRIBE_MODEL", "openai/whisper-small")
        device = _get_env("AUDIO_TRANSCRIBE_DEVICE", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = 0
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1
        elif device == "cpu":
            device = -1

        _asr_pipeline = await asyncio.to_thread(
            pipeline,
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )
    return _asr_pipeline


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


async def _extract_audio_from_video(file_id: str, video_path: Path) -> Path:
    output_path = _uploads_dir / f"{file_id}_audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    await asyncio.to_thread(
        subprocess.run,
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


async def _transcribe_audio(audio_path: Path) -> dict:
    asr = await _get_asr_pipeline()

    def run():
        return asr(
            str(audio_path),
            return_timestamps=True,
            chunk_length_s=int(_get_env("AUDIO_TRANSCRIBE_CHUNK_SECONDS", "30")),
            batch_size=int(_get_env("AUDIO_TRANSCRIBE_BATCH_SIZE", "8")),
        )

    result = await asyncio.to_thread(run)
    chunks = []
    for chunk in result.get("chunks", []) or []:
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp
        chunks.append(
            {
                "start": start,
                "end": end,
                "text": (chunk.get("text") or "").strip(),
            }
        )

    if not chunks and result.get("text"):
        chunks.append({"start": None, "end": None, "text": result["text"].strip()})

    return {"text": (result.get("text") or "").strip(), "chunks": chunks}


async def _write_transcript_artifact(
    file_id: str, original_name: str, transcript: dict
) -> Path:
    transcript_path = _uploads_dir / f"{file_id}_transcript.md"
    lines = [f"# Transcript for {original_name}", ""]
    for chunk in transcript["chunks"]:
        start = _format_timestamp(chunk["start"])
        end = _format_timestamp(chunk["end"])
        prefix = f"[{start}-{end}] " if chunk["start"] is not None else ""
        lines.append(f"{prefix}{chunk['text']}")
    async with aiofiles.open(transcript_path, "w") as f:
        await f.write("\n".join(lines).strip() + "\n")
    return transcript_path


def _transcript_hits(file_record: dict, transcript: dict) -> list[dict]:
    hits = []
    for index, chunk in enumerate(transcript.get("chunks", []), start=1):
        text = chunk["text"]
        if not text:
            continue
        hits.append(
            {
                "engine": "document",
                "collection_id": file_record["collection_id"],
                "file_id": file_record["id"],
                "score": max(0.1, 1.0 - (index * 0.02)),
                "snippet": text[:400],
                "locator": {
                    "kind": "timestamp",
                    "start": _format_timestamp(chunk["start"]),
                    "end": _format_timestamp(chunk["end"]),
                },
                "modality": file_record["modality"],
            }
        )
    return hits[:8]


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

    async def update(self, file_id: str, **fields) -> dict | None:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            updated = None
            for record in records:
                if record["id"] == file_id:
                    record.update(fields)
                    updated = record
                    break
            await self._save(records)
            return updated

    async def get(self, file_id: str) -> dict | None:
        for record in await self.load():
            if record["id"] == file_id:
                return record
        return None

    async def queryable_records(
        self,
        collection_ids: list[str] | None = None,
        target: str = "auto",
    ) -> list[dict]:
        records = await self.load()
        allowed_collections = set(collection_ids) if collection_ids else None
        result = []
        for record in records:
            if not record.get("queryable"):
                continue
            if allowed_collections and record.get("collection_id") not in allowed_collections:
                continue
            engine = record.get("engine")
            if target == "document" and engine != "document":
                continue
            if target == "video" and engine != "video":
                continue
            result.append(record)
        return result

    async def remove(self, file_id: str) -> dict | None:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            removed = None
            new_records = []
            for record in records:
                if record["id"] == file_id:
                    removed = record
                else:
                    new_records.append(record)
            if removed is None:
                return None
            await self._save(new_records)
            return removed


_manifest = FileManifest(_working_dir)


@dataclass
class QueryEnvelope:
    answer: str
    hits: list[dict]
    engine: str
    raw: dict


class DocumentEngineAdapter:
    engine_name = "document"

    async def ingest_file(self, file_path: Path, doc_id: str) -> dict:
        rag = await _get_rag()
        try:
            await rag.process_document_complete(
                file_path=str(file_path), backend=_mineru_backend, doc_id=doc_id
            )
        except Exception:
            await rag._ensure_lightrag_initialized()
            try:
                await rag.lightrag.adelete_by_doc_id(doc_id)
            except Exception as cleanup_exc:
                logging.warning("Failed to roll back document %s: %s", doc_id, cleanup_exc)
            raise
        summary = rag.image_processing_summary()
        capabilities = ["text"]
        if summary["succeeded"]:
            capabilities.append("image")
        return {
            "engine_doc_id": doc_id,
            "image_processing": summary,
            "capabilities": capabilities,
        }

    async def query(self, prompt: str, mode: str = "mix") -> QueryEnvelope:
        rag = await _get_rag()
        await rag._ensure_lightrag_initialized()
        result = await rag.aquery(prompt, mode=mode)
        return QueryEnvelope(
            answer=str(result).strip(),
            hits=[],
            engine=self.engine_name,
            raw={"result": result},
        )

    async def delete(self, engine_doc_id: str | None) -> None:
        if not engine_doc_id:
            return None
        rag = await _get_rag()
        await rag._ensure_lightrag_initialized()
        await rag.lightrag.adelete_by_doc_id(engine_doc_id)
        return None


class VideoEngineAdapter:
    engine_name = "video"

    def __init__(self):
        self.timeout = float(_get_env("VIDEO_ENGINE_TIMEOUT", "300"))
        self.segment_length = _video_segment_length
        self.base_workdir = Path(_working_dir) / "video_engine"
        self.base_workdir.mkdir(parents=True, exist_ok=True)
        self._engines: dict[str, object] = {}
        self._llm_config = None
        self._lock = asyncio.Lock()

    def configured(self) -> bool:
        return _video_engine_enabled

    def _workspace(self, file_id: str) -> Path:
        return self.base_workdir / file_id

    async def _get_llm_config(self):
        if self._llm_config is not None:
            return self._llm_config

        from videorag._llm import LLMConfig

        st_model = await _get_st_model()
        embedding_dim = st_model.get_sentence_embedding_dimension()
        api_base = _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1")
        llm_model = _get_env("LLM_MODEL", "hf.co/unsloth/Qwen3.5-2B-GGUF")
        api_key = _get_env("OPENAI_API_KEY", "docker-model-runner")

        async def embed_func(texts: list[str], **kwargs):
            return await asyncio.to_thread(
                lambda: st_model.encode(texts, show_progress_bar=False)
            )

        async def complete_func(model_name, prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = list(history_messages or [])
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return await _call_openai_chat(
                api_base=api_base,
                model=model_name,
                api_key=api_key,
                messages=messages,
                max_completion_tokens=kwargs.get("max_completion_tokens")
                or kwargs.get("max_tokens")
                or 2048,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                presence_penalty=kwargs.get("presence_penalty"),
                frequency_penalty=kwargs.get("frequency_penalty"),
            )

        self._llm_config = LLMConfig(
            embedding_func_raw=embed_func,
            embedding_model_name=_get_env("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dim=embedding_dim,
            embedding_max_token_size=8192,
            embedding_batch_num=8,
            embedding_func_max_async=4,
            query_better_than_threshold=0.2,
            best_model_func_raw=complete_func,
            best_model_name=llm_model,
            best_model_max_token_size=32768,
            best_model_max_async=4,
            cheap_model_func_raw=complete_func,
            cheap_model_name=llm_model,
            cheap_model_max_token_size=32768,
            cheap_model_max_async=4,
        )
        return self._llm_config

    async def _get_engine(self, file_id: str):
        engine = self._engines.get(file_id)
        if engine is not None:
            return engine

        async with self._lock:
            engine = self._engines.get(file_id)
            if engine is not None:
                return engine

            from videorag import VideoRAG
            from videorag._storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage

            llm_config = await self._get_llm_config()
            workspace = self._workspace(file_id)
            workspace.mkdir(parents=True, exist_ok=True)
            engine = VideoRAG(
                working_dir=str(workspace),
                llm=llm_config,
                key_string_value_json_storage_cls=JsonKVStorage,
                vector_db_storage_cls=NanoVectorDBStorage,
                vs_vector_db_storage_cls=NanoVectorDBStorage,
                graph_storage_cls=NetworkXStorage,
                video_segment_length=self.segment_length,
                video_embedding_dim=llm_config.embedding_dim,
            )
            engine.load_caption_model(debug=True)
            self._engines[file_id] = engine
            return engine

    async def health(self) -> dict:
        vision_health = await _probe_vision_model()
        media_tools = _media_tool_status()
        return {
            "configured": self.configured(),
            "ok": self.configured()
            and bool(vision_health.get("ok"))
            and bool(media_tools.get("ffmpeg"))
            and bool(media_tools.get("ffprobe")),
            "segment_length": self.segment_length,
            "media_tools": media_tools,
            "vision": vision_health,
        }

    async def ingest_file(self, file_id: str, file_path: Path, collection_id: str) -> dict:
        if not self.configured():
            raise RuntimeError("Video engine not configured")
        engine = await self._get_engine(file_id)
        try:
            result = await asyncio.to_thread(engine.insert_video, [str(file_path)])
        except Exception as exc:
            error = RuntimeError(str(exc))
            error.stage = getattr(exc, "stage", "video_ingest")
            error.details = getattr(exc, "details", {})
            raise error from exc
        result = result or {}
        result.setdefault("engine_doc_id", file_id)
        result.setdefault("video_stage", "indexed")
        return result

    async def query(
        self, prompt: str, records: list[dict], top_k: int
    ) -> QueryEnvelope:
        if not self.configured() or not records:
            return QueryEnvelope(answer="", hits=[], engine=self.engine_name, raw={})
        from videorag import QueryParam

        answers = []
        hits = []
        per_video = []
        for record in records:
            file_id = record.get("engine_doc_id") or record["id"]
            engine = await self._get_engine(file_id)
            payload = await asyncio.to_thread(
                engine.query,
                prompt,
                QueryParam(mode="videorag", top_k=max(top_k, 4), wo_reference=False),
            )
            answer = ""
            references = []
            if isinstance(payload, dict):
                answer = (payload.get("answer") or "").strip()
                references = payload.get("references") or []
            else:
                answer = str(payload).strip()

            if answer:
                answers.append(f"{record['original_name']}: {answer}")
            for index, ref in enumerate(references[:top_k], start=1):
                hits.append(
                    {
                        "engine": "video",
                        "collection_id": record["collection_id"],
                        "file_id": record["id"],
                        "score": max(0.1, 1.0 - (index * 0.02)),
                        "snippet": str(ref.get("content", ""))[:400],
                        "video_name": ref.get("video_name") or record["original_name"],
                        "timestamp": f"{ref.get('start_time')} - {ref.get('end_time')}",
                        "segment_id": ref.get("segment_id"),
                    }
                )
            per_video.append({"file_id": record["id"], "answer": answer, "references": references[:top_k]})
        return QueryEnvelope(
            answer="\n\n".join(answer for answer in answers if answer),
            hits=hits[:top_k],
            engine=self.engine_name,
            raw={"videos": per_video},
        )

    async def delete(self, engine_doc_id: str | None) -> None:
        if not engine_doc_id:
            return None
        self._engines.pop(engine_doc_id, None)
        workspace = self._workspace(engine_doc_id)
        await asyncio.to_thread(shutil.rmtree, workspace, True)
        return None


_document_engine = DocumentEngineAdapter()
_video_engine = VideoEngineAdapter()


async def _ingest_document(file_id: str, file_path: Path) -> dict:
    result = await _document_engine.ingest_file(file_path, doc_id=file_id)
    return {
        "status": "done",
        "engine": "document",
        "engine_doc_id": result.get("engine_doc_id"),
        "ingested_at": _iso_now(),
        "queryable": True,
        "ingest_path": "document",
        "capabilities": result.get("capabilities", ["text"]),
        "image_processing": result.get("image_processing"),
    }


async def _ingest_audio(file_id: str, file_path: Path, record: dict) -> dict:
    await _manifest.update(file_id, status="transcribing")
    transcript = await _transcribe_audio(file_path)
    transcript_path = await _write_transcript_artifact(
        file_id, record["original_name"], transcript
    )
    await _manifest.update(file_id, status="ingesting")
    transcript_doc_id = f"{file_id}-audio"
    await _document_engine.ingest_file(transcript_path, doc_id=transcript_doc_id)
    return {
        "status": "done",
        "engine": "document",
        "engine_doc_id": transcript_doc_id,
        "ingested_at": _iso_now(),
        "queryable": True,
        "ingest_path": "audio-transcript",
        "capabilities": ["text", "timestamps"],
        "derived_paths": [str(transcript_path)],
        "transcript_preview": transcript["text"][:4000],
        "transcript_chunks": transcript["chunks"][:128],
    }


async def _ingest_video(file_id: str, file_path: Path, record: dict) -> dict:
    if not _video_engine.configured():
        raise RuntimeError("Video engine disabled")
    await _manifest.update(file_id, status="indexing-video", video_stage="starting")
    ingest_result = await _video_engine.ingest_file(
        file_id=file_id,
        file_path=file_path,
        collection_id=record["collection_id"],
    )
    return {
        "status": "done",
        "engine": "video",
        "engine_doc_id": ingest_result.get("engine_doc_id"),
        "ingested_at": _iso_now(),
        "queryable": True,
        "ingest_path": "video",
        "capabilities": ["video", "timestamps"],
        "audio_probe": ingest_result.get("audio_probe"),
        "video_stage": ingest_result.get("video_stage", "indexed"),
        "video_error_stage": None,
        "video_audio_summary": ingest_result.get("audio_summary"),
        "video_asr_summary": ingest_result.get("asr_summary"),
        "video_segments_indexed": ingest_result.get("segments_indexed"),
    }


async def _ingest_background(file_id: str, file_path: Path) -> None:
    log = logging.getLogger("ingest")
    record = await _manifest.get(file_id)
    if not record:
        _ingest_tasks.pop(file_id, None)
        return

    log.info("Starting ingestion: %s (%s)", file_path.name, file_id)
    await _manifest.update(file_id, status="routing")
    try:
        modality = record["modality"]
        if modality == "audio":
            updates = await _ingest_audio(file_id, file_path, record)
        elif modality == "video":
            updates = await _ingest_video(file_id, file_path, record)
        else:
            updates = await _ingest_document(file_id, file_path)

        updates.setdefault("modality", modality)
        log.info("Ingestion done: %s (%s)", file_path.name, file_id)
        await _manifest.update(file_id, **updates)
    except Exception as exc:
        log.exception("Ingestion failed: %s (%s)", file_path.name, file_id)
        await _manifest.update(
            file_id,
            status="error",
            error=str(exc),
            queryable=False,
            video_stage=getattr(exc, "stage", None),
            video_error_stage=getattr(exc, "stage", None),
            audio_probe=getattr(exc, "details", {}).get("audio_probe"),
            video_audio_summary=getattr(exc, "details", {}).get("audio_summary"),
            video_asr_summary=getattr(exc, "details", {}).get("asr_summary"),
        )
    finally:
        _ingest_tasks.pop(file_id, None)


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG-Anywhere — Media Router</title>
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
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; max-width: 60rem; }
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
  .badge-pending { background: rgba(245,158,11,0.15); color: var(--warning); }
  .badge-error { background: rgba(239,68,68,0.15); color: var(--error); }
  .spinner { width: 10px; height: 10px; border: 2px solid currentColor;
    border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .del-btn { background: none; border: none; color: var(--muted); cursor: pointer;
    font-size: 1rem; padding: 0.25rem 0.5rem; border-radius: 4px; transition: color 0.2s; }
  .del-btn:hover { color: var(--error); }
  .empty { text-align: center; color: var(--muted); padding: 3rem; font-size: 0.875rem; }
  .table-wrap { background: var(--surface); border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden; }
  .mono { font-family: ui-monospace, SFMono-Regular, monospace; color: var(--muted); }
</style>
</head>
<body>
<h1>RAG-Anywhere</h1>
<p class="subtitle">Uploads are routed by modality: documents go to RAGAnything, audio is transcribed into the document graph, and video is indexed by the in-process VideoEngineAdapter.</p>

<div class="drop-zone" id="dropZone">
  <div>&#128249; Drop files here</div>
  <p>Documents, audio, or video. Video ingest requires native video indexing and returns timestamped segments.</p>
  <label class="drop-btn">Browse files<input type="file" id="fileInput" multiple hidden></label>
</div>

<div class="table-wrap">
  <table>
    <thead><tr>
      <th>Name</th><th>Type</th><th>Path</th><th>Engine</th><th>Status</th><th>Uploaded</th><th></th>
    </tr></thead>
    <tbody id="tbody"><tr><td colspan="7" class="empty">No files yet.</td></tr></tbody>
  </table>
</div>

<script>
const tbody = document.getElementById('tbody');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const polls = {};

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}
function badge(status) {
  const spin = `<div class="spinner"></div>`;
  if (status === 'done') return `<span class="badge badge-done">&#10003; Done</span>`;
  if (status === 'error') return `<span class="badge badge-error">&#10007; Error</span>`;
  return `<span class="badge badge-pending">${spin} ${status}</span>`;
}
function render(files) {
  if (!files.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No files yet.</td></tr>';
    return;
  }
  tbody.innerHTML = files.map(f => `
    <tr id="row-${f.id}">
      <td title="${f.original_name}">${f.original_name}</td>
      <td>${f.modality || 'document'}</td>
      <td class="mono">${f.ingest_path || 'pending'}</td>
      <td>${f.engine || 'pending'}</td>
      <td id="status-${f.id}">${badge(f.status)}</td>
      <td>${fmtDate(f.uploaded_at)}</td>
      <td><button class="del-btn" onclick="delFile('${f.id}')" title="Remove">&#128465;</button></td>
    </tr>`).join('');
  files.forEach(f => {
    if (!['done', 'error'].includes(f.status)) startPoll(f.id);
  });
}
async function load() {
  const res = await fetch('/api/files');
  render(await res.json());
}
function startPoll(id) {
  if (polls[id]) return;
  polls[id] = setInterval(async () => {
    const res = await fetch(`/api/files/${id}/status`);
    if (!res.ok) { clearInterval(polls[id]); delete polls[id]; return; }
    const data = await res.json();
    const row = document.getElementById(`status-${id}`);
    if (row) row.innerHTML = badge(data.status);
    if (['done', 'error'].includes(data.status)) {
      clearInterval(polls[id]);
      delete polls[id];
      await load();
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
async function delFile(id) {
  await fetch(`/api/files/${id}`, { method: 'DELETE' });
  clearInterval(polls[id]);
  delete polls[id];
  await load();
}
fileInput.addEventListener('change', () => { [...fileInput.files].forEach(upload); fileInput.value=''; });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  [...e.dataTransfer.files].forEach(upload);
});
load();
</script>
</body>
</html>"""


@asynccontextmanager
async def _lifespan(app):
    records = await _manifest.load()
    for record in records:
        if record["status"] not in {"done", "error"}:
            await _manifest.update(
                record["id"], status="error", error="Server restarted during ingestion"
            )
    yield


mcp = FastMCP("rag-anywhere", host="0.0.0.0", lifespan=_lifespan)


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
    collection_id = form.get("collection_id", _default_collection_id)
    file_id = uuid4().hex[:8]
    safe_name = f"{file_id}_{re.sub(r'[^a-zA-Z0-9._-]', '_', upload.filename)}"
    dest = _uploads_dir / safe_name
    async with aiofiles.open(dest, "wb") as f:
        await f.write(await upload.read())

    modality = _classify_file(dest)
    record = {
        "id": file_id,
        "name": safe_name,
        "original_name": upload.filename,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "uploaded_at": _iso_now(),
        "status": "uploading",
        "ingested_at": None,
        "error": None,
        "collection_id": collection_id,
        "modality": modality,
        "ingest_path": "pending",
        "engine": None,
        "engine_doc_id": None,
        "queryable": False,
        "capabilities": [],
        "derived_paths": [],
        "fallback_reason": None,
        "image_processing": None,
    }
    await _manifest.add(record)
    _ingest_tasks[file_id] = asyncio.create_task(_ingest_background(file_id, dest))
    return JSONResponse(
        {"id": file_id, "status": "uploading", "modality": modality}, status_code=202
    )


@mcp.custom_route("/api/files/{file_id}/status", methods=["GET"])
async def api_file_status(request: Request) -> Response:
    record = await _manifest.get(request.path_params["file_id"])
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(
        {
            k: record.get(k)
            for k in (
                "id",
                "status",
                "error",
                "ingested_at",
                "modality",
                "ingest_path",
                "engine",
                "fallback_reason",
            )
        }
    )


@mcp.custom_route("/api/files/{file_id}", methods=["DELETE"])
async def api_delete_file(request: Request) -> Response:
    file_id = request.path_params["file_id"]
    record = await _manifest.get(file_id)
    if not record:
        return JSONResponse({"ok": False}, status_code=404)

    task = _ingest_tasks.pop(file_id, None)
    if task:
        task.cancel()

    if record.get("engine") == "video":
        try:
            await _video_engine.delete(record.get("engine_doc_id"))
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    elif record.get("engine") == "document":
        try:
            await _document_engine.delete(record.get("engine_doc_id"))
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)

    removed = await _manifest.remove(file_id)
    for path_str in [record.get("path"), *(record.get("derived_paths") or [])]:
        if not path_str:
            continue
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:
            logging.warning("Failed to delete local artifact: %s", path_str)
    return JSONResponse({"ok": bool(removed)})


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health(request: Request) -> Response:
    video_health = await _video_engine.health()
    vision_health = await _probe_vision_model()
    return JSONResponse(
        {
            "ok": True,
            "video_engine_enabled": _video_engine_enabled,
            "video_engine": video_health,
            "vision_model": _get_env("VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest"),
            "vision_api_base": _get_env(
                "VISION_API_BASE",
                _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1"),
            ),
            "vision_required": _vision_required(),
            "vision_ready": bool(vision_health.get("ok")),
            "vision_last_probe": vision_health,
            "media_tools": _media_tool_status(),
        }
    )


async def _ingest_paths(paths: list[str], recursive: bool = True) -> dict:
    added = []
    errors = []
    for item in paths:
        path = Path(item)
        try:
            if path.is_dir():
                for child in path.rglob("*") if recursive else path.glob("*"):
                    if child.is_file():
                        file_id = uuid4().hex[:8]
                        record = {
                            "id": file_id,
                            "name": child.name,
                            "original_name": child.name,
                            "path": str(child),
                            "size_bytes": child.stat().st_size,
                            "uploaded_at": _iso_now(),
                            "status": "queued",
                            "ingested_at": None,
                            "error": None,
                            "collection_id": _default_collection_id,
                            "modality": _classify_file(child),
                            "ingest_path": "pending",
                            "engine": None,
                            "engine_doc_id": None,
                            "queryable": False,
                            "capabilities": [],
                            "derived_paths": [],
                            "fallback_reason": None,
                            "image_processing": None,
                        }
                        await _manifest.add(record)
                        await _ingest_background(file_id, child)
                        added.append(str(child))
            elif path.is_file():
                file_id = uuid4().hex[:8]
                record = {
                    "id": file_id,
                    "name": path.name,
                    "original_name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "uploaded_at": _iso_now(),
                    "status": "queued",
                    "ingested_at": None,
                    "error": None,
                    "collection_id": _default_collection_id,
                    "modality": _classify_file(path),
                    "ingest_path": "pending",
                    "engine": None,
                    "engine_doc_id": None,
                    "queryable": False,
                    "capabilities": [],
                    "derived_paths": [],
                    "fallback_reason": None,
                    "image_processing": None,
                }
                await _manifest.add(record)
                await _ingest_background(file_id, path)
                added.append(str(path))
            else:
                errors.append(f"Path not found: {item}")
        except Exception as exc:
            errors.append(f"{item}: {exc}")
    return {"added": added, "errors": errors}


async def _document_query_enabled(records: list[dict], target: str) -> bool:
    if target == "video":
        return False
    return any(record.get("engine") == "document" for record in records)


async def _video_query_enabled(records: list[dict], target: str) -> bool:
    if target == "document":
        return False
    if target == "video":
        return any(record.get("engine") == "video" for record in records)
    return any(record.get("engine") == "video" for record in records)


async def _federate_answer(prompt: str, envelopes: list[QueryEnvelope]) -> str:
    if not envelopes:
        return ""
    if len(envelopes) == 1:
        return envelopes[0].answer

    llm_func = await _get_llm_func()
    evidence = []
    for envelope in envelopes:
        evidence.append(
            {
                "engine": envelope.engine,
                "answer": envelope.answer,
                "hits": envelope.hits[:4],
            }
        )
    return await llm_func(
        f"Question: {prompt}\n\n"
        f"Combine the evidence below into one concise answer. Cite engine names when useful.\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=True)}"
    )


async def _fallback_hits(records: list[dict]) -> list[dict]:
    hits = []
    for record in records:
        if record.get("ingest_path") != "audio-transcript":
            continue
        if record.get("transcript_chunks"):
            hits.extend(_transcript_hits(record, {"chunks": record["transcript_chunks"]}))
    return hits[:8]


@mcp.tool()
async def query(
    query: str,
    mode: str = "mix",
    target: str = "auto",
    collection_ids: list[str] | None = None,
    top_k: int = 8,
) -> dict:
    """Query the routed RAG system."""
    selected_records = await _manifest.queryable_records(
        collection_ids=collection_ids, target=target
    )
    envelopes: list[QueryEnvelope] = []
    hits: list[dict] = []

    if await _document_query_enabled(selected_records, target):
        document_envelope = await _document_engine.query(query, mode=mode)
        envelopes.append(document_envelope)
        hits.extend(await _fallback_hits(selected_records))

    if await _video_query_enabled(selected_records, target):
        video_records = [record for record in selected_records if record.get("engine") == "video"]
        video_envelope = await _video_engine.query(query, video_records, top_k)
        if video_envelope.answer or video_envelope.hits:
            envelopes.append(video_envelope)
            hits.extend(video_envelope.hits[:top_k])

    if not envelopes and hits:
        answer = "Relevant transcript segments found."
    else:
        answer = await _federate_answer(query, envelopes)

    return {
        "result": answer,
        "hits": hits[:top_k],
        "engines_used": [envelope.engine for envelope in envelopes],
        "collections": sorted({record["collection_id"] for record in selected_records}),
    }


@mcp.tool()
async def ingest(paths: list[str], recursive: bool = True) -> dict:
    """Ingest documents, audio, or video files into the routed RAG system."""
    return await _ingest_paths(paths=paths, recursive=recursive)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
