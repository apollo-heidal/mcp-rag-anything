"""
RAG-Anywhere server with modality-aware routing and federated query support.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import aiofiles
import httpx
import nest_asyncio
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
_ARCHIVE_EXTENSIONS = {".zip"}

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
_embed_func_cache = None
_embed_lock = asyncio.Lock()
_vision_probe_state = {
    "checked_at": None,
    "ok": None,
    "error": None,
    "model": None,
    "response_preview": None,
}
_VISION_PROBE_TTL_SECONDS = 60
_VISION_PROBE_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X6Z0AAAAASUVORK5CYII="


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _ARCHIVE_EXTENSIONS:
        return "archive"
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


def _base_url() -> str:
    return _get_env("APP_BASE_URL", "http://localhost:8000").rstrip("/")


def _media_tool_status() -> dict:
    return {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }


def _vision_required() -> bool:
    return _get_env("VISION_REQUIRED", "true").lower() == "true"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _grant_secret() -> bytes:
    return _get_env("RAG_SESSION_SECRET", "change-me-for-production").encode("utf-8")


def _sign_grant_payload(payload: str) -> str:
    digest = hmac.new(_grant_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _mint_grant_token(grant_id: str, grant_type: str, expires_at: str) -> str:
    payload = json.dumps(
        {"id": grant_id, "type": grant_type, "exp": expires_at},
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded_payload = (
        base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8").rstrip("=")
    )
    signature = _sign_grant_payload(encoded_payload)
    return f"{encoded_payload}.{signature}"


def _parse_grant_token(token: str) -> tuple[str, str, str]:
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Invalid grant token") from exc
    expected = _sign_grant_payload(encoded_payload)
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Grant signature mismatch")
    try:
        padded = encoded_payload + "=" * (-len(encoded_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        return payload["id"], payload["type"], payload["exp"]
    except Exception as exc:
        raise ValueError("Invalid grant token") from exc


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


async def _build_embed_func() -> tuple:
    """Return (embed_async_func, embedding_dim).

    Prefers the DMR embeddings API (EMBEDDING_API_BASE + EMBEDDING_MODEL).
    Falls back to a local SentenceTransformer if the API is not configured.
    """
    import numpy as np

    api_base = _get_env("EMBEDDING_API_BASE", "")
    embed_model = _get_env("EMBEDDING_MODEL", "")

    if api_base and embed_model:
        import httpx

        # DMR's llama.cpp backend has n_batch=512 by default — any single
        # input exceeding ~512 tokens causes a 500 error.  Truncate inputs
        # to stay safely under the limit (≈ 4 chars/token).
        _EMBED_MAX_CHARS = 1200  # ~450 tokens — safe margin under DMR n_batch=512

        # Serialise requests: DMR handles one embedding batch at a time.
        _embed_semaphore = asyncio.Semaphore(1)

        async def _embed_via_api(texts: list[str]):
            truncated = [t[:_EMBED_MAX_CHARS] for t in texts]
            max_retries = 5
            async with _embed_semaphore:
                for attempt in range(max_retries):
                    try:
                        async with httpx.AsyncClient(timeout=120) as client:
                            resp = await client.post(
                                f"{api_base}embeddings",
                                json={"model": embed_model, "input": truncated},
                            )
                            if resp.status_code != 200:
                                logging.warning(
                                    "Embedding API returned %d (batch=%d, max_chars=%s): %s",
                                    resp.status_code,
                                    len(truncated),
                                    [len(t) for t in truncated[:3]],
                                    resp.text[:200],
                                )
                            resp.raise_for_status()
                            data = resp.json()
                        sorted_data = sorted(data["data"], key=lambda d: d["index"])
                        return np.array(
                            [d["embedding"] for d in sorted_data], dtype=np.float32
                        )
                    except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                        if attempt < max_retries - 1:
                            wait = [3, 5, 10, 15][
                                attempt
                            ]  # keep total <60s (LightRAG worker timeout)
                            logging.warning(
                                "Embedding attempt %d/%d failed (%s), retrying in %ds…",
                                attempt + 1,
                                max_retries,
                                exc,
                                wait,
                            )
                            await asyncio.sleep(wait)
                        else:
                            raise

        # Probe the API to discover the actual embedding dimension
        probe_dim = int(_get_env("EMBEDDING_DIM", "0"))
        if not probe_dim:
            import httpx as _httpx

            async with _httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{api_base}embeddings",
                    json={"model": embed_model, "input": ["dim probe"]},
                )
                resp.raise_for_status()
                probe_dim = len(resp.json()["data"][0]["embedding"])
        logging.info(
            "Using DMR embeddings API: model=%s dim=%d", embed_model, probe_dim
        )
        return _embed_via_api, probe_dim

    # Fallback: local SentenceTransformer (CPU)
    logging.warning(
        "EMBEDDING_API_BASE not configured — falling back to local "
        "SentenceTransformer on CPU. This is significantly slower than "
        "using Docker Model Runner."
    )
    from sentence_transformers import SentenceTransformer

    st_model_name = _get_env("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    st_model = await asyncio.to_thread(SentenceTransformer, st_model_name)
    dim = st_model.get_sentence_embedding_dimension()

    async def _embed_local(texts: list[str]):
        return await asyncio.to_thread(
            lambda: st_model.encode(texts, show_progress_bar=False)
        )

    return _embed_local, dim


async def _build_vision_func():
    api_base = _get_env(
        "VISION_API_BASE", _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1")
    )
    vision_model = _get_env("VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest")
    api_key = _get_env(
        "VISION_API_KEY", _get_env("OPENAI_API_KEY", "docker-model-runner")
    )

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
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

        max_tokens = (
            kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 1024
        )
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
        and (now - datetime.fromisoformat(checked_at)).total_seconds()
        < _VISION_PROBE_TTL_SECONDS
    ):
        return dict(_vision_probe_state)

    async with _vision_probe_lock:
        checked_at = _vision_probe_state.get("checked_at")
        if (
            not force
            and checked_at
            and _vision_probe_state.get("model") == vision_model
            and (now - datetime.fromisoformat(checked_at)).total_seconds()
            < _VISION_PROBE_TTL_SECONDS
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
        """
        _getattr is this working?
        """
        return getattr(self._processor, name)

    async def generate_description_only(self, *args, **kwargs):
        self.attempted += 1
        modal_content = args[0] if args else kwargs.get("modal_content")
        try:
            (
                enhanced_caption,
                entity_info,
            ) = await self._processor.generate_description_only(*args, **kwargs)
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
            raise RuntimeError(
                "Image metadata extraction fell back to placeholder content"
            )

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
        if image_processor and not isinstance(
            image_processor, StrictImageModalProcessor
        ):
            self._rag.modal_processors["image"] = StrictImageModalProcessor(
                image_processor
            )

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
        if (
            _vision_required()
            and summary["attempted"]
            and (summary["failed"] or summary["succeeded"] != summary["attempted"])
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

        embed_fn, embedding_dim = await _build_embed_func()
        embed_func = EmbeddingFunc(
            embedding_dim=embedding_dim, max_token_size=8192, func=embed_fn
        )

        lightrag_kwargs = {
            "vector_storage": "MilvusVectorDBStorage",
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2,
            },
        }

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embed_func,
            lightrag_kwargs=lightrag_kwargs,
        )
        _rag = StrictRAGAnything(rag)
        # Schedule recovery of false-done documents now that RAG is ready
        asyncio.create_task(_recover_false_done_documents())
        return _rag


def _whisper_api_base() -> str:
    return _get_env("WHISPER_API_BASE", "")


async def _get_asr_model():
    """Fallback: load local PyTorch Whisper when no API is configured."""
    global _asr_pipeline
    if _asr_pipeline is not None:
        return _asr_pipeline
    async with _asr_lock:
        if _asr_pipeline is not None:
            return _asr_pipeline

        logging.warning(
            "WHISPER_API_BASE not configured — falling back to local "
            "PyTorch Whisper on CPU. This is significantly slower than "
            "using a whisper.cpp server with Metal/GPU acceleration."
        )

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = _get_env("AUDIO_TRANSCRIBE_MODEL", "openai/whisper-small")
        device = _get_env("AUDIO_TRANSCRIBE_DEVICE", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        def _load():
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            model.to(device)
            return model, processor

        _asr_pipeline = await asyncio.to_thread(_load)
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


_AUDIO_CHUNK_SECONDS = 300  # 5 minutes — whisper.cpp struggles with very long audio


async def _get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0.0


async def _split_audio_chunks(audio_path: Path, chunk_secs: int) -> list[Path]:
    """Split audio into chunk_secs-length WAV segments. Returns list of paths."""
    duration = await _get_audio_duration(audio_path)
    if duration <= 0:
        return [audio_path]
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))
    segments = []
    start = 0.0
    idx = 0
    while start < duration:
        out = tmp_dir / f"chunk_{idx:04d}.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(chunk_secs),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if out.exists() and out.stat().st_size > 0:
            segments.append((start, out))
        start += chunk_secs
        idx += 1
    return segments


async def _transcribe_single_chunk(url: str, audio_file: Path) -> dict:
    """Transcribe one audio chunk via the whisper API."""
    import httpx

    async with httpx.AsyncClient(timeout=600) as client:
        with open(audio_file, "rb") as f:
            resp = await client.post(
                url,
                files={"file": (audio_file.name, f, "audio/wav")},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                },
            )
        resp.raise_for_status()
    return resp.json()


async def _transcribe_audio_via_api(audio_path: Path) -> dict:
    """Transcribe using an OpenAI-compatible whisper API endpoint.

    For audio longer than _AUDIO_CHUNK_SECONDS, splits into chunks first
    to avoid whisper.cpp memory/encoding failures on very long audio.
    """
    api_base = _whisper_api_base()
    url = f"{api_base}/inference"

    duration = await _get_audio_duration(audio_path)
    if duration > _AUDIO_CHUNK_SECONDS:
        logging.info(
            "Audio %.0fs > %ds threshold — splitting into chunks for transcription",
            duration,
            _AUDIO_CHUNK_SECONDS,
        )
        segments = await _split_audio_chunks(audio_path, _AUDIO_CHUNK_SECONDS)
        all_chunks = []
        all_text_parts = []
        for offset_secs, chunk_path in segments:
            try:
                data = await _transcribe_single_chunk(url, chunk_path)
            except Exception as exc:
                logging.warning("Whisper chunk %s failed: %s", chunk_path.name, exc)
                continue
            finally:
                chunk_path.unlink(missing_ok=True)
            text = data.get("text", "").strip()
            if text:
                all_text_parts.append(text)
            for seg in data.get("segments", []):
                s = seg.get("start")
                e = seg.get("end")
                t = seg.get("text", "").strip()
                if s is not None and e is not None and t:
                    all_chunks.append(
                        {
                            "start": s + offset_secs,
                            "end": e + offset_secs,
                            "text": t,
                        }
                    )
        # Cleanup temp dir
        if segments:
            segments[0][1].parent.rmdir()  # empty after unlinking chunks
        full_text = " ".join(all_text_parts)
        if not all_chunks and full_text:
            all_chunks.append({"start": None, "end": None, "text": full_text})
        return {"text": full_text, "chunks": all_chunks}

    # Short audio — send directly
    data = await _transcribe_single_chunk(url, audio_path)
    full_text = data.get("text", "").strip()
    chunks = []
    for seg in data.get("segments", []):
        chunks.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
        )
    if not chunks and full_text:
        chunks.append({"start": None, "end": None, "text": full_text})
    return {"text": full_text, "chunks": chunks}


async def _transcribe_audio_local(audio_path: Path) -> dict:
    """Fallback: transcribe using local PyTorch Whisper model."""
    import torch
    from transformers.pipelines.audio_utils import ffmpeg_read

    model, processor = await _get_asr_model()
    device = next(model.parameters()).device

    def run():
        with open(audio_path, "rb") as f:
            audio = ffmpeg_read(f.read(), sampling_rate=16000)
        inputs = processor(
            audio,
            return_tensors="pt",
            sampling_rate=16000,
            return_attention_mask=True,
            truncation=False,
            padding="max_length",
        )
        inputs = inputs.to(device)
        generated_ids = model.generate(
            **inputs,
            return_timestamps=True,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
        )
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            output_offsets=True,
        )
        return decoded

    decoded = await asyncio.to_thread(run)

    chunks = []
    full_text = ""
    if decoded:
        entry = decoded[0]
        if isinstance(entry, dict):
            full_text = entry.get("text", "").strip()
            for offset in entry.get("offsets", []):
                chunks.append(
                    {
                        "start": offset.get("timestamp", (None, None))[0],
                        "end": offset.get("timestamp", (None, None))[1],
                        "text": offset.get("text", "").strip(),
                    }
                )
        else:
            full_text = str(entry).strip()

    if not chunks and full_text:
        chunks.append({"start": None, "end": None, "text": full_text})

    return {"text": full_text, "chunks": chunks}


async def _transcribe_audio(audio_path: Path) -> dict:
    if _whisper_api_base():
        return await _transcribe_audio_via_api(audio_path)
    return await _transcribe_audio_local(audio_path)


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
            if (
                allowed_collections
                and record.get("collection_id") not in allowed_collections
            ):
                continue
            engine = record.get("engine")
            if target == "document" and engine != "document":
                continue
            if target == "video" and engine != "video":
                continue
            result.append(record)
        return result

    async def remove_by_parent(self, parent_archive_id: str) -> list[dict]:
        async with self._lock:
            records = json.loads(self._path.read_text()) if self._path.exists() else []
            removed = []
            kept = []
            for record in records:
                if record.get("parent_archive_id") == parent_archive_id:
                    removed.append(record)
                else:
                    kept.append(record)
            if removed:
                await self._save(kept)
            return removed

    async def find_by_hash(self, content_hash: str) -> dict | None:
        for record in await self.load():
            if record.get("content_hash") == content_hash:
                return record
        return None

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


class GrantStore:
    def __init__(self, working_dir: str):
        self._path = Path(working_dir) / "grants.json"
        self._lock = asyncio.Lock()

    async def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        async with aiofiles.open(self._path) as f:
            return json.loads(await f.read())

    async def _save(self, grants: dict[str, dict]) -> None:
        async with aiofiles.open(self._path, "w") as f:
            await f.write(json.dumps(grants, indent=2))

    async def issue(
        self,
        *,
        grant_type: str,
        subject: str,
        ttl_seconds: int,
        metadata: dict | None = None,
        max_uses: int | None = None,
    ) -> dict:
        async with self._lock:
            grants = await self._load()
            grant_id = secrets.token_urlsafe(12)
            expires_at = (_now_utc()).timestamp() + ttl_seconds
            grant = {
                "id": grant_id,
                "type": grant_type,
                "subject": subject,
                "issued_at": _iso_now(),
                "expires_at": datetime.fromtimestamp(
                    expires_at, tz=timezone.utc
                ).isoformat(),
                "metadata": metadata or {},
                "max_uses": max_uses,
                "uses": 0,
                "used_at": None,
                "revoked": False,
            }
            grants[grant_id] = grant
            await self._save(grants)
            grant["token"] = _mint_grant_token(
                grant_id, grant_type, grant["expires_at"]
            )
            return grant

    async def get_valid(self, token: str, expected_type: str) -> dict:
        grant_id, grant_type, expires_at = _parse_grant_token(token)
        if grant_type != expected_type:
            raise ValueError("Grant type mismatch")
        if datetime.fromisoformat(expires_at) <= _now_utc():
            raise ValueError("Grant expired")
        async with self._lock:
            grants = await self._load()
            grant = grants.get(grant_id)
            if not grant or grant.get("revoked"):
                raise ValueError("Grant not found")
            if grant.get("type") != expected_type:
                raise ValueError("Grant type mismatch")
            if grant.get("expires_at") != expires_at:
                raise ValueError("Grant expiration mismatch")
            if datetime.fromisoformat(grant["expires_at"]) <= _now_utc():
                raise ValueError("Grant expired")
            max_uses = grant.get("max_uses")
            if max_uses is not None and grant.get("uses", 0) >= max_uses:
                raise ValueError("Grant already used")
            return grant

    async def mark_used(self, grant_id: str) -> None:
        async with self._lock:
            grants = await self._load()
            grant = grants.get(grant_id)
            if not grant:
                return
            grant["uses"] = int(grant.get("uses", 0)) + 1
            grant["used_at"] = _iso_now()
            grants[grant_id] = grant
            await self._save(grants)


_grants = GrantStore(_working_dir)


def _request_grant_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.query_params.get("grant")


async def _require_grant(
    request: Request, expected_type: str
) -> tuple[dict | None, Response | None]:
    token = _request_grant_token(request)
    if not token:
        return None, JSONResponse(
            {"error": f"missing {expected_type} grant"}, status_code=401
        )
    try:
        grant = await _grants.get_valid(token, expected_type)
    except Exception as exc:
        return None, JSONResponse({"error": str(exc)}, status_code=401)
    return grant, None


async def _issue_ui_session(subject: str, ttl_seconds: int | None = None) -> dict:
    ttl = ttl_seconds or int(_get_env("UI_SESSION_TTL_SECONDS", "900"))
    grant = await _grants.issue(
        grant_type="ui_session",
        subject=subject,
        ttl_seconds=ttl,
        metadata={"surface": "ui"},
        max_uses=None,
    )
    grant["url"] = f"{_base_url()}/ui?grant={quote(grant['token'])}"
    return grant


async def _issue_upload_link(
    subject: str,
    *,
    collection_id: str | None = None,
    filename: str | None = None,
    ttl_seconds: int | None = None,
) -> dict:
    ttl = ttl_seconds or int(_get_env("UPLOAD_LINK_TTL_SECONDS", "900"))
    grant = await _grants.issue(
        grant_type="upload",
        subject=subject,
        ttl_seconds=ttl,
        metadata={
            "collection_id": collection_id or _default_collection_id,
            "filename": filename,
        },
        max_uses=1,
    )
    grant["upload_url"] = f"{_base_url()}/api/upload/{quote(grant['token'])}"
    return grant


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

        # --- Cascade prevention ---
        # LightRAG re-processes ALL failed/processing docs on every insert,
        # causing GPU contention and potentially destroying existing embeddings.
        # Temporarily mark them as "processed" so the cascade skips them.
        await rag._ensure_lightrag_initialized()
        shielded: dict[str, dict] = {}
        try:
            from lightrag.base import DocStatus as _DocStatus

            for status_val in (_DocStatus.FAILED, _DocStatus.PROCESSING):
                docs = await rag.lightrag.doc_status.get_docs_by_status(status_val)
                for did, status_obj in (docs or {}).items():
                    if did == doc_id:
                        continue
                    # Save original as dict for restore (convert enums to strings)
                    orig = {}
                    for k, v in (
                        status_obj.__dict__ if hasattr(status_obj, "__dict__") else {}
                    ).items():
                        orig[k] = v.value if isinstance(v, _DocStatus) else v
                    if orig:
                        shielded[did] = orig
            if shielded:
                shield_upsert = {
                    did: {**orig, "status": "processed"}
                    for did, orig in shielded.items()
                }
                await rag.lightrag.doc_status.upsert(shield_upsert)
                logging.info(
                    "Cascade prevention: shielded %d docs from re-processing",
                    len(shielded),
                )
        except Exception:
            logging.warning("Cascade prevention: failed to shield docs", exc_info=True)
            shielded.clear()

        try:
            await rag.process_document_complete(
                file_path=str(file_path), backend=_mineru_backend, doc_id=doc_id
            )
        except Exception:
            await rag._ensure_lightrag_initialized()
            try:
                await rag.lightrag.adelete_by_doc_id(doc_id)
            except Exception as cleanup_exc:
                logging.warning(
                    "Failed to roll back document %s: %s", doc_id, cleanup_exc
                )
            raise
        finally:
            # Restore shielded docs to their original status
            if shielded:
                try:
                    await rag.lightrag.doc_status.upsert(shielded)
                    logging.info(
                        "Cascade prevention: restored %d shielded docs",
                        len(shielded),
                    )
                except Exception:
                    logging.warning(
                        "Cascade prevention: failed to restore shielded docs",
                        exc_info=True,
                    )

        # Verify LightRAG actually stored embeddings — process_document_complete
        # may silently swallow embedding failures and return normally while
        # internally marking the doc as "failed" in kv_store_doc_status.
        await rag._ensure_lightrag_initialized()
        doc_entry = await rag.lightrag.doc_status.get_by_id(doc_id)
        if doc_entry and isinstance(doc_entry, dict):
            internal_status = doc_entry.get("status", "")
            if internal_status == "failed":
                error_msg = doc_entry.get("error_msg", "unknown embedding failure")
                raise RuntimeError(
                    f"LightRAG embedding failed for {doc_id}: {error_msg}"
                )

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

        embed_fn, embedding_dim = await _build_embed_func()
        api_base = _get_env("LLM_API_BASE", "http://localhost:12434/engines/v1")
        llm_model = _get_env("LLM_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest")
        api_key = _get_env("OPENAI_API_KEY", "docker-model-runner")

        async def embed_func(texts: list[str], **kwargs):
            return await embed_fn(texts)

        async def complete_func(
            model_name, prompt, system_prompt=None, history_messages=None, **kwargs
        ):
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
            embedding_model_name=_get_env("EMBEDDING_MODEL", "ai/mxbai-embed-large"),
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
            from videorag._storage import (
                JsonKVStorage,
                MilvusVectorDBStorage,
                MilvusVectorDBVideoSegmentStorage,
                NetworkXStorage,
            )

            vdb_cls = MilvusVectorDBStorage
            vs_vdb_cls = MilvusVectorDBVideoSegmentStorage

            llm_config = await self._get_llm_config()
            workspace = self._workspace(file_id)
            workspace.mkdir(parents=True, exist_ok=True)
            engine = VideoRAG(
                working_dir=str(workspace),
                llm=llm_config,
                key_string_value_json_storage_cls=JsonKVStorage,
                vector_db_storage_cls=vdb_cls,
                vs_vector_db_storage_cls=vs_vdb_cls,
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

    async def ingest_file(
        self, file_id: str, file_path: Path, collection_id: str
    ) -> dict:
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
            per_video.append(
                {
                    "file_id": record["id"],
                    "answer": answer,
                    "references": references[:top_k],
                }
            )
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
    await _manifest.update(file_id, status="ingesting")
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


async def _ingest_archive(file_id: str, file_path: Path, record: dict) -> dict:
    from dedup import compute_file_hash
    from zip_handler import extract_recursive

    await _manifest.update(file_id, status="extracting")
    extract_dir = _uploads_dir / f"{file_id}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = await extract_recursive(file_path, extract_dir)
    total = len(extracted_files)

    child_ids = []
    skipped = []
    errors = []
    for idx, child_path in enumerate(extracted_files, 1):
        await _manifest.update(file_id, status=f"processing {idx}/{total}")
        try:
            content_hash = await compute_file_hash(child_path)
            existing = await _manifest.find_by_hash(content_hash)
            if existing:
                skipped.append(
                    {
                        "name": child_path.name,
                        "duplicate_of": existing["id"],
                        "original_name": existing["original_name"],
                    }
                )
                continue

            child_id = uuid4().hex[:8]
            safe_name = _safe_filename(child_id, child_path.name)
            child_dest = _uploads_dir / safe_name
            shutil.copy2(str(child_path), str(child_dest))

            child_record = _build_file_record(
                file_id=child_id,
                safe_name=safe_name,
                original_name=child_path.name,
                path=child_dest,
                collection_id=record["collection_id"],
                content_hash=content_hash,
                parent_archive_id=file_id,
            )
            child_record["status"] = "queued"
            await _manifest.add(child_record)
            await _ingest_background(child_id, child_dest)
            child_ids.append(child_id)
        except Exception as exc:
            errors.append({"name": child_path.name, "error": str(exc)})

    shutil.rmtree(str(extract_dir), ignore_errors=True)
    summary = f"done ({len(child_ids)} ingested, {len(skipped)} skipped, {len(errors)} errors)"
    return {
        "status": "done",
        "archive_progress": summary,
        "files_extracted": total,
        "files_ingested": len(child_ids),
        "files_skipped": len(skipped),
        "skipped_details": skipped,
        "child_ids": child_ids,
        "ingested_at": _iso_now(),
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
        if modality == "archive":
            updates = await _ingest_archive(file_id, file_path, record)
        elif modality == "audio":
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
  .page-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.25rem; }
  .page-header h1 { margin-bottom: 0; }
  .btn-attu { margin-left: auto; display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border: 1px solid var(--border); border-radius: var(--radius);
    color: var(--muted); font-size: 0.78rem; text-decoration: none; transition: all 0.15s; }
  .btn-attu:hover { color: var(--accent); border-color: var(--accent); }
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
  .badge-error-wrap { position: relative; display: inline-block; }
  .badge-error-wrap:hover .error-tooltip { display: block; }
  .error-tooltip {
    display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    margin-bottom: 6px; padding: 0.5rem 0.75rem; max-width: 300px; width: max-content;
    background: #1e1e2e; color: var(--error); border: 1px solid var(--error); border-radius: 6px;
    font-size: 0.75rem; line-height: 1.4; white-space: pre-wrap; word-break: break-word; z-index: 10;
  }
  .status-detail { display: block; font-size: 0.65rem; color: var(--muted); margin-top: 2px; }
  .upload-status { display: none; margin-top: 1.25rem; text-align: left; }
  .upload-status.active { display: block; }
  .upload-label { font-size: 0.8rem; color: var(--text); margin-bottom: 0.4rem; }
  .upload-bar-track { width: 100%; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .upload-bar-fill { height: 100%; width: 0%; background: var(--accent); border-radius: 3px; transition: width 0.15s ease; }
  .upload-pct { font-size: 0.7rem; color: var(--muted); margin-top: 0.25rem; text-align: right; }
  .upload-error { display: none; margin-top: 1rem; padding: 0.6rem 1rem; background: rgba(239,68,68,0.12);
    border: 1px solid var(--error); border-radius: var(--radius); color: var(--error); font-size: 0.8rem; text-align: left; }
  .upload-error.active { display: block; }
</style>
</head>
<body>
<div class="page-header"><h1>RAG-Anywhere</h1><!-- ATTU_LINK --></div>
<p class="subtitle">Uploads are routed by modality: documents go to RAGAnything, audio is transcribed into the document graph, and video is indexed by the in-process VideoEngineAdapter. Access to this library is granted by a signed session link.</p>

<div class="drop-zone" id="dropZone">
  <div>&#128249; Drop files here</div>
  <p>Documents, audio, video, or ZIP archives. Video ingest requires native video indexing and returns timestamped segments.</p>
  <label class="drop-btn">Browse files<input type="file" id="fileInput" multiple hidden></label>
  <div class="upload-status" id="uploadStatus">
    <div class="upload-label" id="uploadLabel">Uploading…</div>
    <div class="upload-bar-track"><div class="upload-bar-fill" id="uploadFill"></div></div>
    <div class="upload-pct" id="uploadPct">0%</div>
  </div>
  <div class="upload-error" id="uploadError"></div>
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
const grant = new URLSearchParams(window.location.search).get('grant');

function withGrant(url) {
  const target = new URL(url, window.location.origin);
  if (grant) target.searchParams.set('grant', grant);
  return `${target.pathname}${target.search}`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function badge(status, data) {
  data = data || {};
  const spin = `<div class="spinner"></div>`;
  if (status === 'done') return `<span class="badge badge-done">&#10003; Done</span>`;
  if (status === 'duplicate') return `<span class="badge badge-pending">&#9888; Duplicate</span>`;
  if (status === 'error') {
    const tip = data.error ? `<span class="error-tooltip">${esc(data.error)}</span>` : '';
    return `<span class="badge-error-wrap"><span class="badge badge-error">&#10007; Error</span>${tip}</span>`;
  }
  let label = status;
  let detail = '';
  if (status === 'routing') { label = 'Classifying\u2026'; }
  else if (status === 'transcribing') { label = 'Transcribing audio\u2026'; }
  else if (status === 'extracting') { label = 'Extracting archive\u2026'; }
  else if (status && status.startsWith('processing')) {
    label = 'Processing archive\u2026';
    const m = status.match(/processing\\s+(\\d+\\/\\d+)/);
    if (m) detail = m[1];
  }
  else if (status === 'indexing-video') {
    label = 'Indexing video\u2026';
    if (data.video_stage) detail = data.video_stage;
  }
  else if (status === 'ingesting') { label = 'Ingesting document\u2026'; }
  const detailHtml = detail ? `<span class="status-detail">${esc(detail)}</span>` : '';
  return `<span class="badge badge-pending">${spin} ${esc(label)}</span>${detailHtml}`;
}
function render(files) {
  if (!files.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No files yet.</td></tr>';
    return;
  }
  tbody.innerHTML = files.map(f => `
    <tr id="row-${f.id}">
      <td title="${f.original_name}">${f.parent_archive_id ? '&nbsp;&nbsp;&#8627; ' : ''}${f.original_name}</td>
      <td>${f.modality || 'document'}</td>
      <td class="mono">${f.ingest_path || 'pending'}</td>
      <td>${f.engine || 'pending'}</td>
      <td id="status-${f.id}">${badge(f.status, f)}</td>
      <td>${fmtDate(f.uploaded_at)}</td>
      <td>${f.status === 'error' ? `<button class="del-btn" onclick="retryFile('${f.id}')" title="Retry" style="margin-right:4px">&#8635;</button>` : ''}<button class="del-btn" onclick="delFile('${f.id}')" title="Remove">&#128465;</button></td>
    </tr>`).join('');
  files.forEach(f => {
    if (!['done', 'error', 'duplicate'].includes(f.status)) startPoll(f.id);
  });
}
async function load() {
  const res = await fetch(withGrant('/api/files'));
  render(await res.json());
}
function startPoll(id) {
  if (polls[id]) return;
  polls[id] = setInterval(async () => {
    const res = await fetch(withGrant(`/api/files/${id}/status`));
    if (!res.ok) { clearInterval(polls[id]); delete polls[id]; return; }
    const data = await res.json();
    const row = document.getElementById(`status-${id}`);
    if (row) row.innerHTML = badge(data.status, data);
    if (['done', 'error', 'duplicate'].includes(data.status)) {
      clearInterval(polls[id]);
      delete polls[id];
      await load();
    }
  }, 2000);
}
const uploadState = { active: 0, loaded: 0, total: 0, perFile: {} };
const uploadStatusEl = document.getElementById('uploadStatus');
const uploadLabel = document.getElementById('uploadLabel');
const uploadFill = document.getElementById('uploadFill');
const uploadPct = document.getElementById('uploadPct');
const uploadErrorEl = document.getElementById('uploadError');
let errorTimer = null;

function showUploadError(msg) {
  uploadErrorEl.textContent = msg;
  uploadErrorEl.classList.add('active');
  if (errorTimer) clearTimeout(errorTimer);
  errorTimer = setTimeout(() => uploadErrorEl.classList.remove('active'), 5000);
}

function updateProgress() {
  const total = Object.values(uploadState.perFile).reduce((s, f) => s + f.total, 0);
  const loaded = Object.values(uploadState.perFile).reduce((s, f) => s + f.loaded, 0);
  const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
  uploadFill.style.width = pct + '%';
  uploadPct.textContent = pct + '%';
  if (uploadState.active === 1) {
    const name = Object.values(uploadState.perFile).find(f => !f.done)?.name || '';
    uploadLabel.textContent = 'Uploading ' + name + '\u2026';
  } else {
    uploadLabel.textContent = 'Uploading ' + uploadState.active + ' files\u2026';
  }
}

function upload(file) {
  const fid = Math.random().toString(36).slice(2, 10);
  uploadState.perFile[fid] = { name: file.name, loaded: 0, total: file.size || 1, done: false };
  uploadState.active++;
  uploadStatusEl.classList.add('active');
  updateProgress();

  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', withGrant('/api/upload'));

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      uploadState.perFile[fid].loaded = e.loaded;
      uploadState.perFile[fid].total = e.total;
      updateProgress();
    }
  };

  function finish() {
    uploadState.perFile[fid].done = true;
    uploadState.perFile[fid].loaded = uploadState.perFile[fid].total;
    uploadState.active--;
    if (uploadState.active <= 0) {
      uploadState.active = 0;
      uploadStatusEl.classList.remove('active');
      uploadState.perFile = {};
    }
    updateProgress();
  }

  xhr.onload = async () => {
    finish();
    if (xhr.status >= 400) {
      let msg = 'Upload failed (' + xhr.status + ')';
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
      showUploadError(msg);
      return;
    }
    let data;
    try { data = JSON.parse(xhr.responseText); } catch { await load(); return; }
    await load();
    if (data.status === 'duplicate') {
      const msg = '"' + data.original_name + '" is a duplicate of "' + data.existing_name + '" (uploaded ' + fmtDate(data.uploaded_at) + ')';
      alert(msg);
      return;
    }
    startPoll(data.id);
  };

  xhr.onerror = () => {
    finish();
    showUploadError('Upload failed for "' + file.name + '" \u2014 network error.');
  };

  xhr.ontimeout = () => {
    finish();
    showUploadError('Upload timed out for "' + file.name + '".');
  };

  xhr.send(fd);
}
async function retryFile(id) {
  await fetch(withGrant(`/api/files/${id}/retry`), { method: 'POST' });
  await load();
}
async function delFile(id) {
  await fetch(withGrant(`/api/files/${id}`), { method: 'DELETE' });
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
if (!grant) {
  tbody.innerHTML = '<tr><td colspan="7" class="empty">Missing session grant. Request a new UI link from MCP.</td></tr>';
  dropZone.style.pointerEvents = 'none';
  dropZone.style.opacity = '0.6';
} else {
  load();
}
</script>
</body>
</html>"""


async def _recover_false_done_documents():
    """Mark stale in-progress documents as errors, then detect documents marked
    'done' in files.json but 'failed' in LightRAG's internal doc_status
    (embedding failures that were silently swallowed). Re-queue them."""
    log = logging.getLogger("ingest")

    # Mark any documents left in non-terminal state as errors (server restart)
    records = await _manifest.load()
    stale = [r for r in records if r["status"] not in {"done", "error"}]
    if stale:
        log.info("Recovery: %d stale documents from interrupted ingestion", len(stale))
        for record in stale:
            await _manifest.update(
                record["id"], status="error", error="Server restarted during ingestion"
            )
        log.info("Recovery: marked %d stale documents as error", len(stale))

    log.info("Recovery: starting false-done document scan...")
    try:
        rag = await _get_rag()
        await rag._ensure_lightrag_initialized()
    except Exception:
        log.exception("Recovery: failed to initialize RAG — skipping recovery")
        return
    log.info("Recovery: RAG initialized, scanning documents...")

    try:
        records = await _manifest.load()
        requeued = 0
        for record in records:
            if record.get("modality") in ("archive", "video"):
                continue
            # Recover: (a) false-done docs with failed LightRAG status, or
            #          (b) error docs where the error was an embedding/LightRAG failure
            if record["status"] == "done":
                doc_id = record.get("engine_doc_id") or record["id"]
                doc_entry = await rag.lightrag.doc_status.get_by_id(doc_id)
                if not doc_entry or not isinstance(doc_entry, dict):
                    continue
                if doc_entry.get("status") != "failed":
                    continue
            elif record["status"] == "error":
                err = (record.get("error", "") or "").lower()
                recoverable = (
                    "embedding" in err
                    or "lightrag" in err
                    or "server restarted" in err
                    or "vdb" in err
                    or "500 internal server error" in err
                )
                if not recoverable:
                    continue
                # Check if LightRAG actually has this doc as processed — if so,
                # the cascade may have fixed it already. Just update files.json.
                doc_id = record.get("engine_doc_id") or record["id"]
                doc_entry = await rag.lightrag.doc_status.get_by_id(doc_id)
                if (
                    doc_entry
                    and isinstance(doc_entry, dict)
                    and doc_entry.get("status") == "processed"
                    and doc_entry.get("chunks_list")
                ):
                    log.info(
                        "Recovery: %s (%s) already processed in LightRAG with %d chunks — marking done",
                        record["id"],
                        record.get("original_name", "?"),
                        len(doc_entry["chunks_list"]),
                    )
                    await _manifest.update(
                        record["id"], status="done", error=None, queryable=True
                    )
                    requeued += 1
                    continue
                # Not already processed — fall through to re-ingest
            else:
                continue
            doc_id = record.get("engine_doc_id") or record["id"]
            file_path = Path(record["path"])
            if not file_path.exists():
                log.warning("Cannot recover %s — file missing: %s", doc_id, file_path)
                continue
            log.warning(
                "Recovering document %s (%s)",
                record["id"],
                record.get("original_name", file_path.name),
            )
            # Clean any stale entry from LightRAG so it doesn't cascade-retry
            try:
                await rag.lightrag.adelete_by_doc_id(doc_id)
            except Exception:
                pass
            await _manifest.update(
                record["id"], status="pending", error=None, queryable=False
            )
            # Process sequentially to avoid overwhelming DMR with concurrent
            # LLM + embedding requests (causes 500 errors).
            log.info("Recovery: re-ingesting %s...", record["id"])
            await _ingest_background(record["id"], file_path)
            requeued += 1
        if requeued:
            log.info("Recovery: completed %d false-done documents", requeued)
        else:
            log.info("Recovery: no false-done documents found")
    except Exception:
        log.exception("Recovery: unexpected error during document recovery scan")


mcp = FastMCP("rag-anywhere", host="0.0.0.0")


def _safe_filename(file_id: str, original_name: str) -> str:
    return f"{file_id}_{re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)}"


def _build_file_record(
    *,
    file_id: str,
    safe_name: str,
    original_name: str,
    path: Path,
    collection_id: str,
    content_hash: str | None = None,
    parent_archive_id: str | None = None,
) -> dict:
    return {
        "id": file_id,
        "name": safe_name,
        "original_name": original_name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "uploaded_at": _iso_now(),
        "status": "uploading",
        "ingested_at": None,
        "error": None,
        "collection_id": collection_id,
        "modality": _classify_file(path),
        "ingest_path": "pending",
        "engine": None,
        "engine_doc_id": None,
        "queryable": False,
        "capabilities": [],
        "derived_paths": [],
        "fallback_reason": None,
        "image_processing": None,
        "content_hash": content_hash,
        "parent_archive_id": parent_archive_id,
    }


async def _store_upload_file(upload, collection_id: str) -> dict:
    from dedup import compute_file_hash

    file_id = uuid4().hex[:8]
    safe_name = _safe_filename(file_id, upload.filename)
    dest = _uploads_dir / safe_name
    async with aiofiles.open(dest, "wb") as f:
        await f.write(await upload.read())

    content_hash = await compute_file_hash(dest)
    existing = await _manifest.find_by_hash(content_hash)
    if existing:
        dest.unlink(missing_ok=True)
        return {
            "id": None,
            "status": "duplicate",
            "duplicate_of": existing["id"],
            "original_name": upload.filename,
            "existing_name": existing["original_name"],
            "uploaded_at": existing.get("uploaded_at"),
        }

    record = _build_file_record(
        file_id=file_id,
        safe_name=safe_name,
        original_name=upload.filename,
        path=dest,
        collection_id=collection_id,
        content_hash=content_hash,
    )
    await _manifest.add(record)
    _ingest_tasks[file_id] = asyncio.create_task(_ingest_background(file_id, dest))
    return {"id": file_id, "status": "uploading", "modality": record["modality"]}


@mcp.custom_route("/ui", methods=["GET"])
async def ui_index(request: Request) -> Response:
    # _, error = await _require_grant(request, "ui_session")
    # if error:
    #     return error
    html = _UI_HTML
    attu_link = '<a href="/database" class="btn-attu">&#9881; Milvus Dashboard</a>'
    html = html.replace("<!-- ATTU_LINK -->", attu_link)
    return HTMLResponse(html)


@mcp.custom_route("/api/files", methods=["GET"])
async def api_list_files(request: Request) -> Response:
    _, error = await _require_grant(request, "ui_session")
    if error:
        return error
    return JSONResponse(await _manifest.load())


@mcp.custom_route("/api/upload", methods=["POST"])
async def api_upload(request: Request) -> Response:
    grant, error = await _require_grant(request, "ui_session")
    if error:
        return error
    form = await request.form()
    upload = form["file"]
    collection_id = form.get(
        "collection_id",
        grant.get("metadata", {}).get("collection_id") or _default_collection_id,
    )
    result = await _store_upload_file(upload, collection_id)
    status_code = 200 if result.get("status") == "duplicate" else 202
    return JSONResponse(result, status_code=status_code)


@mcp.custom_route("/api/upload/{grant_token}", methods=["POST"])
async def api_upload_with_grant(request: Request) -> Response:
    token = request.path_params["grant_token"]
    try:
        grant = await _grants.get_valid(token, "upload")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=401)

    form = await request.form()
    upload = form["file"]
    collection_id = (
        grant.get("metadata", {}).get("collection_id") or _default_collection_id
    )
    result = await _store_upload_file(upload, collection_id)
    await _grants.mark_used(grant["id"])
    status_code = 200 if result.get("status") == "duplicate" else 202
    return JSONResponse(result, status_code=status_code)


@mcp.custom_route("/api/files/{file_id}/status", methods=["GET"])
async def api_file_status(request: Request) -> Response:
    _, error = await _require_grant(request, "ui_session")
    if error:
        return error
    record = await _manifest.get(request.path_params["file_id"])
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)
    info = {
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
            "video_stage",
        )
    }
    if record.get("modality") == "archive":
        info["archive_progress"] = record.get("archive_progress")
        info["child_ids"] = record.get("child_ids")
    return JSONResponse(info)


@mcp.custom_route("/api/files/{file_id}/retry", methods=["POST"])
async def api_retry_file(request: Request) -> Response:
    _, error = await _require_grant(request, "ui_session")
    if error:
        return error
    file_id = request.path_params["file_id"]
    record = await _manifest.get(file_id)
    if not record:
        return JSONResponse({"error": "not found"}, status_code=404)
    if file_id in _ingest_tasks:
        return JSONResponse({"error": "already processing"}, status_code=409)
    file_path = Path(record["path"])
    if not file_path.exists():
        return JSONResponse({"error": "source file missing"}, status_code=410)

    # Clean stale LightRAG state so it doesn't cascade-retry old failures
    doc_id = record.get("engine_doc_id") or file_id
    if record.get("engine") == "document":
        try:
            rag = await _get_rag()
            await rag._ensure_lightrag_initialized()
            await rag.lightrag.adelete_by_doc_id(doc_id)
        except Exception:
            pass

    await _manifest.update(file_id, status="pending", error=None, queryable=False)
    _ingest_tasks[file_id] = asyncio.create_task(_ingest_background(file_id, file_path))
    return JSONResponse({"ok": True, "status": "pending"})


@mcp.custom_route("/api/files/{file_id}", methods=["DELETE"])
async def api_delete_file(request: Request) -> Response:
    _, error = await _require_grant(request, "ui_session")
    if error:
        return error
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

    # Cascade delete children of an archive
    if record.get("modality") == "archive":
        children = await _manifest.remove_by_parent(file_id)
        for child in children:
            child_task = _ingest_tasks.pop(child["id"], None)
            if child_task:
                child_task.cancel()
            if child.get("engine") == "video":
                try:
                    await _video_engine.delete(child.get("engine_doc_id"))
                except Exception:
                    pass
            elif child.get("engine") == "document":
                try:
                    await _document_engine.delete(child.get("engine_doc_id"))
                except Exception:
                    pass
            for p in [child.get("path"), *(child.get("derived_paths") or [])]:
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass

    removed = await _manifest.remove(file_id)
    for path_str in [record.get("path"), *(record.get("derived_paths") or [])]:
        if not path_str:
            continue
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:
            logging.warning("Failed to delete local artifact: %s", path_str)
    return JSONResponse({"ok": bool(removed)})


def _vector_health() -> dict:
    """Probe Milvus connectivity."""
    uri = _get_env("MILVUS_URI", "http://localhost:19530")
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=uri)
        collections = client.list_collections()
        return {
            "backend": "milvus",
            "ok": True,
            "uri": uri,
            "collections": len(collections),
        }
    except Exception as exc:
        return {"backend": "milvus", "ok": False, "uri": uri, "error": str(exc)}


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health(request: Request) -> Response:
    video_health = await _video_engine.health()
    vision_health = await _probe_vision_model()
    vector_health = _vector_health()
    return JSONResponse(
        {
            "ok": True,
            "video_engine_enabled": _video_engine_enabled,
            "control_plane": {
                "base_url": _base_url(),
                "ui_session_ttl_seconds": int(
                    _get_env("UI_SESSION_TTL_SECONDS", "900")
                ),
                "upload_link_ttl_seconds": int(
                    _get_env("UPLOAD_LINK_TTL_SECONDS", "900")
                ),
            },
            "vector_db": vector_health,
            "video_engine": video_health,
            "vision_model": _get_env(
                "VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest"
            ),
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
    from dedup import compute_file_hash

    added = []
    skipped = []
    errors = []

    async def _ingest_single(file_path: Path):
        content_hash = await compute_file_hash(file_path)
        existing = await _manifest.find_by_hash(content_hash)
        if existing:
            skipped.append(
                {
                    "path": str(file_path),
                    "duplicate_of": existing["id"],
                    "original_name": existing["original_name"],
                }
            )
            return

        file_id = uuid4().hex[:8]
        record = _build_file_record(
            file_id=file_id,
            safe_name=file_path.name,
            original_name=file_path.name,
            path=file_path,
            collection_id=_default_collection_id,
            content_hash=content_hash,
        )
        record["status"] = "queued"
        await _manifest.add(record)
        await _ingest_background(file_id, file_path)
        added.append(str(file_path))

    for item in paths:
        path = Path(item)
        try:
            if path.is_dir():
                for child in path.rglob("*") if recursive else path.glob("*"):
                    if child.is_file():
                        await _ingest_single(child)
            elif path.is_file():
                await _ingest_single(path)
            else:
                errors.append(f"Path not found: {item}")
        except Exception as exc:
            errors.append(f"{item}: {exc}")
    return {"added": added, "skipped": skipped, "errors": errors}


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
            hits.extend(
                _transcript_hits(record, {"chunks": record["transcript_chunks"]})
            )
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
        video_records = [
            record for record in selected_records if record.get("engine") == "video"
        ]
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
async def create_ui_session_link(
    requested_for: str = "local-user",
    ttl_seconds: int = 900,
) -> dict:
    """Create a short-lived signed link for the browser UI."""
    grant = await _issue_ui_session(requested_for, ttl_seconds=ttl_seconds)
    return {
        "url": grant["url"],
        "expires_at": grant["expires_at"],
        "subject": grant["subject"],
        "grant_type": grant["type"],
    }


@mcp.tool()
async def create_upload_link(
    requested_for: str = "local-user",
    filename: str = "/absolute/path/to/file",
    collection_id: str | None = None,
    ttl_seconds: int = 900,
) -> dict:
    """Create a one-time upload URL for agents to push files over HTTP."""
    grant = await _issue_upload_link(
        requested_for,
        collection_id=collection_id,
        filename=filename,
        ttl_seconds=ttl_seconds,
    )
    curl_example = f"curl -f -X POST -F file=@{shlex.quote(filename)} {shlex.quote(grant['upload_url'])}"
    return {
        "upload_url": grant["upload_url"],
        "expires_at": grant["expires_at"],
        "subject": grant["subject"],
        "grant_type": grant["type"],
        "collection_id": grant["metadata"].get("collection_id"),
        "filename_hint": grant["metadata"].get("filename"),
        "curl_example": curl_example,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
