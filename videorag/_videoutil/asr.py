import logging
import os

_ASR_MODEL = None


def _whisper_api_base() -> str:
    return os.environ.get("WHISPER_API_BASE", "")


def _get_asr_model():
    """Fallback: load local PyTorch Whisper when no API is configured."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL

    logging.warning(
        "WHISPER_API_BASE not configured — falling back to local "
        "PyTorch Whisper on CPU. This is significantly slower than "
        "using a whisper.cpp server with Metal/GPU acceleration."
    )

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_name = os.environ.get("AUDIO_TRANSCRIBE_MODEL", "openai/whisper-small")
    device = os.environ.get("AUDIO_TRANSCRIBE_DEVICE", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    _ASR_MODEL = (model, processor)
    return _ASR_MODEL


def _transcribe_segment_via_api(audio_file):
    """Transcribe a single audio file using the whisper API."""
    import requests

    api_base = _whisper_api_base()
    url = f"{api_base}/inference"
    with open(audio_file, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(audio_file), f, "audio/mpeg")},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
            },
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()


def _transcribe_segment_local(audio_file):
    """Fallback: transcribe using local PyTorch Whisper model."""
    from transformers.pipelines.audio_utils import ffmpeg_read

    model, processor = _get_asr_model()
    device = next(model.parameters()).device

    with open(audio_file, "rb") as f:
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


def _parse_api_result(data):
    """Parse whisper API verbose_json response into timestamped chunk strings."""
    chunks = []
    for seg in data.get("segments", []):
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()
        if start is not None and end is not None and text:
            chunks.append(f"[{start:.2f}s -> {end:.2f}s] {text}")
    if not chunks:
        text = data.get("text", "").strip()
        if text:
            chunks.append(text)
    return chunks


def _parse_local_result(decoded):
    """Parse local Whisper decoded output into timestamped chunk strings."""
    chunks = []
    if decoded:
        entry = decoded[0]
        if isinstance(entry, dict):
            for offset in entry.get("offsets", []):
                ts = offset.get("timestamp", (None, None))
                start, end = ts if ts else (None, None)
                if start is None or end is None:
                    continue
                chunks.append(f"[{start:.2f}s -> {end:.2f}s] {offset.get('text', '').strip()}")
            if not chunks:
                text = entry.get("text", "").strip()
                if text:
                    chunks.append(text)
        else:
            text = str(entry).strip()
            if text:
                chunks.append(text)
    return chunks


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    use_api = bool(_whisper_api_base())
    cache_path = os.path.join(working_dir, "_cache", video_name)

    transcripts = {}
    summary = {
        "segments_total": len(segment_index2name),
        "segments_with_audio_files": 0,
        "segments_with_transcripts": 0,
        "segments_failed": [],
    }
    for index in segment_index2name:
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        summary["segments_with_audio_files"] += 1

        try:
            if use_api:
                data = _transcribe_segment_via_api(audio_file)
                chunks = _parse_api_result(data)
            else:
                decoded = _transcribe_segment_local(audio_file)
                chunks = _parse_local_result(decoded)
        except Exception as exc:
            logging.warning("ASR failed for %s: %s", audio_file, exc)
            transcripts[index] = ""
            summary["segments_failed"].append({"segment_index": index, "error": str(exc)})
            continue

        transcripts[index] = "\n".join(chunks) if chunks else ""
        if transcripts[index].strip():
            summary["segments_with_transcripts"] += 1

    return transcripts, summary
