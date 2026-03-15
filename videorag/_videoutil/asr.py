import logging
import os

import torch
from transformers import pipeline

_ASR_PIPELINE = None


def _get_asr_pipeline():
    global _ASR_PIPELINE
    if _ASR_PIPELINE is not None:
        return _ASR_PIPELINE

    model_name = os.environ.get("AUDIO_TRANSCRIBE_MODEL", "openai/whisper-small")
    device = os.environ.get("AUDIO_TRANSCRIBE_DEVICE", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = 0
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
    elif device == "cpu":
        device = -1

    _ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    return _ASR_PIPELINE


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    asr = _get_asr_pipeline()
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
            result = asr(
                audio_file,
                return_timestamps=True,
                chunk_length_s=int(os.environ.get("AUDIO_TRANSCRIBE_CHUNK_SECONDS", "30")),
                batch_size=int(os.environ.get("AUDIO_TRANSCRIBE_BATCH_SIZE", "8")),
            )
        except Exception as exc:
            logging.warning("ASR failed for %s: %s", audio_file, exc)
            transcripts[index] = ""
            summary["segments_failed"].append({"segment_index": index, "error": str(exc)})
            continue

        chunks = []
        for chunk in result.get("chunks", []) or []:
            start, end = chunk.get("timestamp") or (None, None)
            if start is None or end is None:
                continue
            chunks.append(f"[{start:.2f}s -> {end:.2f}s] {chunk.get('text', '').strip()}")
        transcripts[index] = "\n".join(chunks) if chunks else (result.get("text") or "").strip()
        if transcripts[index].strip():
            summary["segments_with_transcripts"] += 1

    return transcripts, summary
