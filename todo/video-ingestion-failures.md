# Video ingestion failures (5 files)

## Problem
5 video files failed ingestion. These use a different pipeline path (video engine → ffmpeg audio extraction → whisper transcription → LightRAG) and likely hit different failure modes than the document pipeline.

## Investigation needed
- Check if failures are in the ffmpeg extraction step, whisper transcription, or LightRAG ingestion
- Review error messages in files.json for each of the 5 files
- Determine if the long-audio chunking fix (now applied to `_transcribe_audio_via_api`) also covers the video engine's transcription path, or if that path calls whisper differently

## Likely fix
The video engine at `_extract_audio_from_video()` already segments into 30s chunks, so the whisper long-audio issue shouldn't apply. Failures are more likely:
- Corrupt/unsupported video format
- ffmpeg extraction failure
- GPU contention during concurrent processing
