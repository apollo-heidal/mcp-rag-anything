import base64
import os

import httpx
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image


def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    return [Image.fromarray(v.astype("uint8")).resize((1280, 720)) for v in frames]


def _image_to_data_url(image: Image.Image) -> str:
    import io

    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _chat_with_vision(prompt: str, frames: list[Image.Image]) -> str:
    api_base = os.environ.get(
        "VISION_API_BASE",
        os.environ.get("LLM_API_BASE", "http://localhost:12434/engines/v1"),
    ).rstrip("/")
    model = os.environ.get("VISION_MODEL", "docker.io/local/qwen3.5-2b-vlm:latest")
    api_key = os.environ.get(
        "VISION_API_KEY",
        os.environ.get("OPENAI_API_KEY", "docker-model-runner"),
    )

    content = [{"type": "image_url", "image_url": {"url": _image_to_data_url(frame), "detail": "high"}} for frame in frames]
    content.append({"type": "text", "text": prompt})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 1024,
    }
    response = httpx.post(f"{api_base}/chat/completions", json=payload, headers=headers, timeout=180.0)
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Vision model returned no choices")
    return (choices[0].get("message") or {}).get("content") or ""


def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info):
    caption_result = {}
    with VideoFileClip(video_path) as video:
        for index in segment_index2name:
            frame_times = segment_times_info[index]["frame_times"]
            video_frames = encode_video(video, frame_times)
            segment_transcript = transcripts[index]
            query = (
                "The transcript of the current video segment is:\n"
                f"{segment_transcript}\n\n"
                "Provide a concise visual description of the segment in English."
            )
            segment_caption_text = _chat_with_vision(query, video_frames)
            caption_result[index] = segment_caption_text.replace("\n", " ").strip()
    return caption_result


def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = "-".join(segment_name.split("-")[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments


def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    caption_result = {}
    for this_segment in retrieved_segments:
        video_name = "_".join(this_segment.split("_")[:-1])
        index = this_segment.split("_")[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split("-")
        start, end = eval(timestamp[0]), eval(timestamp[1])
        with VideoFileClip(video_path) as video:
            frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
            video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        query = (
            "The transcript of the current video segment is:\n"
            f"{segment_transcript}\n\n"
            "Provide a detailed visual description that is specifically useful for answering this request:\n"
            f"{refine_knowledge}"
        )
        this_caption = _chat_with_vision(query, video_frames).replace("\n", " ").strip()
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"

    return caption_result
