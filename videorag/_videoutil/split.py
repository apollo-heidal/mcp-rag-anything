import json
import os
import shutil
import subprocess
import time

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from .._utils import logger


def _slice_clip(video: VideoFileClip, start: float, end: float):
    if hasattr(video, "subclip"):
        return video.subclip(start, end)
    return video.subclipped(start, end)


def _probe_audio_stream(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return {
            "has_audio": None,
            "tool": "ffprobe",
            "error": "ffprobe not found",
        }
    except subprocess.CalledProcessError as exc:
        return {
            "has_audio": None,
            "tool": "ffprobe",
            "error": (exc.stderr or str(exc)).strip(),
        }

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return {
            "has_audio": False,
            "tool": "ffprobe",
            "codec": None,
            "channels": None,
            "sample_rate": None,
            "error": None,
        }

    stream = streams[0]
    return {
        "has_audio": True,
        "tool": "ffprobe",
        "codec": stream.get("codec_name"),
        "channels": stream.get("channels"),
        "sample_rate": stream.get("sample_rate"),
        "error": None,
    }


def _extract_audio_ffmpeg(video_path, audio_path, start, end):
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        audio_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg audio extraction failed").strip())
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("ffmpeg audio extraction produced no output")


def _extract_audio_moviepy(subvideo, audio_path):
    subaudio = subvideo.audio
    if subaudio is None:
        raise RuntimeError("MoviePy reported no audio object on segment")
    subaudio.write_audiofile(audio_path, codec="mp3", verbose=False, logger=None)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("MoviePy audio extraction produced no output")


def split_video(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
):
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)
    
    audio_probe = _probe_audio_stream(video_path)
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    audio_summary = {
        "attempted": bool(audio_probe.get("has_audio")),
        "segments_total": 0,
        "segments_with_audio": 0,
        "segments_failed": [],
        "strategy": "ffmpeg_then_moviepy",
        "probe": audio_probe,
    }
    with VideoFileClip(video_path) as video:
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))
        # if the last segment is shorter than 5 seconds, we merged it to the last segment
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]
        
        for start in tqdm(start_times, desc=f"Spliting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = _slice_clip(video, start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False)
            frame_times += start
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)}
            audio_summary["segments_total"] += 1

            audio_file_base_name = segment_index2name[f"{segment_index}"]
            audio_file = f'{audio_file_base_name}.{audio_output_format}'
            audio_path = os.path.join(video_segment_cache_path, audio_file)
            if audio_probe.get("has_audio"):
                try:
                    _extract_audio_ffmpeg(video_path, audio_path, start, end)
                    audio_summary["segments_with_audio"] += 1
                except Exception as ffmpeg_exc:
                    try:
                        _extract_audio_moviepy(subvideo, audio_path)
                        audio_summary["segments_with_audio"] += 1
                    except Exception as moviepy_exc:
                        failure = {
                            "segment_index": str(segment_index),
                            "start": start,
                            "end": end,
                            "ffmpeg_error": str(ffmpeg_exc),
                            "moviepy_error": str(moviepy_exc),
                        }
                        audio_summary["segments_failed"].append(failure)
                        logger.warning(
                            "Failed to extract audio for video %s (%s-%s). ffprobe_has_audio=%s ffmpeg=%s moviepy=%s",
                            video_name,
                            start,
                            end,
                            audio_probe.get("has_audio"),
                            failure["ffmpeg_error"],
                            failure["moviepy_error"],
                        )

            segment_index += 1

    return {
        "segment_index2name": segment_index2name,
        "segment_times_info": segment_times_info,
        "audio_probe": audio_probe,
        "audio_summary": audio_summary,
    }

def saving_video_segments(
    video_name,
    video_path,
    working_dir,
    segment_index2name,
    segment_times_info,
    error_queue,
    video_output_format='mp4',
):
    try:
        with VideoFileClip(video_path) as video:
            video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
            for index in tqdm(segment_index2name, desc=f"Saving Video Segments {video_name}"):
                start, end = segment_times_info[index]["timestamp"][0], segment_times_info[index]["timestamp"][1]
                video_file = f'{segment_index2name[index]}.{video_output_format}'
                subvideo = _slice_clip(video, start, end)
                subvideo.write_videofile(os.path.join(video_segment_cache_path, video_file), codec='libx264', verbose=False, logger=None)
    except Exception as e:
        error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
        raise RuntimeError
