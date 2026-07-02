"""Shared video helpers used across M1, M2, and M3."""

import os
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from moviepy import (
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)


def get_video_duration(path: str) -> float:
    """Return duration of a video file in seconds."""
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        raise ValueError(f"Could not read FPS from: {path}")
    return frame_count / fps


def loop_video_to_duration(src_path: str, target_sec: float, output_path: str) -> str:
    """
    Loop or trim a video clip to exactly target_sec seconds.

    Returns:
        Absolute path to the output file.
    """
    src_path = str(Path(src_path).resolve())
    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clip = VideoFileClip(src_path)
    src_duration = clip.duration

    if src_duration >= target_sec:
        result = clip.subclipped(0, target_sec)
    else:
        n_loops = math.ceil(target_sec / src_duration)
        looped = concatenate_videoclips([clip] * n_loops)
        result = looped.subclipped(0, target_sec)

    result.write_videofile(output_path, logger=None)
    clip.close()
    return output_path


def static_image_to_clip(image_path: str, duration_sec: float, fps: int = 25) -> ImageClip:
    """
    Convert a static image (.png or .jpg) to a MoviePy ImageClip.

    Returns:
        ImageClip of the specified duration.
    """
    image_path = str(Path(image_path).resolve())
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return ImageClip(image_path, duration=duration_sec).with_fps(fps)


def crossfade_clips(clip_a, clip_b, n_frames: int, fps: int = 25):
    """
    Blend the tail of clip_a into the head of clip_b over n_frames.
    Returns a single concatenated clip with the crossfade baked in.
    """
    fade_duration = n_frames / fps

    # Trim the overlap region from each clip
    tail_a = clip_a.subclipped(clip_a.duration - fade_duration, clip_a.duration)
    head_b = clip_b.subclipped(0, fade_duration)

    # Build per-frame blend
    frames_a = [tail_a.get_frame(t) for t in np.linspace(0, fade_duration, n_frames, endpoint=False)]
    frames_b = [head_b.get_frame(t) for t in np.linspace(0, fade_duration, n_frames, endpoint=False)]

    blended_frames = []
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        alpha = i / n_frames
        blended_frames.append((fa * (1 - alpha) + fb * alpha).astype(np.uint8))

    from moviepy import VideoClip

    def make_frame(t):
        idx = int(t * fps)
        idx = min(idx, len(blended_frames) - 1)
        return blended_frames[idx]

    transition = VideoClip(make_frame, duration=fade_duration).with_fps(fps)

    body_a = clip_a.subclipped(0, clip_a.duration - fade_duration)
    body_b = clip_b.subclipped(fade_duration, clip_b.duration)

    return concatenate_videoclips([body_a, transition, body_b])


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
