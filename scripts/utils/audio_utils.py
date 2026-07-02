"""Shared audio helpers used across M3."""

import os
import subprocess
from pathlib import Path

import mutagen
from mutagen.mp3 import MP3
from mutagen.wave import WAVE


def get_audio_duration(path: str) -> float:
    """
    Return duration of an audio file in seconds.
    Supports .mp3, .wav, and most formats mutagen can read.
    """
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = Path(path).suffix.lower()
    if ext == ".mp3":
        audio = MP3(path)
    elif ext == ".wav":
        audio = WAVE(path)
    else:
        audio = mutagen.File(path)
        if audio is None:
            raise ValueError(f"Unsupported audio format: {path}")

    return audio.info.length


def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Convert any audio file to .wav using ffmpeg.
    Requires ffmpeg to be on PATH.

    Returns:
        Absolute path to the output .wav file.
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    return output_path


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
