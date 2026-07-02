"""edge-tts (Microsoft Neural TTS) — free, no API key.

generate_audio(text, output_dir, voice) -> dict, following the project's
function contract. Reused by the M3 playback controller, which sets the
speaking-segment length S equal to the returned `duration`.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import edge_tts

try:  # shared helper (preferred); fall back to ffprobe if run standalone
    from scripts.utils.audio_utils import get_audio_duration
except Exception:  # pragma: no cover
    get_audio_duration = None


def _slugify(text: str, limit: int = 40) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_") else "" for c in text).strip()
    return ("_".join(keep.split()) or "utterance")[:limit]


def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def generate_audio(
    text: str,
    output_dir: str,
    voice: str = "en-US-JennyNeural",
    filename: Optional[str] = None,
) -> dict:
    """Synthesize `text` to speech with edge-tts (saved as .mp3).

    Args:
        text:       the words to speak (non-empty).
        output_dir: directory to write the audio into (created if needed).
        voice:      edge-tts neural voice id, e.g. "en-US-JennyNeural".
        filename:   optional output filename; auto-derived from `text` if None.

    Returns:
        {
          "output_path": str,    # absolute path to the .mp3
          "duration":    float,  # seconds — feed this in as M2's S
          "engine":      "edge-tts",
          "voice":       str,
          "metadata":    {"text": str},
        }
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    os.makedirs(output_dir, exist_ok=True)

    name = filename or f"{_slugify(text)}.mp3"
    if not name.lower().endswith(".mp3"):
        name += ".mp3"
    out_path = str(Path(output_dir, name).resolve())

    async def _run() -> None:
        await edge_tts.Communicate(text, voice).save(out_path)

    try:
        asyncio.run(_run())
    except RuntimeError:
        # already inside an event loop (e.g. Jupyter) — use a fresh one
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"edge-tts produced no audio at {out_path}")

    duration = get_audio_duration(out_path) if get_audio_duration else _ffprobe_duration(out_path)
    return {
        "output_path": out_path,
        "duration": float(duration),
        "engine": "edge-tts",
        "voice": voice,
        "metadata": {"text": text},
    }


if __name__ == "__main__":
    res = generate_audio(
        "Hello, I am your virtual assistant. How can I help you today?",
        output_dir="outputs/m3/edge_tts",
    )
    print(json.dumps(res, indent=2))
