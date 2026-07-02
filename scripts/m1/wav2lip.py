"""
Wav2Lip local inference — generates a talking animation with accurate lip sync.

Pipeline: static image → short looped video (FFmpeg) → Wav2Lip inference with silent audio.

First run: clones Wav2Lip repo, downloads wav2lip_gan.pth (~400 MB), installs dependencies.
Subsequent runs: skips setup.

Note: Wav2Lip's face area outside the mouth region may appear softer/blurrier than
LivePortrait or SadTalker — this is a known trade-off for its superior lip sync accuracy.

CPU is supported (slow). Use the Colab notebook for GPU speed.
"""

import os
import sys
import glob
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPO_DIR = MODELS_DIR / "wav2lip"
CHECKPOINT_PATH = REPO_DIR / "checkpoints" / "wav2lip_gan.pth"
SETUP_FLAG = REPO_DIR / ".setup_done"

# wav2lip_gan.pth — HuggingFace mirror (Google Drive is rate-limited)
_CHECKPOINT_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"


def _setup() -> None:
    if SETUP_FLAG.exists():
        return

    MODELS_DIR.mkdir(exist_ok=True)
    if not REPO_DIR.exists():
        print("[Wav2Lip] Cloning repo (one-time setup)...")
        subprocess.run(
            ["git", "clone", "https://github.com/Rudrabha/Wav2Lip", str(REPO_DIR)],
            check=True,
        )

    # Wav2Lip's requirements.txt pins torch==1.1.0, numpy==1.17.1, librosa==0.7.0 —
    # all from 2019 and incompatible with Python 3.11. Skip it entirely and install
    # only what inference actually needs, using modern compatible versions.

    # Step 1: PyTorch CPU.
    print("[Wav2Lip] Installing PyTorch (CPU)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        check=True,
    )

    # Step 2: Runtime dependencies (modern versions, no pinning).
    print("[Wav2Lip] Installing runtime dependencies...")
    _WAV2LIP_DEPS = [
        "librosa",        # audio processing (numba pulled in automatically as a dep)
        "face_alignment", # face detection & landmarks
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + _WAV2LIP_DEPS,
        check=True,
    )

    print("[Wav2Lip] Downloading checkpoint wav2lip_gan.pth (~400 MB, one-time)...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        import urllib.request
        print(f"  From: {_CHECKPOINT_URL}")
        urllib.request.urlretrieve(_CHECKPOINT_URL, str(CHECKPOINT_PATH))
        print(f"  Saved to: {CHECKPOINT_PATH}")

    SETUP_FLAG.touch()
    print("[Wav2Lip] Setup complete.")


def _image_to_video(image_path: str, output_video: str, duration: float = 5.0, fps: int = 25) -> str:
    """Convert a static image to a short looped video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", str(duration),
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg image-to-video failed:\n{result.stderr}")
    return output_video


def _make_silent_wav(duration_sec: float = 5.0, sample_rate: int = 16000) -> str:
    """Write a temporary silent WAV and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    silence = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)
    wavfile.write(tmp.name, sample_rate, silence)
    return tmp.name


def generate_animation(
    headshot_path: str,
    output_dir: str,
    driver_audio: str = "",
    fps: int = 25,
    duration_sec: float = 5.0,
) -> dict:
    """
    Generate a talking animation from a static headshot using Wav2Lip.

    Args:
        headshot_path: Path to source PNG/JPG headshot.
        output_dir:    Directory where output .mp4 will be saved.
        driver_audio:  Path to a WAV/MP3 file. Leave empty for a silent driver.
        fps:           Output frames per second.
        duration_sec:  Duration when using silent driver.

    Returns:
        {
            "output_path": str,       # absolute path to output .mp4
            "duration":    float,
            "engine":      "wav2lip",
            "fps":         int,
            "metadata":    dict
        }

    Raises:
        FileNotFoundError: if headshot does not exist.
        RuntimeError:      if FFmpeg or Wav2Lip inference fails.
    """
    _setup()

    headshot_path = str(Path(headshot_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    if not os.path.exists(headshot_path):
        raise FileNotFoundError(f"Headshot not found: {headshot_path}")

    os.makedirs(output_dir, exist_ok=True)

    name = Path(headshot_path).stem
    output_path = os.path.join(output_dir, f"{name}_talking.mp4")

    # Step 1: image → short video
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    _image_to_video(headshot_path, tmp_video, duration=duration_sec, fps=fps)

    # Step 2: silent driver audio
    silent_tmp = None
    if not driver_audio:
        silent_tmp = _make_silent_wav(duration_sec)
        driver_audio = silent_tmp
    else:
        driver_audio = str(Path(driver_audio).resolve())
        if not os.path.exists(driver_audio):
            raise FileNotFoundError(f"Driver audio not found: {driver_audio}")

    print(f"[Wav2Lip] Processing {name} ...")

    cmd = [
        sys.executable, str(REPO_DIR / "inference.py"),
        "--checkpoint_path", str(CHECKPOINT_PATH),
        "--face", tmp_video,
        "--audio", driver_audio,
        "--outfile", output_path,
        "--fps", str(fps),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))

    # Cleanup temporaries
    os.unlink(tmp_video)
    if silent_tmp:
        os.unlink(silent_tmp)

    if result.returncode != 0:
        raise RuntimeError(
            f"Wav2Lip inference failed for '{name}'.\n"
            f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output file not found after inference: {output_path}")

    print(f"[Wav2Lip] Done → {output_path}")
    return {
        "output_path": output_path,
        "duration": duration_sec,
        "engine": "wav2lip",
        "fps": fps,
        "metadata": {
            "driver_audio": driver_audio if not silent_tmp else "silent",
            "source": headshot_path,
            "checkpoint": str(CHECKPOINT_PATH),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate talking animation with Wav2Lip (local CPU/GPU)"
    )
    parser.add_argument("--headshot", required=True, help="Path to source PNG headshot")
    parser.add_argument("--audio", default="", help="Path to driver audio (leave blank for silent)")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "m1" / "wav2lip"),
    )
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    result = generate_animation(args.headshot, args.output_dir, driver_audio=args.audio, fps=args.fps)
    print(f"\nResult: {result}")
