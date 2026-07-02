"""
SadTalker local inference — generates a talking animation driven by a silent audio track.

First run: clones SadTalker repo, downloads checkpoints (~300 MB), installs dependencies.
Subsequent runs: skips setup.

CPU is supported via --cpu flag (slow, ~10-30 min per image). Use Colab for GPU speed.
"""

import os
import sys
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPO_DIR = MODELS_DIR / "sadtalker"
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"
GFPGAN_DIR = REPO_DIR / "gfpgan" / "weights"
SETUP_FLAG = REPO_DIR / ".setup_done"

# Checkpoint URLs from GitHub Releases (v0.0.2-rc) — the official source per download_models.sh
_ST_BASE = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
_CHECKPOINT_URLS = {
    "SadTalker_V0.0.2_256.safetensors": f"{_ST_BASE}/SadTalker_V0.0.2_256.safetensors",
    "SadTalker_V0.0.2_512.safetensors": f"{_ST_BASE}/SadTalker_V0.0.2_512.safetensors",
    "mapping_00229-model.pth.tar":       f"{_ST_BASE}/mapping_00229-model.pth.tar",
    "mapping_00109-model.pth.tar":       f"{_ST_BASE}/mapping_00109-model.pth.tar",
}
_GFPGAN_URLS = {
    "alignment_WFLW_4HG.pth":       "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    "detection_Resnet50_Final.pth":  "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "GFPGANv1.4.pth":               "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth":          "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}


def _download_url(url: str, dest_path: str) -> None:
    """Download a file from a public URL with a progress indicator."""
    import urllib.request
    print(f"  Downloading {Path(dest_path).name} ...", flush=True)
    urllib.request.urlretrieve(url, dest_path)


def _setup() -> None:
    if SETUP_FLAG.exists():
        return

    MODELS_DIR.mkdir(exist_ok=True)
    if not REPO_DIR.exists():
        print("[SadTalker] Cloning repo (one-time setup)...")
        subprocess.run(
            ["git", "clone", "https://github.com/OpenTalker/SadTalker", str(REPO_DIR)],
            check=True,
        )

    # SadTalker's requirements.txt has old version pins incompatible with Python 3.11.
    # We skip it and install only what inference actually needs, using modern versions.

    # Step 1: PyTorch CPU (not listed in SadTalker's requirements at all).
    print("[SadTalker] Installing PyTorch (CPU)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        check=True,
    )

    # Step 2: Runtime dependencies (modern compatible versions, no pinning).
    print("[SadTalker] Installing runtime dependencies...")
    _SADTALKER_DEPS = [
        "librosa",        # audio processing
        "resampy",        # audio resampling
        "pydub",          # audio I/O
        "yacs",           # config system
        "kornia",         # image processing ops
        "face_alignment", # face landmark detection
        "basicsr",        # base super-resolution (used by gfpgan)
        "gfpgan",         # face restoration (enhancer)
        "safetensors",    # checkpoint loading
        "av",             # video decode
        "scikit-image",   # image utilities
        "imageio",        # image/video I/O
        "imageio-ffmpeg", # ffmpeg backend for imageio
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + _SADTALKER_DEPS,
        check=True,
    )

    print("[SadTalker] Downloading checkpoints (~300 MB, one-time)...")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    GFPGAN_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in _CHECKPOINT_URLS.items():
        dest = str(CHECKPOINTS_DIR / fname)
        if not os.path.exists(dest):
            _download_url(url, dest)

    for fname, url in _GFPGAN_URLS.items():
        dest = str(GFPGAN_DIR / fname)
        if not os.path.exists(dest):
            _download_url(url, dest)

    SETUP_FLAG.touch()
    print("[SadTalker] Setup complete.")


def _make_silent_wav(duration_sec: float = 5.0, sample_rate: int = 16000) -> str:
    """Write a temporary silent WAV file and return its path."""
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
    use_enhancer: bool = True,
    cpu_only: bool = False,
) -> dict:
    """
    Generate a talking animation from a static headshot using SadTalker.

    Args:
        headshot_path: Path to source PNG/JPG headshot.
        output_dir:    Directory where output .mp4 will be saved.
        driver_audio:  Path to a WAV/MP3 file that drives the motion. Leave empty
                       to auto-generate a silent driver (produces natural idle motion).
        fps:           Output frames per second.
        duration_sec:  Duration of the silent driver if driver_audio is not provided.
        use_enhancer:  Apply GFPGAN face restoration (better quality, slower).
        cpu_only:      Force CPU inference (slow but works without a GPU).

    Returns:
        {
            "output_path": str,        # absolute path to output .mp4
            "duration":    float,
            "engine":      "sadtalker",
            "fps":         int,
            "metadata":    dict
        }
    """
    _setup()

    headshot_path = str(Path(headshot_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    if not os.path.exists(headshot_path):
        raise FileNotFoundError(f"Headshot not found: {headshot_path}")

    silent_tmp = None
    if not driver_audio:
        silent_tmp = _make_silent_wav(duration_sec)
        driver_audio = silent_tmp
    else:
        driver_audio = str(Path(driver_audio).resolve())
        if not os.path.exists(driver_audio):
            raise FileNotFoundError(f"Driver audio not found: {driver_audio}")

    os.makedirs(output_dir, exist_ok=True)

    name = Path(headshot_path).stem
    output_path = os.path.join(output_dir, f"{name}_talking.mp4")
    sadtalker_result_dir = str(REPO_DIR / "results" / name)

    cmd = [
        sys.executable, str(REPO_DIR / "inference.py"),
        "--driven_audio", driver_audio,
        "--source_image", headshot_path,
        "--result_dir", sadtalker_result_dir,
        "--still",
        "--preprocess", "full",
    ]
    if use_enhancer:
        cmd += ["--enhancer", "gfpgan"]
    if cpu_only:
        cmd.append("--cpu")

    print(f"[SadTalker] Processing {name} ...")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))

    # Retry without enhancer if GFPGAN caused the failure
    if result.returncode != 0 and use_enhancer and "gfpgan" in result.stderr.lower():
        print("[SadTalker] GFPGAN failed; retrying without enhancer...")
        cmd_no_enhancer = [c for c in cmd if c not in ("--enhancer", "gfpgan")]
        result = subprocess.run(cmd_no_enhancer, capture_output=True, text=True, cwd=str(REPO_DIR))

    if result.returncode != 0:
        raise RuntimeError(
            f"SadTalker inference failed for '{name}'.\n"
            f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}"
        )

    # Find the generated output and copy to standardized path
    candidates = glob.glob(os.path.join(sadtalker_result_dir, "**", "*.mp4"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No output video found in {sadtalker_result_dir}")
    shutil.copy2(max(candidates, key=os.path.getmtime), output_path)

    if silent_tmp:
        os.unlink(silent_tmp)

    print(f"[SadTalker] Done → {output_path}")
    return {
        "output_path": output_path,
        "duration": duration_sec,
        "engine": "sadtalker",
        "fps": fps,
        "metadata": {
            "driver_audio": driver_audio if not silent_tmp else "silent",
            "source": headshot_path,
            "enhancer": use_enhancer,
            "cpu_only": cpu_only,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate talking animation with SadTalker (local CPU/GPU)"
    )
    parser.add_argument("--headshot", required=True, help="Path to source PNG headshot")
    parser.add_argument("--audio", default="", help="Path to driver audio (leave blank for silent)")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "m1" / "sadtalker"),
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no_enhancer", action="store_true", help="Skip GFPGAN face restoration")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only inference")
    args = parser.parse_args()

    result = generate_animation(
        args.headshot, args.output_dir,
        driver_audio=args.audio, fps=args.fps,
        use_enhancer=not args.no_enhancer, cpu_only=args.cpu,
    )
    print(f"\nResult: {result}")
