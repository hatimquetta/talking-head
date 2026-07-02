"""
M1 Wav2Lip — static headshot → talking animation (accurate lip sync)
Run: python test/M1_wav2lip.py --headshot headshots/person_01.png

Pipeline: PNG → looped MP4 (FFmpeg) → Wav2Lip inference with silent audio → output MP4
Note: face area outside the mouth may appear softer than LivePortrait — known Wav2Lip trade-off.
"""

import os, sys, glob, subprocess, threading, time, tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
REPO_DIR        = PROJECT_ROOT / "models" / "wav2lip"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "m1" / "wav2lip"
CHECKPOINT_PATH = REPO_DIR / "checkpoints" / "wav2lip_gan.pth"

# ── Checkpoint URL ─────────────────────────────────────────────────────────────
# HuggingFace mirror — Google Drive is rate-limited and unreliable
_CHECKPOINT_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m:02d}m {s:02d}s"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"

def _step(msg: str):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _run(label: str, cmd: list, **kwargs):
    print(f"  -> {label} ...", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, **kwargs)
    print(f"     done in {_fmt(time.time() - t0)}", flush=True)

def _progress_printer(stop_event: threading.Event, start: float, interval: int = 30):
    while not stop_event.wait(interval):
        print(f"  [still running... {_fmt(time.time() - start)} elapsed]", flush=True)

def _patch_wav2lip():
    """librosa.filters.mel() made positional args keyword-only in librosa 0.9+."""
    target = REPO_DIR / "audio.py"
    if not target.exists():
        return
    OLD = "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
    NEW = "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"
    txt = target.read_text(encoding="utf-8")
    if OLD in txt:
        target.write_text(txt.replace(OLD, NEW), encoding="utf-8")
        print("  Patched audio.py (librosa 0.9+ keyword-only args)")


# ── Step 1: Clone repo + install packages (runs once) ─────────────────────────
def setup():
    flag = REPO_DIR / ".setup_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Setup already done, skipping.")
        return

    _step("SETUP — one-time install (takes a few minutes)")

    if not REPO_DIR.exists():
        _run("Cloning Wav2Lip repo",
             ["git", "clone", "https://github.com/Rudrabha/Wav2Lip", str(REPO_DIR)])

    # Wav2Lip's requirements.txt pins torch==1.1.0 from 2019 — incompatible with Python 3.11.
    # Skip it and install only what inference needs with modern versions.
    _run("Installing PyTorch (CPU build)", [sys.executable, "-m", "pip", "install",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu"])

    _run("Installing runtime dependencies", [sys.executable, "-m", "pip", "install",
         "librosa", "face_alignment"])

    flag.touch()
    print("  Setup complete.")


# ── Step 2: Download checkpoint from HuggingFace (~400 MB, runs once) ─────────
def download_checkpoint():
    flag = REPO_DIR / ".checkpoint_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Checkpoint already downloaded, skipping.")
        return

    _step("CHECKPOINT — downloading wav2lip_gan.pth (~400 MB) from HuggingFace")
    import urllib.request

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        print(f"  downloading wav2lip_gan.pth ...", end=" ", flush=True)
        t0 = time.time()
        urllib.request.urlretrieve(_CHECKPOINT_URL, str(CHECKPOINT_PATH))
        size_mb = CHECKPOINT_PATH.stat().st_size / 1024 / 1024
        print(f"{size_mb:.0f} MB in {_fmt(time.time() - t0)}", flush=True)
    else:
        print(f"  already have wav2lip_gan.pth", flush=True)

    flag.touch()
    print("  Checkpoint ready.")


# ── Helper: image → looped video ───────────────────────────────────────────────
def _image_to_video(image_path: str, output_video: str, duration: float, fps: int) -> str:
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-t", str(duration),
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg image-to-video failed:\n{result.stderr}")
    return output_video


# ── Step 3: Run inference ──────────────────────────────────────────────────────
def generate_animation(headshot_path: str,
                       output_dir: str = str(OUTPUT_DIR),
                       driver_audio: str = "",
                       fps: int = 25,
                       duration_sec: float = 5.0) -> dict:
    """
    headshot_path : path to source PNG/JPG
    output_dir    : where to save the output .mp4
    driver_audio  : WAV/MP3 to drive lip sync (blank = auto-generate silent driver)
    fps           : output frames per second
    duration_sec  : duration when using silent driver

    Returns dict with output_path, engine, fps, metadata.
    """
    setup()
    download_checkpoint()
    _patch_wav2lip()

    headshot_path = str(Path(headshot_path).resolve())
    output_dir    = str(Path(output_dir).resolve())

    if not os.path.exists(headshot_path):
        raise FileNotFoundError(f"Headshot not found: {headshot_path}")

    name       = Path(headshot_path).stem
    final_path = os.path.join(output_dir, f"{name}_talking.mp4")
    os.makedirs(output_dir, exist_ok=True)

    # Temp files for intermediate video and silent audio
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    silent_tmp = None

    _step(f"INFERENCE — {name}  |  {fps}fps  |  {duration_sec}s")
    print(f"  source  : {headshot_path}")

    _step("IMAGE -> VIDEO — converting static image to looped video clip")
    t0 = time.time()
    _image_to_video(headshot_path, tmp_video, duration_sec, fps)
    print(f"  FFmpeg done in {_fmt(time.time() - t0)}", flush=True)

    if not driver_audio:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        silence = np.zeros(int(16000 * duration_sec), dtype=np.int16)
        wavfile.write(tmp.name, 16000, silence)
        silent_tmp   = tmp.name
        driver_audio = silent_tmp
        print(f"  audio   : silent (generated {duration_sec}s @ 16kHz)", flush=True)
    else:
        driver_audio = str(Path(driver_audio).resolve())
        print(f"  audio   : {driver_audio}", flush=True)

    print(f"  NOTE: CPU inference is slow (~5-20 min). Progress below every 30s.")

    cmd = [
        sys.executable, str(REPO_DIR / "inference.py"),
        "--checkpoint_path", str(CHECKPOINT_PATH),
        "--face",    tmp_video,
        "--audio",   driver_audio,
        "--outfile", final_path,
        "--fps",     str(fps),
    ]

    start_time = time.time()
    stop_event = threading.Event()
    ticker     = threading.Thread(target=_progress_printer, args=(stop_event, start_time, 30), daemon=True)
    ticker.start()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))

    stop_event.set()
    elapsed = time.time() - start_time

    # Cleanup temp files — wrapped in try/except because Windows holds file handles
    # briefly after subprocess.run() returns, causing PermissionError on fast cleanup.
    for tmp in (tmp_video, silent_tmp):
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass

    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip inference failed after {_fmt(elapsed)}.\n"
                           f"STDERR:\n{result.stderr[-2000:]}")

    if not os.path.exists(final_path):
        raise RuntimeError(f"No output mp4 found at expected path: {final_path}")

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"\n  DONE in {_fmt(elapsed)}")
    print(f"  Output : {final_path}  ({size_mb:.2f} MB)")

    return {
        "output_path": final_path,
        "engine":      "wav2lip",
        "fps":         fps,
        "elapsed_sec": round(elapsed, 1),
        "metadata":    {"source": headshot_path, "duration_sec": duration_sec,
                        "checkpoint": str(CHECKPOINT_PATH)},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Wav2Lip: headshot -> talking animation")
    p.add_argument("--headshot",   required=True,  help="Path to source PNG (e.g. headshots/person_01.png)")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    p.add_argument("--audio",      default="",     help="Path to driver WAV/MP3 (blank = silent)")
    p.add_argument("--fps",        type=int,   default=25)
    p.add_argument("--duration",   type=float, default=5.0, help="Duration in seconds (silent driver only)")
    args = p.parse_args()

    overall_start = time.time()
    print(f"{'='*60}")
    print(f"  Wav2Lip M1 Pipeline")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = generate_animation(
        args.headshot, args.output_dir,
        driver_audio=args.audio,
        fps=args.fps,
        duration_sec=args.duration,
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {_fmt(time.time() - overall_start)}")
    print(f"  Output: {result['output_path']}")
    print(f"{'='*60}")
