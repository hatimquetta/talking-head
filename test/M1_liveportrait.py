"""
M1 LivePortrait — static headshot → talking animation
Run: python test/M1_liveportrait.py --headshot headshots/person_01.png
"""

import os, sys, glob, pickle, subprocess, threading, time
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_DIR     = PROJECT_ROOT / "models" / "liveportrait"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "m1" / "liveportrait"
DRIVING_DIR  = REPO_DIR / "assets" / "examples" / "driving"
WEIGHTS_DIR  = REPO_DIR / "pretrained_weights"

# ── HuggingFace weights (KwaiVGI org renamed to KlingTeam) ────────────────────
HF_REPO = "KlingTeam/LivePortrait"
WEIGHT_FILES = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "liveportrait/landmark.onnx",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt(seconds: float) -> str:
    """Format seconds as 1m 23s or 45s."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

def _step(msg: str):
    """Print a timestamped status line."""
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _run(label: str, cmd: list, **kwargs):
    """Run a subprocess with before/after timing."""
    print(f"  -> {label} ...", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, **kwargs)
    print(f"     done in {_fmt(time.time() - t0)}", flush=True)

def _progress_printer(stop_event: threading.Event, start: float, interval: int = 30):
    """Background thread: prints elapsed time every `interval` seconds."""
    while not stop_event.wait(interval):
        print(f"  [still running... {_fmt(time.time() - start)} elapsed]", flush=True)


# ── Step 1: Clone repo + install packages (runs once) ─────────────────────────
def setup():
    flag = REPO_DIR / ".setup_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Setup already done, skipping.")
        return

    _step("SETUP — one-time install (takes a few minutes)")

    if not REPO_DIR.exists():
        _run("Cloning LivePortrait repo", ["git", "clone",
             "https://github.com/KwaiVGI/LivePortrait", str(REPO_DIR)])

    # PyTorch CPU — not in LivePortrait's requirements so we install manually
    _run("Installing PyTorch (CPU build)", [sys.executable, "-m", "pip", "install",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu"])

    # requirements_base.txt only — the full requirements.txt pulls onnxruntime-gpu (needs CUDA)
    _run("Installing base requirements", [sys.executable, "-m", "pip", "install",
         "-r", str(REPO_DIR / "requirements_base.txt")])

    # CPU onnxruntime replaces onnxruntime-gpu
    _run("Installing onnxruntime (CPU) + huggingface_hub", [sys.executable, "-m", "pip", "install",
         "onnxruntime", "transformers==4.38.0", "huggingface_hub"])

    flag.touch()
    print(f"  Setup complete.")


# ── Step 2: Download model weights from HuggingFace (~600 MB, runs once) ──────
def download_weights():
    flag = REPO_DIR / ".weights_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Weights already downloaded, skipping.")
        return

    _step("WEIGHTS — downloading 6 files from HuggingFace KlingTeam/LivePortrait (~600 MB total)")

    from huggingface_hub import hf_hub_download
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for f in WEIGHT_FILES:
        dest = WEIGHTS_DIR / f
        if dest.exists():
            print(f"  already have {Path(f).name}", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {Path(f).name} ...", end=" ", flush=True)
        t0 = time.time()
        hf_hub_download(repo_id=HF_REPO, filename=f, local_dir=str(WEIGHTS_DIR))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"{size_mb:.0f} MB in {_fmt(time.time() - t0)}", flush=True)

    flag.touch()
    print(f"  All weights ready. Total: {_fmt(time.time() - total_start)}")


# ── Step 3: Build hybrid driving template (runs once) ─────────────────────────
# Why hybrid? talking.pkl has 10× more head/shoulder rotation than stock d5.pkl,
# but its lip expression is flat. d8.pkl has the best expression variety.
# We combine: rotation from talking.pkl + expression from d8.pkl.
def build_driving_template():
    out = DRIVING_DIR / "talking_with_movement.pkl"
    if out.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Driving template already built, skipping.")
        return

    _step("DRIVING TEMPLATE — building hybrid (talking.pkl rotation + d8.pkl expression)")

    with open(DRIVING_DIR / "talking.pkl", "rb") as f:
        talking = pickle.load(f)
    with open(DRIVING_DIR / "d8.pkl", "rb") as f:
        d8 = pickle.load(f)

    n = talking["n_frames"]
    print(f"  talking.pkl: {n} frames @ {talking['output_fps']}fps "
          f"(10x more head rotation than stock d5.pkl)")
    print(f"  d8.pkl: {len(d8['motion'])} frames resampled → {n} frames (best expression range)")

    # Resample d8's 275 expressions down to talking's 100 frames
    idx    = np.linspace(0, len(d8["motion"]) - 1, n).astype(int)
    d8_exp = np.stack([d8["motion"][i]["exp"] for i in idx])

    # Blink: wider open eyes, quick close
    t     = np.linspace(0, 4 * np.pi, n)
    blink = np.where(np.abs(np.sin(t * 0.7)) > 0.97, 0.03, 0.28).astype(np.float32)

    # Lip movement: ~4 syllables/sec (20 cycles over 5s) with peak 0.40
    # — fast enough to look like real speech, high enough to show teeth
    syllable = np.abs(np.sin(t * 5.0))
    envelope = 0.45 + 0.55 * np.abs(np.sin(t * 0.8))   # phrase-level amplitude variation
    lip      = (syllable * envelope * 0.40).astype(np.float32)
    lip      = np.where(lip < 0.05, 0.0, lip)           # hard-close (stop consonants)

    hybrid = {
        "n_frames":     n,
        "output_fps":   talking["output_fps"],
        "motion":       [{"scale": talking["motion"][i]["scale"],
                          "R_d":   talking["motion"][i]["R"],   # R→R_d (updated format)
                          "exp":   d8_exp[i],
                          "t":     talking["motion"][i]["t"]}
                         for i in range(n)],
        "c_d_eyes_lst": [np.array([[b, b]], dtype=np.float32) for b in blink],
        "c_d_lip_lst":  [np.array([[float(lip[i])]], dtype=np.float32) for i in range(n)],
    }

    with open(out, "wb") as f:
        pickle.dump(hybrid, f)
    print(f"  Saved -> {out.name}  ({n} frames, {n / talking['output_fps']:.1f}s @ {talking['output_fps']}fps)")


# ── Step 4: Run inference ──────────────────────────────────────────────────────
def generate_animation(headshot_path: str,
                       output_dir: str = str(OUTPUT_DIR),
                       driving_template: str = "",
                       driving_multiplier: float = 1.2) -> dict:
    """
    headshot_path     : path to source PNG/JPG
    output_dir        : where to save the output .mp4
    driving_template  : path to .pkl or .mp4 driver (blank = auto hybrid template)
    driving_multiplier: 1.0 = normal, 1.3 = more expressive (>1.5 may distort)

    Returns dict with output_path, engine, driving, metadata.
    """
    setup()
    download_weights()
    build_driving_template()

    headshot_path = str(Path(headshot_path).resolve())
    output_dir    = str(Path(output_dir).resolve())

    if driving_template:
        driving = str(Path(driving_template).resolve())
    elif (DRIVING_DIR / "talking_with_movement.pkl").exists():
        driving = str(DRIVING_DIR / "talking_with_movement.pkl")
    else:
        driving = str(DRIVING_DIR / "d8.pkl")  # fallback

    if not os.path.exists(headshot_path):
        raise FileNotFoundError(f"Headshot not found: {headshot_path}")
    if not os.path.exists(driving):
        raise FileNotFoundError(f"Driving template not found: {driving}")

    name     = Path(headshot_path).stem
    work_dir = os.path.join(output_dir, name)
    os.makedirs(work_dir, exist_ok=True)

    _step(f"INFERENCE — {name}  |  driver: {Path(driving).name}  |  multiplier: {driving_multiplier}")
    print(f"  source  : {headshot_path}")
    print(f"  driving : {driving}")
    print(f"  output  : {work_dir}")
    print(f"  NOTE: CPU inference is slow (~12 min for 100 frames). Progress below every 30s.")

    cmd = [
        sys.executable, str(REPO_DIR / "inference.py"),
        "--source",             headshot_path,
        "--driving",            driving,
        "--output_dir",         work_dir,
        "--flag_force_cpu",
        "--no-flag_use_half_precision",
        "--driving_multiplier", str(driving_multiplier),
    ]
    if driving.endswith(".mp4"):
        cmd.append("--flag_crop_driving_video")

    env        = {**os.environ, "PYTHONUTF8": "1"}
    start_time = time.time()

    # Background thread prints elapsed time every 30s so we know it's still running
    stop_event = threading.Event()
    ticker     = threading.Thread(target=_progress_printer, args=(stop_event, start_time, 30), daemon=True)
    ticker.start()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR), env=env)

    stop_event.set()
    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"LivePortrait inference failed after {_fmt(elapsed)}.\n"
                           f"STDERR:\n{result.stderr[-2000:]}")

    # LivePortrait writes: name--template.mp4 (animated) + name--template_concat.mp4 (side-by-side)
    # Sort by mtime descending → the plain animated output is written last and comes first
    mp4s = sorted(glob.glob(os.path.join(work_dir, "**", "*.mp4"), recursive=True),
                  key=os.path.getmtime, reverse=True)
    if not mp4s:
        raise RuntimeError(f"No output mp4 found in {work_dir}")

    final_path = os.path.join(output_dir, f"{name}_talking.mp4")
    os.replace(mp4s[0], final_path)

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"\n  DONE in {_fmt(elapsed)}")
    print(f"  Output : {final_path}  ({size_mb:.2f} MB)")

    return {
        "output_path": final_path,
        "engine":      "liveportrait",
        "driving":     Path(driving).name,
        "elapsed_sec": round(elapsed, 1),
        "metadata":    {"source": headshot_path, "multiplier": driving_multiplier},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LivePortrait: headshot -> talking animation")
    p.add_argument("--headshot",   required=True,  help="Path to source PNG (e.g. headshots/person_01.png)")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR))
    p.add_argument("--template",   default="",     help="Custom .pkl or .mp4 driving file")
    p.add_argument("--multiplier", type=float, default=1.0, help="Motion amplifier (1.0-1.5)")
    args = p.parse_args()

    overall_start = time.time()
    print(f"{'='*60}")
    print(f"  LivePortrait M1 Pipeline")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = generate_animation(args.headshot, args.output_dir, args.template, args.multiplier)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {_fmt(time.time() - overall_start)}")
    print(f"  Output: {result['output_path']}")
    print(f"{'='*60}")
