"""
M1 SadTalker — static headshot → talking animation
Run: python test/M1_sadtalker.py --headshot headshots/person_01.png
"""

import os, sys, glob, shutil, subprocess, threading, time, tempfile, wave, random as _rnd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
REPO_DIR        = PROJECT_ROOT / "models" / "sadtalker"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "m1" / "sadtalker"
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"
GFPGAN_DIR      = REPO_DIR / "gfpgan" / "weights"

# ── Checkpoint URLs ────────────────────────────────────────────────────────────
_ST_BASE = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
_CHECKPOINTS = {
    "SadTalker_V0.0.2_256.safetensors": f"{_ST_BASE}/SadTalker_V0.0.2_256.safetensors",
    "SadTalker_V0.0.2_512.safetensors": f"{_ST_BASE}/SadTalker_V0.0.2_512.safetensors",
    "mapping_00229-model.pth.tar":       f"{_ST_BASE}/mapping_00229-model.pth.tar",
    "mapping_00109-model.pth.tar":       f"{_ST_BASE}/mapping_00109-model.pth.tar",
}
_GFPGAN = {
    "alignment_WFLW_4HG.pth":      "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "GFPGANv1.4.pth":              "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth":         "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}

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

def _patch_basicsr():
    """basicsr uses torchvision.transforms.functional_tensor removed in torchvision 0.16+."""
    import site
    OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    NEW = "from torchvision.transforms.functional import rgb_to_grayscale"
    for sp in site.getsitepackages():
        target = Path(sp) / "basicsr" / "data" / "degradations.py"
        if target.exists():
            txt = target.read_text(encoding="utf-8")
            if OLD in txt:
                target.write_text(txt.replace(OLD, NEW), encoding="utf-8")
                print("  Patched basicsr degradations.py (torchvision 0.16+ compat)")
            return

def _patch_sadtalker_numpy():
    """
    SadTalker has two numpy 1.24+ incompatibilities:
    1. np.float/np.int/np.bool aliases removed in 1.24 (my_awing_arch.py etc.)
    2. np.array([..., arr_1d, ...]) raises ValueError for inhomogeneous shape
       when arr_1d is shape (1,) — preprocess.py line 101, POS returns (2,1) stack.
    """
    # Patch 1: removed np.float aliases
    alias_targets = [
        REPO_DIR / "src" / "face3d" / "util" / "my_awing_arch.py",
        REPO_DIR / "src" / "face3d" / "util" / "util.py",
        REPO_DIR / "src" / "face3d" / "models" / "networks.py",
    ]
    alias_replacements = [
        ("np.float,",  "float,"),
        ("np.float)",  "float)"),
        ("np.int,",    "int,"),
        ("np.int)",    "int)"),
        ("np.bool,",   "bool,"),
        ("np.bool)",   "bool)"),
        ("np.complex,","complex,"),
    ]
    for target in alias_targets:
        if not target.exists():
            continue
        txt = target.read_text(encoding="utf-8")
        patched = txt
        for old, new in alias_replacements:
            patched = patched.replace(old, new)
        if patched != txt:
            target.write_text(patched, encoding="utf-8")
            print(f"  Patched {target.name} (numpy 1.24+ compat)")

    # Patch 2: POS() returns t of shape (2,1); t[0]/t[1] are (1,) arrays which
    # numpy 1.24+ refuses to mix with scalars in np.array([...]).
    preprocess = REPO_DIR / "src" / "face3d" / "util" / "preprocess.py"
    if preprocess.exists():
        txt = preprocess.read_text(encoding="utf-8")
        OLD = "trans_params = np.array([w0, h0, s, t[0], t[1]])"
        NEW = "trans_params = np.array([w0, h0, s, float(t[0]), float(t[1])])"
        if OLD in txt:
            preprocess.write_text(txt.replace(OLD, NEW), encoding="utf-8")
            print("  Patched preprocess.py (t[0]/t[1] inhomogeneous array fix)")


# ── Step 1: Clone repo + install packages (runs once) ─────────────────────────
def setup():
    flag = REPO_DIR / ".setup_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Setup already done, skipping.")
        return

    _step("SETUP — one-time install (takes a few minutes)")

    if not REPO_DIR.exists():
        _run("Cloning SadTalker repo",
             ["git", "clone", "https://github.com/OpenTalker/SadTalker", str(REPO_DIR)])

    # SadTalker's requirements.txt has pins from 2022 incompatible with Python 3.11.
    # Skip it and install only what inference needs with modern versions.
    _run("Installing PyTorch (CPU build)", [sys.executable, "-m", "pip", "install",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cpu"])

    _run("Installing runtime dependencies", [sys.executable, "-m", "pip", "install",
         "librosa", "resampy", "pydub", "yacs", "kornia", "face_alignment",
         "basicsr", "gfpgan", "safetensors", "av",
         "scikit-image", "imageio", "imageio-ffmpeg"])

    flag.touch()
    print("  Setup complete.")


# ── Step 2: Download checkpoints from GitHub Releases (runs once) ─────────────
def download_checkpoints():
    flag = REPO_DIR / ".checkpoints_done"
    if flag.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Checkpoints already downloaded, skipping.")
        return

    _step("CHECKPOINTS — downloading from GitHub Releases v0.0.2-rc + GFPGAN weights")
    import urllib.request

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    GFPGAN_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for fname, url in _CHECKPOINTS.items():
        dest = CHECKPOINTS_DIR / fname
        if dest.exists():
            print(f"  already have {fname}", flush=True)
            continue
        print(f"  downloading {fname} ...", end=" ", flush=True)
        t0 = time.time()
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"{size_mb:.0f} MB in {_fmt(time.time() - t0)}", flush=True)

    for fname, url in _GFPGAN.items():
        dest = GFPGAN_DIR / fname
        if dest.exists():
            print(f"  already have {fname}", flush=True)
            continue
        print(f"  downloading {fname} ...", end=" ", flush=True)
        t0 = time.time()
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"{size_mb:.0f} MB in {_fmt(time.time() - t0)}", flush=True)

    flag.touch()
    print(f"  All checkpoints ready. Total: {_fmt(time.time() - total_start)}")


# ── Step 3: Run inference ──────────────────────────────────────────────────────
def generate_animation(headshot_path: str,
                       output_dir: str = str(OUTPUT_DIR),
                       driver_audio: str = "",
                       fps: int = 25,
                       duration_sec: float = 5.0,
                       use_enhancer: bool = True,
                       cpu_only: bool = True) -> dict:
    """
    headshot_path : path to source PNG/JPG
    output_dir    : where to save the output .mp4
    driver_audio  : WAV/MP3 to drive animation (blank = auto-generate silent driver)
    fps           : output frames per second
    duration_sec  : duration of silent driver if no audio provided
    use_enhancer  : apply GFPGAN face restoration (better quality, ~2x slower)
    cpu_only      : force CPU inference (pass --cpu to SadTalker)

    Returns dict with output_path, engine, fps, metadata.
    """
    setup()
    download_checkpoints()
    _patch_basicsr()
    _patch_sadtalker_numpy()

    headshot_path = str(Path(headshot_path).resolve())
    output_dir    = str(Path(output_dir).resolve())

    if not os.path.exists(headshot_path):
        raise FileNotFoundError(f"Headshot not found: {headshot_path}")

    # Generate phoneme-diverse driver WAV if no audio provided.
    # SadTalker maps mel features to 3DMM face coefficients.
    # A pure sine tone → sustained "O" → pursed lips frozen in one position.
    # Alternating voiced vowels / stop closures / fricative bursts gives
    # varied mel features → varied mouth shapes → natural-looking speech.
    silent_tmp = None
    if not driver_audio:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        silent_tmp   = tmp.name
        driver_audio = silent_tmp
        sr  = 16000
        n   = int(sr * duration_sec)
        sig = np.zeros(n, dtype=np.float32)
        _rnd.seed(42)
        i   = 0
        while i < n:
            # Voiced vowel segment (~140 ms, Hanning-windowed)
            v_len = min(int(0.14 * sr), n - i)
            if v_len > 0:
                t_v = np.arange(v_len) / sr
                f0  = 130 + 20 * np.sin(2 * np.pi * 0.3 * (i / sr))
                vow = (0.55 * np.sin(2 * np.pi * f0 * t_v) +
                       0.28 * np.sin(2 * np.pi * f0 * 2 * t_v) +
                       0.12 * np.sin(2 * np.pi * f0 * 3 * t_v))
                sig[i:i + v_len] = vow * np.hanning(v_len) * 0.8
            i += v_len
            # Stop closure (30-70 ms silence) → mouth closes
            i += min(_rnd.randint(int(0.03 * sr), int(0.07 * sr)), n - i)
            # Fricative burst (~40 ms broadband noise, 50% of syllables)
            if i < n and _rnd.random() < 0.5:
                f_len = min(int(0.04 * sr), n - i)
                raw   = os.urandom(f_len * 2)
                noise = (np.frombuffer(raw, dtype=np.uint16).astype(np.float32) / 32768.0 - 1.0)
                sig[i:i + f_len] = noise * 0.25
                i += f_len
        mx = np.max(np.abs(sig))
        if mx > 0:
            sig = sig / mx * 0.8
        pcm = (sig * 32767).astype(np.int16)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    else:
        driver_audio = str(Path(driver_audio).resolve())

    name         = Path(headshot_path).stem
    result_dir   = str(REPO_DIR / "results" / name)
    os.makedirs(output_dir, exist_ok=True)

    enhancer_label = "with GFPGAN enhancer" if use_enhancer else "no enhancer"
    cpu_label      = "CPU" if cpu_only else "GPU"
    _step(f"INFERENCE — {name}  |  {enhancer_label}  |  {cpu_label}")
    print(f"  source  : {headshot_path}")
    print(f"  audio   : {'silent (generated)' if silent_tmp else driver_audio}")
    print(f"  NOTE: CPU inference is slow (~10-30 min). Progress below every 30s.")

    cmd = [
        sys.executable, str(REPO_DIR / "inference.py"),
        "--driven_audio",  driver_audio,
        "--source_image",  headshot_path,
        "--result_dir",    result_dir,
        "--still",
        "--preprocess", "full",
    ]
    if use_enhancer:
        cmd += ["--enhancer", "gfpgan"]
    if cpu_only:
        cmd.append("--cpu")

    start_time  = time.time()
    stop_event  = threading.Event()
    ticker      = threading.Thread(target=_progress_printer, args=(stop_event, start_time, 30), daemon=True)
    ticker.start()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))
    stop_event.set()
    elapsed = time.time() - start_time

    # Retry without GFPGAN if enhancer caused the failure
    if result.returncode != 0 and use_enhancer and "gfpgan" in result.stderr.lower():
        print(f"\n  GFPGAN failed; retrying without enhancer ...")
        cmd_no_enh = [c for c in cmd if c not in ("--enhancer", "gfpgan")]
        stop_event2 = threading.Event()
        ticker2 = threading.Thread(target=_progress_printer, args=(stop_event2, time.time(), 30), daemon=True)
        ticker2.start()
        result = subprocess.run(cmd_no_enh, capture_output=True, text=True, cwd=str(REPO_DIR))
        stop_event2.set()
        elapsed = time.time() - start_time

    if result.returncode != 0:
        if silent_tmp:
            try: os.unlink(silent_tmp)
            except OSError: pass
        raise RuntimeError(f"SadTalker inference failed after {_fmt(elapsed)}.\n"
                           f"STDERR:\n{result.stderr[-2000:]}")

    if silent_tmp:
        try: os.unlink(silent_tmp)
        except OSError: pass  # Windows may hold the handle briefly; temp dir cleans up on reboot

    candidates = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
    if not candidates:
        raise RuntimeError(f"No output mp4 found in {result_dir}")

    final_path = os.path.join(output_dir, f"{name}_talking.mp4")
    shutil.copy2(max(candidates, key=os.path.getmtime), final_path)

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"\n  DONE in {_fmt(elapsed)}")
    print(f"  Output : {final_path}  ({size_mb:.2f} MB)")

    return {
        "output_path": final_path,
        "engine":      "sadtalker",
        "fps":         fps,
        "elapsed_sec": round(elapsed, 1),
        "metadata":    {"source": headshot_path, "enhancer": use_enhancer, "cpu_only": cpu_only},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SadTalker: headshot -> talking animation")
    p.add_argument("--headshot",    required=True,  help="Path to source PNG (e.g. headshots/person_01.png)")
    p.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    p.add_argument("--audio",       default="",     help="Path to driver WAV/MP3 (blank = silent)")
    p.add_argument("--fps",         type=int,   default=25)
    p.add_argument("--duration",    type=float, default=5.0, help="Duration in seconds (silent driver only)")
    p.add_argument("--no_enhancer", action="store_true",     help="Skip GFPGAN face restoration")
    p.add_argument("--gpu",         action="store_true",     help="Use GPU (default: CPU)")
    args = p.parse_args()

    overall_start = time.time()
    print(f"{'='*60}")
    print(f"  SadTalker M1 Pipeline")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    result = generate_animation(
        args.headshot, args.output_dir,
        driver_audio=args.audio,
        fps=args.fps,
        duration_sec=args.duration,
        use_enhancer=not args.no_enhancer,
        cpu_only=not args.gpu,
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {_fmt(time.time() - overall_start)}")
    print(f"  Output: {result['output_path']}")
    print(f"{'='*60}")
