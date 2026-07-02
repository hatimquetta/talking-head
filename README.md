# Talking-Head Animation Pipeline

Turn a **static headshot** into a **talking-head video with synchronized speech**, in
three stages:

1. **Animate** — headshot → idle + talking animations (LivePortrait, on GPU).
2. **Sequence** — assemble `idle (3s) → speaking (S) → idle (3s)` as one seamless clip.
3. **Voice + playback** — text → speech, then play voice and animation in sync (separate
   assets) via `playback.html`, with optional muxing via `attach_audio.py`.

Every clip is built from eased **"boomerang" loops** (a slice + its reverse) so it starts
and ends on a neutral frame and loops with no visible seam.

---

## 👉 Start here: `final/`

`final/` is the polished, self-contained deliverable. **Read [`final/README.md`](final/README.md)**
for the full step-by-step guide (Colab GPU → local CLI → browser player).

### Quick start (local tools)

**Prerequisites (one time):**

| Tool | macOS | Linux | Windows |
|---|---|---|---|
| Python 3.10+ | `brew install python` or [python.org](https://www.python.org/downloads/) | `sudo apt install python3 python3-pip` | [python.org](https://www.python.org/downloads/) or Microsoft Store |
| ffmpeg + ffprobe | `brew install ffmpeg` | `sudo apt install ffmpeg` | `winget install Gyan.FFmpeg` |
| Python deps | `pip install -r final/requirements.txt` | same | same |

```bash
# From the repo root — paths use / on all platforms (Python accepts them on Windows too)

# 1. Text → speech (prints the audio duration — use that as S in step 2)
python final/tts.py "Hey there, great to meet you." -g male

# 2. Idle + speaking(S) + idle → silent video
#    (talking clip comes from animations.ipynb / pipeline.ipynb on Colab)
python final/sequencer.py final/headshots/person_01.png final/output/animations/person_01_talking.mp4 8.4

# 3. M3 playback — open the browser player and load idle + talking + mp3
#    macOS:   open final/playback.html
#    Linux:   xdg-open final/playback.html
#    Windows: start final/playback.html

# Optional — bake voice into one shareable mp4
python final/attach_audio.py final/output/sequenced/person_01_talking_sequenced.mp4 final/output/audio/tts_output.mp3
```

The animation stage (`animations.ipynb` or `pipeline.ipynb`) needs a **GPU** and runs on
**Google Colab**; the CLI tools and `playback.html` run **locally** (Python + ffmpeg, plus
internet for TTS).

---

## Repository map

| Path | What it is |
|---|---|
| **`final/`** | ⭐ The deliverable — `animations.ipynb` + `pipeline.ipynb` (Colab), `sequencer.py`, `tts.py`, `playback.html`, `attach_audio.py` (local), sample `headshots/` + `templates/`, and its own README. |
| `test/` | Experimentation notebooks used to build & validate the pipeline. Kept for reference (generated media is git-ignored). |
| `scripts/` | Modular, reusable Python: `m1/` model wrappers, `m3/` TTS, `utils/` shared video/audio helpers. |
| `notebooks/` | Early per-model Colab notebooks (LivePortrait / SadTalker / Wav2Lip) — see [`docs/M1_HOW_TO_RUN.md`](docs/M1_HOW_TO_RUN.md) for the R&D workflow. |
| `docs/` | How-to documentation. |
| `Project_Hatim_2026_Jun.pdf` | Original project brief. |

### What's in `test/` (the R&D trail)
- `M1_all_models.ipynb` — evaluated 5 audio-driven animation models (EchoMimic V2, MuseTalk, VideoReTalking, AniPortrait, SONIC) on Colab.
- `M1_liveportrait_template.ipynb` — the LivePortrait approach that was chosen.
- `M2_boomerang_idle_speaking.ipynb` — the boomerang idle/speaking sequencing.
- `M3_tts_audio.ipynb` — TTS + audio-to-video muxing.
- `pipeline.ipynb` — the full end-to-end run (matured into `final/pipeline.ipynb`).

---

## Requirements

Two `requirements.txt` files exist on purpose:

| File | When to use |
|---|---|
| **`final/requirements.txt`** | Local CLI tools (`tts.py`, `sequencer.py`, `attach_audio.py`) — only `edge-tts`. **Install this for the deliverable.** |
| `requirements.txt` (repo root) | Full R&D / Colab-local environment (torch, gradio, opencv, etc.) — only if you are hacking on the older `test/` or `notebooks/` workflows locally. |

**For the deliverable you need:**

- **Python 3.10+**
- **ffmpeg + ffprobe** on PATH (see table above)
- `pip install -r final/requirements.txt`
- A **GPU (Colab)** for the LivePortrait animation stage — the notebooks install their own stack there.
- **Internet** for the TTS step (Microsoft neural voices).

---

## The three milestones

| # | Focus | Deliverable |
|---|---|---|
| **M1** | Natural speaking animation | LivePortrait idle + talking clips per headshot (`animations.ipynb` / `pipeline.ipynb`) |
| **M2** | Programmatic sequencing | `sequencer.py` — seamless `idle → speaking → idle` |
| **M3** | Synchronized audio + playback | `tts.py` + `playback.html` (primary); `attach_audio.py` (optional mux) |

See [`final/README.md`](final/README.md) for the detailed, copy-paste usage of every tool.
See [`docs/M1_HOW_TO_RUN.md`](docs/M1_HOW_TO_RUN.md) for M1 animation setup (deliverable + legacy notebooks).
