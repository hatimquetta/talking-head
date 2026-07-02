# Talking-Head Animation Pipeline

Turn a **static headshot** into a **talking-head video with synchronized speech**, in
three stages:

1. **Animate** — headshot → idle + talking animations (LivePortrait, on GPU).
2. **Sequence** — assemble `idle (3s) → speaking (S) → idle (3s)` as one seamless clip.
3. **Voice** — text → speech, then attach it to the video (or keep it as a separate track).

Every clip is built from eased **"boomerang" loops** (a slice + its reverse) so it starts
and ends on a neutral frame and loops with no visible seam.

---

## 👉 Start here: `final/`

`final/` is the polished, self-contained deliverable. **Read [`final/README.md`](final/README.md)**
for full usage. In short:

```bash
cd final
pip install -r requirements.txt          # installs edge-tts (needs ffmpeg on PATH too)

# text -> speech (prints the audio duration to use as S)
python tts.py "Hey there, great to meet you." -g male -o greeting.mp3

# idle + speaking(S) + idle  ->  silent video   (talking clip comes from pipeline.ipynb)
python sequencer.py headshots/person_01.png person_01_talking.mp4 8.4 -o person_01_sequenced.mp4

# attach the voice  ->  person_01_sequenced_with_audio.mp4
python attach_audio.py person_01_sequenced.mp4 greeting.mp3
```

The animation stage (`pipeline.ipynb`) needs a **GPU** and runs on **Google Colab**; the
three CLI tools run **locally** (they only need Python + ffmpeg, plus internet for TTS).

---

## Repository map

| Path | What it is |
|---|---|
| **`final/`** | ⭐ The deliverable — `pipeline.ipynb` (batch, Colab) + `sequencer.py`, `tts.py`, `attach_audio.py` (local CLIs) + sample `headshots/` + its own README. |
| `test/` | Experimentation notebooks used to build & validate the pipeline, plus their generated clips in `test/assets/`. Kept for reference. |
| `scripts/` | Modular, reusable Python (pure functions): `m1/` model wrappers, `m3/` TTS, `utils/` shared video/audio helpers. |
| `notebooks/` | Early per-model Colab notebooks (LivePortrait / SadTalker / Wav2Lip). |
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

- **Python 3.10+**
- **ffmpeg + ffprobe** on PATH — `winget install Gyan.FFmpeg` (Win) · `brew install ffmpeg` (mac) · `apt install ffmpeg` (Linux)
- `pip install -r final/requirements.txt` for the local tools.
- A **GPU (Colab)** for the LivePortrait animation stage — `final/pipeline.ipynb` installs everything it needs there.
- **Internet** for the TTS step (Microsoft neural voices).

---

## The three milestones

| # | Focus | Deliverable |
|---|---|---|
| **M1** | Natural speaking animation | LivePortrait idle + talking clips per headshot (`pipeline.ipynb`) |
| **M2** | Programmatic sequencing | `sequencer.py` — seamless `idle → speaking → idle` |
| **M3** | Synchronized audio + playback | `tts.py` + `attach_audio.py` — voice, muxed or separate |

See `final/README.md` for the detailed, copy-paste usage of every tool.
