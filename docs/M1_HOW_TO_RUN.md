# M1: How to Generate Talking Animations

This guide covers **Milestone 1** — turning static headshots into talking (and optional
idle) animations. You do not need to write any code; follow the steps below.

---

## Which workflow should I use?

| Goal | Use this |
|---|---|
| **Run the project deliverable** (recommended) | `final/animations.ipynb` or `final/pipeline.ipynb` — see [Primary workflow](#primary-workflow-deliverable) below |
| **Compare older animation models** (R&D) | `notebooks/M1_*.ipynb` — see [Legacy notebooks](#legacy-notebooks-rd) below |

The deliverable uses **LivePortrait** with driving templates. The legacy notebooks let you
try SadTalker and Wav2Lip as alternatives.

---

## What you need before starting

1. A **Google account** (Google Colab + Google Drive)
2. **Headshot images** — the repo includes `final/headshots/person_01.png` … `person_08.png`
3. **Driving templates** — `final/templates/template_talking.mp4` (required) and
   `template_idle.mp4` (only if you want idle animations)

---

## Primary workflow (deliverable)

### One-time setup: upload files to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Under **My Drive**, create a folder named **`talking_head`** with this layout:

   ```
   My Drive/
   └── talking_head/
       ├── headshots/
       │   ├── person_01.png
       │   ├── person_02.png
       │   └── ...
       └── templates/
           ├── template_talking.mp4
           └── template_idle.mp4    ← only if you need idle animations
   ```

3. Copy from your local clone:
   - `final/headshots/*.png` → `talking_head/headshots/`
   - `final/templates/` → `talking_head/templates/`

   The notebooks write results to `talking_head/output/animations/` on Drive automatically.

### Run the notebook in Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → pick `final/animations.ipynb`
   - *(Or **File → Open notebook → GitHub** and paste the repo URL.)*
3. **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**
4. Edit **Cell 1** — set your headshot names and whether you need idle clips:

   ```python
   HEADSHOTS = ['person_01', 'person_02', 'person_03']   # names WITHOUT extension
   NEED_IDLE = True      # False = talking only (no idle template needed)
   ```

5. **Run Cell 1** — when prompted, **Allow** Google Drive access
6. **Run Cell 2** — first run clones LivePortrait and downloads ~600 MB of weights (a few
   minutes, one time). Then ~30–60 s per animation on a T4. Re-runs skip existing files.

### Collect the results

Animations appear on Drive:

```
My Drive/talking_head/output/animations/
├── person_01_talking.mp4
├── person_01_idle.mp4        (only if NEED_IDLE)
└── ...
```

Download that folder and place the `.mp4` files in your local
**`final/output/animations/`** — that is where the local scripts (`sequencer.py`, etc.)
look for them.

> **Batch everything:** `final/pipeline.ipynb` uses the **same Drive layout** but runs all
> three milestones at once (animate → voice → sequence → attach). Use `animations.ipynb`
> when you only need the animations.

### Next steps (M2 + M3)

After animations are on your machine, continue with [`final/README.md`](../final/README.md)
for local sequencing, TTS, and playback.

**Local prerequisites (macOS / Linux / Windows):**

| Tool | macOS | Linux | Windows |
|---|---|---|---|
| ffmpeg | `brew install ffmpeg` | `sudo apt install ffmpeg` | `winget install Gyan.FFmpeg` |
| Python deps | `pip install -r final/requirements.txt` | same | same |

```bash
# Generate speech (note the printed duration = S)
python final/tts.py "Hey there, great to meet you." -g male

# Build idle → speaking(S) → idle clip
python final/sequencer.py final/headshots/person_01.png final/output/animations/person_01_talking.mp4 8.4

# Open the M3 player
#   macOS:   open final/playback.html
#   Linux:   xdg-open final/playback.html
#   Windows: start final/playback.html
```

---

## Checking output quality

After the notebook finishes, watch the output videos and check:

- [ ] Mouth movement looks natural (not frozen, not mechanical)
- [ ] Eyes blink naturally
- [ ] No severe warping or blurry patches on the face
- [ ] The video loops seamlessly (watch the transition from end back to start)

**To verify the loop:** open the video in VLC → enable Loop (press `L`), or drag the `.mp4`
into a browser tab and let it repeat.

---

## Adding a new headshot

1. Name your image `person_09.png` (or any `person_XX.png` format)
2. Upload it to Google Drive at `talking_head/headshots/`
3. Add `'person_09'` to the `HEADSHOTS` list in Cell 1
4. Re-run Cell 2 (setup does not need to repeat if the runtime is still connected)
5. Download the new animation to `final/output/animations/`

---

## Legacy notebooks (R&D)

The `notebooks/` folder contains earlier per-model experiments. These use a **different**
Drive layout from the deliverable — do not mix them with the `final/` workflow unless you
know what you are doing.

### Legacy Drive layout

```
My Drive/
└── talking_head/
    ├── headshots/          ← upload PNG headshots here
    └── outputs/            ← notebooks create this automatically
        └── m1/
            ├── liveportrait/
            ├── sadtalker/
            └── wav2lip/
```

For LivePortrait in the legacy notebooks, also upload a template video to
`talking_head/template_talking.mp4` (at the **root** of `talking_head/`, not in `templates/`).

### Notebooks

| Notebook | Model | Best for | Driving signal |
|---|---|---|---|
| `notebooks/M1_liveportrait.ipynb` | LivePortrait | Most realistic, natural look | Template video |
| `notebooks/M1_sadtalker.ipynb` | SadTalker | Best 3D head motion | Silent audio (auto-generated) |
| `notebooks/M1_wav2lip.ipynb` | Wav2Lip | Sharpest lip sync | Silent audio (auto-generated) |

### Steps for each legacy notebook

1. Upload the `.ipynb` from `notebooks/` to Colab (or open from GitHub)
2. **Runtime → Change runtime type → T4 GPU → Save**
3. **Runtime → Run all** (or `Ctrl+F9` / `Cmd+F9` on macOS)
4. Approve Google Drive access when prompted
5. Download outputs from `talking_head/outputs/m1/{model}/` on Drive

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Runtime disconnected" | Free Colab disconnects after ~90 min. Re-run from the render cell only — setup does not need to repeat if weights are cached |
| "Template video not found" | Deliverable: ensure `talking_head/templates/template_talking.mp4` exists. Legacy LivePortrait: upload to `talking_head/template_talking.mp4` |
| `ffmpeg` / `ffprobe` not found (local steps) | Install ffmpeg (see table above) and confirm `ffmpeg -version` works in your terminal |
| Output looks blurry around mouth | Expected with Wav2Lip — compare with LivePortrait output |
| "CUDA out of memory" | Runtime → Disconnect and delete runtime → reconnect with T4 GPU |
| Fewer outputs than headshots | Check failure messages in the notebook; re-run the render cell for failed images |
| `pip install` fails on Windows | Use `py -m pip install -r final/requirements.txt` if `python` is not on PATH |
