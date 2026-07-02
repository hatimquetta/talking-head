# Talking-Head Animation Pipeline

Turn a **static headshot** into a **talking-head video with synchronized speech**.

The pipeline has three stages:

| Stage | What it does | Where it runs |
|---|---|---|
| **M1 — Animate** | Headshot → idle + talking animations (LivePortrait) | Google Colab (GPU) — `animations.ipynb` / `pipeline.ipynb` |
| **M2 — Sequence** | Idle (3s) + speaking (S) + idle (3s), seamless loops | Local — `sequencer.py` |
| **M3 — Voice + Playback** | Text → speech, then a player that plays the voice alongside the animation and returns to idle when it ends (assets stay separate) | Local — `tts.py`, `playback.html` |

Every clip is built from **eased "boomerang" loops** (a slice + its reverse) so it
starts and ends on a neutral frame and loops with no visible seam.

---

## Contents

```
final/
├── pipeline.ipynb        # M1–M3 end-to-end for a batch of headshots (Colab GPU)
├── animations.ipynb      # M1 only: LivePortrait talking (+ optional idle) — Colab GPU
├── sequencer.py          # M2 CLI: idle + speaking + idle  ->  6+S video
├── tts.py                # M3 CLI: text  ->  speech audio (male / female voice)
├── playback.html         # M3 player: plays voice + animation together, returns to idle on end
├── attach_audio.py       # M3 optional: bake voice into one mp4 / make a padded separate audio
├── headshots/            # 8 sample client headshots (person_01..08)
├── templates/            # driving videos: template_talking.mp4 + template_idle.mp4
├── output/               # all generated results land here (contents git-ignored)
│   ├── animations/       #   {name}_talking.mp4, {name}_idle.mp4
│   ├── audio/            #   TTS .mp3
│   ├── sequenced/        #   idle + speaking + idle (silent)
│   └── with_audio/       #   final video with voice + a separate padded audio
├── requirements.txt
└── README.md
```

Every tool writes into `final/output/…` by default (regardless of where you run it
from), so you don't need to pass `-o` unless you want a custom path.

---

# How to run — step by step (A → Z)

## Stage 1 — Animate the headshots (Google Colab, GPU)

LivePortrait needs a GPU, so this stage runs on **Google Colab** with your files on
**Google Drive**.

### 1. Create the Drive folder structure
In your Google Drive, under **My Drive**, create a folder named **`talking_head`** laid
out exactly like this:

```
My Drive/
└── talking_head/
    ├── headshots/               <- copy the images from  final/headshots/
    │   ├── person_01.png
    │   ├── person_02.png
    │   └── ...
    └── templates/               <- copy the whole  final/templates/  folder here
        ├── template_talking.mp4
        └── template_idle.mp4        (only if NEED_IDLE)
```

Everything you need to copy up is already in the repo — the Drive layout mirrors it 1:1:
- **Headshots** — copy the images from **`final/headshots/`** into `talking_head/headshots/`
  (the 8 samples are `person_01.png` … `person_08.png`; `.jpg` / `.jpeg` also work).
  For your own people, use front-facing, well-lit photos named `person_XX.png`.
- **Templates** — copy the whole **`final/templates/`** folder into `talking_head/`, so you
  end up with `talking_head/templates/template_talking.mp4` (and `…/template_idle.mp4`).
  They're short talking-head clips whose head/mouth motion LivePortrait copies onto each
  headshot. **`template_idle.mp4` is only needed when `NEED_IDLE = True`.**

### 2. Open the notebook in Colab
Go to <https://colab.research.google.com> → **File ▸ Upload notebook** → pick
`final/animations.ipynb`. *(Or **File ▸ Open notebook ▸ GitHub** and paste the repo URL.)*

### 3. Turn on the GPU
**Runtime ▸ Change runtime type ▸ Hardware accelerator → `T4 GPU` ▸ Save.**

### 4. Edit the config (Cell 1)
Match `HEADSHOTS` to your filenames and choose whether you need the idle animation:
```python
HEADSHOTS = ['person_01', 'person_02', 'person_03']   # names WITHOUT the extension
NEED_IDLE = True      # False = render talking only (no idle template needed)
```
The folder paths at the top already point at `My Drive/talking_head/...` — leave them
as-is unless your Drive layout differs.

### 5. Run Cell 1
A **"Permit this notebook to access your Google Drive files?"** popup appears — click
through and **Allow**, so Colab can read your headshots and write results back.

### 6. Run Cell 2
The **first** run clones LivePortrait and downloads ~600 MB of weights (a few minutes,
one time). Then it renders each headshot — roughly **30–60 s per animation** on a T4.
Re-running is safe: it **skips files that already exist**.

### 7. Collect the results
Animations are written back to your Drive:
```
My Drive/talking_head/output/animations/
├── person_01_talking.mp4
├── person_01_idle.mp4        (only if NEED_IDLE)
└── ...
```

### 8. Download them to your computer
Download that `animations/` folder from Drive and drop the files into your local
**`final/output/animations/`** — that's where the local scripts look for them.

> **Tip:** `pipeline.ipynb` uses the **same Drive layout** but runs *all three* stages
> at once (animate → voice → sequence → attach) for the whole `HEADSHOTS` set. Use
> `animations.ipynb` when you only want the animations.

---

## Stage 2 — Sequence the animation (local)

### 1. Install the prerequisites (one time)
- **Python 3.10+**
- **ffmpeg + ffprobe** on PATH:
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- `pip install -r final/requirements.txt`  (installs `edge-tts` for Stage 3)
- An **internet connection** for the TTS step (Microsoft neural voices).

### 2. Build the idle → speaking → idle clip
Outputs auto-land in `final/output/…`:
```powershell
# arg1 = a headshot image (static 3s ends) OR a *_idle.mp4 (subtle motion)
# arg2 = the talking animation   |   arg3 = S (speaking seconds; use the tts.py duration)
python final\sequencer.py final\headshots\person_01.png final\output\animations\person_01_talking.mp4 8.40
#   -> final/output/sequenced/person_01_talking_sequenced.mp4   (silent, 6+S long)
```

---

## Stage 3 — Voice + playback (local)

The Milestone-3 spec: generate speech, then a **playback system** that plays the audio and
the animation **in sync**, keeps them as **separate assets** (nothing muxed), and returns to
idle when the audio ends.

### 1. Generate the voice
```powershell
python final\tts.py "Hey there, great to meet you." --gender male
#   -> final/output/audio/tts_output.mp3   (also prints the duration = your S)
```

### 2. Play it — open `final/playback.html`
Double-click **`final/playback.html`** (opens in any browser), then:
1. **Idle** — load a headshot image *or* a `_idle.mp4` (optional).
2. **Talking animation** — load the `_talking.mp4`.
3. **TTS audio** — load the `.mp3` from step 1.
4. Press **Play** — the voice plays while the mouth moves; when the voice **ends** it eases
   back to idle. **Stop** returns immediately.

The audio (an `<audio>` element) and the animation (a muted `<video>`) play in parallel and
are **never combined** — exactly what M3 requires.

### Optional — bake into one file
For a single shareable clip (or a separate padded audio track that lines up 1:1 with the
silent video), run `attach_audio.py` on the Stage-2 sequenced video:
```powershell
python final\attach_audio.py final\output\sequenced\person_01_talking_sequenced.mp4 final\output\audio\tts_output.mp3
#   -> final/output/with_audio/..._with_audio.mp4     (one file, muxed)
#   -> final/output/with_audio/..._audio_padded.m4a   (separate, same length as the video)
```

> **Why the duration matters:** the `S` you pass to `sequencer.py` is exactly the number
> `tts.py` prints, so the speaking (mouth-moving) window matches the audio length.

---

## Tool reference

### `sequencer.py` — build the idle → speaking → idle clip (silent)
```
python sequencer.py  IDLE_OR_HEADSHOT  SPEAKING_ANIMATION  S  [-o OUT]
```
- **arg 1** — an **image** (static headshot: a 3s freeze at each end) or a **video**
  (idle animation: an eased 3s boomerang; looped if shorter than 1.5s).
- **arg 2** — the talking animation → eased boomerang of **exactly S** seconds.
- **arg 3** — `S`, the speaking length in seconds.
- Options: `--idle-duration` (3.0), `--crossfade` (0 = hard cut), `--ease-src`/`--ease-out`.

### `tts.py` — text → speech
```
python tts.py "TEXT"  [--gender male|female]  [-o out.mp3]
```
- Human voices: female = `en-US-AvaMultilingualNeural`, male = `en-US-AndrewMultilingualNeural`
  (override with `--voice`). Prints the audio **duration** to use as `S`.

### `playback.html` — the M3 player (no install)
Open in a browser; load an idle (image or `_idle.mp4`), the `_talking.mp4`, and the TTS
`.mp3`, then press **Play**. Plays the voice + animation together and returns to idle when
the audio ends. Audio and video stay as **separate assets** (never muxed).

### `attach_audio.py` — optional one-file export
```
python attach_audio.py  VIDEO  AUDIO  [--offset 3.0]  [--mode both]
```
- `mux` = one .mp4 with audio baked in at 3s · `separate` = audio padded to the video
  length · `both` (default). Handy for sharing — but M3 itself wants the assets kept
  separate, so use `playback.html` for the actual playback.

---

## Batch everything: `pipeline.ipynb` (Colab GPU)

Runs all of M1–M3 for a dictionary of headshots. Uses the **same Drive setup as Stage 1
above**. Cell 1 is the only cell you edit: templates, the `HEADSHOTS = {name: isMale}`
dict, folders, and the shared text. It renders idle + talking per headshot, generates a
**gender-appropriate** voice (male if `isMale`), builds the sequenced clip, and writes
results into `output/animations`, `output/audio`, `output/sequenced`, and `output/with_audio`
(a muxed video **and** a separate padded audio) — the same logic as the CLI tools, batched.
For the live M3 player (voice + animation in sync, auto-return to idle), use `playback.html`
with the `output/animations` clips and an `output/audio` mp3.
