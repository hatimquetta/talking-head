# M1: How to Generate Talking Animations

This guide explains how to generate a talking animation from any static headshot image using the three notebooks in this project. You do not need to write any code — just follow the steps below.

---

## What you need before starting

1. A Google account (for Google Colab and Google Drive)
2. The headshot images (`.png` files, named `person_01.png` through `person_08.png`)
3. For LivePortrait only: one short template video (5–10 seconds of any person talking)

---

## One-time setup: upload files to Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Create this folder structure in your Drive:
   ```
   My Drive/
   └── talking_head/
       ├── headshots/          ← upload your 8 PNG files here
       ├── outputs/            ← notebooks will create this automatically
       └── template_talking.mp4  ← LivePortrait only: upload any ~5s talking-head video here
   ```
3. Drag and drop all 8 PNG headshots into `headshots/`
4. If using LivePortrait: drop your template video at `talking_head/template_talking.mp4`

You only do this once. The notebooks will read from and write to these folders automatically.

---

## Running the notebooks

There are three notebooks — one per animation model. Run them in any order; each is independent.

| Notebook | Model | Best for | Driving signal |
|---|---|---|---|
| `M1_liveportrait.ipynb` | LivePortrait | Most realistic, natural look | Template video |
| `M1_sadtalker.ipynb` | SadTalker | Best 3D head motion | Silent audio (auto-generated) |
| `M1_wav2lip.ipynb` | Wav2Lip | Sharpest lip sync | Silent audio (auto-generated) |

### Steps for each notebook

1. **Open the notebook in Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Click **File → Open notebook → Upload**, and upload the `.ipynb` file from the `notebooks/` folder
   - Or: open Google Drive, navigate to the notebook file, and double-click it

2. **Enable GPU**
   - Click **Runtime → Change runtime type**
   - Set Hardware accelerator to **T4 GPU**
   - Click **Save**

3. **Run all cells in order**
   - Click **Runtime → Run all** (or press `Ctrl+F9`)
   - The first cell (install/setup) takes 3–8 minutes — this is normal
   - When prompted "Allow this notebook to access Google Drive?", click **Connect to Google Drive** and approve

4. **Wait for batch processing to finish**
   - You will see progress like: `Processing: person_01.png ... Done → /content/drive/...`
   - All 8 animations are saved directly to your Google Drive under `outputs/m1/{model}/`

5. **Download the outputs**
   - Go to Google Drive → `talking_head/outputs/m1/{model}/`
   - Download all `.mp4` files
   - Place them in your local project: `outputs/m1/{model}/person_01_talking.mp4`, etc.

---

## Checking output quality

After each notebook finishes, watch the output videos and check:

- [ ] Mouth movement looks natural (not frozen, not mechanical)
- [ ] Eyes blink naturally
- [ ] No severe warping or blurry patches on the face
- [ ] The video loops seamlessly (watch the transition from end back to start)

**To verify the loop:** open the video in VLC → enable Loop (press `L`), or drag the `.mp4` into a browser tab and let it repeat.

---

## Adding a new headshot in the future

1. Name your new image `person_09.png` (or any `person_XX.png` format)
2. Upload it to Google Drive at `talking_head/headshots/`
3. Open any notebook in Colab
4. In **Cell 4**, change the batch loop to process only your new file:
   ```python
   headshots = [f"{HEADSHOTS_DIR}/person_09.png"]
   ```
5. Run Cell 4 only (no need to re-run setup)
6. Download the new animation from Drive

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Runtime disconnected" | Free Colab disconnects after ~90 min. Re-run from Cell 4 only — setup does not need to repeat |
| "Template video not found" | LivePortrait only — upload `template_talking.mp4` to `My Drive/talking_head/` |
| Output looks blurry around mouth | Expected with Wav2Lip — compare with LivePortrait or SadTalker output |
| "CUDA out of memory" | Runtime → Disconnect and delete runtime → reconnect with T4 GPU |
| Fewer than 8 outputs | Check the failure messages printed during the batch run; re-run Cell 4 for the failed images |
