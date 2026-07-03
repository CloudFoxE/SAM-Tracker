# Tracker — Video Object & Contour Tracking (SAM2 / SAM3)

A Windows desktop app that tracks objects in videos. You click a few points on the first
frame of a video, and the app (powered by Meta's **SAM2** or **SAM3** AI models) follows
those objects through every frame and exports:

- **CSV files** with the exact outline (contour) coordinates of each object, per frame.
- **MP4 videos** with the tracked objects highlighted.

It was built for tracking GUVs (giant unilamellar vesicles) and other bright particles in
microscopy videos, but works for general object tracking too.

> **This guide assumes no coding experience.** Follow the steps in order and copy‑paste the
> commands exactly.

---

## What you need first

1. **A Windows PC with an NVIDIA graphics card (GPU)** and a recent driver.
   - This app needs an NVIDIA GPU (e.g. RTX 30/40/50 series). Update your driver from
     [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) if unsure.
2. **Git** — the tool used to download the code.
   - Download & install: [git-scm.com/download/win](https://git-scm.com/download/win)
     (accept all defaults).
3. **uv** — the tool that installs Python and everything the app needs, in one command.
   - Open **PowerShell** (press the Start button, type `PowerShell`, press Enter) and paste:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **Close and reopen PowerShell** afterward so it picks up `uv`.
   - (You do **not** need to install Python yourself — `uv` handles that automatically.)
4. **(For the SAM3 model only)** a free **Hugging Face** account:
   [huggingface.co/join](https://huggingface.co/join).

---

## Step 1 — Download the app

In PowerShell, run these two lines (replace the URL with this project's GitHub address):

```powershell
git clone https://github.com/CloudFoxE/SAM-Tracker.git
cd SAM-Tracker
```

`git clone` downloads the code into a new folder; `cd` moves you into it. **Stay in this
folder** for every command below.

---

## Step 2 — Install the app

This one command creates an isolated environment and installs everything (Python 3.13,
PyTorch, the AI libraries, the app). It downloads a few GB, so give it several minutes:

```powershell
uv sync
```

- This installs what you need to run the **SAM3** backend (recommended).
- **If you also want the SAM2 backend** (works fully offline), run this instead/as well:
  ```powershell
  $env:SAM2_BUILD_CUDA="0"; $env:SAM2_BUILD_ALLOW_ERRORS="1"; uv sync --extra dev --extra sam2
  ```
  (The `SAM2_BUILD_*` settings skip an optional component that isn't needed — safe to
  ignore any `_C` warning later.)

You never have to "activate" anything — every command below starts with `uv run`, which
automatically uses the environment `uv` just created.

---

## Step 3 — Get the AI model weights

The app's *code* is installed; now you need the model *weights* (the trained AI files).
**Pick the backend you want** — you can set up either or both.

### Option A — SAM3 (recommended, newer/better)

1. **Get access** (one time): sign in to Hugging Face, open
   **[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)**, and click to
   accept the license / request access. Approval may take a little while.
2. **Get a token:** go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   create a token with the **Read** role, and copy it.
3. **Log in** (one time, in PowerShell):
   ```powershell
   uv run hf auth login
   ```
   Paste the token when asked. The ~3.4 GB model then **downloads automatically the first
   time you run the app** — no manual file handling.

### Option B — SAM2 (works offline, no account needed)

Download a model file into a `checkpoints` folder. The default is the "large" model:

```powershell
mkdir checkpoints
curl.exe -L -o checkpoints\sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
```

(Or just paste that `https://...` link into your web browser to download the file, then move
it into the `checkpoints` folder.) Other sizes are listed in the table further below.

---

## Step 4 — Choose your backend and model (the settings you can change)

All choices live in **one file**: `src/tracker/config/settings.py`. Open it in any text
editor (e.g. Notepad, or the built‑in editor in your file browser). You only ever touch the
lines shown below.

### 4a. Choose SAM2 vs SAM3

Find this line near the top:

```python
TRACKER_BACKEND = os.environ.get("TRACKER_BACKEND", "sam3")
```

It's set to **`"sam3"`** by default (recommended). Change **`"sam3"`** to **`"sam2"`** if you'd
rather use the SAM2 backend. That's the only change needed to switch models. Save the file.

> **No‑edit alternative:** instead of editing the file, you can choose the backend each time
> you launch, straight from PowerShell:
> ```powershell
> $env:TRACKER_BACKEND="sam3"   # or "sam2"
> uv run python src\tracker\app.py
> ```

### 4b. (SAM3 only) Choose which SAM3 model

Find:
```python
TRACKER_MODEL_ID = os.environ.get("TRACKER_MODEL_ID", "facebook/sam3")
```
There are currently **two** official Meta SAM3 model IDs you can select here:
- **`facebook/sam3`** — the default.
- **`facebook/sam3.1`** — the newer update (tracks more objects per pass; better in crowded scenes).

Leave it as `facebook/sam3` unless you want to try `facebook/sam3.1`. (The `facebook/sam-3d-*`
models are a different, incompatible product — single-image 3D reconstruction, not video
tracking — do not use those here.)

### 4c. (SAM2 only) Choose which SAM2 model size

By default SAM2 uses the **large** model you downloaded in Step 3B. To use a different size,
download that file (below) and set two values. The easiest way is in PowerShell before
launching:

```powershell
$env:SAM2_CHECKPOINT_PATH="D:\path\to\your\checkpoints\sam2.1_hiera_base_plus.pt"
$env:SAM2_CONFIG_PATH="configs/sam2.1/sam2.1_hiera_b+.yaml"
uv run python src\tracker\app.py
```

**SAM2 model sizes** (download link = base URL `https://dl.fbaipublicfiles.com/segment_anything_2/092824/` + the file name):

| Size | File name | The `...CONFIG_PATH` (yaml) value | Approx. size |
|---|---|---|---|
| Tiny (fastest) | `sam2.1_hiera_tiny.pt` | `configs/sam2.1/sam2.1_hiera_t.yaml` | ~150 MB |
| Small | `sam2.1_hiera_small.pt` | `configs/sam2.1/sam2.1_hiera_s.yaml` | ~180 MB |
| Base+ | `sam2.1_hiera_base_plus.pt` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | ~320 MB |
| **Large (best, default)** | `sam2.1_hiera_large.pt` | `configs/sam2.1/sam2.1_hiera_l.yaml` | ~857 MB |

> Bigger = more accurate but slower. Large is the default and recommended.

### 4d. (Advanced) Precision — only if you run low on memory

Both models run at full precision (`float32`) by default. If a **long or high‑resolution
video** runs out of memory, freezes, or crashes partway through, switching to **half
precision** (`bfloat16`) roughly halves the memory it uses, with almost no effect on the
results — see the [Troubleshooting](#troubleshooting) note below. Most users never need to
touch this.

---

## Step 5 — Run the app

```powershell
uv run python src\tracker\app.py
```

The window **"Tracker"** opens. The first time you use SAM3 it will pause ~15 seconds (or
longer on first run, while it downloads the model) — this is normal. **Tip:** don't close the
window while the model is still loading.

---

## Step 6 — Use the app

1. **Browse…** and pick a folder containing your videos (`.mp4`, `.avi`, `.mov`, `.mkv`).
   They appear in the left list.
2. **Select a video.** Its first frame shows in the middle.
3. **Mark your object** on that first frame:
   - **Left‑click** on the object you want to track (green point = "this is it").
   - **Right‑click** on areas to exclude (red point = "not this").
   - Add a few points for better accuracy.
4. Click **Generate Mask** to preview, then **Save Mask** when it looks right. The video is
   marked **Complete**. Repeat for each video (or mark ones you don't want as **Skip**).
5. Tick the export options (**tracked videos** and/or **contour CSV files**) and click
   **Start Streamed Analyze & Export**. Progress shows at the bottom.

**Where your results go:** inside the output folder you chose, under `exports/`:
- `exports/tracked_videos/` — the highlighted MP4s.
- `exports/contour_data/<video name>/frame_0000.csv, ...` — the outline coordinates per frame.

---

## Troubleshooting

- **"Model Load Error" / "Failed to load the tracking model."**
  - If using **SAM2**: make sure you ran the SAM2 install line in Step 2 and downloaded a
    checkpoint into `checkpoints/` (Step 3B).
  - If using **SAM3**: make sure you were approved for `facebook/sam3` and ran
    `uv run hf auth login` (Step 3A).
- **It says my GPU / CUDA isn't available.** Update your NVIDIA driver and confirm you have an
  NVIDIA GPU. This app requires one.
- **The app can't find SAM2 after I ran `uv sync` again.** A plain `uv sync` removes the
  optional SAM2 part. Re‑run the SAM2 install line from Step 2 (`uv sync --extra dev --extra sam2`).
- **A `Cannot import name '_C'` warning appears.** Harmless — safe to ignore.
- **The app runs out of memory, freezes, or crashes partway through a long video.**
  Switch the model to **half‑precision** — it roughly halves memory use with almost no effect
  on results. In `src/tracker/config/settings.py`, change the line for the backend you're
  using from `"float32"` to `"bfloat16"`:
  ```python
  SAM3_COMPUTE_DTYPE = os.environ.get("SAM3_COMPUTE_DTYPE", "bfloat16")   # if using SAM3
  SAM2_COMPUTE_DTYPE = os.environ.get("SAM2_COMPUTE_DTYPE", "bfloat16")   # if using SAM2
  ```
  Longer and higher‑resolution videos use more memory, so this is the first thing to try.
- **Something else?** Open an issue on the repository.

---

## For developers

- The tracking backend is pluggable behind a `TrackerBackend` interface
  (`src/tracker/tracking/`), selected by `TRACKER_BACKEND` (`"sam2"` or `"sam3"`).

---

## License

This project's **code** is released under the [MIT License](LICENSE) — you're free to
use, modify, and redistribute it; just keep the copyright notice.

**The AI model weights are licensed separately by Meta, not by this project.** In
particular, **SAM3 is for non‑commercial research use only**, and SAM2 has its own
license terms. Review Meta's licenses before using the models — especially for any
commercial purpose. This MIT license covers only the application code in this repository.
