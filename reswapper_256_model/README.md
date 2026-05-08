# AI Live Face Swap — Prototype

A real-time face-swap pipeline built from scratch using the same core libraries as leading open-source tools (InsightFace + reswapper_256), structured as clean, readable code that demonstrates every stage of the system end-to-end.

**Measured performance (RTX 4060 Laptop / CUDA + FP16):**
- Avg latency: ~190 ms end-to-end
- Inference rate: ~7 FPS (GPU-bound; display runs at 30 FPS smoothly via dedicated thread)
- Face stability: robust to lighting and moderate pose variation

---

## How It Works

### Pipeline Overview

```
Webcam
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Frame Capture (src/capture.py)               │
│  Background thread grabs frames via DirectShow (Windows) │
│  and exposes only the latest one to prevent stale reads. │
└───────────────────────┬─────────────────────────────────┘
                        │  BGR frame + capture timestamp
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Face Detection (src/detector.py)             │
│  InsightFace buffalo_l / SCRFD detects faces using only │
│  the two models the swap needs: det_10g (bounding box + │
│  5-point keypoints) and w600k_r50 (identity embedding). │
│  Three unused models are skipped via allowed_modules.   │
│  Detection runs at 320×320 and is cached every 4 frames │
│  (det_skip=4) — reused for frames in between.          │
└───────────────────────┬─────────────────────────────────┘
                        │  target face object
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — Face Swap (src/swapper.py)                   │
│  reswapper_256.onnx maps the source identity embedding  │
│  onto the target face geometry in a single ONNX forward │
│  pass. The swap region is warped to align with the      │
│  performer's head pose, then blended back into the full │
│  frame automatically (paste_back=True).                 │
└───────────────────────┬─────────────────────────────────┘
                        │  composited BGR frame
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 — Output (src/output.py)                       │
│  Renders a preview window with an optional HUD overlay  │
│  (FPS, latency, execution provider). If --virtual-cam   │
│  is set, the frame is also forwarded to OBS Virtual     │
│  Camera as an RGB feed, making it appear as a standard  │
│  webcam to any streaming platform.                      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
         OBS / Streaming Platform (e.g. Stripchat)
```

### Key Design Decisions

**Three-thread architecture** — Camera capture, inference, and display each run independently. The capture thread overwrites a single `(timestamp, frame)` slot with the latest camera frame. The inference thread picks up the latest frame, runs detect+swap, and writes the result to an output slot. The display loop reads the latest result at 30 FPS regardless of how fast inference completes — so the preview stays smooth even when GPU inference is slower than the camera.

**Detection frame skipping** — Face detection (det_10g) runs every 4 frames (`det_skip=4`); the cached bounding box and keypoints are reused for the 3 frames in between. This cuts detection cost by ~75% with negligible quality loss for a frontal webcam feed.

**Minimal buffalo_l usage** — buffalo_l ships 5 models. INSwapper only needs 2: `det_10g` (bounding box + 5-point keypoints for alignment) and `w600k_r50` (identity embedding for the swap). The other three — `1k3d68`, `2d106det`, `genderage` — are skipped via `allowed_modules=['detection', 'recognition']`.

**FP16 swap model** — The convolution and linear layers in reswapper_256 run in FP16 on CUDA Tensor Cores (~2× throughput). Numerically sensitive ops (`ReduceMean`, `Sqrt`, `Reciprocal`, `Tanh`, `Resize`) stay in FP32 to prevent the NaN / black-box artifact caused by variance underflow in the AdaIN normalisation chain. Convert once with `python convert_fp16.py`, then pass `--fp16` at runtime.

**Latency measurement** — The timestamp is taken at the moment the camera produces the frame (inside the capture thread), not when the main thread reads it. End-to-end latency is therefore the true pipeline delay: camera-out to virtual-camera-in.

**Execution providers** — ONNX Runtime tries providers in the order specified. `cuda` → `['CUDAExecutionProvider', 'CPUExecutionProvider']` means CUDA is used if available, with CPU as an automatic fallback. `directml` covers AMD and Intel GPUs on Windows without requiring CUDA.

**Detection threshold** — Set to 0.7 by default. Lower values detect more faces but introduce false positives that cause the swap to jump between faces ("drift"). Higher values (0.8–0.9) improve stability at the cost of missing partially occluded faces.

---

## Hardware Requirements

| Tier | GPU | Mode | Inference FPS | Avg Latency |
|---|---|---|---|---|
| Measured | RTX 4060 Laptop | CUDA + FP16 | ~7 FPS | ~190 ms |
| Measured | RTX 4060 Laptop | CUDA (FP32) | ~3.5 FPS | ~330 ms |
| Estimated | RTX 4080 / 4090 | CUDA + FP16 | ~20–30 FPS | ~100–150 ms |
| Estimated | RTX 3070 / 3080 | CUDA + FP16 | ~12–18 FPS | ~150–200 ms |
| CPU-only | Any | CPU | 1 – 3 FPS | 2 – 5 s |
| DirectML | AMD/Intel GPU | DirectML | 5 – 15 FPS | 0.3 – 0.6 s |

> Display always runs at 30 FPS via the dedicated inference thread — the preview is smooth regardless of inference rate. Only the frequency of new swapped frames is GPU-bound.

---

## Quick Start

### Prerequisites

- Windows 10 / 11
- Python 3.11 — [python.org/downloads](https://www.python.org/downloads/release/python-3110/)
- NVIDIA GPU: CUDA Toolkit 12.x + cuDNN 8.9.x (for `--execution-provider cuda`)
- OBS Studio with OBS Virtual Camera (optional, for `--virtual-cam` output)

### 1. Set up the environment

```powershell
# GPU build (default)
.\setup.ps1

# CPU-only build (functional test, not performance)
.\setup.ps1 -Cpu
```

The script creates a `.venv`, installs dependencies, and offers to download models.

**Or manually:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# pip install -r requirements-cpu.txt   # CPU only

python download_models.py             # downloads reswapper_256.onnx (~300 MB)
python convert_fp16.py                # converts swap model to FP16 (one-time, ~10 s)
```

> The `buffalo_l` detection model (~200 MB) is downloaded automatically by InsightFace on first run to `~/.insightface/models/buffalo_l/`.

### 2. Run the prototype

Replace `your_face.jpg` with the path to a real face photo (JPG or PNG). You can use
a filename if the photo is in the project folder, or a full path anywhere on your PC:

```powershell
# --- GPU (CUDA) + FP16 — recommended, best performance ---
python run.py --source "your_face.jpg" --execution-provider cuda --fp16
python run.py --source "President_Barack_Obama.jpg" --execution-provider cuda --fp16  # test image male
python run.py --source "Maira.jpg" --execution-provider cuda --fp16  # test image female
python run.py --source "President_Barack_Obama.jpg" --execution-provider cuda --fp16 --max-swap-fps 7 # with limitation on FPS for safety

# --- GPU (CUDA) FP32 — if --fp16 produces visual artifacts ---
python run.py --source "your_face.jpg" --execution-provider cuda

# --- CPU — slow (1–3 FPS), use only if no NVIDIA GPU available ---
python run.py --source "your_face.jpg" --execution-provider cpu

# --- GPU + virtual camera output (streams into OBS / broadcasting software) ---
python run.py --source "your_face.jpg" --execution-provider cuda --fp16 --virtual-cam
```

> The first run will pause ~1–2 minutes while InsightFace downloads the `buffalo_l`
> detection model (~200 MB). This is a one-time download.

Press **Q** in the preview window to stop.

---

## Using with OBS

### Method 1 — Virtual Camera (recommended for video calls and streaming)

The script feeds frames directly into OBS Virtual Camera, which appears as a webcam to any application (Zoom, Teams, Streamlabs, OBS, etc.).

**Important: do NOT start OBS Virtual Camera inside OBS before running the script.**

1. Run the script with `--virtual-cam`:
   ```powershell
   python run.py --source "your_face.jpg" --execution-provider cuda --fp16 --virtual-cam
   ```
2. Wait until the terminal prints: `Virtual camera active: OBS Virtual Camera`
3. Open OBS Studio
4. In the **Sources** panel click **+** → **Video Capture Device**
5. Name it (e.g. "Face Swap") → click **OK**
6. In the **Device** dropdown select **OBS Virtual Camera** → click **OK**
7. The face-swapped feed appears in your OBS scene

> If OBS prints "Failed to start virtual camera", it means OBS is trying to start Virtual Camera while the script already owns it. Click OK on the error and skip starting it — the script already started it. Just add the Video Capture Device source directly.

### Method 2 — Display Capture (fallback, no virtual camera needed)

1. Run the script **without** `--virtual-cam`:
   ```powershell
   python run.py --source "your_face.jpg" --execution-provider cuda --fp16
   ```
2. The preview window titled **"AI Face Swap | press Q to quit"** opens
3. In OBS, Sources → **+** → **Display Capture** → OK → OK
4. Your full screen is captured — position the face swap window where you want it in the OBS canvas

### Stopping the script

- Click the preview window to focus it, then press **Q**
- Or press **Ctrl+C** in the PowerShell terminal

---

## CLI Reference

### `run.py`

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Path to source face image (JPG/PNG) |
| `--camera` | `0` | Webcam device index |
| `--width` | `1280` | Capture width in pixels |
| `--height` | `720` | Capture height in pixels |
| `--fps` | `30` | Target display frame rate |
| `--execution-provider` | `cuda` | `cuda` / `directml` / `cpu` |
| `--model` | `models/reswapper_256.onnx` | Path to ONNX swap model |
| `--fp16` | off | Use FP16 swap model for ~2× speedup (run `convert_fp16.py` first) |
| `--det-size` | `320` | Detection input resolution (`320` or `640`). 320 is faster; use 640 if faces are missed |
| `--virtual-cam` | off | Forward output to OBS Virtual Camera |
| `--no-metrics` | off | Hide FPS / latency HUD overlay |
| `--det-thresh` | `0.7` | Face detection confidence threshold (0–1) |

### `benchmark.py`

Runs the full pipeline for a fixed duration and reports pass/fail against the targets from the technical specification.

```powershell
python benchmark.py --source your_face.jpg --duration 60
```

| Flag | Default | Description |
|---|---|---|
| `--duration` | `60` | Benchmark duration in seconds |
| *(all run.py flags)* | — | Same options apply |

**Pass criteria:**

| Metric | Target |
|---|---|
| Avg FPS | ≥ 30 |
| Avg latency | ≤ 400 ms |
| P95 latency | ≤ 500 ms |
| Face-lock rate | ≥ 95% of frames |

---

## Project Structure

```
AI Live Streaming/
├── src/
│   ├── capture.py          Stage 1 — background-thread camera capture (CAP_DSHOW)
│   ├── detector.py         Stage 2 — InsightFace SCRFD detection (2 of 5 models, det_skip=4)
│   ├── swapper.py          Stage 3 — reswapper_256 face swap (ORT_ENABLE_ALL optimisation)
│   ├── output.py           Stage 4 — preview window + OBS Virtual Camera
│   ├── metrics.py          Rolling FPS counter and latency tracker
│   └── pipeline.py         Inference thread + display loop; HUD overlay; session summary
├── models/
│   ├── reswapper_256.onnx      (download via download_models.py)
│   └── reswapper_256_fp16.onnx (generate via convert_fp16.py)
├── run.py                  CLI entry point
├── benchmark.py            Performance benchmark with pass/fail verdict
├── convert_fp16.py         Converts swap model to mixed-precision FP16 (one-time)
├── download_models.py      Downloads reswapper_256.onnx from HuggingFace
├── setup.ps1               One-shot Windows environment setup script
├── requirements.txt        GPU (CUDA) dependencies
└── requirements-cpu.txt    CPU-only dependencies (testing)
```

---

## Troubleshooting

**`onnxruntime` falls back to CPU (< 3 FPS)**
CUDA / cuDNN version mismatch. Check: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — should include `CUDAExecutionProvider`. Install from the Microsoft CUDA 12 pip index (the command in the manual setup step above) — standard PyPI ships a CUDA 11 build that fails with error 126 on systems with CUDA 12.x.

**`No face detected in source image`**
The source photo must contain a clearly visible, forward-facing face. Re-shoot or use a cleaner crop.

**Virtual camera not found / `virtual camera output could not be started`**
Do NOT start OBS Virtual Camera inside OBS before running the script. The script (pyvirtualcam) must start it first. Correct order: run the script with `--virtual-cam` → wait for "Virtual camera active" message → then open OBS and add a Video Capture Device source. If OBS already has Virtual Camera running, stop it first.

**Face drift / popping at profile angles**
Expected behaviour — current models are trained on near-frontal data. Raise `--det-thresh` to 0.8–0.9 to reduce false detections, and keep the performer's face within ~45 degrees of camera-facing.
