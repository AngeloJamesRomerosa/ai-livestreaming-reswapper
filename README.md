# AI Livestreaming Reswapper

Real-time face swap for live streaming. Upload a source face, start a session, and your webcam feed is face-swapped live — output is served over the browser and picked up by OBS for streaming to Twitch, YouTube, or any RTMP platform.

Runs fully locally. No cloud APIs, no subscriptions.

---

## How It Works

```
Browser Webcam
      │
      │  JPEG frames @ 15–30 fps (WebSocket)
      ▼
┌─────────────────────────────────────────┐
│  FastAPI Server (app.py)                │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Face Detector                  │   │
│  │  InsightFace buffalo_l / SCRFD  │   │
│  │  det_skip=4 (75% cost reduction)│   │
│  └────────────────┬────────────────┘   │
│                   │ target face         │
│  ┌────────────────▼────────────────┐   │
│  │  Face Swapper                   │   │
│  │  reswapper_256 ONNX             │   │
│  │  CUDA / DirectML / CPU          │   │
│  └────────────────┬────────────────┘   │
└───────────────────┼─────────────────────┘
                    │  JPEG frames (WebSocket back to browser)
                    ▼
             Browser Canvas
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   OBS Media Source    OBS Browser Source
   /stream.mjpeg       /viewer.html
          │                   │
          └─────────┬─────────┘
                    ▼
          Twitch / YouTube / RTMP
```

**Performance (RTX 4060 Laptop, CUDA + FP16):**

| Mode | Swap FPS | Latency |
|---|---|---|
| CUDA + FP16 | ~7 FPS | ~190 ms |
| CUDA FP32 | ~3.5 FPS | ~330 ms |
| CPU only | 1–3 FPS | 2–5 s |
| DirectML (AMD/Intel) | 5–15 FPS | 0.3–0.6 s |

> Display runs at full 30 FPS regardless of inference rate — the preview stays smooth even when the GPU produces swapped frames at 7 FPS.

---

## Requirements

- Windows 10 / 11
- Python 3.11
- NVIDIA GPU with CUDA 12.x drivers (recommended — CPU fallback works but is slow)
- A webcam

---

## Setup

### 1. Install Python dependencies

The project uses a shared `.venv` for both the model and the server.

```powershell
# GPU build (recommended)
cd reswapper_256_model
.\setup.ps1
```

Then install the server dependencies into the same venv:

```powershell
cd ..
.\.venv\Scripts\python.exe -m pip install -r requirements-server.txt
```

### 2. Download and convert models

```powershell
cd reswapper_256_model
.\.venv\Scripts\python.exe download_models.py     # downloads reswapper_256.onnx (~529 MB)
.\.venv\Scripts\python.exe convert_fp16.py         # converts to FP16 for ~2x GPU speedup (one-time, ~10s)
cd ..
```

> `buffalo_l` (~200 MB) is downloaded automatically by InsightFace on first session start.

### 3. Configure (optional)

Copy `.env.example` to `.env` and adjust as needed:

```env
SERVER_PORT=8000
STREAM_SECRET=changeme        # key required for the MJPEG URL
EXECUTION_PROVIDER=auto       # auto / cuda / directml / cpu
MAX_SWAP_FPS=10               # cap GPU inference rate to prevent throttling
DET_THRESH=0.7                # face detection confidence threshold (0–1)
DET_SIZE=320                  # detection resolution (320 = faster, 640 = more accurate)
```

`EXECUTION_PROVIDER=auto` detects your GPU automatically. If CUDA is found it uses FP16 automatically. If CUDA fails to initialise it reloads with FP32 on CPU.

---

## Running the Server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Using the Web Interface

1. **Choose Image** — upload a photo of the face you want to swap in. The face must be clearly visible and roughly forward-facing.
2. **Stream Settings** — pick your resolution (144p–1080p) and frame rate (15/25/30 fps) before starting.
3. **Start** — first run downloads InsightFace models (~200 MB, one-time). Subsequent starts take ~5 seconds.
4. Watch the **Status** panel — each component shows its state individually:
   - `Loading…` (blue pulse) while models initialise
   - `Ready` / `Active` (green) when confirmed working
   - `Warning` (yellow) if GPU was requested but fell back to CPU
   - `Failed` (red) on error
5. Watch the **Activity Log** at the bottom for step-by-step progress messages.
6. Once the session is active, your swapped output appears in the right video panel.

---

## OBS Integration

After starting a session, the **OBS Stream** section shows two URLs:

### MJPEG stream (OBS Media Source)
```
http://localhost:8000/stream.mjpeg?sid=<sid>&key=<stream_secret>
```
In OBS: **Sources → + → Media Source** → uncheck "Local File" → paste the URL.

### Browser Source viewer
```
http://localhost:8000/viewer.html?sid=<sid>
```
In OBS: **Sources → + → Browser** → paste the URL → set width/height to match your resolution.

---

## CUDA / GPU Setup

The server registers NVIDIA pip-package DLL paths at startup via `cuda_setup.py`. If CUDA is installed but ONNX falls back to CPU, check the Activity Log for:

```
No CUDA DLLs found — will use CPU if CUDA unavailable
```

If you see this, install the CUDA 12 build of onnxruntime-gpu:

```powershell
.\.venv\Scripts\python.exe -m pip install onnxruntime-gpu ^
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

Then restart the server. The Activity Log will show:

```
CUDA DLLs registered: N handles loaded   ✅
GPU confirmed: CUDA is active            ✅
```

### FP16 model

FP16 is selected automatically when CUDA is confirmed active. It gives ~2× speedup over FP32 on RTX 30/40 series by running convolutions on Tensor Cores. If CUDA fails and the model falls back to CPU, FP32 is reloaded automatically (FP16 is slower on CPU).

---

## Project Structure

```
ai-livestreaming-reswapper/
│
├── app.py                        FastAPI entry point
├── config.py                     Settings loaded from .env
├── cuda_setup.py                 Registers NVIDIA DLL paths before onnxruntime loads
├── requirements-server.txt       Server-only dependencies (FastAPI, uvicorn)
│
├── routes/
│   ├── faces.py                  POST /api/uploadImage
│   ├── session.py                POST /api/session/create  |  /api/session/close
│   ├── status.py                 GET  /api/status
│   ├── log.py                    GET  /api/log  (SSE activity stream)
│   ├── stream.py                 WS   /ws/stream-in  (viewer relay)
│   └── stream_mjpeg.py           GET  /stream.mjpeg  (OBS Media Source)
│
├── providers/
│   └── reswapper/
│       ├── provider.py           Model loading, inference worker, GPU detection
│       └── routes.py             WS /ws/swap  (browser ↔ server frame exchange)
│
├── public/
│   ├── index.html                Main control panel
│   ├── viewer.html               Pop-out viewer (OBS Browser Source)
│   ├── js/session.js             Webcam capture, WebSocket swap, status polling
│   └── css/style.css             Dark UI styles
│
├── uploads/                      Uploaded face images (git-ignored)
│
└── reswapper_256_model/          Original CLI pipeline
    ├── src/
    │   ├── capture.py            Stage 1 — background-thread camera capture
    │   ├── detector.py           Stage 2 — InsightFace SCRFD (det_skip=4)
    │   ├── swapper.py            Stage 3 — reswapper_256 ONNX inference
    │   ├── output.py             Stage 4 — preview window + OBS Virtual Camera
    │   ├── metrics.py            Rolling FPS / latency tracker
    │   └── pipeline.py           Inference thread + display loop
    ├── models/
    │   ├── reswapper_256.onnx         (download via download_models.py)
    │   └── reswapper_256_fp16.onnx    (generate via convert_fp16.py)
    ├── run.py                    CLI entry point (standalone, no server)
    ├── download_models.py        Downloads reswapper_256.onnx from HuggingFace
    ├── convert_fp16.py           Converts swap model to FP16 (one-time)
    ├── setup.ps1                 One-shot Windows environment setup
    ├── requirements.txt          GPU (CUDA 12) dependencies
    └── requirements-cpu.txt      CPU-only dependencies
```

---

## CLI Mode (Standalone, No Browser)

The original pipeline can still run without the server — outputs directly to a preview window and OBS Virtual Camera:

```powershell
cd reswapper_256_model
.\.venv\Scripts\python.exe run.py --source "your_face.jpg" --execution-provider cuda --fp16 --virtual-cam
```

See [`reswapper_256_model/README.md`](reswapper_256_model/README.md) for full CLI documentation.

---

## Troubleshooting

**ONNX falls back to CPU despite NVIDIA GPU**
The Activity Log will show `WARNING: Requested CUDA but running on CPU`. Install the CUDA 12 pip build of onnxruntime-gpu (see [CUDA / GPU Setup](#cuda--gpu-setup) above).

**`No face detected in source image`**
The source photo must contain a clearly visible, roughly forward-facing face. Avoid sunglasses, heavy shadow, or extreme angles.

**Very low FPS (< 1)**
You are on CPU. Check the GPU Provider row in the Status panel. If it shows `Warning`, CUDA did not initialise — see the CUDA setup section.

**Blank output canvas**
The webcam may not have loaded before the WebSocket connected. Stop the session, refresh the page, and start again.

**OBS MJPEG stream shows black or freezes**
Confirm the session is active (Status panel shows `Active` for Swap Stream). The MJPEG URL requires `key=` to match `STREAM_SECRET` in your `.env`.
