# AI Live Streaming — Hardware Specification Report

---

## Current Laptop Specifications

| Component | Specification |
|---|---|
| **Machine** | MTC_CT_01 |
| **OS** | Windows 11 Home (Build 26200) |
| **CPU** | Intel Core i7-13620H (13th Gen, 10-core hybrid) |
| **RAM** | 16 GB |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU |
| **VRAM** | 8 GB GDDR6 |
| **GPU TDP** | 30W (laptop power cap) |
| **NVIDIA Driver** | 536.52 |
| **Storage** | Kingston 1TB NVMe SSD |

---

## Hardware Problems and Incapabilities

### 1. GPU Power Cap (Primary Bottleneck)
The RTX 4060 **Laptop** GPU is capped at **30W TDP** by the manufacturer. The desktop version of the same GPU runs at **115W** — nearly 4× more power. This hard limit means the GPU cannot sustain its full boost clock under continuous inference load.

**Impact:** Sustained inference rate of **~2–4 FPS** instead of the 15–20 FPS a desktop RTX 4060 would achieve.

### 2. GPU Power State Throttling
The NVIDIA driver was defaulting to a low-power state (210 MHz clock) even under active GPU load. This was caused by the default power management mode being set to adaptive rather than performance-oriented.

**Fix applied:** NVIDIA Control Panel → Manage 3D Settings → Power management mode set to **Prefer Maximum Performance**.

**Impact before fix:** ~1.8 FPS. After fix: ~3–4 FPS stable.

### 3. Thermal Throttling Under Sustained Load
Running the GPU at unlimited inference speed causes the 30W thermal budget to be reached within seconds. The driver then throttles the clock to stay within the power envelope.

**Impact:** FPS spikes to ~7 briefly on startup, then drops to ~2 FPS sustained.

**Fix applied:** `--max-swap-fps 4` cap prevents sustained saturation and keeps the GPU at stable 3–4 FPS rather than oscillating between 7 and 1.

### 4. Intel Optimus (iGPU Passthrough)
The laptop uses NVIDIA Optimus — the NVIDIA GPU processes frames but output passes through the Intel iGPU to the display. This adds a memory copy overhead on every frame.

**Impact:** ~10–15% additional latency per frame. A MUX switch (if supported by this laptop model) would bypass this.

### 5. Virtual Camera Driver Conflict
pyvirtualcam (OBS Virtual Camera backend) and OBS Studio cannot both own the virtual camera output simultaneously.

**Workaround:** Run the face-swap script first (it claims the virtual camera), then open OBS. Do not click "Start Virtual Camera" in OBS.

### 6. Third-Party App Integration Limitation (AKOOL)
AKOOL desktop does not expose a camera input selector — it reads only from the physical default webcam. Since our pipeline outputs to a virtual camera device (OBS Virtual Camera), AKOOL cannot receive the face-swapped feed without either:
- A camera routing middleware (e.g. ManyCam)
- AKOOL adding a camera selector to their desktop app
- Using AKOOL's API directly

---

## AI Live Stream System — Minimum and Recommended Specifications

### Minimum Specifications (Functional, low quality)

| Component | Minimum |
|---|---|
| **OS** | Windows 10 64-bit |
| **CPU** | Intel Core i5 (8th Gen) or AMD Ryzen 5 3600 |
| **RAM** | 8 GB |
| **GPU** | NVIDIA GTX 1660 (6 GB VRAM) |
| **GPU TDP** | 120W+ |
| **VRAM** | 6 GB |
| **Storage** | 10 GB free (SSD preferred) |
| **Webcam** | 720p 30 FPS |
| **Expected FPS** | ~3–5 FPS inference |

### Recommended Specifications (Smooth, demo-ready)

| Component | Recommended |
|---|---|
| **OS** | Windows 11 64-bit |
| **CPU** | Intel Core i7 (12th Gen+) or AMD Ryzen 7 5800X+ |
| **RAM** | 16 GB+ |
| **GPU** | NVIDIA RTX 3070 / RTX 4070 (desktop) |
| **GPU TDP** | 150W+ |
| **VRAM** | 8 GB+ |
| **Storage** | SSD, 20 GB free |
| **Webcam** | 1080p 30 FPS |
| **Expected FPS** | ~15–25 FPS inference |

### Ideal Specifications (Production-ready, 30 FPS target)

| Component | Ideal |
|---|---|
| **OS** | Windows 11 64-bit |
| **CPU** | Intel Core i9 (13th Gen+) or AMD Ryzen 9 7900X |
| **RAM** | 32 GB |
| **GPU** | NVIDIA RTX 4080 / 4090 (desktop) |
| **GPU TDP** | 250W+ |
| **VRAM** | 12 GB+ |
| **Storage** | NVMe SSD, 50 GB free |
| **Webcam** | 1080p 60 FPS |
| **Expected FPS** | 25–30 FPS sustained inference |

---

## Summary

The current laptop is capable of running the AI live face-swap pipeline but is constrained by its 30W GPU power budget — a hardware limitation that cannot be resolved in software. For demo and development purposes the system is functional at 3–4 FPS stable. For a production deployment targeting 30 FPS, a desktop GPU with 150W+ TDP is required.
