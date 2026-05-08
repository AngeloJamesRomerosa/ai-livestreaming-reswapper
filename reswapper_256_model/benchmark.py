"""
Pipeline performance benchmark.

Runs the full detect+swap loop against the webcam for a fixed duration and
reports whether the result meets the pass criteria from the technical report:
  - Sustained FPS >= 30
  - Average latency < 400 ms
  - P95 latency < 500 ms
  - Face-lock rate >= 95 % (face detected in >=95% of frames)

Usage:
    python benchmark.py --source face.jpg --duration 60
    python benchmark.py --source face.jpg --duration 60 --execution-provider cuda
"""

import argparse
import os
import sys
import time
import cv2

from src.capture import CameraCapture
from src.detector import FaceDetector
from src.swapper import FaceSwapper
from src.metrics import FPSCounter, LatencyTracker
from src.pipeline import _resolve_providers


PASS_FPS = 30
PASS_LAT_AVG_MS = 400
PASS_LAT_P95_MS = 500
PASS_FACE_LOCK_RATE = 0.95


def run_benchmark(cfg):
    providers = _resolve_providers(cfg.execution_provider)

    capture = CameraCapture(camera_index=cfg.camera, width=cfg.width, height=cfg.height, fps=cfg.fps)
    detector = FaceDetector(providers=providers, det_thresh=cfg.det_thresh)
    swapper  = FaceSwapper(model_path=cfg.model_path, providers=providers)

    fps_counter = FPSCounter(window=300)
    latency     = LatencyTracker(window=300)

    source_face = detector.get_face_from_image(cfg.source)

    capture.start()
    print(f"Warming up (5 s) ...")
    # Allow CUDA JIT to compile kernels on the first few frames so warm-up
    # outliers don't skew the benchmark latency numbers.
    time.sleep(5)

    print(f"Benchmarking for {cfg.duration} s — do not quit early.\n")
    deadline = time.perf_counter() + cfg.duration

    total_frames = 0
    face_detected_frames = 0
    last_seen = None

    while time.perf_counter() < deadline:
        frame_time, frame = capture.read()
        # Skip None (startup) and repeated frame objects (GPU slower than camera)
        if frame is None or frame is last_seen:
            continue
        last_seen = frame
        total_frames += 1

        targets = detector.detect(frame)
        if targets:
            face_detected_frames += 1
            swapper.swap(frame, targets[0], source_face)

        latency.record(frame_time)
        fps_counter.tick()

        elapsed = cfg.duration - (deadline - time.perf_counter())
        # Print a live progress line every 30 frames (~1 s at 30 FPS)
        if total_frames % 30 == 0:
            print(
                f"  {elapsed:5.1f}s | FPS {fps_counter.fps:5.1f} | "
                f"Lat avg {latency.mean_ms:5.0f} ms | "
                f"Face lock {face_detected_frames}/{total_frames}",
                end="\r",
            )

    capture.stop()
    print()

    face_lock_rate = face_detected_frames / total_frames if total_frames else 0.0

    print("\n" + "=" * 52)
    print("  BENCHMARK RESULTS")
    print("=" * 52)
    _row("Avg FPS",         f"{fps_counter.fps:.1f}",      fps_counter.fps >= PASS_FPS,           f">= {PASS_FPS}")
    _row("Avg latency",     f"{latency.mean_ms:.0f} ms",   latency.mean_ms <= PASS_LAT_AVG_MS,    f"<= {PASS_LAT_AVG_MS} ms")
    _row("P95 latency",     f"{latency.p95_ms:.0f} ms",    latency.p95_ms  <= PASS_LAT_P95_MS,    f"<= {PASS_LAT_P95_MS} ms")
    _row("Min / Max lat",   f"{latency.min_ms:.0f} / {latency.max_ms:.0f} ms", True, "—")
    _row("Face lock rate",  f"{face_lock_rate*100:.1f} %", face_lock_rate  >= PASS_FACE_LOCK_RATE, f">= {PASS_FACE_LOCK_RATE*100:.0f} %")
    _row("Total frames",    str(total_frames),              True, "—")
    print("=" * 52)

    passed = all([
        fps_counter.fps >= PASS_FPS,
        latency.mean_ms <= PASS_LAT_AVG_MS,
        latency.p95_ms  <= PASS_LAT_P95_MS,
        face_lock_rate  >= PASS_FACE_LOCK_RATE,
    ])
    verdict = "PASS — pipeline meets production targets." if passed else "FAIL — one or more targets missed."
    print(f"\n  Verdict: {verdict}")
    if not passed:
        if fps_counter.fps < PASS_FPS:
            print("  Tip: FPS low — check GPU utilisation with: nvidia-smi dmon -s u")
        if latency.mean_ms > PASS_LAT_AVG_MS:
            print("  Tip: Latency high — try --execution-provider cuda or disable face enhancers.")
        if face_lock_rate < PASS_FACE_LOCK_RATE:
            print("  Tip: Face lock poor — raise --det-thresh slightly (0.7 → 0.8).")
    print()


def _row(label, value, passed, target):
    status = "PASS" if passed else "FAIL"
    print(f"  {label:<18} {value:<18} [{status}]  target: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Face-swap pipeline benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",   required=True)
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--width",    type=int, default=1280)
    parser.add_argument("--height",   type=int, default=720)
    parser.add_argument("--fps",      type=int, default=30)
    parser.add_argument("--duration", type=int, default=60,
                        help="Benchmark duration in seconds")
    parser.add_argument("--execution-provider",
                        choices=["cuda", "directml", "cpu"], default="cuda")
    parser.add_argument("--model",
                        default=os.path.join("models", "reswapper_256.onnx"))
    parser.add_argument("--det-thresh", type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"[error] Source image not found: {args.source}")
    if not os.path.exists(args.model):
        sys.exit(f"[error] Model not found: {args.model}\n        Run: python download_models.py")

    class _Cfg:
        pass
    cfg = _Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k.replace("-", "_"), v)
    cfg.model_path = args.model

    run_benchmark(cfg)


if __name__ == "__main__":
    main()
