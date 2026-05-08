import cv2
import time
import threading

from .capture import CameraCapture
from .detector import FaceDetector
from .swapper import FaceSwapper
from .output import FrameOutput
from .metrics import FPSCounter, LatencyTracker


class _InferenceWorker(threading.Thread):
    """Background thread: face detection + swap, decoupled from display.

    Detection runs only every det_skip frames; the cached face position is
    reused for intermediate frames so the expensive buffalo_l models run a
    fraction of the time.  The display loop reads whatever the latest
    processed result is, independent of how fast inference completes.
    """

    def __init__(self, detector: FaceDetector, swapper: FaceSwapper,
                 source_face, det_skip: int = 4, max_fps: float = 0):
        super().__init__(daemon=True, name="inference")
        self._detector = detector
        self._swapper = swapper
        self._source_face = source_face
        self._det_skip = det_skip
        self._min_interval = (1.0 / max_fps) if max_fps > 0 else 0

        # Input slot — written by display loop, read by this thread
        self._in_lock = threading.Lock()
        self._in_frame = None
        self._in_time = None
        self._wakeup = threading.Event()

        # Output slot — written by this thread, read by display loop
        self._out_lock = threading.Lock()
        self._out_frame = None
        self._out_time = None

        self._running = True
        self._frame_count = 0
        self._cached_faces = []

    def submit(self, timestamp: float, frame) -> None:
        """Hand the latest camera frame to the worker (non-blocking)."""
        with self._in_lock:
            self._in_frame = frame
            self._in_time = timestamp
        self._wakeup.set()

    def latest_result(self):
        """Return (timestamp, frame) of the most recently processed result."""
        with self._out_lock:
            return self._out_time, self._out_frame

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()  # unblock the wait() so the thread exits cleanly

    def run(self) -> None:
        last_frame = None
        last_inference_time = 0.0
        while self._running:
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()

            with self._in_lock:
                frame = self._in_frame
                frame_time = self._in_time

            if frame is None or frame is last_frame:
                continue

            # Pace inference to max_fps so the GPU never runs at 100%
            # continuously — prevents thermal throttling on power-limited GPUs.
            if self._min_interval > 0:
                elapsed = time.perf_counter() - last_inference_time
                wait = self._min_interval - elapsed
                if wait > 0:
                    time.sleep(wait)

            last_frame = frame
            last_inference_time = time.perf_counter()
            self._frame_count += 1

            # Run detection every det_skip frames; reuse cached faces otherwise.
            # This cuts the buffalo_l model cost by ~75% at det_skip=4.
            if self._frame_count % self._det_skip == 1 or not self._cached_faces:
                self._cached_faces = self._detector.detect(frame)

            if self._cached_faces and self._source_face is not None:
                result = self._swapper.swap(frame, self._cached_faces[0], self._source_face)
            else:
                result = frame

            with self._out_lock:
                self._out_frame = result
                self._out_time = frame_time


class FaceSwapPipeline:
    """Orchestrates the full real-time face-swap pipeline:

        Camera → [inference thread: Detect → Swap] → Display loop → Preview / Virtual Camera

    The inference thread runs detect+swap as fast as the GPU allows.
    The display loop runs independently at the target FPS, always showing
    the latest available result — so the preview stays smooth even when
    GPU inference is slower than the camera frame rate.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        providers = _resolve_providers(cfg.execution_provider)

        self._capture = CameraCapture(
            camera_index=cfg.camera,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )
        det_size = (cfg.det_size, cfg.det_size)
        self._detector = FaceDetector(providers=providers, det_thresh=cfg.det_thresh, det_size=det_size)
        self._swapper = FaceSwapper(model_path=cfg.model_path, providers=providers)
        self._output = FrameOutput(
            window_title="AI Face Swap | press Q to quit",
            use_virtual_cam=cfg.virtual_cam,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )
        self._fps = FPSCounter()
        self._latency = LatencyTracker()
        self._source_face = None

    def load_source(self, image_path: str):
        print(f"Loading source face: {image_path}")
        self._source_face = self._detector.get_face_from_image(image_path)
        print("Source face ready.")

    def run(self):
        self._capture.start()
        self._output.start()

        worker = _InferenceWorker(
            self._detector, self._swapper,
            self._source_face, det_skip=4, max_fps=self._cfg.max_swap_fps,
        )
        worker.start()

        print("Pipeline running — press Q in the preview window to quit.")

        _interval = 1.0 / self._cfg.fps   # target display period (e.g. 33 ms at 30 FPS)
        _last_result_time = None           # track when inference produced a new result

        try:
            while True:
                _tick_start = time.perf_counter()

                # Submit the latest camera frame to the inference thread
                frame_time, frame = self._capture.read()
                if frame is not None:
                    worker.submit(frame_time, frame)

                # Display the latest processed result
                result_time, result = worker.latest_result()

                if result is None:
                    # Inference hasn't produced anything yet — show raw camera
                    if frame is None:
                        if self._output.quit_requested():
                            break
                        continue
                    result = frame
                    result_time = frame_time

                # Only tick metrics when inference has produced a NEW result,
                # not every display frame — avoids inflating latency with stale timestamps.
                if result_time != _last_result_time:
                    self._latency.record(result_time)
                    self._fps.tick()
                    _last_result_time = result_time

                if self._cfg.show_metrics:
                    # Copy before drawing — result is the worker's cached frame and
                    # drawing on it in-place would accumulate HUD text every display tick.
                    display = result.copy()
                    _draw_hud(display, self._fps.fps, self._latency, self._cfg.execution_provider)
                else:
                    display = result

                self._output.send(display)

                if self._output.quit_requested():
                    break

                # Throttle display loop to target FPS so we don't spin the CPU
                # and starve the inference thread running on the same machine.
                elapsed = time.perf_counter() - _tick_start
                sleep_for = _interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            pass
        finally:
            worker.stop()
            worker.join(timeout=2.0)
            self._capture.stop()
            self._output.stop()
            _print_summary(self._fps, self._latency)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_providers(name: str) -> list:
    # ONNX Runtime attempts providers in order; CPU is always the final fallback
    # so the pipeline degrades gracefully if CUDA or DirectML is unavailable.
    return {
        'cuda':     ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'directml': ['DmlExecutionProvider',  'CPUExecutionProvider'],
        'cpu':      ['CPUExecutionProvider'],
    }.get(name, ['CPUExecutionProvider'])


def _draw_hud(frame, fps: float, latency: LatencyTracker, provider: str):
    lines = [
        f"FPS:     {fps:.1f}",
        f"Lat avg: {latency.mean_ms:.0f} ms",
        f"Lat p95: {latency.p95_ms:.0f} ms",
        f"Exec:    {provider.upper()}",
    ]
    y = 28
    for line in lines:
        # Black outline then coloured text for readability on any background
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 80), 1, cv2.LINE_AA)
        y += 26


def _print_summary(fps: FPSCounter, latency: LatencyTracker):
    print("\n--- Session summary ---")
    print(f"  Avg FPS     : {fps.fps:.1f}")
    print(f"  Avg latency : {latency.mean_ms:.0f} ms")
    print(f"  P95 latency : {latency.p95_ms:.0f} ms")
    print(f"  Min / Max   : {latency.min_ms:.0f} ms / {latency.max_ms:.0f} ms")
