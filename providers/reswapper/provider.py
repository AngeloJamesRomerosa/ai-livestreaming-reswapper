import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

_src = str(Path(__file__).parent.parent.parent / "reswapper_256_model")
if _src not in sys.path:
    sys.path.insert(0, _src)

import config
from src.detector import FaceDetector
from src.swapper import FaceSwapper


def _resolve_providers(name: str) -> list:
    return {
        "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "directml": ["DmlExecutionProvider",  "CPUExecutionProvider"],
        "cpu":      ["CPUExecutionProvider"],
    }.get(name, ["CPUExecutionProvider"])


def _auto_detect_provider() -> str:
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return "cuda"
        if "DmlExecutionProvider" in available:
            return "directml"
    except Exception:
        pass
    return "cpu"


def _best_model_path(provider: str) -> str:
    fp16 = Path(config.MODEL_PATH).parent / "reswapper_256_fp16.onnx"
    if provider == "cuda" and fp16.exists():
        return str(fp16)
    return config.MODEL_PATH


def _log(msg: str, level: str = "info"):
    try:
        from routes.log import emit
        emit(msg, level)
    except Exception:
        pass


def _set_state(states: dict, key: str, status: str, detail: str = ""):
    states[key]["status"] = status
    if detail:
        states[key]["detail"] = detail


class _Worker(threading.Thread):
    def __init__(self, detector, swapper, source_face, det_skip=4, max_fps=10):
        super().__init__(daemon=True, name="inference")
        self._detector = detector
        self._swapper = swapper
        self._source_face = source_face
        self._det_skip = det_skip
        self._min_interval = (1.0 / max_fps) if max_fps > 0 else 0
        self._in_lock = threading.Lock()
        self._in_frame = None
        self._wakeup = threading.Event()
        self._out_lock = threading.Lock()
        self._out_frame = None
        self._out_seq = 0
        self._running = True
        self._frame_count = 0
        self._cached_faces = []
        # per-window stats (reset on each get_stats() call)
        self._st_lock       = threading.Lock()
        self._st_submitted  = 0
        self._st_processed  = 0
        self._st_det_ms     = 0.0
        self._st_det_n      = 0
        self._st_swap_ms    = 0.0
        self._st_swap_n     = 0
        self._st_window     = time.perf_counter()

    def submit(self, frame: np.ndarray):
        with self._in_lock:
            self._in_frame = frame
        with self._st_lock:
            self._st_submitted += 1
        self._wakeup.set()

    def latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._out_lock:
            return self._out_seq, self._out_frame

    def stop(self):
        self._running = False
        self._wakeup.set()

    def get_stats(self) -> dict:
        with self._st_lock:
            elapsed  = time.perf_counter() - self._st_window
            proc     = self._st_processed
            sub      = self._st_submitted
            det_avg  = (self._st_det_ms  / self._st_det_n  * 1000) if self._st_det_n  > 0 else 0.0
            swap_avg = (self._st_swap_ms / self._st_swap_n * 1000) if self._st_swap_n > 0 else 0.0
            stats = {
                "worker_fps": proc / elapsed if elapsed > 0 else 0.0,
                "dropped":    sub - proc,
                "det_avg_ms": det_avg,
                "swap_avg_ms": swap_avg,
            }
            self._st_submitted = 0
            self._st_processed = 0
            self._st_det_ms    = 0.0
            self._st_det_n     = 0
            self._st_swap_ms   = 0.0
            self._st_swap_n    = 0
            self._st_window    = time.perf_counter()
            return stats

    def run(self):
        last_frame = None
        last_time = 0.0
        while self._running:
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()
            with self._in_lock:
                frame = self._in_frame
            if frame is None or frame is last_frame:
                continue
            if self._min_interval > 0:
                wait = self._min_interval - (time.perf_counter() - last_time)
                if wait > 0:
                    time.sleep(wait)
            last_frame = frame
            last_time = time.perf_counter()
            self._frame_count += 1
            if self._frame_count % self._det_skip == 1 or not self._cached_faces:
                t0 = time.perf_counter()
                self._cached_faces = self._detector.detect(frame)
                det_ms = time.perf_counter() - t0
                with self._st_lock:
                    self._st_det_ms += det_ms
                    self._st_det_n  += 1
            if self._cached_faces:
                t0 = time.perf_counter()
                result = self._swapper.swap(frame, self._cached_faces[0], self._source_face)
                swap_ms = time.perf_counter() - t0
                with self._st_lock:
                    self._st_swap_ms += swap_ms
                    self._st_swap_n  += 1
            else:
                result = frame
            with self._st_lock:
                self._st_processed += 1
            with self._out_lock:
                self._out_frame = result
                self._out_seq += 1


class ReswapperProvider:
    def __init__(self):
        self._detector: Optional[FaceDetector] = None
        self._swapper: Optional[FaceSwapper] = None
        self._worker: Optional[_Worker] = None
        self._metrics_running = False
        self._metrics_thread: Optional[threading.Thread] = None
        self.loaded = False
        self.active_provider = "cpu"
        self.model_file = ""
        self.components = {
            "face_detector": {"status": "idle", "detail": "InsightFace buffalo_l"},
            "swap_model":    {"status": "idle", "detail": "—"},
            "gpu_provider":  {"status": "idle", "detail": "—"},
            "source_face":   {"status": "idle", "detail": "No face loaded"},
        }

    def _metrics_loop(self):
        while self._metrics_running:
            for _ in range(200):          # 200 × 0.1s = 20s, interruptible
                if not self._metrics_running:
                    return
                time.sleep(0.1)
            if not self._worker:
                continue
            s = self._worker.get_stats()
            _log(
                f"[Server]   Worker: {s['worker_fps']:.1f} fps  |  "
                f"Detect: {s['det_avg_ms']:.0f} ms  |  "
                f"Swap: {s['swap_avg_ms']:.0f} ms  |  "
                f"Dropped: {s['dropped']}",
                "info",
            )

    def load(self):
        name = config.EXECUTION_PROVIDER
        if name == "auto":
            _log("Detecting execution provider…")
            name = _auto_detect_provider()

        providers = _resolve_providers(name)
        model_path = _best_model_path(name)
        self.model_file = Path(model_path).name

        try:
            _set_state(self.components, "face_detector", "loading", "InsightFace buffalo_l")
            _log("Loading face detector…")
            self._detector = FaceDetector(
                providers=providers,
                det_thresh=config.DET_THRESH,
                det_size=(config.DET_SIZE, config.DET_SIZE),
            )
            _set_state(self.components, "face_detector", "ready", "InsightFace buffalo_l")
            _log("Face detector loaded successfully", "success")
        except Exception as e:
            _set_state(self.components, "face_detector", "failed", str(e))
            _log(f"Face detector loading failed: {e}", "error")
            raise

        try:
            _set_state(self.components, "swap_model", "loading", self.model_file)
            if "fp16" in self.model_file:
                _log(f"Loading swap model: {self.model_file} (FP16 — faster performance)…")
            else:
                _log(f"Loading swap model: {self.model_file}…")
            self._swapper = FaceSwapper(model_path=model_path, providers=providers)
        except Exception as e:
            _set_state(self.components, "swap_model", "failed", str(e))
            _log(f"Swap model loading failed: {e}", "error")
            raise

        actual = self._swapper.session.get_providers()
        first = actual[0] if actual else "CPUExecutionProvider"
        if "CUDA" in first:
            self.active_provider = "cuda"
            _set_state(self.components, "gpu_provider", "active", "CUDA")
            _log("GPU confirmed: CUDA is active", "success")
        elif "Dml" in first:
            self.active_provider = "directml"
            _set_state(self.components, "gpu_provider", "active", "DirectML")
            _log("GPU confirmed: DirectML is active", "success")
        else:
            self.active_provider = "cpu"
            _set_state(self.components, "gpu_provider", "warning", "CPU (no GPU)")
            if name != "cpu":
                _log(
                    f"WARNING: Requested {name.upper()} but running on CPU — check your GPU drivers",
                    "warning",
                )
            else:
                _log("Running on CPU", "info")

        # FP16 is a GPU optimisation — it is slower on CPU than FP32.
        # If we loaded FP16 but CUDA didn't initialise, reload with FP32.
        if self.active_provider == "cpu" and "fp16" in self.model_file:
            fp32_path = config.MODEL_PATH
            _log(
                "FP16 model is slower on CPU than FP32 — reloading with FP32 model…",
                "warning",
            )
            _set_state(self.components, "swap_model", "loading", Path(fp32_path).name)
            try:
                self._swapper = FaceSwapper(model_path=fp32_path, providers=providers)
                self.model_file = Path(fp32_path).name
                _log(f"Reloaded with FP32 model: {self.model_file}", "success")
            except Exception as e:
                _set_state(self.components, "swap_model", "failed", str(e))
                _log(f"FP32 reload failed: {e}", "error")
                raise

        _set_state(self.components, "swap_model", "ready", self.model_file)
        _log("Swap model loaded successfully", "success")
        self.loaded = True
        _log("All models ready", "success")

    def set_source_face(self, image_path: str, max_fps: Optional[int] = None):
        if max_fps is None:
            max_fps = config.MAX_SWAP_FPS
        fname = Path(image_path).name
        _set_state(self.components, "source_face", "loading", fname)
        _log(f"Loading source face: {fname}…")
        try:
            source = self._detector.get_face_from_image(image_path)
        except Exception as e:
            _set_state(self.components, "source_face", "failed", str(e))
            _log(f"Source face loading failed: {e}", "error")
            raise
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2)
        fps_label = "uncapped" if max_fps == 0 else f"{max_fps} fps"
        _log(f"Starting inference worker — FPS cap: {fps_label}")
        self._worker = _Worker(
            self._detector, self._swapper, source,
            det_skip=4, max_fps=max_fps,
        )
        self._worker.start()
        self._metrics_running = True
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop, daemon=True, name="metrics"
        )
        self._metrics_thread.start()
        _set_state(self.components, "source_face", "ready", fname)
        _log(f"Source face loaded successfully: {fname}", "success")

    def get_source_latent(self) -> Optional[list]:
        """Returns latent = L2_norm(normed_embedding @ emap) for browser-side inference."""
        if not (self._worker and self._worker._source_face is not None and self._swapper):
            return None
        normed = self._worker._source_face.normed_embedding.reshape((1, -1))
        emap = self._swapper._model.emap
        latent = np.dot(normed, emap)
        latent = latent / np.linalg.norm(latent)
        return latent.tolist()

    def submit_frame(self, frame: np.ndarray):
        if self._worker:
            self._worker.submit(frame)

    def latest_frame(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._worker:
            return self._worker.latest()
        return -1, None

    def stop(self):
        self._metrics_running = False
        if self._metrics_thread:
            self._metrics_thread.join(timeout=1)
            self._metrics_thread = None
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2)
            self._worker = None


_provider = ReswapperProvider()

def get_provider() -> ReswapperProvider:
    return _provider
