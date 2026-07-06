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

    def submit(self, frame: np.ndarray):
        with self._in_lock:
            self._in_frame = frame
        self._wakeup.set()

    def latest(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._out_lock:
            return self._out_seq, self._out_frame

    def stop(self):
        self._running = False
        self._wakeup.set()

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
                self._cached_faces = self._detector.detect(frame)
            if self._cached_faces:
                result = self._swapper.swap(frame, self._cached_faces[0], self._source_face)
            else:
                result = frame
            with self._out_lock:
                self._out_frame = result
                self._out_seq += 1


class ReswapperProvider:
    def __init__(self):
        self._detector: Optional[FaceDetector] = None
        self._swapper: Optional[FaceSwapper] = None
        self._worker: Optional[_Worker] = None
        self.loaded = False
        self.active_provider = "cpu"
        self.model_file = ""
        self.components = {
            "face_detector": {"status": "idle", "detail": "InsightFace buffalo_l"},
            "swap_model":    {"status": "idle", "detail": "—"},
            "gpu_provider":  {"status": "idle", "detail": "—"},
            "source_face":   {"status": "idle", "detail": "No face loaded"},
        }

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

    def set_source_face(self, image_path: str):
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
        self._worker = _Worker(
            self._detector, self._swapper, source,
            det_skip=4, max_fps=config.MAX_SWAP_FPS,
        )
        self._worker.start()
        _set_state(self.components, "source_face", "ready", fname)
        _log(f"Source face loaded successfully: {fname}", "success")

    def submit_frame(self, frame: np.ndarray):
        if self._worker:
            self._worker.submit(frame)

    def latest_frame(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._worker:
            return self._worker.latest()
        return -1, None

    def stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2)
            self._worker = None


_provider = ReswapperProvider()

def get_provider() -> ReswapperProvider:
    return _provider
