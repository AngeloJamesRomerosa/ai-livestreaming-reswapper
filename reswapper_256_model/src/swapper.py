import numpy as np
import onnxruntime
from insightface.model_zoo.inswapper import INSwapper


def _upgrade_cuda(providers: list) -> list:
    """Replace the bare 'CUDAExecutionProvider' string with a (name, options) tuple
    that uses HEURISTIC conv-algo search instead of the default EXHAUSTIVE.

    EXHAUSTIVE tries every cuDNN algorithm on the first inference call — this takes
    1–2 s per unique operator shape and causes the startup stutter.  HEURISTIC picks
    a good algorithm instantly using cuDNN's built-in heuristics; real-world throughput
    is within ~5 % of EXHAUSTIVE for typical face-swap models.
    """
    upgraded = []
    for p in providers:
        if p == 'CUDAExecutionProvider':
            upgraded.append(('CUDAExecutionProvider', {
                'cudnn_conv_algo_search': 'HEURISTIC',
            }))
        else:
            upgraded.append(p)
    return upgraded


def _warmup_session(session: onnxruntime.InferenceSession, n_runs: int = 3) -> None:
    """Run dummy forward passes so CUDA kernels are compiled before live inference.

    Even with HEURISTIC search, the first real run still JIT-compiles CUDA kernels
    and allocates GPU memory.  Running n_runs dummy passes here moves that cost to
    startup (hidden behind the 'Loading source face' message) rather than the first
    live frames — eliminating the visible slowdown at the beginning of each session.
    """
    dummy = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dummy[inp.name] = np.zeros(shape, dtype=np.float32)
    for _ in range(n_runs):
        try:
            session.run(None, dummy)
        except Exception:
            pass


class FaceSwapper:
    """Wrapper around InsightFace's INSwapper loaded directly from a model file.

    We bypass insightface.model_zoo.get_model() because its filename-based type
    detector misclassifies reswapper_256.onnx as ArcFaceONNX (a recognition model).
    Directly instantiating INSwapper forces the correct wrapper regardless of filename.

    The model file (reswapper_256.onnx) must be downloaded separately —
    run download_models.py before first use.
    """

    def __init__(self, model_path: str, providers: list):
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.enable_mem_pattern = True

        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_opts, providers=_upgrade_cuda(providers)
        )

        print("[warm-up] Compiling swap kernels...", end=" ", flush=True)
        _warmup_session(session)
        print("done.")

        self._model = INSwapper(model_file=model_path, session=session)

    def swap(self, frame_bgr, target_face, source_face):
        """Return a copy of frame_bgr with target_face replaced by source_face."""
        return self._model.get(frame_bgr.copy(), target_face, source_face, paste_back=True)
