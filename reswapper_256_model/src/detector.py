import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def _upgrade_cuda(providers: list) -> list:
    """Swap bare 'CUDAExecutionProvider' for a (name, options) tuple with
    HEURISTIC conv-algo search so the first inference call doesn't block for
    1–2 s doing EXHAUSTIVE cuDNN algorithm profiling."""
    upgraded = []
    for p in providers:
        if p == 'CUDAExecutionProvider':
            upgraded.append(('CUDAExecutionProvider', {
                'cudnn_conv_algo_search': 'HEURISTIC',
            }))
        else:
            upgraded.append(p)
    return upgraded


def _provider_name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


class FaceDetector:
    """SCRFD-backed face detector via InsightFace buffalo_l model pack.

    buffalo_l is downloaded automatically to ~/.insightface/models/buffalo_l/
    on first run (~200 MB). Detection threshold of 0.7–0.9 is critical for
    face stability — lower values cause false positives and drift.
    """

    def __init__(self, providers, det_thresh=0.7, det_size=(320, 320)):
        self._thresh = det_thresh
        upgraded = _upgrade_cuda(providers)
        # InsightFace uses ctx_id=0 for the first GPU, -1 to force CPU
        ctx_id = 0 if any(
            _provider_name(p) in ('CUDAExecutionProvider', 'DmlExecutionProvider')
            for p in upgraded
        ) else -1
        # allowed_modules limits buffalo_l to only the two models the swap actually
        # needs: det_10g (bounding box + 5 keypoints) and w600k_r50 (face embedding).
        # The other three — 1k3d68 (3D landmarks), 2d106det (2D landmarks), genderage —
        # run per detected face but are unused by INSwapper, so skipping them is free speedup.
        _root = os.environ.get(
            'INSIGHTFACE_HOME', os.path.expanduser('~/.insightface')
        )
        self._app = FaceAnalysis(
            name='buffalo_l',
            root=_root,
            providers=upgraded,
            allowed_modules=['detection', 'recognition'],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Warm up detection + recognition CUDA kernels with a blank frame.
        # This moves cuDNN compilation to startup rather than the first live frame.
        print("[warm-up] Compiling detector kernels...", end=" ", flush=True)
        _dummy = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
        for _ in range(3):
            self._app.get(_dummy)
        print("done.")

    def detect(self, frame_bgr):
        """Return faces sorted largest-first, filtered by detection score."""
        faces = self._app.get(frame_bgr)
        # Drop low-confidence detections — below ~0.7 the bounding box is unreliable
        # and causes the swap to slip or flicker.
        faces = [f for f in faces if f.det_score >= self._thresh]
        # Largest bounding-box area first so the performer's face is always index 0,
        # even if background faces are also in the frame.
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces

    def get_face_from_image(self, image_path: str):
        """Load an image file and return the largest detected face."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        faces = self.detect(img)
        if not faces:
            raise ValueError(f"No face detected in source image: {image_path}")
        return faces[0]
