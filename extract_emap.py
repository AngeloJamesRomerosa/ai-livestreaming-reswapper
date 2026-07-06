"""Extract emap from the reswapper ONNX model and save as emap.npy.

emap is the last ONNX initializer — a 512×512 matrix that projects ArcFace
normed_embedding into the reswapper latent space.  It never changes between
runs, so we extract it once at build time and load the tiny 1 MB file at
runtime instead of loading the full swap model.

Prefers reswapper_256_fp16.onnx (smaller), falls back to reswapper_256.onnx.
emap is always saved as float32 regardless of source model precision.

Usage (run from repo root):
    python extract_emap.py
"""
import sys
from pathlib import Path
import numpy as np


def main():
    model_dir = Path(__file__).parent / "reswapper_256_model" / "models"
    emap_path = model_dir / "emap.npy"

    if emap_path.exists():
        print(f"[skip] Already present: {emap_path}")
        return

    # Prefer FP16 (smaller, loads faster), fall back to FP32
    fp16 = model_dir / "reswapper_256_fp16.onnx"
    fp32 = model_dir / "reswapper_256.onnx"
    model_path = fp16 if fp16.exists() else fp32

    if not model_path.exists():
        sys.exit(f"[error] No reswapper model found in {model_dir}\n"
                 "Run: cd reswapper_256_model && python download_models.py")

    print(f"Extracting emap from {model_path.name} …")
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        sys.exit("[error] onnx not installed — run: pip install onnx")

    model = onnx.load(str(model_path))
    emap_tensor = model.graph.initializer[-1]
    emap = numpy_helper.to_array(emap_tensor).astype(np.float32)  # always save as fp32

    print(f"  shape: {emap.shape}  dtype: {emap.dtype}")
    np.save(str(emap_path), emap)
    print(f"  Saved → {emap_path}  ({emap_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
