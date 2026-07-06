"""
AI Live Face Swap — prototype entry point.

Usage:
    python run.py --source face.jpg
    python run.py --source face.jpg --execution-provider cuda --virtual-cam
    python run.py --source face.jpg --execution-provider cpu   # slow but functional

Prerequisites:
    1. pip install -r requirements.txt   (GPU) or requirements-cpu.txt (CPU)
    2. python download_models.py         (downloads reswapper_256.onnx ~300 MB)
"""

import argparse
import os
import sys
from dataclasses import dataclass


# Handles must be stored at module level — if they are garbage-collected the
# search path is removed, causing onnxruntime's CUDA provider to fail with error 126.
_DLL_HANDLES = []


def _setup_cuda_dll_paths():
    """Register NVIDIA pip-package DLL folders with Windows before onnxruntime loads.

    pip installs CUDA/cuDNN DLLs into site-packages/nvidia/*/bin/ which Windows
    does not search by default.  Three layers of registration are applied so that
    both Python's DLL loader and onnxruntime's C-level LoadLibrary() find them:
      1. os.add_dll_directory()  — Python import machinery
      2. os.environ['PATH']      — native LoadLibrary() path search
      3. ctypes.WinDLL() preload — puts each DLL into the process module list so
                                   Windows resolves onnxruntime_providers_cuda.dll's
                                   imports without any path search at all.
    """
    if sys.platform != 'win32':
        return
    import site
    import ctypes

    dirs_to_add = []
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, 'nvidia')
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, 'bin')
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    # CUDA Toolkit bin (cudart64_12.dll, cufft64_10.dll, curand64_10.dll, etc.)
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    if os.path.isdir(cuda_bin):
        dirs_to_add.append(cuda_bin)

    for d in dirs_to_add:
        _DLL_HANDLES.append(os.add_dll_directory(d))

    if dirs_to_add:
        os.environ['PATH'] = os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get('PATH', '')

    # Pre-load every CUDA/cuDNN DLL into the process so that when onnxruntime's
    # C runtime calls LoadLibrary("onnxruntime_providers_cuda.dll"), Windows
    # resolves its import table from the already-loaded module list instantly.
    _PRELOAD = [
        # CUDA 12 runtime + math libs
        'cudart64_12.dll',
        'cublas64_12.dll',
        'cublasLt64_12.dll',
        'cufft64_11.dll',
        'curand64_10.dll',
        # cuDNN 9 (merged layout — no separate infer/train sub-DLLs)
        'cudnn64_9.dll',
        'cudnn_ops64_9.dll',
        'cudnn_adv64_9.dll',
        'cudnn_cnn64_9.dll',
        'cudnn_graph64_9.dll',
    ]
    for dll_name in _PRELOAD:
        for d in dirs_to_add:
            full = os.path.join(d, dll_name)
            if os.path.exists(full):
                try:
                    _DLL_HANDLES.append(ctypes.WinDLL(full))
                except OSError:
                    pass
                break


_setup_cuda_dll_paths()


@dataclass
class Config:
    source: str
    camera: int
    width: int
    height: int
    fps: int
    execution_provider: str
    model_path: str
    virtual_cam: bool
    show_metrics: bool
    det_thresh: float
    det_size: int
    max_swap_fps: float
    output_mode: str


def _print_output_instructions(mode: str, virtual_cam: bool):
    if not virtual_cam:
        return
    print()
    if mode == "obs":
        print("[output] Mode: OBS only")
        print("         → Open OBS → Sources + → Video Capture Device → OBS Virtual Camera")
    elif mode == "akool":
        print("[output] Mode: AKOOL only")
        print("         → Open AKOOL → set input to OBS Virtual Camera")
        print("         → AKOOL Virtual Camera carries the enhanced output")
    elif mode == "akool-obs":
        print("[output] Mode: AKOOL → OBS")
        print("         → Open AKOOL → set input to OBS Virtual Camera")
        print("         → Open OBS → Sources + → Video Capture Device → AKOOL Virtual Camera")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Live Face Swap prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Path to source face image (JPG/PNG)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--execution-provider",
                        choices=["cuda", "directml", "cpu"],
                        default="cuda",
                        help="ONNX Runtime execution provider")
    parser.add_argument("--model",
                        default=os.path.join("models", "reswapper_256.onnx"),
                        help="Path to swap model .onnx file")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 swap model (run convert_fp16.py first)")
    parser.add_argument("--virtual-cam", action="store_true",
                        help="Send output to OBS Virtual Camera")
    parser.add_argument("--output-mode",
                        choices=["obs", "akool", "akool-obs"],
                        default="obs",
                        help="Streaming output mode: obs (OBS only), akool (AKOOL only), akool-obs (AKOOL → OBS)")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable FPS/latency overlay")
    parser.add_argument("--det-thresh", type=float, default=0.7,
                        help="Face detection confidence threshold (0–1)")
    parser.add_argument("--det-size", type=int, default=320,
                        help="Detection input resolution (320 or 640). 320 is faster.")
    parser.add_argument("--max-swap-fps", type=float, default=4.0,
                        help="Cap inference rate to stay within GPU thermal budget (0 = unlimited)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"[error] Source image not found: {args.source}")

    model_path = args.model
    if args.fp16:
        fp16_path = model_path.replace(".onnx", "_fp16.onnx")
        if not os.path.exists(fp16_path):
            sys.exit(
                f"[error] FP16 model not found: {fp16_path}\n"
                "        Run:  python convert_fp16.py"
            )
        model_path = fp16_path
        print(f"[fp16] Using FP16 model: {model_path}")

    if not os.path.exists(model_path):
        sys.exit(
            f"[error] Model not found: {model_path}\n"
            "        Run:  python download_models.py\n"
            "        Or download reswapper_256.onnx from:\n"
            "        https://huggingface.co/crmbz/reswapper_256.onnx"
        )

    cfg = Config(
        source=args.source,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        execution_provider=args.execution_provider,
        model_path=model_path,
        virtual_cam=args.virtual_cam,
        show_metrics=not args.no_metrics,
        det_thresh=args.det_thresh,
        det_size=args.det_size,
        max_swap_fps=args.max_swap_fps,
        output_mode=args.output_mode,
    )

    _print_output_instructions(cfg.output_mode, cfg.virtual_cam)

    # Deferred import keeps startup fast — insightface and onnxruntime are heavy
    # and only need to load after argument validation has already passed.
    from src.pipeline import FaceSwapPipeline
    pipeline = FaceSwapPipeline(cfg)
    pipeline.load_source(cfg.source)
    pipeline.run()


if __name__ == "__main__":
    main()
