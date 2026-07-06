import os
import sys

_DLL_HANDLES = []


def setup():
    """Register NVIDIA pip-package DLL folders with Windows before onnxruntime loads.

    pip installs CUDA/cuDNN DLLs into site-packages/nvidia/*/bin/ which Windows
    does not search by default.  Must be called before any onnxruntime import.
    """
    if sys.platform != "win32":
        return

    import site
    import ctypes

    dirs_to_add = []
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)

    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    if os.path.isdir(cuda_bin):
        dirs_to_add.append(cuda_bin)

    for d in dirs_to_add:
        try:
            _DLL_HANDLES.append(os.add_dll_directory(d))
        except OSError:
            pass

    if dirs_to_add:
        os.environ["PATH"] = (
            os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get("PATH", "")
        )

    _PRELOAD = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_graph64_9.dll",
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
