"""
Convert a float32 ONNX swap model to a float16 model.

The conversion uses keep_io_types=True so that the model still accepts and
returns float32 tensors at its boundaries — InsightFace's INSwapper feeds
float32 inputs, so the external interface must stay float32.

Internally the weights and arithmetic run in float16, which lets CUDA Tensor
Cores process them ~2× faster than float32 on Ampere/Ada GPUs (RTX 30/40).

Usage:
    python convert_fp16.py
    python convert_fp16.py --input models/reswapper_256.onnx --output models/reswapper_256_fp16.onnx
"""

import argparse
import os
import sys


def convert(input_path: str, output_path: str):
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        sys.exit(
            "[error] Required packages not found.\n"
            "        Run:  pip install onnx onnxconverter-common"
        )

    if not os.path.exists(input_path):
        sys.exit(f"[error] Input model not found: {input_path}")

    print(f"Loading  : {input_path}")
    model_fp32 = onnx.load(input_path)

    print("Converting to float16 (keep_io_types=True) ...")
    # keep_io_types=True — input/output tensors remain float32 so that
    # InsightFace's preprocessing code (which always produces float32 blobs)
    # works unchanged.  Only the internal weights and ops become float16.
    #
    # op_block_list keeps numerically sensitive ops in FP32.  Without this,
    # Exp/Pow overflow in FP16 (max value 65504) and BatchNormalization loses
    # enough precision to produce NaN activations, which the final clip+cast
    # converts to 0 — the "black box" artifact over the face.
    # reswapper_256 implements AdaIN normalisation as primitive ops (no
    # InstanceNormalization node).  The chain is:
    #   ReduceMean → Sub → Mul → ReduceMean → Sqrt → Reciprocal → Mul → Add
    # In FP16 the variance easily underflows to 0, making Sqrt(0)=0 and
    # Reciprocal(0)=Inf; subsequent Mul/Add then produce NaN.
    # Conv and Gemm (the expensive ops) stay in FP16 for Tensor Core speedup.
    _KEEP_FP32 = [
        'ReduceMean',   # mean and variance computation — needs FP32 precision
        'Reciprocal',   # 1/std — Inf if std underflows to 0 in FP16
        'Sqrt',         # std = sqrt(var+eps) — sqrt of near-zero underflows in FP16
        'Div',          # any explicit division
        'Tanh',         # output activation — NaN input → NaN output
        'Resize',       # scale/size params must stay float32 per ONNX spec
    ]
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=_KEEP_FP32,
    )

    print(f"Saving   : {output_path}")
    onnx.save(model_fp16, output_path)
    print("Done.")

    in_mb  = os.path.getsize(input_path)  / 1024 / 1024
    out_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nModel size: {in_mb:.1f} MB → {out_mb:.1f} MB ({out_mb/in_mb*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert swap model to FP16")
    parser.add_argument("--input",  default=os.path.join("models", "reswapper_256.onnx"))
    parser.add_argument("--output", default=os.path.join("models", "reswapper_256_fp16.onnx"))
    args = parser.parse_args()
    convert(args.input, args.output)
