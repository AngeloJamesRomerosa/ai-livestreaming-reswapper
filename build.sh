#!/usr/bin/env bash
# Render build script — runs once at deploy time.
# Downloads the reswapper ONNX model and generates the FP16 variant.
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Preparing model directory ==="
mkdir -p reswapper_256_model/models

echo "=== Downloading reswapper_256.onnx from HuggingFace ==="
cd reswapper_256_model
python download_models.py

echo "=== Converting to FP16 ==="
python convert_fp16.py
cd ..

echo "=== Extracting emap (1 MB projection matrix for latent computation) ==="
python extract_emap.py

echo "=== Pre-downloading InsightFace buffalo_l into project directory ==="
export INSIGHTFACE_HOME="$(pwd)/.insightface_home"
python -c "
import os
from insightface.app import FaceAnalysis
root = os.environ.get('INSIGHTFACE_HOME', os.path.expanduser('~/.insightface'))
print('Downloading to:', root)
app = FaceAnalysis(name='buffalo_l', root=root, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
print('buffalo_l ready at', root)
"

echo "=== Converting w600k_r50 to FP16 to halve RAM usage (330 MB -> 165 MB) ==="
python -c "
import os, onnx
from onnxconverter_common import convert_float_to_float16
root = os.environ.get('INSIGHTFACE_HOME', os.path.expanduser('~/.insightface'))
path = os.path.join(root, 'models/buffalo_l/w600k_r50.onnx')
print(f'Converting {path} ...')
model = onnx.load(path)
model_fp16 = convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, path)
print(f'Done. {os.path.getsize(path)/1024/1024:.1f} MB')
"

echo "=== Build complete ==="
