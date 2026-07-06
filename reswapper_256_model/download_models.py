"""
Download swap model files into the models/ directory.

Available models (all from netrunner-exe/Insight-Swap-models-onnx on HuggingFace):

  reswapper_256.onnx   (~300 MB)  INSwapper retrain at 256px.

Usage:
  python download_models.py   # downloads reswapper_256

The buffalo_l detection/recognition pack (~200 MB) is downloaded automatically
by InsightFace on first run and cached in ~/.insightface/models/buffalo_l/.
"""

import argparse
import os
import sys
import requests
from tqdm import tqdm

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_HF_BASE = "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main"

KNOWN_MODELS = {
    "reswapper_256": f"{_HF_BASE}/reswapper_256.onnx",
}


def download_file(url: str, dest: str, label: str) -> bool:
    """Download url → dest with a progress bar. Returns True on success."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=label
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"  [fail] {url}\n         {exc}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_model(name: str) -> bool:
    filename = f"{name}.onnx"
    dest = os.path.join(MODELS_DIR, filename)

    if os.path.exists(dest):
        print(f"[skip] Already present: {dest}")
        return True

    url = KNOWN_MODELS.get(name)
    if url is None:
        print(f"[error] Unknown model '{name}'. Available: {', '.join(KNOWN_MODELS)}")
        return False

    print(f"\nDownloading {filename} ...")
    print(f"Source: {url}")
    if download_file(url, dest, filename):
        print(f"[ok] Saved to: {dest}")
        return True

    print("\n" + "=" * 60)
    print(f"  Automatic download failed for {filename}.")
    print(f"  Manual: download from\n    {url}")
    print(f"  and place it at:\n    {dest}")
    print("=" * 60)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download face-swap ONNX models")
    parser.add_argument("--model", choices=list(KNOWN_MODELS), default="reswapper_256",
                        help="Which model to download (default: reswapper_256)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if download_model(args.model):
        print("\nModel ready.")
        print("Run:  python run.py --source <face.jpg> --model models\\reswapper_256.onnx")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
