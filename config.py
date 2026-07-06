import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
STREAM_SECRET = os.getenv("STREAM_SECRET", "secret")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).parent / "reswapper_256_model" / "models" / "reswapper_256.onnx"),
)
EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDER", "auto")
MAX_SWAP_FPS = int(os.getenv("MAX_SWAP_FPS", 10))
DET_THRESH = float(os.getenv("DET_THRESH", 0.7))
DET_SIZE = int(os.getenv("DET_SIZE", 320))
