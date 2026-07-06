from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_MODELS = {
    "det_10g.onnx":          lambda: Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx",
    "reswapper_256.onnx":    lambda: Path("reswapper_256_model") / "models" / "reswapper_256.onnx",
    "reswapper_256_fp16.onnx": lambda: Path("reswapper_256_model") / "models" / "reswapper_256_fp16.onnx",
}


@router.get("/models/{model_name}")
async def serve_model(model_name: str):
    if model_name not in _MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    path = _MODELS[model_name]()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not on disk: {model_name}")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f"attachment; filename={model_name}",
        },
    )
