import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
import config
from routes.log import emit

router = APIRouter()

@router.post("/api/uploadImage")
async def upload_image(file: UploadFile = File(...)):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "face.jpg").suffix or ".jpg"
    fname = f"{uuid.uuid4().hex}{suffix}"
    dest = config.UPLOAD_DIR / fname
    dest.write_bytes(await file.read())
    emit(f"Face image uploaded: {file.filename or fname}", "info")
    return {"path": str(dest), "url": f"/uploads/{fname}"}
