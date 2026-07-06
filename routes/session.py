import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from routes import state
from routes.log import emit
from providers.reswapper.provider import get_provider
import config

router = APIRouter()

class CreateReq(BaseModel):
    face_path: str
    sid: Optional[str] = None

@router.post("/api/session/create")
async def session_create(req: CreateReq):
    sid = req.sid or uuid.uuid4().hex
    sess = state.create(sid)
    provider = get_provider()
    emit(f"Session create requested (sid: {sid[:8]}…)")
    if not provider.loaded:
        emit("First run — loading models (this may take ~30s)…")
        provider.load()
    try:
        provider.set_source_face(req.face_path)
    except Exception as e:
        state.remove(sid)
        emit(f"Session failed: {e}", "error")
        raise HTTPException(status_code=400, detail=str(e))
    sess.face_path = req.face_path
    sess.active = True
    emit(f"Session active (sid: {sid[:8]}…)", "success")
    return {"sid": sid, "provider": provider.active_provider, "stream_secret": config.STREAM_SECRET}

class CloseReq(BaseModel):
    sid: str

@router.post("/api/session/close")
async def session_close(req: CloseReq):
    emit(f"Session closed (sid: {req.sid[:8]}…)")
    state.remove(req.sid)
    return {"ok": True}
