import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from routes import state
import config

router = APIRouter()

@router.get("/stream.mjpeg")
async def mjpeg(request: Request, sid: str = "", key: str = ""):
    if key != config.STREAM_SECRET:
        raise HTTPException(status_code=403, detail="Invalid stream key")

    async def frames():
        last_ts = 0.0
        while not await request.is_disconnected():
            sess = state.get(sid)
            if sess and sess.latest_frame and sess.latest_frame_ts != last_ts:
                last_ts = sess.latest_frame_ts
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + sess.latest_frame
                    + b"\r\n"
                )
            else:
                await asyncio.sleep(0.033)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
