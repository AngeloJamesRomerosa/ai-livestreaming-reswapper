import asyncio
import time
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routes import state
from routes.log import emit
from providers.reswapper.provider import get_provider

router = APIRouter()

@router.websocket("/ws/swap")
async def ws_swap(ws: WebSocket, sid: str = ""):
    await ws.accept()
    sess = state.get(sid)
    if not sess or not sess.active:
        await ws.close(code=4004)
        return

    provider = get_provider()
    loop = asyncio.get_event_loop()
    emit(f"Swap stream connected (sid: {sid[:8]}…)", "success")

    async def recv():
        while True:
            data = await ws.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                await loop.run_in_executor(None, provider.submit_frame, frame)

    async def send():
        last_seq = -1
        while True:
            seq, result = await loop.run_in_executor(None, provider.latest_frame)
            if result is not None and seq != last_seq:
                last_seq = seq
                _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                out = buf.tobytes()
                await ws.send_bytes(out)
                sess.latest_frame = out
                sess.latest_frame_ts = time.time()
                dead = set()
                for viewer in list(sess.viewers):
                    try:
                        await viewer.send_bytes(out)
                    except Exception:
                        dead.add(viewer)
                sess.viewers -= dead
            else:
                await asyncio.sleep(0.033)

    try:
        await asyncio.gather(recv(), send())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        emit(f"Swap stream disconnected (sid: {sid[:8]}…)")
