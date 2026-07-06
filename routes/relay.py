import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routes import state

router = APIRouter()


@router.websocket("/ws/relay")
async def ws_relay(ws: WebSocket, sid: str = ""):
    """Receive locally-inferred JPEG frames from the browser and broadcast to
    OBS viewers + store as latest_frame for the MJPEG endpoint."""
    await ws.accept()
    sess = state.get(sid)
    if not sess or not sess.active:
        await ws.close(code=4004)
        return
    try:
        while True:
            data = await ws.receive_bytes()
            sess.latest_frame    = data
            sess.latest_frame_ts = time.time()
            dead = set()
            for viewer in list(sess.viewers):
                try:
                    await viewer.send_bytes(data)
                except Exception:
                    dead.add(viewer)
            sess.viewers -= dead
    except (WebSocketDisconnect, Exception):
        pass
