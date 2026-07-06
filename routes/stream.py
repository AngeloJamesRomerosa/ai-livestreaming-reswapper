import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routes import state

router = APIRouter()

@router.websocket("/ws/stream-in")
async def stream_in(ws: WebSocket, sid: str = ""):
    await ws.accept()
    sess = state.get(sid)
    if sess is None:
        await ws.close(code=4004)
        return
    sess.viewers.add(ws)
    try:
        while True:
            await asyncio.sleep(30)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        sess.viewers.discard(ws)
