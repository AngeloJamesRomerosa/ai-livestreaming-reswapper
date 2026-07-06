import asyncio
from collections import deque
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_history: deque = deque(maxlen=400)
_subscribers: set = set()
_main_loop: asyncio.AbstractEventLoop = None


def init_loop(loop: asyncio.AbstractEventLoop):
    global _main_loop
    _main_loop = loop


def emit(message: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    data = f"data: {level}|{ts}|{message}\n\n"
    _history.append(data)
    if _main_loop and not _main_loop.is_closed():
        _main_loop.call_soon_threadsafe(_broadcast, data)


def _broadcast(data: str):
    dead = set()
    for q in _subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.add(q)
    _subscribers.difference_update(dead)


@router.get("/api/log")
async def log_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=300)
    _subscribers.add(q)

    async def generate():
        for entry in list(_history):
            yield entry
        try:
            while not await request.is_disconnected():
                try:
                    data = q.get_nowait()
                    yield data
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
