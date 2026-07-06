import cuda_setup
cuda_setup.setup()

import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from routes.faces import router as faces_router
from routes.session import router as session_router
from routes.status import router as status_router
from routes.log import router as log_router, init_loop, emit
from routes.stream import router as stream_router
from routes.stream_mjpeg import router as mjpeg_router
from providers.reswapper.routes import router as swap_router

app = FastAPI(title="AI Face Swap Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in [faces_router, session_router, status_router, log_router,
          stream_router, mjpeg_router, swap_router]:
    app.include_router(r)

Path("uploads").mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/", StaticFiles(directory="public", html=True), name="public")


@app.on_event("startup")
async def on_startup():
    init_loop(asyncio.get_running_loop())
    emit("Server started", "success")
    dll_count = len(cuda_setup._DLL_HANDLES)
    if dll_count > 0:
        emit(f"CUDA DLLs registered: {dll_count} handles loaded", "success")
    else:
        emit("No CUDA DLLs found — will use CPU if CUDA unavailable", "warning")
    emit("Waiting for session to load models…", "info")
