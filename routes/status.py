from fastapi import APIRouter
from providers.reswapper.provider import get_provider

router = APIRouter()

@router.get("/api/status")
def get_status():
    p = get_provider()
    return {
        "loaded": p.loaded,
        "provider": p.active_provider,
        "model": p.model_file,
        "components": p.components,
    }
