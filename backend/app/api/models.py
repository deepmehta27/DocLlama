# app/api/models.py
from fastapi import APIRouter
from ..core.ollama import list_local_models
from ..core.settings import settings

router = APIRouter()

@router.get("/models")
async def models():
    names = await list_local_models()
    embeddings = [n for n in names if "embed" in n or "nomic" in n]
    generation = [n for n in names if n not in embeddings]
    return {"generation": generation, "embeddings": embeddings, "default": settings.default_model}
