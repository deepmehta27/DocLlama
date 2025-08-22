from fastapi import APIRouter
import httpx
from ..core.settings import settings

router = APIRouter()

@router.get("/models")
async def list_models():
    # Ask Ollama for local models
    url = f"{settings.ollama_url}/api/tags"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    all_models = [m.get("name", "") for m in data.get("models", [])]
    # Simple split: the embedding model we’ll use vs. everything else
    embeddings = [m for m in all_models if "embed" in m or "nomic" in m]
    generation = [m for m in all_models if m not in embeddings]

    return {
        "generation": generation,
        "embeddings": embeddings,
        "default": settings.generation_default
    }
