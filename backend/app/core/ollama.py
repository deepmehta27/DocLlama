# app/core/ollama.py
import json
import httpx
from typing import AsyncIterator, List
from .settings import settings

async def list_local_models() -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{settings.ollama_url}/api/tags")
        r.raise_for_status()
    data = r.json()
    return [m.get("name","") for m in data.get("models",[])]

async def embed_texts(texts: List[str], model: str | None = None) -> list[list[float]]:
    em = model or settings.embeddings_model
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for t in texts:
            r = await client.post(f"{settings.ollama_url}/api/embeddings",
                                  json={"model": em, "prompt": t})
            r.raise_for_status()
            out.append(r.json()["embedding"])
    return out

async def generate_once(prompt: str, model: str | None = None) -> str:
    m = model or settings.default_model
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{settings.ollama_url}/api/generate",
                              json={"model": m, "prompt": prompt, "stream": False})
        r.raise_for_status()
        return r.json().get("response","")

async def stream_chat(messages: list[dict], model: str | None = None) -> AsyncIterator[str]:
    m = model or settings.default_model
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{settings.ollama_url}/api/chat",
                                 json={"model": m, "messages": messages, "stream": True}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("done"): break
                    msg = data.get("message") or {}
                    token = msg.get("content","")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue
