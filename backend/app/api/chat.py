# app/api/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .types import ChatRequest
from ..core.ollama import generate_once, stream_chat

router = APIRouter()

@router.post("/chat_once")
async def chat_once(body: ChatRequest):
    txt = await generate_once(body.prompt or "", model=body.model)
    return {"text": txt, "model": body.model}

@router.post("/chat")
async def chat(body: ChatRequest):
    # prefer messages; fallback to prompt
    if body.messages:
        async def gen():
            async for tok in stream_chat([m.model_dump() for m in body.messages], model=body.model):
                yield f"data: {tok}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        # emulate streaming with single shot (kept simple)
        txt = await generate_once(body.prompt or "", model=body.model)
        async def gen_once():
            yield f"data: {txt}\n\n"
        return StreamingResponse(gen_once(), media_type="text/event-stream")
