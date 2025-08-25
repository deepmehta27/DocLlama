# app/api/types.py
from typing import List, Literal, Optional
from pydantic import BaseModel

Role = Literal["system","user","assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
