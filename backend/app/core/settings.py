# app/core/settings.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    default_model: str = os.getenv("DOCLLAMA_DEFAULT_MODEL", "llama3")
    cors_origins: list[str] = ["http://localhost:5173"]

    data_dir: str = os.getenv("DATA_DIR", "./data")
    pdf_dir:  str = os.path.join(data_dir, "pdfs")
    text_dir: str = os.path.join(data_dir, "text")
    chunk_dir:str = os.path.join(data_dir, "chunks")
    chroma_dir:str= os.path.join(data_dir, "chroma")
    embeddings_model: str = "nomic-embed-text"

settings = Settings()

# ensure dirs exist on import
os.makedirs(settings.pdf_dir, exist_ok=True)
os.makedirs(settings.text_dir, exist_ok=True)
os.makedirs(settings.chunk_dir, exist_ok=True)
os.makedirs(settings.chroma_dir, exist_ok=True)
