# app/db/chroma.py
import chromadb
from chromadb.config import Settings
from ..core.settings import settings

_client = chromadb.Client(Settings(
    persist_directory=settings.chroma_dir,
    anonymized_telemetry=False
))
_collection = _client.get_or_create_collection("docllama")

def collection():
    return _collection
