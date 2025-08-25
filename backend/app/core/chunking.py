# app/core/chunking.py
import uuid, json, os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .settings import settings

def safe_name(name: str) -> str:
    return os.path.basename(name).replace("..","")

def pdf_to_text(pdf_path: str) -> tuple[str, int, int]:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    full = "\n\n".join(texts)
    return full, len(texts), len(full)

def pdf_to_chunks(pdf_path: str, chunk_chars: int = 1200, overlap: int = 200) -> list[dict]:
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    out: list[dict] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        for part in splitter.split_text(txt):
            out.append({"id": str(uuid.uuid4()), "page": page_idx, "text": part})
    return out

def write_chunks_jsonl(base_name: str, chunks: list[dict]) -> str:
    out_path = os.path.join(settings.chunk_dir, f"{os.path.splitext(base_name)[0]}.chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path
