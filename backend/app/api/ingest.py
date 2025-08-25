# app/api/ingest.py
from fastapi import APIRouter, File, UploadFile
import os
from ..core.settings import settings
from ..core.chunking import safe_name, pdf_to_text, pdf_to_chunks, write_chunks_jsonl
from ..core.ollama import embed_texts
from ..db.chroma import collection

router = APIRouter()

@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []
    col = collection()

    for f in files:
        if (f.content_type or "").lower() not in ("application/pdf", "application/octet-stream"):
            results.append({"file": f.filename, "status": "skipped (not a PDF)"})
            continue

        safe = safe_name(f.filename or "document.pdf")
        pdf_path = os.path.join(settings.pdf_dir, safe)
        blob = await f.read()
        with open(pdf_path, "wb") as w:
            w.write(blob)

        text, pages, chars = pdf_to_text(pdf_path)
        txt_path = os.path.join(settings.text_dir, os.path.splitext(safe)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as w:
            w.write(text)

        chunks = pdf_to_chunks(pdf_path, chunk_chars=1200, overlap=200)
        chunk_path = write_chunks_jsonl(safe, chunks)

        ids       = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metas     = [{"file": safe, "page": c["page"]} for c in chunks]
        vecs      = await embed_texts(documents)

        col.upsert(ids=ids, documents=documents, metadatas=metas, embeddings=vecs)

        results.append({
            "file": safe,
            "pages": pages,
            "chars": chars,
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "chunk_count": len(chunks),
            "chunk_path": chunk_path,
            "indexed": len(chunks),
            "status": "ok"
        })

    return {"accepted": len(files), "results": results}
