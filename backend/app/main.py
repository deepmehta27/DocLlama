from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os, json, httpx
from pypdf import PdfReader

# Create the FastAPI app
app = FastAPI(title="DocLlama API")

# Set up environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DOCLLAMA_DEFAULT_MODEL", "llama3")

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- simple storage dirs ----
DATA_DIR = os.getenv("DATA_DIR", "./data")
PDF_DIR  = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "text")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True) 

def _safe_name(name: str) -> str:
    # keep filename safe & local
    return os.path.basename(name).replace("..","")

async def _save_upload(file: UploadFile, dest_path: str) -> int:
    blob = await file.read()
    with open(dest_path, "wb") as w:
        w.write(blob)
    return len(blob)

def _pdf_to_text(pdf_path: str) -> tuple[str, int, int]:
    """returns (full_text, pages, chars)"""
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    full = "\n\n".join(texts)
    return full, len(texts), len(full)

# Define the root endpoint
@app.get("/")
def root():
    return {"name": "DocLlama", "status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
    data = r.json()
    names = [m.get("name","") for m in data.get("models",[])]
    embeddings = [n for n in names if "embed" in n or "nomic" in n]
    generation = [n for n in names if n not in embeddings]
    return {"generation": generation, "embeddings": embeddings, "default": DEFAULT_MODEL}

@app.post("/chat_once")
async def chat_once(body: dict):
    prompt = body.get("prompt", "")
    model = body.get("model", DEFAULT_MODEL)
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
        return {"text": r.json().get("response",""), "model": model}

# ↓ UPDATED: supports either {prompt} OR {messages}
@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    model = body.get("model", DEFAULT_MODEL)
    messages = body.get("messages")
    prompt = body.get("prompt")

    # choose Ollama endpoint/payload
    if messages:  # chat API with roles
        api_url = f"{OLLAMA_URL}/api/chat"
        payload = {"model": model, "messages": messages, "stream": True}
        # tip: messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
    else:         # fallback to raw prompt
        api_url = f"{OLLAMA_URL}/api/generate"
        payload = {"model": model, "prompt": prompt or "", "stream": True}

    async def eventgen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", api_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)

                        # handle both /api/generate and /api/chat streaming shapes
                        token = ""
                        if "response" in data:                     # /api/generate
                            token = data.get("response","")
                        elif "message" in data:                     # /api/chat
                            msg = data["message"] or {}
                            token = msg.get("content","") or ""

                        if token:
                            yield f"data: {token}\n\n"
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    return StreamingResponse(eventgen(), media_type="text/event-stream")

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """
    Upload 1..N PDFs. Saves to ./data/pdfs and writes extracted text to ./data/text/*.txt
    Returns quick stats + a short preview.
    """
    results = []
    for f in files:
        if (f.content_type or "").lower() not in ("application/pdf", "application/octet-stream"):
            results.append({"file": f.filename, "status": "skipped (not a PDF)"})
            continue

        safe = _safe_name(f.filename or "document.pdf")
        pdf_path = os.path.join(PDF_DIR, safe)
        _ = await _save_upload(f, pdf_path)

        # extract text
        text, pages, chars = _pdf_to_text(pdf_path)
        txt_name = os.path.splitext(safe)[0] + ".txt"
        txt_path = os.path.join(TEXT_DIR, txt_name)
        with open(txt_path, "w", encoding="utf-8") as w:
            w.write(text)

        results.append({
            "file": safe,
            "pages": pages,
            "chars": chars,
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "preview": (text[:300] + ("…" if len(text) > 300 else "")),
            "status": "ok"
        })

    return {"accepted": len(files), "results": results}