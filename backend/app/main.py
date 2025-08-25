# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.settings import settings
from .api.health import router as health_router
from .api.models import router as models_router
from .api.chat import router as chat_router
from .api.ingest import router as ingest_router
from .api.search import router as search_router

app = FastAPI(title="DocLlama API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"name": "DocLlama", "status": "ok", "docs": "/docs", "health": "/health"}

# mount feature routers
app.include_router(health_router, prefix="")
app.include_router(models_router, prefix="")
app.include_router(chat_router,    prefix="")
app.include_router(ingest_router,  prefix="")
app.include_router(search_router,  prefix="")

# --- Dev entrypoint with auto-reload ---
if __name__ == "__main__":
    import os
    from pathlib import Path
    import uvicorn

    HERE = Path(__file__).resolve()
    BACKEND_DIR = HERE.parent.parent
    APP_DIR = BACKEND_DIR / "app"
    os.chdir(BACKEND_DIR)

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_on = os.getenv("RELOAD", "1") == "1"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_on,
        reload_dirs=[str(APP_DIR)],
        reload_excludes=["data/*", "**/__pycache__/**"],
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
