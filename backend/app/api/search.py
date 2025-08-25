# app/api/search.py
from fastapi import APIRouter, Query
from ..core.ollama import embed_texts
from ..db.chroma import collection

router = APIRouter()

@router.get("/search")
async def search(q: str = Query(...), k: int = 5):
    col = collection()
    qvec = await embed_texts([q])
    res = col.query(query_embeddings=qvec, n_results=k)

    docs   = res.get("documents", [[]])[0]
    metas  = res.get("metadatas", [[]])[0]
    dists  = res.get("distances", [[]])[0] if "distances" in res else [None]*len(docs)

    return {
        "query": q,
        "k": k,
        "results": [
            {"text": d, "meta": m, "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]
    }
