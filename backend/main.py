# backend/main.py
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pinecone.pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ---------- Config (env vars) ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "wiki-ir")
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
CHUNK_PREFIX = "TITLE: {title}\n\n"  # same prefix used during upsert
DEFAULT_FETCH_K = int(os.getenv("DEFAULT_FETCH_K", 50))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV environment variables")

# ---------- Helpers ----------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        if not np.isfinite(norm) or norm == 0:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / (norm + eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(np.isfinite(norms) & (norms > 0), norms, 1.0)
    return arr / (safe_norms + eps)

def debug_stats(x, name: str):
    arr = np.array(x)
    return {
        "name": name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "n_nan": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "n_inf": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }

# ---------- Load model & Pinecone client (global) ----------
print("Loading embedding model:", EMB_MODEL_NAME)
model = SentenceTransformer(EMB_MODEL_NAME)

print("Initializing Pinecone client...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except TypeError:
    pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# ---------- FastAPI app ----------
app = FastAPI(title="RAG Retrieval API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request/Response models ----------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K
    fetch_k: Optional[int] = DEFAULT_FETCH_K
    namespace: Optional[str] = None

class ChunkResult(BaseModel):
    id: str
    page: str
    chunk_index: int
    score: float
    snippet: str
    full_text: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]

# ---------- Search endpoint ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    top_k = max(1, min(100, int(req.top_k or DEFAULT_TOP_K)))
    fetch_k = max(top_k, min(1000, int(req.fetch_k or DEFAULT_FETCH_K)))

    # encode query
    prefixed_q = CHUNK_PREFIX.format(title="") + q
    try:
        q_embs = model.encode([prefixed_q], batch_size=1, show_progress_bar=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query encoding failed: {e}")
    q_emb = np.array(q_embs[0], dtype=np.float32)
    q_emb = l2_normalize(q_emb)

    # fetch candidates from Pinecone
    try:
        res = index.query(vector=q_emb.tolist(), top_k=fetch_k, include_metadata=True, namespace=req.namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    matches = res.get("matches", []) or []
    if not matches:
        return SearchResponse(query=q, results=[])

    # build candidate texts (truncate only for encoding safety, not for stored metadata)
    candidate_texts = []
    for m in matches:
        meta = (m.get("metadata") or {})
        chunk_text = meta.get("text", "") or ""
        page = meta.get("page", "") or ""
        # only truncate for encoding if extremely long
        max_chunk_chars = 2000
        chunk_for_encode = chunk_text if len(chunk_text) <= max_chunk_chars else chunk_text[:max_chunk_chars]
        candidate_texts.append(CHUNK_PREFIX.format(title=page) + chunk_for_encode)

    try:
        cand_embs_raw = model.encode(candidate_texts, batch_size=32, show_progress_bar=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate encoding failed: {e}")

    cand_embs = np.array(cand_embs_raw, dtype=np.float32)
    if cand_embs.ndim == 1:
        cand_embs = cand_embs.reshape(1, -1)
    if cand_embs.dtype == np.object_:
        try:
            cand_embs = np.vstack([np.array(x, dtype=np.float32) for x in cand_embs])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed converting candidate embeddings to numeric array: {e}")

    cand_embs = l2_normalize(cand_embs)
    q_np = l2_normalize(np.array(q_emb, dtype=np.float32).reshape(-1))

    # debug
    print("DEBUG:", debug_stats(cand_embs, "cand_embs"), debug_stats(q_np, "q_np"))

    finite_mask = np.isfinite(cand_embs).all(axis=1)
    if not finite_mask.all():
        valid_idx = np.where(finite_mask)[0].tolist()
        if len(valid_idx) == 0:
            raise HTTPException(status_code=500, detail="All candidate embeddings are invalid (NaN/inf).")
        cand_embs = cand_embs[valid_idx]
        matches = [matches[i] for i in valid_idx]

    if not np.isfinite(q_np).all():
        raise HTTPException(status_code=500, detail="Query embedding contains NaN/inf.")

    sims = np.dot(cand_embs, q_np)
    if (np.isnan(sims) | np.isinf(sims)).any():
        sims = np.where(np.isfinite(sims), sims, -1e6)

    take_k = min(top_k, len(sims))
    order = np.argsort(-sims)[:take_k]

    out = []
    for idx in order:
        m = matches[int(idx)]
        md = (m.get("metadata") or {})
        stored_text = str(md.get("text", ""))  # full stored chunk from Pinecone
        # snippet: short preview for UI
        preview = stored_text[:300] if len(stored_text) > 300 else stored_text
        # debug log length to check whether Pinecone holds full or partial text
        print(f"DEBUG: match id={m.get('id')} stored_text_len={len(stored_text)}")
        out.append(ChunkResult(
            id=str(m.get("id", "")),
            page=str(md.get("page", "")),
            chunk_index=int(md.get("chunk_index", -1)),
            score=float(sims[int(idx)]),
            snippet=preview,
            full_text=stored_text
        ))

    return SearchResponse(query=q, results=out)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME}
