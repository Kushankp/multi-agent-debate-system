import os
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone.pinecone import Pinecone
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
print("Agent_A: Loading embedding model:", EMB_MODEL_NAME)
model = SentenceTransformer(EMB_MODEL_NAME)

print("Agent_A: Initializing Pinecone client...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except TypeError:
    pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# ---------- Public API for the agent ----------
def search(
    query: str,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a full retrieval: encodes query, queries Pinecone, re-embeds candidates,
    ranks by dot-product and returns a dictionary with 'query' and 'results' as a list of dicts:
    each result has id, page, chunk_index, score, snippet, full_text.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query")

    top_k = max(1, min(100, int(top_k or DEFAULT_TOP_K)))
    fetch_k = max(top_k, min(1000, int(fetch_k or DEFAULT_FETCH_K)))

    # encode query (with same CHUNK_PREFIX convention)
    prefixed_q = CHUNK_PREFIX.format(title="") + q
    q_embs = model.encode([prefixed_q], batch_size=1, show_progress_bar=False)
    q_emb = np.array(q_embs[0], dtype=np.float32)
    q_emb = l2_normalize(q_emb)

    # fetch candidates from Pinecone
    res = index.query(vector=q_emb.tolist(), top_k=fetch_k, include_metadata=True, namespace=namespace)
    matches = res.get("matches", []) or []
    if not matches:
        return {"query": q, "results": []}

    # build candidate texts (truncate only for encoding safety, not for stored metadata)
    candidate_texts = []
    for m in matches:
        meta = (m.get("metadata") or {})
        chunk_text = meta.get("text", "") or ""
        page = meta.get("page", "") or ""
        max_chunk_chars = 2000
        chunk_for_encode = chunk_text if len(chunk_text) <= max_chunk_chars else chunk_text[:max_chunk_chars]
        candidate_texts.append(CHUNK_PREFIX.format(title=page) + chunk_for_encode)

    cand_embs_raw = model.encode(candidate_texts, batch_size=32, show_progress_bar=False)
    cand_embs = np.array(cand_embs_raw, dtype=np.float32)
    if cand_embs.ndim == 1:
        cand_embs = cand_embs.reshape(1, -1)
    if cand_embs.dtype == np.object_:
        # defensive conversion if SentenceTransformer returns object arrays
        cand_embs = np.vstack([np.array(x, dtype=np.float32) for x in cand_embs])

    cand_embs = l2_normalize(cand_embs)
    q_np = l2_normalize(np.array(q_emb, dtype=np.float32).reshape(-1))

    # debug prints (kept for parity with original)
    print("Agent_A DEBUG:", debug_stats(cand_embs, "cand_embs"), debug_stats(q_np, "q_np"))

    finite_mask = np.isfinite(cand_embs).all(axis=1)
    if not finite_mask.all():
        valid_idx = np.where(finite_mask)[0].tolist()
        if len(valid_idx) == 0:
            raise RuntimeError("All candidate embeddings are invalid (NaN/inf).")
        cand_embs = cand_embs[valid_idx]
        matches = [matches[i] for i in valid_idx]

    if not np.isfinite(q_np).all():
        raise RuntimeError("Query embedding contains NaN/inf.")

    sims = np.dot(cand_embs, q_np)
    if (np.isnan(sims) | np.isinf(sims)).any():
        sims = np.where(np.isfinite(sims), sims, -1e6)

    take_k = min(top_k, len(sims))
    order = np.argsort(-sims)[:take_k]

    out_results = []
    for idx in order:
        m = matches[int(idx)]
        md = (m.get("metadata") or {})
        stored_text = str(md.get("text", ""))  # full stored chunk from Pinecone
        preview = stored_text[:300] if len(stored_text) > 300 else stored_text
        print(f"Agent_A DEBUG: match id={m.get('id')} stored_text_len={len(stored_text)}")
        out_results.append({
            "id": str(m.get("id", "")),
            "page": str(md.get("page", "")),
            "chunk_index": int(md.get("chunk_index", -1)),
            "score": float(sims[int(idx)]),
            "snippet": preview,
            "full_text": stored_text
        })

    return {"query": q, "results": out_results}


def health() -> Dict[str, str]:
    return {"status": "ok", "index": INDEX_NAME}
