# app/services/Agent_A.py
import os
import re
import time
import logging
import traceback
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from pinecone.pinecone import Pinecone
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("Agent_A")

# ---------- Config (env vars) ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "wiki-ir")
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
CHUNK_PREFIX = os.getenv("CHUNK_PREFIX", "TITLE: {title}\n\n")
DEFAULT_FETCH_K = int(os.getenv("DEFAULT_FETCH_K", 50))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

# Groq generation config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = os.getenv("GROQ_CHAT_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 1000))
# approximate context char budget for building the prompt (simple heuristic)
DEFAULT_CONTEXT_CHARS = int(os.getenv("CONTEXT_CHARS", 3000))

if not PINECONE_API_KEY or not PINECONE_ENV:
    logger.error("Missing Pinecone configuration: PINECONE_API_KEY or PINECONE_ENV not set")
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV environment variables")

logger.info("Agent_A config: index=%s emb_model=%s groq_model=%s", INDEX_NAME, EMB_MODEL_NAME, DEFAULT_GROQ_MODEL)

def normalize_answer_text(s: str) -> str:
    if not s:
        return ""
    # remove leading separators like '---' or '___' and surrounding whitespace/newlines
    s = re.sub(r'^\s*[-_]{2,}\s*', '', s)
    # collapse 3+ newlines into 2
    s = re.sub(r'\n{3,}', '\n\n', s)
    # strip leading/trailing whitespace
    return s.strip()

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
logger.info("Agent_A: Loading embedding model: %s", EMB_MODEL_NAME)
model = SentenceTransformer(EMB_MODEL_NAME)

logger.info("Agent_A: Initializing Pinecone client...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except TypeError:
    pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

# ---------- Existing search function (kept mostly as-is) ----------
def search(
    query: str,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty query")

    top_k = max(1, min(100, int(top_k or DEFAULT_TOP_K)))
    fetch_k = max(top_k, min(1000, int(fetch_k or DEFAULT_FETCH_K)))

    logger.debug("search: query=%s top_k=%d fetch_k=%d namespace=%s", q, top_k, fetch_k, namespace)

    # encode query (with same CHUNK_PREFIX convention)
    prefixed_q = CHUNK_PREFIX.format(title="") + q
    q_embs = model.encode([prefixed_q], batch_size=1, show_progress_bar=False)
    q_emb = np.array(q_embs[0], dtype=np.float32)
    q_emb = l2_normalize(q_emb)

    # fetch candidates from Pinecone
    logger.debug("search: querying pinecone index=%s", INDEX_NAME)
    res = index.query(vector=q_emb.tolist(), top_k=fetch_k, include_metadata=True, namespace=namespace)
    matches = res.get("matches", []) or []
    logger.info("search: pinecone returned %d matches", len(matches))
    if not matches:
        return {"query": q, "results": []}

    # build candidate texts (truncate only for encoding safety)
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
        cand_embs = np.vstack([np.array(x, dtype=np.float32) for x in cand_embs])

    cand_embs = l2_normalize(cand_embs)
    q_np = l2_normalize(np.array(q_emb, dtype=np.float32).reshape(-1))

    logger.debug("Agent_A DEBUG: %s %s", debug_stats(cand_embs, "cand_embs"), debug_stats(q_np, "q_np"))

    finite_mask = np.isfinite(cand_embs).all(axis=1)
    if not finite_mask.all():
        valid_idx = np.where(finite_mask)[0].tolist()
        if len(valid_idx) == 0:
            raise RuntimeError("All candidate embeddings are invalid (NaN/inf).")
        cand_embs = cand_embs[valid_idx]
        matches = [matches[i] for i in valid_idx]
        logger.warning("search: dropped invalid candidate embeddings, remaining=%d", len(matches))

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
        out_results.append({
            "id": str(m.get("id", "")),
            "page": str(md.get("page", "")),
            "chunk_index": int(md.get("chunk_index", -1)),
            "score": float(sims[int(idx)]),
            "snippet": preview,
            "full_text": stored_text
        })

    logger.info("search: returning %d results", len(out_results))
    return {"query": q, "results": out_results}

def health() -> Dict[str, str]:
    return {"status": "ok", "index": INDEX_NAME}

# ---------- New: build_context to assemble top-K snippets into a single context ----------
def build_context_from_results(results: List[Dict[str, Any]], max_chars: int = DEFAULT_CONTEXT_CHARS) -> Tuple[str, List[str]]:
    """
    Concatenate high-score snippets with source metadata until max_chars is reached.
    Returns (context_str, list_of_sources)
    """
    parts: List[str] = []
    included_sources: List[str] = []
    cur_len = 0
    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        txt = (r.get("full_text") or r.get("snippet") or "").strip()
        if not txt:
            continue
        source = r.get("page") or r.get("id") or "unknown"
        header = f"[Source: {source} | score:{r.get('score', 0):.3f}]\n"
        part = header + txt + "\n\n"
        if cur_len + len(part) > max_chars:
            remaining = max_chars - cur_len - 20
            if remaining > 50:
                truncated = txt[:remaining].rsplit(" ", 1)[0] + "..."
                parts.append(header + truncated + "\n\n")
                included_sources.append(source)
            logger.debug("build_context: truncated context at source=%s", source)
            break
        parts.append(part)
        included_sources.append(source)
        cur_len += len(part)

    context = "".join(parts)
    logger.info("build_context: built context len=%d chars using %d sources", len(context), len(included_sources))
    return context, included_sources

# ---------- New: robust Groq call + extractor ----------
def call_groq_chat(messages: List[Dict[str, str]], model: str = DEFAULT_GROQ_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0, retries: int = 4, timeout: int = 60) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in environment")
        raise RuntimeError("GROQ_API_KEY not set in environment")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    backoff = 1.0
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("call_groq_chat: attempt=%d model=%s max_tokens=%d", attempt, model, max_tokens)
            r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            last_exception = e
            logger.warning("call_groq_chat: exception on request: %s", e)
            time.sleep(backoff)
            backoff *= 2.0
            continue

        # Log status and body for non-200
        if r.status_code != 200:
            logger.warning("call_groq_chat: non-200 status=%d body=%s", r.status_code, r.text[:1000])
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2.0
                continue
            # other errors -> raise
            try:
                r.raise_for_status()
            except Exception as e:
                logger.error("call_groq_chat: raising after non-retryable status: %s", e)
                raise

        # Try parse JSON
        try:
            parsed = r.json()
            logger.debug("call_groq_chat: parsed response keys=%s", list(parsed.keys()))
            return parsed
        except ValueError:
            logger.error("call_groq_chat: invalid JSON response: %s", r.text[:2000])
            raise RuntimeError("Invalid JSON from Groq: " + r.text[:2000])

    # all retries failed
    if last_exception:
        logger.error("call_groq_chat: all retries failed; last exception: %s", last_exception)
        raise RuntimeError(f"Failed to call Groq: {last_exception}")
    raise RuntimeError("Failed to call Groq: non-200 responses")

def extract_assistant_text(raw: Dict[str, Any]) -> str:
    """
    Support different possible shapes returned by Groq / OpenAI-compatible APIs.
    Prioritize: choices[].message.content -> choices[].text -> output -> generated_text
    """
    if not raw:
        return ""

    # choices -> message.content / text / delta fragments
    try:
        choices = raw.get("choices", [])
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            # new-style: message.content
            if isinstance(first.get("message"), dict) and first["message"].get("content"):
                return first["message"]["content"]
            # older openai: text
            if first.get("text"):
                return first["text"]
            # streaming delta fragments
            if isinstance(first.get("delta"), dict):
                # prefer content or text
                if first["delta"].get("content"):
                    return first["delta"].get("content")
                if first["delta"].get("text"):
                    return first["delta"].get("text")
    except Exception as e:
        logger.debug("extract_assistant_text: error parsing choices: %s", e)

    # other shapes: 'output' or 'generated_text'
    if raw.get("output"):
        out = raw["output"]
        if isinstance(out, list):
            try:
                return "\n".join([str(x) for x in out if x])
            except Exception:
                return str(out)
        if isinstance(out, dict) and out.get("text"):
            return out.get("text")
        if isinstance(out, str):
            return out

    if raw.get("generated_text"):
        return raw.get("generated_text")

    logger.debug("extract_assistant_text: no assistant text found in raw response. raw keys: %s", list(raw.keys()))
    return ""

def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a JSON object from free text. Returns Python dict or None.
    Useful when LLM output accidentally includes JSON in plain text.
    """
    if not text:
        return None
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try find the first {...} or [...] block
    try:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            candidate = m.group(1)
            return json.loads(candidate)
    except Exception:
        pass
    return None

# ---------- New: high-level generate function that runs retrieval + generation ----------
def generate_answer(
    query: str,
    top_k: int = 5,
    fetch_k: int = 50,
    namespace: Optional[str] = None,
    model: Optional[str] = DEFAULT_GROQ_MODEL,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    context_chars: Optional[int] = DEFAULT_CONTEXT_CHARS,
) -> Dict[str, Any]:
    """
    Agent A generate function updated to behave like Agent B:
      - retrieves context,
      - asks the LLM to produce a single-paragraph debate-style reply (2-6 sentences) that directly addresses
        the question and the retrieved context,
      - returns {"query", "answer", "raw"} where `answer` is the ready-to-display debate reply.
    """
    logger.info("generate_answer (Agent A - debate mode): query=%s top_k=%d fetch_k=%d model=%s", (query or "")[:200], top_k, fetch_k, model)

    try:
        # Normalize defaults (defensive)
        if model is None:
            model = DEFAULT_GROQ_MODEL
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        if context_chars is None:
            context_chars = DEFAULT_CONTEXT_CHARS

        if not query or not query.strip():
            raise ValueError("Empty query")

        # 1) Retrieval
        retrieved = search(query=query, top_k=top_k, fetch_k=fetch_k, namespace=namespace)
        results = retrieved.get("results", []) if isinstance(retrieved, dict) else []

        logger.info("generate_answer: retrieval returned %d results (top_k=%d fetch_k=%d)", len(results), top_k, fetch_k)
        if not results:
            logger.info("generate_answer: no retrieval results; returning empty answer with raw error")
            return {"query": query, "answer": "", "raw": {"error": "no_retrieval_results"}}

        # 2) Build compact context
        context_str, sources = build_context_from_results(results, max_chars=context_chars)
        logger.debug("generate_answer: context preview:\n%s", (context_str[:2000] + "...") if len(context_str) > 2000 else context_str)

        # 3) Compose system + user instructions that force a debate-style reply
        system_msg = (
     "You are ADAM, a live debate respondent. Follow these rules exactly:\n"
    "1) Output a single natural paragraph, 2–6 sentences total. Do NOT output lists, headers, or any meta-instructions.\n"
    "2) Begin with a single, short, assertive claim sentence (one sentence). DO NOT start with filler openings such as "
    "'While', 'Although', 'However', 'It\\'s true that', or similar — the first token must be a direct claim.\n"
    "3) In the next 1–2 sentences, rebut or advance the opponent's last claim (if present) or state a clear opening stance. "
    "Use the provided context to *synthesize* evidence; do NOT quote, cite, or name any specific book, paper, author, website, or source title. "
    "Paraphrase across the retrieved context instead of copying single-resource lines.\n"
    "4) If facts are uncertain, append a single short parenthetical note <= 120 chars at the very end (e.g. '(Some sources may be older.)').\n"
    "5) Be concise, forceful, and evidence-aware. Output only the reply text (no JSON, no extra fields)."
)

        user_msg = (
             "Context (retrieved evidence):\n"
    f"{context_str}\n\n"
    "Question: " + query + "\n\n"
    "If there is an opponent argument available, it will be provided below. "
    "Respond directly to that argument (cite context if possible). Otherwise, state your opening stance.\n\n"
    f"Opponent (if any):\n{{opponent_text}}\n\n"   # replace with the latest opponent text when you call generate
    "Now produce the single-paragraph debate reply following the system rules above."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        logger.debug("generate_answer: calling groq with messages length=%d chars", sum(len(m.get("content","")) for m in messages))

        # 4) Call Groq chat
        raw = call_groq_chat(messages=messages, model=model, max_tokens=max_tokens)
        logger.debug("generate_answer: raw response keys=%s", list(raw.keys()) if isinstance(raw, dict) else type(raw))

        # 5) Extract assistant text and normalize
        assistant_text = extract_assistant_text(raw) or ""
        if not assistant_text:
            logger.warning("generate_answer: empty assistant text from model")
            return {"query": query, "answer": "", "raw": {"error": "empty_answer", "payload": raw}}

        # Keep only the first paragraph (in case model outputs extra)
        assistant_text = assistant_text.strip()
        first_paragraph = assistant_text.split("\n\n")[0].strip()

        # Ensure it is not a meta-instruction — if it looks like JSON or contains 'b_args' key, attempt to extract prompt
        try:
            parsed_blob = extract_json_blob(first_paragraph)
            if isinstance(parsed_blob, dict):
                # try common keys
                if "b_args" in parsed_blob and isinstance(parsed_blob["b_args"], dict) and parsed_blob["b_args"].get("prompt"):
                    reply_text = parsed_blob["b_args"]["prompt"].strip()
                elif parsed_blob.get("prompt"):
                    reply_text = parsed_blob.get("prompt").strip()
                else:
                    reply_text = first_paragraph
            else:
                reply_text = first_paragraph
        except Exception:
            reply_text = first_paragraph

        # Short defensive normalization: collapse many blank lines and trim
        reply_text = normalize_answer_text(reply_text)

        # Log & print for visibility
        logger.info("generate_answer: model generated reply length=%d chars", len(reply_text))
        logger.debug("generate_answer: reply preview:\n%s", (reply_text[:2000] + "...") if len(reply_text) > 2000 else reply_text)

        print("\n" + "="*80)
        print("🔍 RETRIEVED CONTEXT (truncated to 1500 chars):")
        print(context_str[:1500] + ("..." if len(context_str) > 1500 else ""))
        print("="*80)
        print("🤖 AGENT A (debate reply):")
        print(reply_text)
        print("="*80 + "\n")

        # 6) Return same shape as before
        return {"query": query, "answer": reply_text, "raw": raw}

    except Exception as e:
        logger.error("generate_answer: unhandled exception: %s", e)
        logger.debug("generate_answer: traceback: %s", traceback.format_exc())
        return {
            "query": query,
            "answer": "",
            "raw": {"error": "exception_in_generate", "message": str(e), "traceback": traceback.format_exc()[:2000]}
        }
