# main.py (fixed: includes CreateSessionReq and SubmitArgReq + /debate/3-round)
import os
import logging
import traceback
import time
import datetime
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.services.Agent_B import generate_answer as agent_generate_b

load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv("MAIN_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("main")

# Import the existing search/generate agent (unchanged)
from app.services.Agent_A import search as agent_search, health as agent_health, generate_answer as agent_generate

# Import the Tavily adapter and Judge implementation (same as before)
try:
    from app.workers.search_adapter import web_search_tavily
except Exception as e:
    logger.exception("Failed to import app.workers.search_adapter: %s", e)
    web_search_tavily = None

try:
    from app.services.judge import JudgeGroq, EphemeralStore
except Exception as e:
    logger.exception("Failed to import app.workers.judge (make sure file is at app/workers/judge.py): %s", e)
    # fallback minimal classes (unchanged from your original)
    class EphemeralStore:
        def __init__(self, ttl_seconds: int = 300):
            self._args = {}
            self._searches = {}
            self.ttl = ttl_seconds
        def put_argument(self, argument_id, record): self._args[argument_id] = {"record":record,"created_at":time.time()}
        def get_argument(self, argument_id): return self._args.get(argument_id, {}).get("record")
        def delete_argument(self, argument_id): self._args.pop(argument_id, None)
        def put_search(self, search_id, record): self._searches[search_id] = {"record":record,"created_at":time.time()}
        def delete_search(self, search_id): self._searches.pop(search_id, None)

    class JudgeGroq:
        def __init__(self, store, web_search_fn=None, groq_model=None, max_search_results=5):
            self.store = store
            self.web_search = web_search_fn
        def submit_temporary_argument(self, agent, text):
            arg_id = f"{agent}_{int(time.time()*1000)}"
            self.store.put_argument(arg_id, {"argument_id":arg_id,"agent":agent,"text":text,"created_at":datetime.datetime.utcnow().isoformat()})
            return arg_id
        def run_round_judge(self, a_id, b_id, search_results=None):
            a = self.store.get_argument(a_id); b = self.store.get_argument(b_id)
            if not a or not b:
                self.store.delete_argument(a_id); self.store.delete_argument(b_id)
                raise ValueError("Missing args")
            la=len(a["text"].split()); lb=len(b["text"].split()); total=max(1,la+lb)
            A=int(100*la/total); B=100-A
            parsed={"winner":"A" if A>B else "B" if B>A else "tie","score":{"A":A,"B":B},"explanation":"fallback demo","evidence_summary":[],"tools_used":[]}
            self.store.delete_argument(a_id); self.store.delete_argument(b_id)
            return {"parsed":parsed,"valid":True,"judge_record":{"judge_id":f"fake_{int(time.time()*1000)}"}}

# FastAPI app
app = FastAPI(title="RAG Retrieval + Generation API")
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
    top_k: Optional[int] = None
    fetch_k: Optional[int] = None
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

class GenerateRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    fetch_k: Optional[int] = 50
    namespace: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    context_chars: Optional[int] = None

class GenerateResponse(BaseModel):
    query: str
    answer: str
    raw: Optional[dict] = None

# ---------- MISSING models (now included) ----------
class CreateSessionReq(BaseModel):
    topic: Optional[str] = None
    ttl_seconds: Optional[int] = 300

class SubmitArgReq(BaseModel):
    agent: str
    text: str

# Debate request/response
class DebateRequest(BaseModel):
    initial_query: str
    top_k: Optional[int] = 5
    fetch_k: Optional[int] = 50
    namespace: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    context_chars: Optional[int] = None

class Turn(BaseModel):
    agent: str
    text: str
    raw: Optional[dict] = None

class Debate3Response(BaseModel):
    rounds: List[Turn]  # sequence of 6 turns: A1,B1,A2,B2,A3,B3
    final: Dict[str, Any]

# Session store (unchanged)
class SessionStore:
    def __init__(self):
        self._s: Dict[str, Dict[str, Any]] = {}
    def create(self, topic=None, ttl_seconds=300):
        sid = f"sess_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl_seconds)).isoformat()
        self._s[sid] = {"session_id":sid,"topic":topic,"participants":[],"round":1,"arguments":[],"search_ids":[],"judge_result":None,"created_at":datetime.datetime.utcnow().isoformat(),"expires_at":expires_at}
        logger.info("Created session %s", sid)
        return self._s[sid]
    def get(self, sid):
        s=self._s.get(sid)
        if not s: return None
        if "expires_at" in s and datetime.datetime.fromisoformat(s["expires_at"]) < datetime.datetime.utcnow():
            self._s.pop(sid,None); return None
        return s
    def update(self, sid, session):
        self._s[sid]=session
    def delete(self, sid):
        self._s.pop(sid, None)

session_store = SessionStore()

# initialize ephemeral store + judge instance
ephemeral_store = EphemeralStore(ttl_seconds=300)
judge_instance = JudgeGroq(store=ephemeral_store, web_search_fn=(web_search_tavily if web_search_tavily else None))

# ---------- Endpoints (unchanged search/generate) ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        out = agent_search(query=req.query, top_k=req.top_k, fetch_k=req.fetch_k, namespace=req.namespace)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("/search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    if not isinstance(out, dict):
        raise HTTPException(status_code=500, detail="Search failed: invalid agent response")
    return SearchResponse(query=out.get("query",""), results=[ChunkResult(**r) for r in out.get("results",[])])

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request):
    try:
        out = agent_generate(query=req.query, top_k=req.top_k or 5, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("/generate failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")
    if out is None or not isinstance(out, dict):
        raise HTTPException(status_code=502, detail="Generation failed: agent returned invalid data")
    return GenerateResponse(query=out.get("query",""), answer=out.get("answer",""), raw=out.get("raw", None))

# ---------- Agent B generation endpoint (unchanged) ----------
@app.post("/agent-b/generate")
def agent_b_generate(request: GenerateRequest):
    """
    Generate Agent B’s argument. Returns parsed b_args as top-level field `b_args`.
    """
    try:
        out = agent_generate_b(
            query=request.query,
            top_k=request.top_k or 3,
            fetch_k=request.fetch_k or 50,
            namespace=request.namespace,
            model=request.model,
            max_tokens=request.max_tokens,
            context_chars=request.context_chars,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("/agent-b/generate failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Agent B generation failed: {e}")

    if out is None or not isinstance(out, dict):
        raise HTTPException(status_code=502, detail="Agent B returned invalid data")

    raw = out.get("raw") or {}
    parsed_b_args = raw.get("parsed_b_args") if isinstance(raw, dict) else None

    return {
        "query": out.get("query", ""),
        "answer": out.get("answer", ""),       # legacy client support
        "b_args": parsed_b_args,               # parsed dict or null
        "assistant_text": raw.get("assistant_text"),
        "llm_raw": raw.get("llm_raw"),
        "raw": raw,                            # full raw debug for inspection
    }

# ---------- NEW: 3-round debate orchestration ----------
@app.post("/debate/3-round", response_model=Debate3Response)
def debate_three_rounds(req: DebateRequest):
    """
    Run a 3-round debate:
      A1 = Agent A on initial_query
      B1 = Agent B on A1.answer
      A2 = Agent A on B1.answer
      B2 = Agent B on A2.answer
      A3 = Agent A on B2.answer
      B3 = Agent B on A3.answer

    Returns the sequence of turns (A1,B1,A2,B2,A3,B3) with text + raw debug.
    """
    if not req.initial_query or not req.initial_query.strip():
        raise HTTPException(status_code=400, detail="initial_query is required")

    rounds = []
    try:
        # Round 1: A1
        a1 = agent_generate(query=req.initial_query, top_k=req.top_k or 5, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "A", "text": a1.get("answer",""), "raw": a1.get("raw")})

        # B1
        b1_query = a1.get("answer","")
        b1 = agent_generate_b(query=b1_query, top_k=req.top_k or 3, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "B", "text": b1.get("answer",""), "raw": b1.get("raw")})

        # A2
        a2_query = b1.get("answer","")
        a2 = agent_generate(query=a2_query, top_k=req.top_k or 5, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "A", "text": a2.get("answer",""), "raw": a2.get("raw")})

        # B2
        b2_query = a2.get("answer","")
        b2 = agent_generate_b(query=b2_query, top_k=req.top_k or 3, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "B", "text": b2.get("answer",""), "raw": b2.get("raw")})

        # A3
        a3_query = b2.get("answer","")
        a3 = agent_generate(query=a3_query, top_k=req.top_k or 5, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "A", "text": a3.get("answer",""), "raw": a3.get("raw")})

        # B3
        b3_query = a3.get("answer","")
        b3 = agent_generate_b(query=b3_query, top_k=req.top_k or 3, fetch_k=req.fetch_k or 50, namespace=req.namespace, model=req.model, max_tokens=req.max_tokens, context_chars=req.context_chars)
        rounds.append({"agent": "B", "text": b3.get("answer",""), "raw": b3.get("raw")})

        final = {
            "A3": a3.get("answer",""),
            "B3": b3.get("answer",""),
            "summary_note": "3-round debate finished"
        }

        # Return as Pydantic-compatible dicts
        return Debate3Response(rounds=[Turn(**r) for r in rounds], final=final)

    except Exception as e:
        logger.exception("/debate/3-round failed: %s", e)
        raise HTTPException(status_code=500, detail=f"3-round debate failed: {e}")

# ---------- Session endpoints and judge (unchanged) ----------
@app.post("/session")
def create_session(req: CreateSessionReq):
    try:
        s = session_store.create(topic=req.topic, ttl_seconds=req.ttl_seconds or 300)
        return {"session_id": s["session_id"], "expires_at": s["expires_at"]}
    except Exception as e:
        logger.exception("/session create failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

@app.post("/session/{session_id}/argument")
def submit_argument(session_id: str, req: SubmitArgReq):
    s = session_store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if req.agent not in ("A","B"):
        raise HTTPException(status_code=400, detail="agent must be 'A' or 'B'")
    arg_id = f"{req.agent}_{int(time.time()*1000)}"
    arg_record = {"argument_id": arg_id, "agent": req.agent, "text": req.text, "created_at": datetime.datetime.utcnow().isoformat()}
    s["arguments"] = [a for a in s["arguments"] if a["agent"] != req.agent] + [arg_record]
    if req.agent not in s["participants"]:
        s["participants"].append(req.agent)
    session_store.update(session_id, s)
    return {"session_id": session_id, "argument_id": arg_id, "agent": req.agent}

@app.post("/session/{session_id}/judge")
def run_judge_on_session(session_id: str):
    s = session_store.get(session_id)
    if not s:
        logger.warning("/session/%s/judge: session not found", session_id)
        raise HTTPException(status_code=404, detail="Session not found or expired")

    agents_present = {a["agent"] for a in s["arguments"]}
    if not {"A", "B"}.issubset(agents_present):
        raise HTTPException(status_code=400, detail="Both Agent A and Agent B must submit arguments first")

    # pick latest arguments
    a_arg = next(a for a in s["arguments"] if a["agent"] == "A")
    b_arg = next(a for a in s["arguments"] if a["agent"] == "B")

    # --- NEW: use the session topic/query to call Tavily once and store results in session ---
    search_query = (s.get("topic") or "").strip()
    # Safety: ensure search_query is not too long for Tavily
    if search_query and len(search_query) > 380:
        search_query = search_query[:380].rsplit(" ", 1)[0]
    search_results = []
    if search_query and (web_search_tavily is not None):
        try:
            logger.info("Running Tavily search for session %s: len=%d", session_id, len(search_query))
            search_results = web_search_tavily(search_query, top_k=6) or []
        except Exception as e:
            logger.exception("/session/%s/judge: Tavily search failed: %s", session_id, e)
            # continue: we will still run the judge but with empty evidence
            search_results = []

    # store search results in session for UI / audit
    s["search_query"] = search_query
    s["search_results"] = search_results
    session_store.update(session_id, s)

    # submit to ephemeral judge and run (pass search_results into judge)
    try:
        argA_temp_id = judge_instance.submit_temporary_argument("A", a_arg["text"])
        argB_temp_id = judge_instance.submit_temporary_argument("B", b_arg["text"])
    except Exception as e:
        logger.exception("/session/%s/judge: failed to submit temporary arguments: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to submit arguments: {e}")

    try:
        # pass search_results into judge so judge uses them rather than calling web_search again
        out = judge_instance.run_round_judge(argA_temp_id, argB_temp_id, search_results=search_results)
    except Exception as e:
        logger.exception("/session/%s/judge: judge run failed: %s", session_id, e)
        # attempt cleanup of ephemeral arguments
        try:
            ephemeral_store.delete_argument(argA_temp_id)
            ephemeral_store.delete_argument(argB_temp_id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Judge run failed: {e}")

    # attach minimal judge result to session for frontend traceability (minimal info)
    try:
        s["judge_result"] = out.get("judge_record") if isinstance(out, dict) else out
        session_store.update(session_id, s)
    except Exception:
        logger.exception("Failed to attach judge result to session %s", session_id)

    # Normalize response for frontend (same as before)
    parsed = out.get("parsed") if isinstance(out, dict) else None
    if parsed and not parsed.get("error"):
        winner = parsed.get("winner")
        scoreA = parsed.get("score", {}).get("A", parsed.get("score_a", 0))
        scoreB = parsed.get("score", {}).get("B", parsed.get("score", 0))
        explanation = parsed.get("explanation") or parsed.get("reason", "")
        evidence = parsed.get("evidence_summary", parsed.get("evidence", []))
        return {
            "parsed": parsed,
            "valid": out.get("valid", False),
            "winner": winner,
            "score_a": scoreA,
            "score_b": scoreB,
            "explanation": explanation,
            "evidence_summary": evidence,
            "judge_record": out.get("judge_record"),
            "search_query": s.get("search_query"),
            "search_results": s.get("search_results"),
        }
    else:
        # if model returned invalid / error, also return session search results to help debugging
        if isinstance(out, dict):
            out["search_query"] = s.get("search_query")
            out["search_results"] = s.get("search_results")
        return out

# health
@app.get("/health")
def health():
    return agent_health()
