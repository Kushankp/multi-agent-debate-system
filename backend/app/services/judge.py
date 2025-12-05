import os
import json
import re
import time
import datetime
import threading
import logging
import socket
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError, validator
from typing import Literal

from app.workers.search_adapter import web_search_tavily
from app.services.groq_client import call_groq_chat, extract_assistant_text, DEFAULT_GROQ_MODEL, DEFAULT_MAX_TOKENS
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("judge-groq")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Pydantic schema for judge output
# ---------------------------
class EvidenceItem(BaseModel):
    for_: Literal["A", "B"] = Field(..., alias="for")
    source_title: str
    url: str
    reason: str

class Score(BaseModel):
    A: int
    B: int

    @validator("A", "B")
    def check_range(cls, v):
        if not isinstance(v, int):
            raise ValueError("Score must be integer")
        if v < 0 or v > 100:
            raise ValueError("Score must be 0..100")
        return v

    @validator("B")
    def check_sum(cls, v, values):
        if "A" in values:
            if values["A"] + v != 100:
                raise ValueError("Scores must sum to 100")
        return v

class JudgeOutput(BaseModel):
    winner: Literal["A", "B", "tie", "inconclusive"]
    score: Score
    explanation: str
    evidence_summary: List[EvidenceItem]
    tools_used: List[str]

# ---------------------------
# Ephemeral store for arguments & search results
# ---------------------------
class EphemeralStore:
    def __init__(self, ttl_seconds: int = 300):
        self._args: Dict[str, Dict[str, Any]] = {}
        self._searches: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()

    def _start_timer(self, delay: float, target, *args, **kwargs):
        t = threading.Timer(delay, target, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        return t

    def put_argument(self, argument_id: str, record: Dict[str, Any]):
        with self._lock:
            self._args[argument_id] = {"record": record, "created_at": time.time()}
        self._start_timer(self.ttl + 1, self._expire_argument, argument_id)

    def get_argument(self, argument_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._args.get(argument_id)
            if not entry:
                return None
            return entry["record"]

    def delete_argument(self, argument_id: str):
        with self._lock:
            self._args.pop(argument_id, None)

    def _expire_argument(self, argument_id: str):
        with self._lock:
            entry = self._args.get(argument_id)
            if not entry:
                return
            if time.time() - entry["created_at"] > self.ttl:
                logger.info("EphemeralStore: expiring argument %s", argument_id)
                self._args.pop(argument_id, None)

    def put_search(self, search_id: str, record: Dict[str, Any]):
        with self._lock:
            self._searches[search_id] = {"record": record, "created_at": time.time()}
        self._start_timer(self.ttl + 1, self._expire_search, search_id)

    def get_search(self, search_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._searches.get(search_id)
            return entry["record"] if entry else None

    def delete_search(self, search_id: str):
        with self._lock:
            self._searches.pop(search_id, None)

    def _expire_search(self, search_id: str):
        with self._lock:
            entry = self._searches.get(search_id)
            if not entry:
                return
            if time.time() - entry["created_at"] > self.ttl:
                logger.info("EphemeralStore: expiring search %s", search_id)
                self._searches.pop(search_id, None)

# ---------------------------
# Utility
# ---------------------------
def extract_json_blob(text: str) -> Optional[str]:
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else None

def build_evidence_block(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        title = (r.get("title") or r.get("source_title") or "").replace("\n", " ")
        published = r.get("published", r.get("published_date", "unknown"))
        snippet = (r.get("snippet") or r.get("content") or "")[:400].replace("\n", " ")
        url = r.get("url") or ""
        lines.append(f"{title} — {published} — {snippet} — {url}")
    return "\n".join(lines)

def _print_summary(parsed: Dict[str, Any], top_n: int = 3) -> None:
    """
    Print a concise judge summary: winner, scores, explanation, top evidence.
    Uses both logger.info and stdout print for visibility.
    """
    try:
        winner = parsed.get("winner", "unknown")
        score = parsed.get("score", {})
        explanation = parsed.get("explanation", "")
        evidence = parsed.get("evidence_summary", []) or []

        lines = []
        lines.append("\n" + "="*50)
        lines.append("JUDGE SUMMARY")
        lines.append(f"Winner: {winner}")
        if isinstance(score, dict):
            lines.append(f"Scores — A: {score.get('A')} | B: {score.get('B')}")
        else:
            lines.append(f"Scores: {score}")
        lines.append("")
        if explanation:
            lines.append("Explanation:")
            lines.append(explanation.strip())
            lines.append("")
        if evidence:
            lines.append("Top evidence cited:")
            for e in evidence[:top_n]:
                side = e.get("for")
                title = e.get("source_title") or e.get("title") or "<no title>"
                url = e.get("url") or "<no url>"
                reason = e.get("reason") or ""
                lines.append(f"- ({side}) {title}\n  {url}\n  {reason}")
        else:
            lines.append("No evidence summary provided.")
        lines.append("="*50 + "\n")

        out = "\n".join(lines)
        # print to stdout for immediate visibility and also log as info
        print(out)
        logger.info("Judge summary:\n%s", out)
    except Exception as e:
        logger.exception("Failed to print judge summary: %s", e)

# ---------------------------
# Judge that uses call_groq_chat, same approach as Agent A
# ---------------------------
class JudgeGroq:
    def __init__(self,
                 store: EphemeralStore,
                 web_search_fn = web_search_tavily,
                 groq_model: str = DEFAULT_GROQ_MODEL,
                 max_search_results: int = 5,
                 groq_url: Optional[str] = None):
        self.store = store
        self.web_search = web_search_fn
        self.groq_model = groq_model
        self.max_search_results = max_search_results
        # use same default as Agent_A
        self.groq_url = (groq_url or
                         os.getenv("GROQ_CHAT_URL") or
                         "https://api.groq.com/openai/v1/chat/completions")

    def submit_temporary_argument(self, agent: str, text: str) -> str:
        ts = int(time.time() * 1000)
        arg_id = f"{agent}_{ts}"
        record = {
            "argument_id": arg_id,
            "agent": agent,
            "text": text,
            "tokens": len(text.split()),
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        self.store.put_argument(arg_id, record)
        logger.info("Stored temporary argument %s (ttl=%ds)", arg_id, self.store.ttl)
        return arg_id

    def run_round_judge(self,
                        argA_id: str,
                        argB_id: str,
                        search_results: Optional[List[Dict[str, Any]]] = None,
                        max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
        argA = self.store.get_argument(argA_id)
        argB = self.store.get_argument(argB_id)
        if not argA or not argB:
            self.store.delete_argument(argA_id); self.store.delete_argument(argB_id)
            raise ValueError("One or both arguments not found (expired or missing)")

        # Determine evidence (like before)
        search_ids: List[str] = []
        try:
            if search_results is None:
                qA = " ".join(argA["text"].split()[:60])
                qB = " ".join(argB["text"].split()[:60])
                if len(qA) > 380: qA = qA[:380].rsplit(" ", 1)[0]
                if len(qB) > 380: qB = qB[:380].rsplit(" ", 1)[0]

                searchA = self.web_search(qA, top_k=self.max_search_results) if self.web_search else []
                searchB = self.web_search(qB, top_k=self.max_search_results) if self.web_search else []

                ts = int(time.time() * 1000)
                sidA = f"sA_{ts}"; sidB = f"sB_{ts}"
                self.store.put_search(sidA, {"search_id": sidA, "query": qA, "results": searchA, "created_at": datetime.datetime.utcnow().isoformat()})
                self.store.put_search(sidB, {"search_id": sidB, "query": qB, "results": searchB, "created_at": datetime.datetime.utcnow().isoformat()})

                topA = searchA[:3]; topB = searchB[:3]
                search_ids = [sidA, sidB]
            else:
                ts = int(time.time() * 1000)
                sid = f"s_{ts}"
                self.store.put_search(sid, {"search_id": sid, "query": None, "results": search_results, "created_at": datetime.datetime.utcnow().isoformat()})
                top = (search_results or [])[:6]
                topA = top[:3]
                topB = top[3:6]
                search_ids = [sid]
        except Exception as e:
            self.store.delete_argument(argA_id); self.store.delete_argument(argB_id)
            logger.exception("Web search failed: %s", e)
            raise

        # build prompt messages
        system_msg = {
            "role": "system",
            "content": (
                "You are a neutral, auditable judge. Return ONLY valid JSON matching the schema:\n"
                "{\n"
                '  "winner": "A" | "B" | "tie" | "inconclusive",\n'
                '  "score": {"A": int(0-100), "B": int(0-100)},\n'
                '  "explanation": "2-3 sentence external explanation (no internal chain-of-thought)",\n'
                '  "evidence_summary": [{"for":"A"|"B","source_title":"...","url":"...","reason":"short reason"}],\n'
                '  "tools_used": ["tavily"]\n'
                "}\n"
                "Scores must sum to 100. Be concise. Do NOT output internal chain-of-thought or any tool internals."
            )
        }

        user_content = "\n".join([
            "ARGUMENT A:",
            argA["text"],
            "",
            "EVIDENCE FOR A (from Tavily search):",
            build_evidence_block(topA),
            "",
            "ARGUMENT B:",
            argB["text"],
            "",
            "EVIDENCE FOR B (from Tavily search):",
            build_evidence_block(topB),
            "",
            "JUDGE TASK: Using ONLY the arguments and the evidence provided above, decide which agent's position is better supported.",
            "Return only the JSON described in the system message."
        ])

        messages = [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": user_content}
        ]

        # Defensive logging of environment visibility (helpful for debugging)
        logger.info("Judge env check: GROQ_API_KEY set=%s GROQ_CHAT_URL=%s groq_url(internal)=%s",
                    bool(os.environ.get("GROQ_API_KEY")),
                    os.environ.get("GROQ_CHAT_URL"),
                    self.groq_url)

        # ---- Primary call: use call_groq_chat exactly like Agent A (no fallbacks) ----
        resp_json = None
        try:
            resp_json = call_groq_chat(messages=messages,
                                      model=(self.groq_model or DEFAULT_GROQ_MODEL).strip(),
                                      max_tokens=max_tokens,
                                      temperature=0.0)
        except Exception as e:
            # cleanup ephemeral sensitive data before re-raising
            try:
                self.store.delete_argument(argA_id)
                self.store.delete_argument(argB_id)
                for sid in search_ids:
                    self.store.delete_search(sid)
            except Exception:
                pass
            logger.exception("Model call failed (Agent A style): %s", e)
            raise

        # Extract model content
        content = None
        try:
            content = resp_json["choices"][0]["message"]["content"]
        except Exception:
            try:
                content = resp_json["choices"][0].get("text") or json.dumps(resp_json)
            except Exception:
                content = json.dumps(resp_json)

        # parse JSON (safe)
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            blob = extract_json_blob(content)
            if blob:
                try:
                    parsed = json.loads(blob)
                except Exception:
                    parsed = {"error": "invalid_json", "raw_text": content}
            else:
                parsed = {"error": "invalid_json", "raw_text": content}

        # validate with pydantic
        valid = True
        validation_error = None
        try:
            if "error" in parsed:
                valid = False
            else:
                JudgeOutput.parse_obj(parsed)
        except ValidationError as ve:
            valid = False
            validation_error = ve

        # judge record (minimal, safe)
        judge_record = {
            "judge_id": f"judge_{int(time.time()*1000)}",
            "arguments_consumed": [argA_id, argB_id],
            "search_ids": search_ids,
            "constructed_context": user_content[:4000],
            "model_response": parsed,
            "raw_text": content,
            "model_used": self.groq_model,
            "created_at": datetime.datetime.utcnow().isoformat()
        }

        # cleanup ephemeral arguments and searches (always)
        try:
            self.store.delete_argument(argA_id)
            self.store.delete_argument(argB_id)
            for sid in search_ids:
                self.store.delete_search(sid)
        except Exception:
            logger.exception("Failed to cleanup ephemeral artifacts for judge run")

        if not valid:
            logger.warning("Judge output invalid: %s", validation_error)
            # print a compact error summary
            try:
                print("\n=== JUDGE RUN INVALID ===")
                print("Validation error:", validation_error)
                print("Raw model content preview:", (content or "")[:1000])
                print("=========================\n")
            except Exception:
                pass
            return {"parsed": parsed, "valid": False, "error": str(validation_error), "judge_record": judge_record}
        else:
            logger.info("Judge returned valid result: winner=%s", parsed.get("winner"))
            # Print a concise human-friendly summary to stdout and logs
            try:
                _print_summary(parsed, top_n=3)
            except Exception:
                logger.exception("Failed to print summary")
            return {"parsed": parsed, "valid": True, "judge_record": judge_record}


# ---------------------------
# Example quick-run (demo)
# ---------------------------
if __name__ == "__main__":
    store = EphemeralStore(ttl_seconds=300)
    judge = JudgeGroq(store=store, web_search_fn=web_search_tavily, groq_model=DEFAULT_GROQ_MODEL, max_search_results=5)

    a_id = judge.submit_temporary_argument("A", "Proposition: adopt model X because it is robust and fast.")
    b_id = judge.submit_temporary_argument("B", "Opposition: adopt model Y because it is cheaper and simpler to deploy.")

    try:
        out = judge.run_round_judge(a_id, b_id)
        print(json.dumps(out, indent=2))
    except Exception as e:
        logger.exception("Run failed: %s", e)
