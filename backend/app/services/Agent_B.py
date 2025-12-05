# app/services/Agent_B.py
import os
import re
import time
import json
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple

# reuse Agent A's search/groq helpers to avoid duplication & mismatches
try:
    from app.services.Agent_A import (
        search as agent_a_search,
        build_context_from_results,
        call_groq_chat,
        extract_assistant_text,
        DEFAULT_GROQ_MODEL,
        DEFAULT_MAX_TOKENS,
    )
except Exception as e:
    # If import fails, raise an explicit error so developer notices
    raise RuntimeError(f"Agent_B: failed to import dependencies from Agent_A: {e}")

# Logging
logging.basicConfig(level=os.getenv("AGENT_B_LOG_LEVEL", "INFO"))
logger = logging.getLogger("Agent_B")

# Config (you can override via env if needed)
DEFAULT_TOP_K = int(os.getenv("AGENT_B_TOP_K", "5"))
DEFAULT_FETCH_K = int(os.getenv("AGENT_B_FETCH_K", "50"))
DEFAULT_CONTEXT_CHARS = int(os.getenv("AGENT_B_CONTEXT_CHARS", "2000"))

def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    """
    Try parse JSON from free text. Returns Python object or None.
    """
    if not text:
        return None
    text = text.strip()
    # direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # attempt to find first {...} or [...] blob
    try:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            candidate = m.group(1)
            return json.loads(candidate)
    except Exception:
        pass
    return None

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Collapse excessive whitespace and trim
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def build_b_prompt_from_context(query: str, context_str: str) -> Tuple[str, str]:
    """
    Build messages (system + user) for Agent B LLM call.
    Returns (system_text, user_text)

    Instructions:
      - Output ONLY a single JSON object with top-level key 'b_args'.
      - 'b_args' must include:
         - 'prompt': a single-paragraph, fluent debate reply (2-6 sentences) that directly rebuts or advances
           the argument provided by Agent A using the context. It should read like a debater speaking.
         - 'notes': optional short parenthetical note about uncertainty or evidence (max ~240 chars).
      - DO NOT include 'confidence' or 'followup_questions' in the visible reply.
      - The 'prompt' must NOT be a header, list, or meta-instruction — it must be the actual rebuttal text.
      - Return ONLY valid JSON and nothing else.
    """
    system_text = (
"You are Agent B — a live debate respondent. Output ONLY valid JSON with a top-level key 'b_args'. "
    "Inside 'b_args' include exactly one field: 'prompt' (string). Do NOT return any other top-level keys.\n"
    "Follow these rules exactly and nothing else:\n"
    "1) 'b_args.prompt' must be a single natural paragraph, 2–6 sentences total. Do NOT output lists, headers, or meta-instructions.\n"
    "2) Begin with a single, short, assertive claim sentence (one sentence). DO NOT start with filler openings such as 'While', 'Although', 'However', 'It's true that', or similar — the first token must be a direct claim.\n"
    "3) In the next 1–3 sentences, directly rebut or advance Agent A's last argument (the Agent A text is provided in the user message). Use the provided context to *synthesize* evidence; do NOT quote, cite, or name any specific book, paper, author, website, or source title. Paraphrase across the retrieved context rather than copying single-resource lines.\n"
    "4) Do NOT include 'notes', parentheses of uncertainty, confidence scores, or follow-up questions in 'b_args.prompt'. If you must record uncertainty, return it only in the raw debug (llm_raw) but not in the visible prompt.\n"
    "5) Be concise, forceful, and evidence-aware. Return ONLY a single valid JSON object and nothing else. Example:\n"
    '{ "b_args": { "prompt": "Claim sentence. Evidence sentence. Concluding sentence." } }'
    )

    user_text = (
        f"Agent A argument (query):\n{query}\n\n"
        f"Context (retrieved and summarized):\n{context_str}\n\n"
        "Now produce the JSON object exactly as instructed above."
    )

    return system_text, user_text

def _safe_int_or_default(val, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default

def _extract_assistant_text_from_raw(raw_resp) -> str:
    """
    Try multiple ways to obtain assistant text from the model response.
    """
    assistant_text = ""
    try:
        # preferred helper (Agent_A provides it)
        if extract_assistant_text:
            assistant_text = extract_assistant_text(raw_resp) or ""
            if assistant_text:
                return assistant_text
    except Exception:
        pass

    # try common shapes
    try:
        if isinstance(raw_resp, dict):
            choices = raw_resp.get("choices", [])
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # new-style: message.content
                if isinstance(first.get("message"), dict) and first["message"].get("content"):
                    return first["message"]["content"]
                if first.get("text"):
                    return first.get("text")
                # delta fragments
                if isinstance(first.get("delta"), dict):
                    return first["delta"].get("content") or first["delta"].get("text") or ""
            # fallback to stringified 'output' or 'generated_text'
            if raw_resp.get("output"):
                out = raw_resp.get("output")
                if isinstance(out, str):
                    return out
                if isinstance(out, list):
                    return "\n".join([str(x) for x in out])
                if isinstance(out, dict):
                    return out.get("text") or str(out)
            if raw_resp.get("generated_text"):
                return raw_resp.get("generated_text")
    except Exception:
        pass

    # last resort: stringify whole raw_resp
    try:
        return str(raw_resp)
    except Exception:
        return ""

def generate_answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    namespace: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    context_chars: Optional[int] = DEFAULT_CONTEXT_CHARS,
) -> Dict[str, Any]:
    """
    Agent B generation entrypoint. Returns dict in the same shape as Agent A:
      { "query": query, "answer": "<string>", "raw": {...} }

    Behavior:
      - Run retrieval using Agent A's `search` to get context.
      - Build a compact context via build_context_from_results.
      - Ask Groq (via call_groq_chat) to produce JSON under 'b_args'.
      - Parse and return structured results. Fallback: auto-wrap free-text into b_args.prompt.
      - Visible reply DOES NOT include confidence or follow-ups.
    """
    start_ts = time.time()

    # Defensive normalization of optional inputs
    top_k = _safe_int_or_default(top_k, DEFAULT_TOP_K)
    fetch_k = _safe_int_or_default(fetch_k, DEFAULT_FETCH_K)
    model_to_use = model or DEFAULT_GROQ_MODEL
    max_tokens_to_use = _safe_int_or_default(max_tokens, DEFAULT_MAX_TOKENS)
    context_chars = _safe_int_or_default(context_chars, DEFAULT_CONTEXT_CHARS)

    logger.info(
        "Agent_B.generate_answer: query=%s top_k=%d fetch_k=%d model=%s",
        (query or "")[:200],
        top_k,
        fetch_k,
        model_to_use,
    )

    try:
        if not query or not query.strip():
            raise ValueError("Empty query for Agent B")

        # 1) Retrieval using Agent A's search (ensures embedding/index compatibility)
        try:
            retrieved = agent_a_search(query=query, top_k=top_k, fetch_k=fetch_k, namespace=namespace)
            results = retrieved.get("results", []) if isinstance(retrieved, dict) else []
            logger.info("Agent_B: retrieval returned %d chunks", len(results))
        except Exception as e:
            logger.exception("Agent_B: retrieval failed: %s", e)
            results = []

        if not results:
            context_str = "(no retrieved context)"
            sources = []
        else:
            context_str, sources = build_context_from_results(results, max_chars=context_chars)

        # 2) Build LLM prompt instructing JSON output under 'b_args'
        sys_txt, user_txt = build_b_prompt_from_context(query, context_str)
        messages = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_txt},
        ]

        # 3) Call Groq chat
        raw_resp = None
        parsed_b_args = None
        assistant_text = ""

        try:
            raw_resp = call_groq_chat(
                messages=messages, model=model_to_use, max_tokens=max_tokens_to_use, temperature=0.0
            )
            assistant_text = _extract_assistant_text_from_raw(raw_resp) or ""
            logger.debug("Agent_B: assistant_text preview (first 2000 chars): %s", assistant_text[:2000])
            # extra debug to help locate source of JSON wrapper
            logger.debug("Agent_B: assistant_text (startswith 200 chars): %s", assistant_text[:200])

            # parse JSON from assistant_text
            parsed = extract_json_blob(assistant_text)
            if parsed and isinstance(parsed, dict) and "b_args" in parsed:
                parsed_b_args = parsed["b_args"]
            else:
                # if model returned top-level prompt/notes, accept those keys
                if parsed and isinstance(parsed, dict) and any(k in parsed for k in ("prompt", "notes")):
                    parsed_b_args = parsed
                else:
                    parsed_b_args = None
        except Exception as e:
            logger.exception("Agent_B: LLM call failed: %s", e)
            raw_resp = {"error": str(e)}
            assistant_text = _extract_assistant_text_from_raw(raw_resp)

        # If LLM returned free-text but no JSON, convert it into b_args
        parsing_info = {"was_wrapped_from_text": False}
        if not parsed_b_args and assistant_text and assistant_text.strip():
            safe_prompt = assistant_text.strip()
            if len(safe_prompt) > 3000:
                safe_prompt = safe_prompt[:3000].rsplit(" ", 1)[0] + "..."
            # Keep notes internally but do not expose them in the user-facing answer.
            parsed_b_args = {
                "prompt": safe_prompt,
                "notes": ""
            }
            parsing_info["was_wrapped_from_text"] = True

        # 4) If parsed_b_args exists, produce a user-friendly debate reply in `answer`
        if parsed_b_args:
            # 1) Try to extract a structured prompt from assistant_text if it contains JSON
            chosen_prompt = None
            try:
                parsed_from_assistant = extract_json_blob(assistant_text) if assistant_text else None
                if isinstance(parsed_from_assistant, dict):
                    if "b_args" in parsed_from_assistant and isinstance(parsed_from_assistant["b_args"], dict):
                        chosen_prompt = parsed_from_assistant["b_args"].get("prompt")
                    elif "prompt" in parsed_from_assistant:
                        chosen_prompt = parsed_from_assistant.get("prompt")
            except Exception:
                chosen_prompt = None

            # 2) If no prompt found inside assistant_text JSON, fall back to parsed_b_args
            if not chosen_prompt:
                chosen_prompt = parsed_b_args.get("prompt") or parsed_b_args.get("instruction") or None

            # 3) Final fallback: use the raw assistant_text (cleaned)
            if chosen_prompt and isinstance(chosen_prompt, str) and chosen_prompt.strip():
                answer_text = chosen_prompt.strip()
            else:
                answer_text = assistant_text.strip() if assistant_text and assistant_text.strip() else ""

            # 4) Remove any visible "Notes: ..." artifact or appended parenthetical referencing auto-wrapping.
            answer_text = re.sub(
                r'Notes:\s*Converted from free-text LLM output.*', '', answer_text, flags=re.IGNORECASE
            ).strip()
            answer_text = re.sub(
                r'\(\s*Converted from free-text LLM output.*\)', '', answer_text, flags=re.IGNORECASE
            ).strip()

            # 5) Remove any leftover empty parentheses and normalize whitespace
            answer_text = re.sub(r'\(\s*\)', '', answer_text).strip()
            answer_text = normalize_text(answer_text)

            raw_out = {
                "llm_raw": raw_resp,
                "assistant_text": assistant_text,
                "parsed_b_args": parsed_b_args,
                "retrieved_count": len(results),
                "parsing_info": parsing_info,
            }
            runtime = time.time() - start_ts
            raw_out["runtime_sec"] = runtime
            return {"query": query, "answer": answer_text, "raw": raw_out}

        # 5) Fallback heuristic (should be rare now)
        cleaned = (query or "").replace("\n", " ").strip()
        head = cleaned[:240] + ("…" if len(cleaned) > 240 else "")
        num_hits = len(results)
        fallback_prompt = (
            f"Agent B summary: {head}\n\n"
            f"Counterpoint: These claims require evidence verification — check recency, sample bias, and applicability. "
            f"Retrieved {num_hits} supporting documents. If you want stricter claims, ask for sources and recency."
        )
        fallback_b_args = {
            "prompt": fallback_prompt,
            "notes": "Fallback generated by Agent B due to LLM parsing failure or missing JSON."
        }
        raw_out = {
            "llm_raw": raw_resp,
            "assistant_text": assistant_text,
            "parsed_b_args": None,
            "retrieved_count": num_hits,
            "fallback_used": True,
        }
        runtime = time.time() - start_ts
        raw_out["runtime_sec"] = runtime
        return {"query": query, "answer": json.dumps(fallback_b_args, ensure_ascii=False), "raw": raw_out}

    except Exception as e:
        logger.error("Agent_B.generate_answer: exception: %s", e)
        logger.debug("trace: %s", traceback.format_exc())
        return {
            "query": query,
            "answer": "",
            "raw": {"error": "exception_in_agent_b", "message": str(e), "traceback": traceback.format_exc()[:2000]},
        }


# For local quick test
if __name__ == "__main__":
    sample = "AI-generated images often drift into abstract faces over repeated transformations — why?"
    out = generate_answer(sample)
    print(json.dumps(out, indent=2, ensure_ascii=False))
