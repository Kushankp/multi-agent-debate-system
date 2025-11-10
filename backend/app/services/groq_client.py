import os
import socket
import time
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger("groq-client")
logging.basicConfig(level=logging.INFO)

# configuration (use .env)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Default to the same endpoint used by Agent_A (hostname, no IP fallbacks)
GROQ_CHAT_URL = os.environ.get("GROQ_CHAT_URL", "https://api.groq.com/openai/v1/chat/completions")
# NOTE: make sure there is no trailing space in the default model name
DEFAULT_GROQ_MODEL = os.environ.get("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "900"))
GROQ_MOCK = os.environ.get("GROQ_MOCK", "0") == "1"


def _resolve_host(url: str) -> None:
    """Log DNS lookup status for host in URL; do not raise to avoid failing fast."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname
    if not host:
        logger.warning("Invalid Groq URL (no host): %s", url)
        return
    try:
        socket.getaddrinfo(host, 443)
        logger.debug("DNS resolve OK for host: %s", host)
    except socket.gaierror as dns_e:
        # Log as warning but do not raise — let requests raise on actual network call.
        logger.warning("DNS resolution warning for Groq host '%s': %s", host, dns_e)


def call_groq_chat(messages: List[Dict[str, str]],
                   model: str = DEFAULT_GROQ_MODEL,
                   max_tokens: int = DEFAULT_MAX_TOKENS,
                   temperature: float = 0.0,
                   retries: int = 4,
                   timeout: int = 60,
                   groq_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Robust Groq HTTP wrapper.
    - Uses only the primary configured URL (no fallback).
    - Retries with exponential backoff for transient codes (429,5xx).
    - Returns parsed JSON or raises RuntimeError with a clear message.
    """
    # dev mock: return canned response for local UI dev when network unavailable
    if GROQ_MOCK:
        logger.info("GROQ_MOCK=1: returning canned response for development")
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"winner":"A","score":{"A":60,"B":40},"explanation":"(mock) Agent A made slightly stronger case","evidence_summary":[],"tools_used":["mock"]}'
                    }
                }
            ]
        }

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    # Use provided groq_url param or environment primary only
    groq_url = (groq_url or os.environ.get("GROQ_CHAT_URL") or GROQ_CHAT_URL).strip()
    if not groq_url:
        raise RuntimeError("GROQ_CHAT_URL not configured")

    # Defensive: strip model name
    model = (model or DEFAULT_GROQ_MODEL).strip()

    # DNS pre-check: log but do not fail the run (keeps behavior consistent with Agent_A)
    try:
        _resolve_host(groq_url)
    except Exception as e:
        # _resolve_host now logs warnings; this except is defensive in case of unexpected errors
        logger.warning("Unexpected error during DNS pre-check (continuing): %s", e)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    backoff = 1.0
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("call_groq_chat POST %s attempt=%d model=%s", groq_url, attempt, model)
            r = requests.post(groq_url, headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            last_exc = e
            logger.warning("call_groq_chat: request exception: %s", e)
            time.sleep(backoff)
            backoff *= 2.0
            continue

        if r.status_code != 200:
            logger.warning("call_groq_chat: non-200 status=%d body=%s", r.status_code, (r.text or "")[:1000])
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2.0
                continue
            # non-retryable -> raise with body included
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Groq returned HTTP {r.status_code}: {e}; body={r.text[:1000]}")

        try:
            parsed = r.json()
            return parsed
        except ValueError:
            raise RuntimeError("Invalid JSON response from Groq: " + (r.text or "")[:2000])

    if last_exc:
        raise RuntimeError(f"Failed to call Groq after retries; last exception: {last_exc}")
    raise RuntimeError("Failed to call Groq: unknown error")


def extract_assistant_text(raw: Dict[str, Any]) -> str:
    """Normalize Groq/OpenAI-like response shapes -> assistant content text."""
    if not raw:
        return ""
    try:
        choices = raw.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first.get("message"), dict) and first["message"].get("content"):
                return first["message"]["content"]
            if first.get("text"):
                return first["text"]
            if isinstance(first.get("delta"), dict):
                return first["delta"].get("content") or first["delta"].get("text") or ""
    except Exception:
        logger.debug("extract_assistant_text: choices parse failed", exc_info=True)

    if raw.get("output"):
        out = raw["output"]
        if isinstance(out, list):
            return "\n".join(str(x) for x in out if x)
        if isinstance(out, dict) and out.get("text"):
            return out.get("text")
        if isinstance(out, str):
            return out
    if raw.get("generated_text"):
        return raw.get("generated_text")
    return ""
