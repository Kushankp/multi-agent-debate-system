import axios from "axios";

const BACKEND_URL = "http://127.0.0.1:8000";

function safeTrimText(t, max = 10000) {
  if (!t) return "";
  let s = String(t).replace(/\s+/g, " ").trim();
  if (s.length > max) s = s.slice(0, max) + " …";
  return s;
}

/** Clean assistant text from server */
function cleanAssistantText(s) {
  if (!s) return "";
  s = String(s);
  s = s.replace(/^\s*([-_*]{2,})\s*/m, ""); // remove leading '---' or similar
  s = s.replace(/\n{3,}/g, "\n\n"); // collapse huge blank gaps
  return s.trim();
}

/** Collapse excessive blank lines but keep paragraph breaks */
function normalizeAnswerSpacing(text) {
  if (!text) return "";
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Fetches an answer from the backend.
 * Uses /generate by default and falls back to /search on error.
 */
export async function fetchAgentA(query, opts = {}) {
  const {
    model,
    max_tokens,
    context_chars,
    top_k = 3,
    fetch_k = 20,
  } = opts;

  const body = { query, top_k, fetch_k };
  if (model) body.model = model;
  if (max_tokens) body.max_tokens = max_tokens;
  if (context_chars) body.context_chars = context_chars;

  try {
    const res = await axios.post(`${BACKEND_URL}/generate`, body, { timeout: 30000 });
    const data = res.data || {};

    // Optional: debug raw backend response
    console.debug("AgentA /generate →", JSON.stringify(data).slice(0, 2000));

    // Clean and normalize answer text
    let answer = data.answer ? String(data.answer) : "";
    answer = cleanAssistantText(answer);
    answer = normalizeAnswerSpacing(answer);

    // If we got a clean model answer → just return it
    if (answer) return answer;

    // Fallback: run /search if generation fails
    const fallback = await axios.post(`${BACKEND_URL}/search`, { query, top_k, fetch_k }, { timeout: 15000 });
    const results = fallback.data?.results || [];
    if (!results.length) return "I couldn't find relevant documents for that query.";

    const parts = results.map((r, i) => {
      const title = r.page || `Source ${i + 1}`;
      const snippet = safeTrimText(r.snippet || r.full_text || "", 480);
      return `${i + 1}. ${title}\n\n${snippet}`;
    });
    return parts.join("\n\n");
  } catch (err) {
    console.error("agentA /generate error", err?.response?.data || err.message || err);
    try {
      const res2 = await axios.post(`${BACKEND_URL}/search`, { query, top_k, fetch_k }, { timeout: 15000 });
      const results = res2.data?.results || [];
      if (!results.length) return "I couldn't find relevant documents for that query (and generation failed).";

      const parts = results.map((r, i) => {
        const title = r.page || `Source ${i + 1}`;
        const snippet = safeTrimText(r.snippet || r.full_text || "", 480);
        return `${i + 1}. ${title}\n\n${snippet}`;
      });
      return parts.join("\n\n");
    } catch (err2) {
      console.error("agentA /search fallback error", err2?.response?.data || err2.message || err2);
      return "Agent A encountered an error fetching evidence.";
    }
  }
}
