// agenta.jsx
import axios from "axios";
import { generateAgentBReply } from "./AgentB";

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
 * Primary: POST /generate  -> returns data.answer
 * Fallback: POST /search  -> builds a snippet list
 *
 * Returns cleaned string (Agent A text).
 */
export async function fetchAgentA(query, opts = {}) {
  const {
    model,
    max_tokens,
    context_chars,
    top_k = 3,
    fetch_k = 20,
    timeout = 30000,
  } = opts;

  if (!query || !String(query).trim()) {
    throw new Error("fetchAgentA: query required");
  }

  const body = { query, top_k, fetch_k };
  if (model) body.model = model;
  if (max_tokens) body.max_tokens = max_tokens;
  if (context_chars) body.context_chars = context_chars;

  try {
    const res = await axios.post(`${BACKEND_URL}/generate`, body, { timeout });
    const data = res.data || {};

    // Optional debug
    console.debug("AgentA /generate →", JSON.stringify(data).slice(0, 2000));

    let answer = data.answer ? String(data.answer) : "";
    answer = cleanAssistantText(answer);
    answer = normalizeAnswerSpacing(answer);

    if (answer) return answer;

    // Fallback to search if generation returned empty
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
    // Try search fallback on network/500 errors
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

/**
 * Client-orchestrated 3-round debate fallback.
 * Uses fetchAgentA and generateAgentBReply in sequence:
 *   A1 = fetchAgentA(initialQuery)
 *   B1 = generateAgentBReply(A1)
 *   A2 = fetchAgentA(B1)
 *   B2 = generateAgentBReply(A2)
 *   A3 = fetchAgentA(B2)
 *   B3 = generateAgentBReply(A3)
 *
 * Returns { rounds: [{agent, text, raw}, ...], final: {...} }
 */
export async function runThreeRoundDebateClient(initialQuery, opts = {}) {
  if (!initialQuery || !String(initialQuery).trim()) {
    throw new Error("initialQuery required");
  }

  const rounds = [];

  // A1
  const a1Text = await fetchAgentA(initialQuery, opts);
  rounds.push({ agent: "A", text: cleanAssistantText(a1Text), raw: { source: "client-generate", meta: { seed: initialQuery } } });

  // B1
  const b1Text = await generateAgentBReply(a1Text, initialQuery);
  rounds.push({ agent: "B", text: cleanAssistantText(b1Text), raw: { source: "client-agent-b", meta: {} } });

  // A2
  const a2Text = await fetchAgentA(b1Text, opts);
  rounds.push({ agent: "A", text: cleanAssistantText(a2Text), raw: { source: "client-generate", meta: {} } });

  // B2
  const b2Text = await generateAgentBReply(a2Text, b1Text);
  rounds.push({ agent: "B", text: cleanAssistantText(b2Text), raw: { source: "client-agent-b", meta: {} } });

  // A3
  const a3Text = await fetchAgentA(b2Text, opts);
  rounds.push({ agent: "A", text: cleanAssistantText(a3Text), raw: { source: "client-generate", meta: {} } });

  // B3
  const b3Text = await generateAgentBReply(a3Text, b2Text);
  rounds.push({ agent: "B", text: cleanAssistantText(b3Text), raw: { source: "client-agent-b", meta: {} } });

  const final = { A3: a3Text, B3: b3Text, summary_note: "client-side 3-round debate finished" };
  return { rounds, final };
}

/**
 * Server-orchestrated 3-round debate.
 * Calls backend /debate/3-round and returns server response (expected { rounds, final }).
 */
export async function runThreeRoundDebateServer(initialQuery, opts = {}) {
  if (!initialQuery || !String(initialQuery).trim()) {
    throw new Error("initialQuery required");
  }

  const payload = {
    initial_query: initialQuery,
    top_k: opts.top_k || 5,
    fetch_k: opts.fetch_k || 50,
    namespace: opts.namespace,
    model: opts.model,
    max_tokens: opts.max_tokens,
    context_chars: opts.context_chars,
  };

  const res = await axios.post(`${BACKEND_URL}/debate/3-round`, payload, { timeout: opts.timeout || 120000 });
  return res.data;
}

/**
 * High-level helper: try server orchestration first, fallback to client orchestration.
 * Returns { rounds: [...], final: {...} }.
 */
export async function runThreeRoundDebate(initialQuery, opts = {}) {
  // Prefer server orchestration
  try {
    const serverRes = await runThreeRoundDebateServer(initialQuery, opts);
    if (serverRes && Array.isArray(serverRes.rounds)) {
      // Normalize texts
      serverRes.rounds = serverRes.rounds.map((r) => ({ agent: r.agent, text: cleanAssistantText(r.text || ""), raw: r.raw || {} }));
      return serverRes;
    }
    console.warn("runThreeRoundDebate: server returned unexpected shape, falling back to client loop", serverRes);
  } catch (err) {
    console.warn("runThreeRoundDebate: server orchestration failed, falling back to client loop:", err?.message || err);
  }

  // fallback to client-managed loop
  return await runThreeRoundDebateClient(initialQuery, opts);
}
