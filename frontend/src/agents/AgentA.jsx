import axios from "axios";

const BACKEND_URL = "http://127.0.0.1:8000";

function safeTrimText(t, max = 1000) {
  if (!t) return "";
  let s = String(t).replace(/\s+/g, " ").trim();
  if (s.length > max) s = s.slice(0, max) + " …";
  return s;
}

export async function fetchAgentA(query) {
  try {
    const res = await axios.post(`${BACKEND_URL}/search`, { query, top_k: 3, fetch_k: 20 });
    const results = res.data?.results || [];
    if (results.length === 0) return "I couldn't find relevant documents for that query.";

    const parts = results.map((r, i) => {
      const title = r.page || `Source ${i + 1}`;
      const snippet = safeTrimText(r.snippet || r.full_text || "", 480);
      return `${i + 1}. ${title}\n\n${snippet}`;
    });
    return parts.join("\n\n---\n\n");
  } catch (err) {
    console.error("agentA error", err);
    return "Agent A encountered an error fetching evidence.";
  }
}
