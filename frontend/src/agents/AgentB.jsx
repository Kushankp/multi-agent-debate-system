// agentb.jsx
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
 * Generate Agent B reply to Agent A's text.
 *
 * Signature kept to match your existing code:
 *   generateAgentBReply(agentAText, query = "")
 *
 * Returns a cleaned string containing the prompt + optional notes/followups.
 */
export async function generateAgentBReply(agentAText, query = "") {
  if (!agentAText || agentAText.trim() === "") {
    return "Agent B has no evidence to rebut; please provide more context.";
  }

  try {
    // Build payload. Main.py expects { query: ... } but including text/question_id is harmless.
    const body = {
      query: query || agentAText.slice(0, 200),
      text: agentAText,
      question_id: `q_${Date.now()}`,
      // you can add top_k/fetch_k/model here if desired
    };

    const resp = await axios.post(`${BACKEND_URL}/agent-b/generate`, body, { timeout: 30000 });
    const data = resp.data || {};

    // Robustly find parsed b_args in a few possible places (main.py returns top-level b_args)
    const bArgs =
      data.b_args ||
      data.parsed_b_args ||
      (data.raw && data.raw.parsed_b_args) ||
      (data.raw && data.raw.parsed && data.raw.parsed.b_args) ||
      {};

    // Extract prompt (several fallbacks)
    let promptRaw = "";
    if (bArgs && bArgs.prompt) promptRaw = bArgs.prompt;
    else if (data.answer) promptRaw = data.answer;
    else if (data.assistant_text) promptRaw = data.assistant_text;
    else if (data.raw && data.raw.assistant_text) promptRaw = data.raw.assistant_text;
    else promptRaw = "(no prompt generated)";

    const prompt = cleanAssistantText(String(promptRaw));

    // notes & followups
    const notes = bArgs && bArgs.notes ? String(bArgs.notes) : (data.notes || "");
    const followups = bArgs && Array.isArray(bArgs.followup_questions) ? bArgs.followup_questions : (data.followup_questions || []);

    // Format friendly reply similar to your original
    let summary = `\n\n${prompt}\n\n`;
    if (notes && String(notes).trim()) summary += `Notes: ${String(notes).trim()}\n\n`;
    if (followups && Array.isArray(followups) && followups.length > 0) {
      summary += "Follow-up questions:\n• " + followups.map((q) => safeTrimText(q, 800)).join("\n• ");
    }

    const out = summary.trim();
    return normalizeAnswerSpacing(out);
  } catch (err) {
    console.error("Agent B request failed:", err?.response?.data || err.message || err);
    const message = err?.response?.data || err?.message || String(err);
    return `Agent B failed to generate a reply: ${message}`;
  }
}
