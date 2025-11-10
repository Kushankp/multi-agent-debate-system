export async function generateAgentBReply(agentAText, query = "") {
  if (!agentAText || agentAText.trim() === "") {
    return "Agent B has no evidence to rebut; please provide more context.";
  }

  try {
    // --- 1️⃣ Build Agent A's argument payload ---
    const aArgs = {
      query: query || agentAText.slice(0, 200),  // short fallback query
      text: agentAText,
      question_id: `q_${Date.now()}`,
    };

    // --- 2️⃣ POST to Agent B backend ---
    // Adjust the URL/port if Agent B runs elsewhere
    const resp = await fetch("http://127.0.0.1:8000/agent-b/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(aArgs),
    });

    if (!resp.ok) {
      const errTxt = await resp.text();
      console.error("Agent B backend error:", errTxt);
      return `Agent B error: ${resp.status} ${resp.statusText}`;
    }

    // --- 3️⃣ Parse Agent B response ---
    const data = await resp.json();

    // expected structure: { b_args: {...}, raw_llm_response: {...} }
    const bArgs = data.b_args || {};
    const prompt = bArgs.prompt || "(no prompt generated)";
    const confidence = bArgs.confidence != null ? ` (confidence ${bArgs.confidence})` : "";

    // --- 4️⃣ Format friendly reply ---
    const summary =
      `Agent B Reply${confidence}:\n\n${prompt}\n\n` +
      (bArgs.notes ? `Notes: ${bArgs.notes}\n\n` : "") +
      (bArgs.followup_questions && bArgs.followup_questions.length
        ? "Follow-up questions:\n• " + bArgs.followup_questions.join("\n• ")
        : "");

    return summary.trim();
  } catch (err) {
    console.error("Agent B request failed:", err);
    return `Agent B failed to generate a reply: ${err.message || err}`;
  }
}
