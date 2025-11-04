export async function generateAgentBReply(agentAText, query = "") {
  // Simple heuristic summarization + counterpoint
  if (!agentAText || agentAText.trim() === "") {
    return "Agent B has no evidence to rebut; please provide more context.";
  }

  // take the first meaningful chunk and produce a counter-argument
  const cleaned = agentAText.replace(/\s+/g, " ").trim();
  const head = cleaned.slice(0, 240);
  const reply = `Agent B summary: ${head}${cleaned.length > 240 ? "…" : ""}\n\nCounterpoint: While those points are valid, the evidence needs context and real-world validation. For example, check sources for recency, sample bias, and applicability to the specific query.`;
  // small artificial delay to feel natural
  await new Promise((r) => setTimeout(r, 500));
  return reply;
}
