import React, { useEffect, useState, useRef } from "react";
import ConversationPanel from "./ConversationPanel";
import JudgeAgent from "../agents/JudgeAgent";   
import * as agentA from "../agents/AgentA";
import * as agentB from "../agents/AgentB";

export default function ConversationLayout({ query }) {
  const [messages, setMessages] = useState([]); // {speaker: 'A'|'B', text, time}
  const [loading, setLoading] = useState(false);
  const firstRun = useRef(true);

  // whenever query changes, start a new debate exchange
    useEffect(() => {
    if (!query || query.trim() === "") return;

    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setMessages([]); // start fresh for each query

      try {
        // ---- A1 ----
        const a1Reply = await agentA.fetchAgentA(query);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "A", text: a1Reply, time: new Date().toISOString() },
        ]);

        // small pause to feel natural
        await new Promise((r) => setTimeout(r, 600));
        if (cancelled) return;

        // ---- B1 ----
        const b1Reply = await agentB.generateAgentBReply(a1Reply, query);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "B", text: b1Reply, time: new Date().toISOString() },
        ]);

        await new Promise((r) => setTimeout(r, 600));
        if (cancelled) return;

        // ---- A2 ----
        const a2Reply = await agentA.fetchAgentA(b1Reply);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "A", text: a2Reply, time: new Date().toISOString() },
        ]);

        await new Promise((r) => setTimeout(r, 600));
        if (cancelled) return;

        // ---- B2 ----
        const b2Reply = await agentB.generateAgentBReply(a2Reply, b1Reply);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "B", text: b2Reply, time: new Date().toISOString() },
        ]);

        await new Promise((r) => setTimeout(r, 600));
        if (cancelled) return;

        // ---- A3 ----
        const a3Reply = await agentA.fetchAgentA(b2Reply);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "A", text: a3Reply, time: new Date().toISOString() },
        ]);

        await new Promise((r) => setTimeout(r, 600));
        if (cancelled) return;

        // ---- B3 ----
        const b3Reply = await agentB.generateAgentBReply(a3Reply, b2Reply);
        if (cancelled) return;
        setMessages((m) => [
          ...m,
          { speaker: "B", text: b3Reply, time: new Date().toISOString() },
        ]);
      } catch (err) {
        console.error("Debate exchange error", err);
        if (!cancelled) {
          setMessages((m) => [
            ...m,
            { speaker: "A", text: "Error during exchange.", time: new Date().toISOString() },
          ]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => { cancelled = true; };
  }, [query]);

  // Optionally, allow user to make a new round (Agent A reads B and responds) — omitted here for simplicity.
  // pass messages to conversation panel and judge
  return (
    <main className="max-w-6xl mx-auto px-4">
      <ConversationPanel messages={messages} loading={loading} />
      <div className="mt-6">
        <JudgeAgent
          agentA={messages.filter(m => m.speaker === "A").map(m => m.text).join("\n\n")}
          agentB={messages.filter(m => m.speaker === "B").map(m => m.text).join("\n\n")}
          query={query}
        />
      </div>
    </main>
  );
}
