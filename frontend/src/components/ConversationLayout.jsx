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
        // Agent A produces first reply based on query
        const aReply = await agentA.fetchAgentA(query);
        if (cancelled) return;
        const aMsg = { speaker: "A", text: aReply, time: new Date().toISOString() };
        setMessages((m) => [...m, aMsg]);

        // small pause to feel natural
        await new Promise((r) => setTimeout(r, 700));

        // Agent B reads A reply and makes a counter reply
        const bReply = await agentB.generateAgentBReply(aReply, query);
        if (cancelled) return;
        const bMsg = { speaker: "B", text: bReply, time: new Date().toISOString() };
        setMessages((m) => [...m, bMsg]);
      } catch (err) {
        console.error("Debate exchange error", err);
        setMessages((m) => [...m, { speaker: "A", text: "Error during exchange.", time: new Date().toISOString() }]);
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
