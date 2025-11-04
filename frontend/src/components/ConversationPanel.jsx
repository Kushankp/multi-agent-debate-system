// src/components/ConversationPanel.jsx
import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

export default function ConversationPanel({ messages = [], loading = false }) {
  const panelRef = useRef(null);

  useEffect(() => {
    if (panelRef.current) panelRef.current.scrollTop = panelRef.current.scrollHeight;
  }, [messages, loading]);

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm text-gray-500">
          <span className="font-medium">Debate</span>
          <span className="ml-2 text-xs text-gray-400">Agents exchanging arguments</span>
        </div>
        <div className="text-xs text-gray-400">Live</div>
      </div>

      <div className="panel rounded-2xl p-4 min-h-[380px] flex flex-col">
        <div
          ref={panelRef}
          className="panel-body flex-1 overflow-auto scrollbar-thin space-y-6 pb-4"
        >
          {messages.length === 0 && (
            <div className="text-gray-500">No exchange yet. Ask a question to start the debate.</div>
          )}

          {messages.map((m, i) => {
            // speaker "A" => left, "B" => right
            const align = m.speaker === "A" ? "left" : "right";
            const rowClass = `msg-row ${align === "right" ? "msg-right" : "msg-left"}`;

            return (
              <div key={i} className={rowClass}>
                <MessageBubble
                  text={m.text}
                  time={m.time}
                  avatarLabel={m.speaker}
                  align={align}
                />
              </div>
            );
          })}

          {loading && (
            <div className="text-sm text-gray-500">Agents are preparing their responses…</div>
          )}
        </div>
      </div>
    </section>
  );
}
