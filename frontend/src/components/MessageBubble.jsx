import React from "react";

export default function MessageBubble({ text, time, avatarLabel, align }) {
  const isRight = align === "right";
  const bubbleClass = isRight ? "bubble-b" : "bubble-a";

  return (
    <div
      className={`flex items-end gap-2 ${
        isRight ? "justify-end text-right" : "justify-start text-left"
      }`}
    >
      {!isRight && <div className="avatar agent shrink-0">{avatarLabel}</div>}

      <div className={`msg-bubble ${bubbleClass}`}>
        <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
        {time && <div className="msg-time">{new Date(time).toLocaleTimeString()}</div>}
      </div>

      {isRight && <div className="avatar user shrink-0">{avatarLabel}</div>}
    </div>
  );
}
