import React, { useState } from "react";

export default function QueryForm({ onSubmit }) {
  const [q, setQ] = useState("");

  const handle = (e) => {
    e.preventDefault();
    const t = q.trim();
    if (!t) return;
    onSubmit(t);
    // keep the query visible for editing
  };

  const clear = () => setQ("");

  return (
    <form onSubmit={handle} className="max-w-6xl mx-auto mb-6 px-4">
      <div className="input-card rounded-2xl shadow flex items-center gap-3 px-4 py-3">
        <svg className="search-icon w-6 h-6 text-sky-500" viewBox="0 0 24 24" fill="none">
          <path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>

        <input
          type="text"
          className="flex-grow h-10 px-3 rounded-lg focus:outline-none placeholder:opacity-80 bg-transparent text-gray-800 dark:text-gray-100"
          placeholder="Enter your topic or question for the debate..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />

        {q && (
          <button type="button" onClick={clear} className="text-sm text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" title="Clear">
            Clear
          </button>
        )}

        <button type="submit" className="ml-2 ask-btn px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg font-medium shadow">
          Ask
        </button>
      </div>
    </form>
  );
}
