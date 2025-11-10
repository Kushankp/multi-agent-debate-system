import React, { useState, useEffect } from "react";
import axios from "axios";

export default function QueryForm({ onSubmit }) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => {
    try {
      return localStorage.getItem("sessionId") || null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    // If parent passes a query in some flows, you can prefill here by reading props (not used now)
  }, []);

  const createSession = async (topic) => {
    const BASE = "http://127.0.0.1:8000"; // adjust if needed
    const res = await axios.post(`${BASE}/session`, { topic, ttl_seconds: 300 });
    return res.data.session_id;
  };

  const handle = async (e) => {
    e.preventDefault();
    const t = q.trim();
    if (!t) return;

    setLoading(true);
    try {
      // create new session for this query and store it
      const sid = await createSession(t);
      try {
        localStorage.setItem("sessionId", sid);
      } catch {}
      setSessionId(sid);

      // call parent's onSubmit with (query, sessionId)
      if (typeof onSubmit === "function") onSubmit(t, sid);
    } catch (err) {
      console.error("Failed to create session:", err);
      // still call onSubmit with query only so caller can decide fallback behavior
      if (typeof onSubmit === "function") onSubmit(t, null);
    } finally {
      setLoading(false);
      // keep the query visible for editing (per your original behavior)
    }
  };

  const clear = () => setQ("");

  const clearSession = () => {
    try {
      localStorage.removeItem("sessionId");
    } catch {}
    setSessionId(null);
  };

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
          disabled={loading}
        />

        {q && (
          <button type="button" onClick={clear} className="text-sm text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" title="Clear">
            Clear
          </button>
        )}

        {sessionId && (
          <button
            type="button"
            onClick={clearSession}
            className="text-sm text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 ml-2"
            title="Clear saved session"
          >
            Clear session
          </button>
        )}

        <button
          type="submit"
          className="ml-2 ask-btn px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg font-medium shadow disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Creating…" : "Ask"}
        </button>
      </div>
      {sessionId && (
        <div className="mt-2 text-xs text-gray-500">
          Current session: <span className="font-mono">{sessionId}</span>
        </div>
      )}
    </form>
  );
}
