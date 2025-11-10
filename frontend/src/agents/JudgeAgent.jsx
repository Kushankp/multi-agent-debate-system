import React, { useState } from "react";
import axios from "axios";

export default function JudgeAgent({ agentA, agentB, query }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const BASE = "http://127.0.0.1:8000"; // adjust if your API is hosted elsewhere

  const normalizeResponse = (data) => {
    // Extract search results if backend returned them
    const searchQuery = data?.search_query ?? data?.parsed?.search_query ?? null;
    const searchResults = data?.search_results ?? data?.parsed?.search_results ?? [];

    // Case 1: our ephemeral backend returns { parsed: {...}, valid: true, judge_record: {...} }
    if (data && data.parsed) {
      const parsed = data.parsed;
      return {
        winner: parsed.winner ?? null,
        scoreA: parsed.score?.A ?? parsed.score_a ?? 0,
        scoreB: parsed.score?.B ?? parsed.score_b ?? 0,
        explanation: parsed.explanation ?? parsed.reason ?? "",
        evidence: Array.isArray(parsed.evidence_summary) ? parsed.evidence_summary : [],
        searchQuery,
        searchResults,
        raw: data,
      };
    }

    // Case 2: flattened shape returned by the FastAPI wrapper:
    // { parsed, valid, winner, score_a, score_b, explanation, evidence_summary, search_query, search_results, ... }
    if (data && (data.winner || data.score_a !== undefined || data.score !== undefined)) {
      return {
        winner: data.winner ?? null,
        scoreA: data.score?.A ?? data.score_a ?? 0,
        scoreB: data.score?.B ?? data.score_b ?? 0,
        explanation: data.explanation ?? data.reason ?? data.parsed?.explanation ?? "",
        evidence: data.evidence_summary ?? data.parsed?.evidence_summary ?? [],
        searchQuery,
        searchResults,
        raw: data,
      };
    }

    // Unknown shape: return something safe
    return {
      winner: null,
      scoreA: 0,
      scoreB: 0,
      explanation: "",
      evidence: [],
      searchQuery,
      searchResults,
      raw: data,
    };
  };

  const createSession = async (topic = "") => {
    const res = await axios.post(`${BASE}/session`, { topic, ttl_seconds: 300 });
    return res.data.session_id;
  };

  const submitArg = async (sessionId, agent, text) => {
    const res = await axios.post(`${BASE}/session/${sessionId}/argument`, { agent, text });
    return res.data.argument_id;
  };

  const runJudgeOnSession = async (sessionId) => {
    const res = await axios.post(`${BASE}/session/${sessionId}/judge`);
    return res.data;
  };

  const handleJudge = async () => {
    setLoading(true);
    setResult(null);

    // Basic validation
    if (!agentA || !agentB) {
      setResult({
        winner: "error",
        scoreA: 0,
        scoreB: 0,
        explanation: "Both agentA and agentB must be provided.",
        evidence: [],
        raw: null,
      });
      setLoading(false);
      return;
    }

    try {
      // 1) Create session (store the user query/topic)
      const sid = await createSession(query || "");
      setSessionId(sid);

      // 2) Submit both arguments (A then B)
      await submitArg(sid, "A", agentA);
      await submitArg(sid, "B", agentB);

      // 3) Trigger judge run on session
      const payload = await runJudgeOnSession(sid);

      // Normalize and show
      const normalized = normalizeResponse(payload);
      setResult(normalized);
    } catch (err) {
      console.error("Judge flow error", err);
      // Try to show any server error body
      const serverData = err?.response?.data ?? err?.message ?? String(err);
      setResult({
        winner: "error",
        scoreA: 0,
        scoreB: 0,
        explanation: "Failed to run judge: " + (serverData?.detail ?? serverData),
        evidence: [],
        searchQuery: null,
        searchResults: [],
        raw: serverData,
      });
    } finally {
      setLoading(false);
    }
  };

  const pretty = (w) =>
    w === "A" ? "Agent A" : w === "B" ? "Agent B" : w === "tie" ? "Tie" : w === "error" ? "Error" : "—";

  return (
    <div className="max-w-6xl mx-auto px-0">
<div className="flex flex-col items-center justify-center text-center py-6">
  <div className="text-sm text-gray-500 mb-1">Judge</div>
  <div className="text-sm text-gray-700 mb-3">
    Decide which agent made the stronger case.
  </div>

  <button
    onClick={handleJudge}
    disabled={loading}
    className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg shadow disabled:opacity-60"
  >
    {loading ? "Judging..." : "Judge"}
  </button>
</div>



      {sessionId && (
        <div className="mt-2 text-xs text-gray-500">Session: <span className="font-mono">{sessionId}</span></div>
      )}

      {result && (
        <div className="panel mt-3 p-4 rounded-lg border space-y-4">
          {/* Search query & Tavily results */}
          {result.searchQuery && (
            <div>
              <div className="text-sm text-gray-500">Search query</div>
              <div className="text-sm text-gray-700 mt-1 break-words">{result.searchQuery}</div>
            </div>
          )}


          {/* Judge result */}
          <div>
            <div className="text-sm text-gray-500">Result</div>

            <div className="mt-2 flex items-center gap-4">
              <div className="text-xl font-semibold">{pretty(result.winner)}</div>
              <div className="text-sm text-gray-600">{result.explanation}</div>
            </div>

            <div className="mt-2 text-xs text-gray-400">
              Scores — A: {Number(result.scoreA).toFixed(2)} | B: {Number(result.scoreB).toFixed(2)}
            </div>
          </div>

          {/* Model evidence (from judge JSON) */}
          {result.evidence && result.evidence.length > 0 && (
            <div>
              <div className="text-sm text-gray-500 mb-1">Top evidence cited by judge</div>
              <ul className="space-y-2">
                {result.evidence.map((e, idx) => {
                  const side = e.for ?? e.for_ ?? e.for ?? "A";
                  return (
                    <li key={idx} className="p-2 rounded-md bg-gray-50 border">
                      <div className="text-sm font-medium">
                        {side === "A" ? "For A" : side === "B" ? "For B" : side}
                      </div>
                      <div className="text-sm text-gray-700">{e.source_title ?? e.title ?? e.url}</div>
                      {e.url && (
                        <a className="text-xs text-indigo-600 break-all" href={e.url} target="_blank" rel="noreferrer">
                          {e.url}
                        </a>
                      )}
                      <div className="text-xs text-gray-500 mt-1">{e.reason ?? e.snippet ?? ""}</div>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {/* debug: raw payload toggle (hidden by default, uncomment to show) */}
          {/* <pre className="mt-3 text-xs text-gray-400">{JSON.stringify(result.raw, null, 2)}</pre> */}
        </div>
      )}
    </div>
  );
}
