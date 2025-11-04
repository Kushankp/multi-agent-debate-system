import React, { useState } from "react";
import axios from "axios";

export default function JudgeAgent({ agentA, agentB, query }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleJudge = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post("http://127.0.0.1:8000/judge", {
        agent_a_text: agentA || "",
        agent_b_text: agentB || "",
        query: query || "",
      });
      setResult(res.data);
    } catch (err) {
      console.error("Judge error", err);
      setResult({ winner: "error", reason: "Could not reach judge endpoint", score_a: 0, score_b: 0 });
    } finally {
      setLoading(false);
    }
  };

  const pretty = (w) => (w === "A" ? "Agent A" : w === "B" ? "Agent B" : w === "tie" ? "Tie" : "—");

  return (
    <div className="max-w-6xl mx-auto px-0">
      <div className="panel rounded-lg p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="text-sm text-gray-500">Judge</div>
          <div className="mt-1 text-sm text-gray-700">Decide which agent made the stronger case.</div>
        </div>

        <div className="flex items-center gap-3">
          <button onClick={handleJudge} disabled={loading} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg shadow">
            {loading ? "Judging..." : "Judge"}
          </button>
        </div>
      </div>

      {result && (
        <div className="panel mt-3 p-4">
          <div className="text-sm text-gray-500">Result</div>
          <div className="mt-2 flex items-center gap-4">
            <div className="text-xl font-semibold">{pretty(result.winner)}</div>
            <div className="text-sm text-gray-600">{result.reason}</div>
          </div>
          <div className="mt-2 text-xs text-gray-400">Scores — A: {Number(result.score_a).toFixed(2)} | B: {Number(result.score_b).toFixed(2)}</div>
        </div>
      )}
    </div>
  );
}
