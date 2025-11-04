import React, { useEffect, useState } from "react";

export default function Header() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("prefers-dark");
    if (saved !== null) setDark(saved === "1");
    else setDark(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);

  useEffect(() => {
    if (dark) {
      document.body.classList.add("dark");
      localStorage.setItem("prefers-dark", "1");
    } else {
      document.body.classList.remove("dark");
      localStorage.setItem("prefers-dark", "0");
    }
  }, [dark]);

  return (
    <header className="max-w-6xl mx-auto mb-8 px-4">
      <div className="hero flex items-center justify-between gap-4">
        <div>
          <h1 className="title">Debate — Agent A <span className="opacity-80">vs</span> Agent B</h1>
          <p className="subtitle">
            Type a topic and watch Agent A and Agent B exchange messages in a single conversation box.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm text-white/90 hidden md:inline">Theme</span>
          <button
            aria-label="Toggle theme"
            onClick={() => setDark((d) => !d)}
            className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-3 py-2 rounded-lg"
          >
            {dark ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m8.66-9h-1M4.34 12h-1M18.36 18.36l-.7-.7M6.34 6.34l-.7-.7M18.36 5.64l-.7.7M6.34 17.66l-.7.7M12 5a7 7 0 100 14 7 7 0 000-14z" />
              </svg>
            )}
            <span className="text-sm text-white/90">{dark ? "Dark" : "Light"}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
