import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import Home from "./pages/Home.jsx";
import Inspect from "./pages/Inspect.jsx";
import Rules from "./pages/Rules.jsx";

const TABS = ["Home", "Inspect", "Rules"];

export default function App() {
  const [tab, setTab] = useState("Home");
  const [apiUp, setApiUp] = useState(null);

  useEffect(() => {
    getHealth().then(() => setApiUp(true)).catch(() => setApiUp(false));
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-600 font-mono text-lg font-bold text-white">cv</div>
            <div>
              <p className="font-mono font-bold leading-none">cv-advisor</p>
              <p className="text-xs text-zinc-500">right CV splitter, every dataset</p>
            </div>
          </div>
          <nav className="flex gap-1 rounded-lg bg-zinc-900 p-1 ring-1 ring-zinc-800">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-4 py-1.5 text-sm font-medium ${tab === t ? "bg-emerald-600 text-white" : "text-zinc-400 hover:text-white"}`}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-6 flex items-center gap-2 text-xs">
          <span className={`h-2 w-2 rounded-full ${apiUp ? "bg-emerald-400" : apiUp === false ? "bg-red-400" : "bg-zinc-600"}`} />
          <span className="text-zinc-500">
            {apiUp ? "API connected (localhost:8000)" : apiUp === false ? "API offline — start with `cv-advisor serve`" : "Checking API…"}
          </span>
        </div>
        {tab === "Home" && <Home />}
        {tab === "Inspect" && <Inspect />}
        {tab === "Rules" && <Rules apiUp={apiUp} />}
      </main>

      <footer className="border-t border-zinc-800 py-4 text-center text-xs text-zinc-600">
        cv-advisor · MIT · deterministic CV recommendations for ML engineers
      </footer>
    </div>
  );
}
