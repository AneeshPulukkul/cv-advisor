import { useState } from "react";
import { postAdvise } from "../api.js";

function Badge({ children, tone }) {
  const tones = {
    green: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30",
    yellow: "bg-yellow-500/10 text-yellow-300 ring-yellow-500/30",
    zinc: "bg-zinc-800 text-zinc-200 ring-zinc-700",
  };
  return (
    <span className={`inline-block rounded-full px-3 py-1 font-mono text-sm ring-1 ${tones[tone] || tones.zinc}`}>
      {children}
    </span>
  );
}

export default function Inspect() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [repeated, setRepeated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    if (!file) return setError("Choose a .csv or .parquet file first.");
    if (!target.trim()) return setError("Enter the target column name.");
    setLoading(true);
    try {
      setResult(await postAdvise(file, target.trim(), repeated));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const rec = result?.recommendation;
  const prof = result?.profile;

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
        <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto]">
          <label className="block">
            <span className="text-sm font-medium text-zinc-300">Dataset (.csv / .parquet, ≤100 MB)</span>
            <input
              type="file"
              accept=".csv,.parquet,.pq"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="mt-1 block w-full text-sm text-zinc-300 file:mr-3 file:rounded-lg file:border-0 file:bg-emerald-600 file:px-4 file:py-2 file:text-white hover:file:bg-emerald-500"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-zinc-300">Target column</span>
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="e.g. price"
              className="mt-1 block w-full rounded-lg bg-zinc-950 px-3 py-2 text-zinc-100 ring-1 ring-zinc-700 placeholder:text-zinc-600 focus:outline-none focus:ring-emerald-500"
            />
          </label>
          <label className="flex items-end gap-2 pb-2 text-sm text-zinc-300">
            <input type="checkbox" checked={repeated} onChange={(e) => setRepeated(e.target.checked)} className="h-4 w-4 accent-emerald-500" />
            Prefer repeated
          </label>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="mt-4 rounded-lg bg-emerald-600 px-5 py-2 font-semibold text-white disabled:opacity-50 hover:bg-emerald-500"
        >
          {loading ? "Profiling…" : "Recommend splitter"}
        </button>
        {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
      </form>

      {rec && prof && (
        <div className="space-y-4">
          <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
            <div className="flex flex-wrap items-center gap-3">
              <Badge tone="green">{rec.recommended_splitter}</Badge>
              <code className="text-sm text-zinc-400">{JSON.stringify(rec.parameters)}</code>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-zinc-300">{rec.rationale}</p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
              <h3 className="font-semibold text-white">Detected patterns</h3>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-300">
                {(rec.detected_patterns || []).map((p) => <li key={p}>{p}</li>)}
                {rec.detected_patterns?.length === 0 && <li className="text-zinc-500">Plain i.i.d. table — no leakage structure.</li>}
              </ul>
            </div>
            <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-yellow-500/20">
              <h3 className="font-semibold text-white">Leakage risks</h3>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-300">
                {(rec.leakage_risks || []).map((r) => <li key={r}>{r}</li>)}
                {rec.leakage_risks?.length === 0 && <li className="text-zinc-500">None beyond the usual.</li>}
              </ul>
            </div>
          </div>

          <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
            <h3 className="font-semibold text-white">Profile</h3>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
              {[["Rows", prof.n_rows.toLocaleString()], ["Cols", prof.n_cols], ["Task", prof.task_type],
                ["Memory MB", prof.memory_mb.toFixed(2)], ["Time col", prof.time_col || "—"],
                ["Group col", prof.group_col || "—"], ["Spatial", prof.has_spatial ? `${prof.lat_col},${prof.lon_col}` : "—"],
                ["Minority ratio", prof.minority_ratio ?? "—"]].map(([k, v]) => (
                <div key={k} className="rounded-lg bg-zinc-950 p-2 ring-1 ring-zinc-800">
                  <dt className="text-xs text-zinc-500">{k}</dt>
                  <dd className="font-mono text-zinc-100">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
            <h3 className="font-semibold text-white">Run it</h3>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-950 p-4 text-sm text-zinc-100 ring-1 ring-zinc-800">
              <code>{rec.code_snippet}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
