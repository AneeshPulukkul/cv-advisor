import { useEffect, useState } from "react";
import { getRules } from "../api.js";

export default function Rules({ apiUp }) {
  const [rules, setRules] = useState(null);

  useEffect(() => {
    getRules().then(setRules).catch(() => {});
  }, []);

  const fallback = {
    precedence: ["Time", "Group", "Spatial", "Imbalance", "Skewed-regression", "Tiny-N", "Default i.i.d."],
    thresholds: { imbalance_minority_ratio: 0.05, skew: 1.0, tiny_n: 50, small_n: 200, large_n: 100000, profile_sample_limit: 500000 },
    splitters: ["TimeSeriesSplit", "PurgedGroupTimeSeriesSplit", "GroupKFold", "StratifiedGroupKFold", "StratifiedKFold", "RepeatedStratifiedKFold", "BinnedStratifiedKFold", "KFold", "LeaveOneOut"],
  };
  const r = rules || fallback;

  return (
    <div className="space-y-4">
      {!apiUp && (
        <p className="rounded-lg bg-yellow-500/10 p-3 text-sm text-yellow-300 ring-1 ring-yellow-500/30">
          API offline — showing baked-in defaults. Start it with <code>cv-advisor serve</code>.
        </p>
      )}
      <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
        <h2 className="font-semibold text-white">Precedence (first match wins)</h2>
        <ol className="mt-3 space-y-2">
          {r.precedence.map((p, i) => (
            <li key={p} className="flex items-center gap-3 text-sm text-zinc-200">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-xs font-bold text-white">{i + 1}</span>
              {p}
            </li>
          ))}
        </ol>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
          <h2 className="font-semibold text-white">Thresholds</h2>
          <dl className="mt-2 space-y-1 font-mono text-sm text-zinc-300">
            {Object.entries(r.thresholds).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2"><dt className="text-zinc-500">{k}</dt><dd>{String(v)}</dd></div>
            ))}
          </dl>
        </div>
        <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
          <h2 className="font-semibold text-white">Supported splitters</h2>
          <div className="mt-2 flex flex-wrap gap-2">
            {r.splitters.map((s) => (
              <span key={s} className="rounded-full bg-zinc-800 px-3 py-1 font-mono text-xs text-zinc-200 ring-1 ring-zinc-700">{s}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
