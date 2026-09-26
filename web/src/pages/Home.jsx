function Code({ children }) {
  return (
    <pre className="overflow-x-auto rounded-lg bg-zinc-950 p-4 text-sm leading-relaxed text-zinc-100 ring-1 ring-zinc-800">
      <code>{children}</code>
    </pre>
  );
}

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="rounded-2xl bg-gradient-to-br from-emerald-950 via-zinc-900 to-zinc-900 p-8 ring-1 ring-zinc-800">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Stop guessing your <span className="text-emerald-400">cross-validation</span> strategy.
        </h1>
        <p className="mt-3 max-w-2xl text-zinc-300">
          <span className="font-mono text-emerald-300">cv-advisor</span> inspects any tabular dataset,
          detects time / group / spatial / imbalance structure, and recommends the optimal
          scikit-learn splitter — with the leakage rationale and copy-pasteable code.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-xs">
          {["TimeSeriesSplit", "StratifiedGroupKFold", "PurgedGroupTimeSeriesSplit", "BinnedStratifiedKFold"].map((s) => (
            <span key={s} className="rounded-full bg-zinc-800 px-3 py-1 font-mono text-zinc-200 ring-1 ring-zinc-700">
              {s}
            </span>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
          <h2 className="font-semibold text-white">1 · CLI</h2>
          <p className="mt-1 text-sm text-zinc-400">Terminal dashboard with rich tables.</p>
          <div className="mt-3"><Code>{`pip install cv-advisor\ncv-advisor inspect data.csv --target price`}</Code></div>
        </div>
        <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
          <h2 className="font-semibold text-white">2 · Python API</h2>
          <p className="mt-1 text-sm text-zinc-400">Drop into training scripts and notebooks.</p>
          <div className="mt-3"><Code>{`from cv_advisor import CVAdvisor\n\nadvisor = CVAdvisor("price")\nrec = advisor.advise(df, "price")\nsplitter = advisor.get_splitter(df, rec)`}</Code></div>
        </div>
        <div className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
          <h2 className="font-semibold text-white">3 · This web UI</h2>
          <p className="mt-1 text-sm text-zinc-400">Upload a CSV, get the verdict visually.</p>
          <div className="mt-3"><Code>{`cv-advisor serve --port 8000\nnpm run dev  # web/`}</Code></div>
        </div>
      </section>

      <section className="rounded-xl bg-zinc-900 p-5 ring-1 ring-zinc-800">
        <h2 className="font-semibold text-white">How it decides</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Deterministic rule tree — <span className="font-mono">Time &gt; Group &gt; Spatial &gt; Imbalance &gt; Skewed-reg &gt; Tiny-N &gt; Default</span>.
          See the <span className="font-semibold text-zinc-200">Rules</span> tab for thresholds, or open the{" "}
          <span className="font-semibold text-zinc-200">Inspect</span> tab to try your own CSV.
        </p>
      </section>
    </div>
  );
}
