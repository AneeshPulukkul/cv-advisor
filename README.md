# cv-advisor

Inspect any tabular dataset (CSV / Parquet / DataFrame) and deterministically recommend,
justify, and instantiate the optimal **scikit-learn** cross-validation strategy.

Built as an open-source reference for ML engineers: leakage-aware, explainable, tested.

- 📖 **[USAGE.md](USAGE.md)** — how to use the tool (CLI, API, web UI, decision tables, troubleshooting)
- 🛠 **[CONTRIBUTING.md](CONTRIBUTING.md)** — engineer setup (env, tests, lint, security, publish checklist)
- ▶️ **[examples/](examples/)** — runnable end-to-end scripts with asserted guarantees

## Pipeline

```
[Input Data] → [Data Profiler] → [Decision Engine] → [Recommendation Report]
                                                    ↘ [sklearn Splitter]
```

Precedence: **Time > Group > Spatial > Imbalance > Skewed-regression > Tiny-N > Default i.i.d.**

## Install

```bash
pip install -e ".[dev]"
```

## CLI

```bash
cv-advisor inspect path/to/data.csv --target price
cv-advisor inspect data.parquet --target fraud --format json
cv-advisor inspect data.csv --target fraud --repeated   # RepeatedStratifiedKFold for imbalance/small-N
cv-advisor serve --port 8000                            # backend for the web UI
```

## Web UI (v0.2)

React + Vite + Tailwind. Home page shows how to use the tool; Inspect tab uploads a CSV
and renders the recommendation; Rules tab shows the live decision tree from `/api/rules`.

```bash
cv-advisor serve --port 8000   # terminal 1: API
cd web && npm install && npm run dev   # terminal 2: UI at http://localhost:5173
```

## Python API

```python
import pandas as pd
from cv_advisor import CVAdvisor

df = pd.read_csv("data.csv")
advisor = CVAdvisor(target_col="price")
rec = advisor.advise(df, "price")
print(rec.recommended_splitter, rec.parameters)
print(rec.rationale)
print(rec.code_snippet)  # <-- run this: it wires groups=/time-sorting for you
```

`rec.code_snippet` is the runnable pattern — plain `cross_val_score(model, X, y,
cv=splitter)` only fits i.i.d. splitters. Grouped splits need
`cv=splitter.split(X, y, groups=df[rec.group_col])`; temporal splits must sort
by `rec.time_col` first and (for purged splits) receive the time-bearing frame.
See [USAGE.md](USAGE.md) for the exact grouped/temporal forms.

## Decisions (differentiators)

- `PurgedGroupTimeSeriesSplit` (`splits.py`): sklearn has no purged/embargoed group-time split — custom `BaseCrossValidator`.
- `BinnedStratifiedKFold` (`splits.py`): quantile-bin regression target → `StratifiedKFold` on bins.
- `RepeatedStratifiedKFold` (v0.2): opt-in via `advise(..., prefer_repeated=True)` / `--repeated` — never default (cost).
- Spatial: KMeans blocks on `(lat, lon)` → `GroupKFold` on blocks. Target columns are excluded from coord detection.
- Defensive profiling: samples to 500k rows, NaN-tolerant, no `x/y` false positives on target-named columns.

## Security

- `pip-audit` on every CI run (report-only) + `npm audit --audit-level=high` for `web/`.
- GitHub Dependabot (pip + npm + actions, weekly) via `.github/dependabot.yml`.
- No known vulnerabilities in project deps at v0.2.0 (`pip audit`: clean, `npm audit`: 0).
- Note: neither tool is a substitute for GitHub's own Dependabot alerts / code scanning —
  enable both in repo settings after push (public repos get them free).

## Tests

```bash
pytest -q
```

Covers: fraud (imbalanced), patients (grouped), housing (spatial), stocks (temporal).
