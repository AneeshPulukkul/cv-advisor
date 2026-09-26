# cv-advisor — User Guide

How to use `cv-advisor` to pick the right cross-validation strategy for any tabular dataset.

## Install

```bash
pip install -e .          # from a clone (PyPI release: pip install cv-advisor)
```

Requires Python ≥ 3.10.

## 1 · CLI (fastest)

```bash
cv-advisor inspect data.csv --target price
cv-advisor inspect data.parquet --target fraud --format json
cv-advisor inspect data.csv --target fraud --repeated   # stabilize small/imbalanced estimates
```

You get: data profile table, primary recommendation with parameters, detected
patterns, leakage risks, the rationale, and copy-pasteable code.

## 2 · Python API

```python
import pandas as pd
from cv_advisor import CVAdvisor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data.csv")
advisor = CVAdvisor(target_col="price")

rec = advisor.advise(df, "price")
print(rec.recommended_splitter, rec.parameters)  # e.g. StratifiedKFold {'n_splits': 5, ...}
print(rec.rationale)
print(rec.code_snippet)                          # paste into your training script

# i.i.d. recommendations only: X/y defined, no groups involved
X = df.drop(columns=["price"]).select_dtypes("number").to_numpy()
y = df["price"].to_numpy()
splitter = advisor.get_splitter(df, rec)
scores = cross_val_score(model, X, y, cv=splitter.split(X, y))
```

`rec.code_snippet` is the authoritative runnable pattern — use it whenever the
recommendation involves groups or time. The two special forms:

```python
# Grouped (patients, stores, devices): entity labels keep folds disjoint
scores = cross_val_score(model, X, y,
    cv=splitter.split(X, y, groups=df[rec.group_col]))

# Temporal: sort FIRST (splitters follow row order, they don't sort)
dfs = df.sort_values(rec.time_col).reset_index(drop=True)
X = dfs.drop(columns=["price"]).select_dtypes("number").to_numpy()
y = dfs["price"].to_numpy()

# ... time-only (TimeSeriesSplit): no groups involved
scores = cross_val_score(model, X, y, cv=splitter.split(X))

# ... time-plus-group (purged): pass the frame so timestamps resolve,
# plus entity labels so folds stay disjoint
cv = splitter.split(dfs, y, groups=dfs[rec.group_col])
scores = cross_val_score(model, X, y, cv=cv)
```

Useful extras:

```python
advisor.profile(df, "price")                     # facts only, no recommendation
advisor.advise(df, "price", prefer_repeated=True)  # RepeatedStratifiedKFold where valid
```

## 3 · Web UI

```bash
cv-advisor serve --port 8000        # terminal 1: API
cd web && npm install && npm run dev  # terminal 2: UI at http://localhost:5173
```

- **Home** — what the tool does and the three ways to use it.
- **Inspect** — upload a CSV/Parquet (≤ 100 MB), enter the target, get the verdict visually.
- **Rules** — live decision tree, thresholds, and supported splitters from `/api/rules`.

## How decisions are made

First match wins: **Time › Group › Spatial › Imbalance › Skewed-regression › Tiny-N › Default i.i.d.**

| Signal | Threshold | Recommendation |
|---|---|---|
| Time + entity, entities interleaved (checked first) | group-time overlap > 0.7 | `TimeSeriesSplit` + caveat (a purged split would yield empty folds) |
| Time column + entity column | ≥ 4 entities | `PurgedGroupTimeSeriesSplit` (purge + embargo) |
| Time column only | — | `TimeSeriesSplit` (sort by time first — splitters follow row order) |
| Repeated entity | — | `GroupKFold`, or `StratifiedGroupKFold` if imbalanced |
| Coordinates (lat/lon) | — | `GroupKFold` over KMeans spatial blocks |
| Class imbalance | minority/majority < 0.05 | `StratifiedKFold` (5 or 10 folds) |
| Skewed regression target | \|skew\| > 1.0 | `BinnedStratifiedKFold` (quantile bins → stratified) |
| Tiny sample | N < 50 | capped stratified folds, or `LeaveOneOut` (regression) |
| Default classification / regression | — | `StratifiedKFold` / `KFold(shuffle=True)` |

Fold counts are feasibility-capped everywhere: never more folds than rows,
minority examples, groups, per-class groups, or spatial blocks. A single-example
minority falls back to plain `KFold` (stratification is mathematically impossible).

## Reading the output

- **Detected patterns** — the structural facts that drove the choice.
- **Leakage risks** — what goes wrong if you use naive random splits instead.
  Take these seriously: identity leakage (same entity both sides), lookahead
  leakage (future in train), spatial leakage (neighbors both sides).
- **Code snippet** — runnable boilerplate. For time data it sorts by the time
  column first; for purged splits it passes the frame (carries timestamps) plus
  `groups=`; for spatial it builds blocks first.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PurgedGroupTimeSeriesSplit produced no usable folds` | Entities overlap in time too fully — follow the message: use `TimeSeriesSplit` (or `GroupKFold` if entities are time-contiguous but few) |
| `time_col=... has N missing timestamps` | Drop or impute those rows — the splitter refuses to silently degrade |
| `n_splits=... infeasible with N groups` | Collect more entities, or accept the `GroupKFold` fallback |
| `X's index does not match` (spatial) | Keep original rows/order, or pass `groups=` explicitly from `make_spatial_blocks` on your filtered frame |
| `Single-example minority` note | Collect more minority samples; current folds are unstratified by necessity |
| API `413` on upload | File exceeds 100 MB — sample it client-side first |

## Worked examples

See [`examples/`](examples/): `fraud_imbalanced.py`, `patients_grouped.py`,
`housing_spatial.py`, `devices_temporal.py`. Each runs end-to-end
(`python examples/<name>.py`) and asserts its core guarantee (no identity leak,
chronology, block holdout).
