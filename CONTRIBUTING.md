# Contributing to cv-advisor — Engineer Setup Guide

## Prereqs

- Python ≥ 3.10, Node ≥ 22, git.

## Backend setup

```bash
git clone <repo-url> cv-advisor && cd cv-advisor
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                       # macOS/Linux
pip install -e ".[dev]"
```

## Everyday commands

```bash
pytest -q                    # full suite (must be green)
ruff check src tests         # lint (must be clean)
python examples/devices_temporal.py   # smoke an end-to-end example
cv-advisor inspect <file> --target <col>   # CLI (console script on PATH;
python -m cv_advisor.cli inspect ...       # ... or via module)
cv-advisor serve --port 8000               # API for the web UI
```

## Web setup

```bash
cd web && npm install && npm run dev   # UI at http://localhost:5173, /api proxied to :8000
npm run build                          # must succeed before pushing web/ changes
```

## Security (do not skip)

```bash
pip-audit -r <(printf 'pandas>=2.0\nnumpy>=1.24\nscikit-learn>=1.3\npydantic>=2.0\nrich>=13.0\ntyper>=0.9\npyarrow>=12.0\nfastapi>=0.110\nuvicorn>=0.29\npython-multipart>=0.0.9\n')
cd web && npm audit --audit-level=high
```

CI runs `pytest`, `ruff`, `pip-audit` (report-only), `npm audit --audit-level=high`,
and the web build on every push/PR. Dependabot (pip + npm + actions, weekly) is
configured in `.github/dependabot.yml`. After pushing, enable Dependabot alerts
and CodeQL code scanning in repo settings.

## Layout

```
src/cv_advisor/
  profiler.py  # Stage 1: facts only (imbalance, skew, time/group/spatial/scale).
               # Sampling cap 500k; NaN-tolerant; targets excluded from time/coord keys.
  engine.py    # Stage 2: pure deterministic rules, precedence
               # Time > Group > Spatial > Imbalance > Skewed-reg > Tiny > Default.
               # Every stratified path feasibility-capped (rows, minority, groups,
               # per-class groups); every path returns (no fall-through).
  splits.py    # Custom BaseCrossValidators: PurgedGroupTimeSeriesSplit (row-level
               # purge via time_col; loud on bad times/empty yield), BinnedStratifiedKFold.
  spatial.py   # KMeans/grid coordinate blocks; inf-filtered; distinct-count capped.
  schema.py    # Pydantic v2: DataProfile, CVRecommendation (self-contained: carries
               # group/time/lat/lon cols, n_bins, and target_col for snippet rebuilds).
  advisor.py   # Stage 3: CVAdvisor.advise/profile/get_splitter + snippet builder.
  cli.py       # inspect / serve commands (Typer + Rich).
  server.py    # FastAPI: /api/health, /api/rules, /api/advise (chunked 100 MB cap).
tests/test_<area>.py, tests/test_review_fixes*.py  # regression files per review round
examples/*.py  # runnable guarantees, not snippets — each must execute cleanly
web/           # React 18 + Vite 6 + Tailwind v4 (Home / Inspect / Rules)
```

## Conventions that review rounds taught us

1. **Never silent-degrade a guarantee.** Purge, stratification, and block counts
   either hold or raise/fall back with a logged risk — no quiet fallback paths.
2. **Splitters follow row order.** Any time-aware snippet/split must sort first;
   say so in text (`leakage_risks`) and in code.
3. **Recommendations must be executable.** `n_splits` caps, group minimums, and
   snippet/parameter sync are enforced and tested — a `rec` that raises on use
   is a bug.
4. **`rec` is self-contained.** Anything `get_splitter` needs (cols, bins,
   target) lives on the model, not in call-stack arguments.
5. **Tests prove the guarantee, not the name.** Prefer behavioral asserts
   (no shared groups across folds, `max(train_t) < min(test_t)`, 413s, fold-count
   equality) over `recommended_splitter == ...` alone.

## Adding a decision rule

1. Add the signal to `DataProfile` + `profiler.py` (with thresholds as module constants).
2. Insert the branch in `engine.decide` respecting precedence; cap its folds;
   ensure all paths return.
3. If it needs data at split time, extend `get_splitter` + snippet builder and
   keep them in sync (see spatial downgrade precedent).
4. Add synthetic tests (advise → splitter → assert guarantee) + docs row in `USAGE.md`.

## Publish checklist

1. `pytest -q`, `ruff`, all `examples/*.py`, `npm run build` green.
2. Bump version in `pyproject.toml`, `src/cv_advisor/__init__.py`,
   `server.py` (`/api/health` + app version), `web/package.json`.
3. Update `USAGE.md`/README for behavior changes.
4. Tag (`git tag v0.x.y`), push, open PR — CI must pass.
5. GitHub: enable Dependabot alerts + CodeQL; publish to PyPI when ready
   (`pip install cv-advisor` then replaces the `pip install -e .` in `USAGE.md`).
