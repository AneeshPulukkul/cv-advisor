"""Sensor drift with deployment waves -> PurgedGroupTimeSeriesSplit.

Sensors are deployed in waves (each wave active for a staggered window), so
entities are roughly time-ordered: expanding-window splits with purge gap +
embargo prevent lookahead AND identity leakage. Rows are shuffled on disk to
prove the snippet's sort-first pattern matters.

Run:  python examples/devices_temporal.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from cv_advisor import CVAdvisor

rng = np.random.default_rng(99)
WAVES = {0: range(60), 1: range(40, 100), 2: range(80, 140)}
frames = []
for wave, days in WAVES.items():
    days = list(days)
    for s in range(2):
        drift = 0.05 * np.arange(len(days)) + rng.normal(0, 0.5, size=len(days))
        frames.append(pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=140, freq="D")[days],
            "device_id": f"wave{wave}-s{s}",
            "load": rng.uniform(0, 100, size=len(days)),
            "reading": drift + rng.normal(0, 0.3, size=len(days)),
        }))
df = pd.concat(frames, ignore_index=True)
df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)  # shuffled on disk!
print(f"rows: {len(df)}, sorted by time: {df['timestamp'].is_monotonic_increasing}")

advisor = CVAdvisor("reading")
prof = advisor.profile(df, "reading")
print(f"group/time overlap: {prof.group_time_overlap:.2f} (waves are staggered, not interleaved)")
rec = advisor.advise(df, "reading")
print("splitter:", rec.recommended_splitter, rec.parameters)

# Follow the snippet: sort first, pass the frame so times resolve.
dfs = df.sort_values(rec.time_col).reset_index(drop=True)
X = dfs[["load"]].to_numpy()
y = dfs["reading"].to_numpy()
groups = dfs[rec.group_col].to_numpy()
splitter = advisor.get_splitter(dfs, rec)
t = dfs[rec.time_col].to_numpy().astype("datetime64[ns]").astype("int64")
for i, (tr, te) in enumerate(splitter.split(dfs, y, groups)):
    assert t[tr].max() < t[te].min(), f"lookahead leak in fold {i}"
    assert not set(groups[tr]) & set(groups[te]), f"identity leak in fold {i}"
    print(f"fold {i}: train={len(tr)} test={len(te)} chrono+disjoint OK")
scores = cross_val_score(Ridge(), X, y,
                         cv=splitter.split(dfs, y, groups), scoring="neg_mean_absolute_error")
print("MAE per fold:", np.round(-scores, 4))
