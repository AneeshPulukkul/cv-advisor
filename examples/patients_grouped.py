"""Grouped + imbalanced patients -> StratifiedGroupKFold.

Same patient, multiple tests: random splits would leak identity.
Run:  python examples/patients_grouped.py
"""

import numpy as np
import pandas as pd

from cv_advisor import CVAdvisor

rng = np.random.default_rng(7)
N_PATIENTS, TESTS = 60, 4
N = N_PATIENTS * TESTS
patient = np.repeat([f"p{i:03d}" for i in range(N_PATIENTS)], TESTS)
# rare disease: only 2 patients positive (all their tests positive) -> 8/232
# minority/majority = 0.034 < 0.05, i.e. the severe-imbalance path
sick = {"p003", "p042"}
df = pd.DataFrame(
    {
        "patient_id": patient,
        "age": rng.integers(20, 80, size=N),
        "marker": rng.normal(size=N),
        "disease": [1 if p in sick else 0 for p in patient],
    }
)

advisor = CVAdvisor("disease")
rec = advisor.advise(df, "disease")
assert rec.recommended_splitter == "StratifiedGroupKFold", rec.recommended_splitter
assert rec.parameters["n_splits"] <= 2  # minority lives in exactly 2 entities
print("splitter:", rec.recommended_splitter, rec.parameters)
print("rationale:", rec.rationale)

X = df[["age", "marker"]].to_numpy()
y = df["disease"].to_numpy()
groups = df["patient_id"].to_numpy()
splitter = advisor.get_splitter(df, rec)
for i, (tr, te) in enumerate(splitter.split(X, y, groups)):
    overlap = set(groups[tr]) & set(groups[te])
    assert not overlap, f"identity leak in fold {i}"
    print(f"fold {i}: train={len(tr)} test={len(te)} "
          f"test-positive-rate={y[te].mean():.3f}")
print("OK: no patient appears on both sides of any fold")
