"""Highly imbalanced fraud classification -> StratifiedKFold (or Repeated).

Run:  python examples/fraud_imbalanced.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import cross_val_score

from cv_advisor import CVAdvisor

rng = np.random.default_rng(42)
N = 2000
df = pd.DataFrame(
    {
        "amount": rng.lognormal(mean=3.0, sigma=1.2, size=N),
        "hour": rng.integers(0, 24, size=N),
        "is_fraud": np.array([0] * 1985 + [1] * 15),  # 0.75% minority
    }
)

advisor = CVAdvisor("is_fraud")
rec = advisor.advise(df, "is_fraud")
print("splitter:", rec.recommended_splitter, rec.parameters)
print("patterns:", rec.detected_patterns)
print("risks:", rec.leakage_risks)

X = df[["amount", "hour"]].to_numpy()
y = df["is_fraud"].to_numpy()
splitter = advisor.get_splitter(df, rec)
folds = list(splitter.split(X, y))
for i, (tr, te) in enumerate(folds):
    # stratification guarantee: minority present on both sides of every fold
    assert y[te].sum() >= 1, f"fold {i} test has no fraud cases"
    assert y[tr].sum() >= 1, f"fold {i} train has no fraud cases"
print(f"OK: all {len(folds)} folds contain fraud cases in train and test")
scores = cross_val_score(LogisticRegression(max_iter=500, class_weight="balanced"), X, y,
                         cv=folds, scoring=make_scorer(recall_score))
print("recall per fold:", np.round(scores, 3))

rec_rep = advisor.advise(df, "is_fraud", prefer_repeated=True)
print("with --repeated:", rec_rep.recommended_splitter, rec_rep.parameters)
