"""End-to-end synthetic cases: fraud, patients, housing, stocks."""

import numpy as np
import pandas as pd

from cv_advisor import CVAdvisor


def _advisor_splits_ok(df, target, groups=None):
    adv = CVAdvisor(target)
    rec = adv.advise(df, target)
    splitter = adv.get_splitter(df, rec)
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).to_numpy()
    if len(X) == 0 or X.shape[1] == 0:
        X = np.zeros((len(df), 1))
    if groups is not None:
        g = df[groups].to_numpy() if isinstance(groups, str) else groups
        splits = list(splitter.split(X, y, g))
    elif rec.recommended_splitter in ("StratifiedKFold", "StratifiedGroupKFold", "BinnedStratifiedKFold"):
        g = df[rec.group_col].to_numpy() if rec.group_col else None
        splits = list(splitter.split(X, y, g))
    else:
        splits = list(splitter.split(X, y))
    assert len(splits) >= 2
    for tr, te in splits:
        assert len(tr) > 0 and len(te) > 0
        assert len(np.intersect1d(tr, te)) == 0
    return rec


def test_fraud_imbalanced():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "fraud": np.array([0] * 1990 + [1] * 10)})
    rec = _advisor_splits_ok(df, "fraud")
    assert rec.recommended_splitter == "StratifiedKFold"


def test_patients_grouped_imbalanced():
    rng = np.random.default_rng(1)
    pids = [f"p{i // 4}" for i in range(400)]
    y = rng.choice([0, 1], size=400, p=[0.97, 0.03])
    df = pd.DataFrame({"patient_id": pids, "x": rng.normal(size=400), "disease": y})
    rec = _advisor_splits_ok(df, "disease", groups="patient_id")
    assert rec.recommended_splitter == "StratifiedGroupKFold"


def test_housing_spatial():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"lat": rng.uniform(32, 42, 300), "lon": rng.uniform(-124, -114, 300),
                       "MedHouseVal": rng.normal(200_000, 50_000, 300)})
    rec = _advisor_splits_ok(df, "MedHouseVal")
    assert rec.recommended_splitter in ("GroupKFold", "BinnedStratifiedKFold", "KFold")


def test_stocks_timeseries():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=300, freq="D"),
                       "price": np.cumsum(np.random.randn(300)) + 100})
    rec = _advisor_splits_ok(df, "price")
    assert rec.recommended_splitter == "TimeSeriesSplit"
