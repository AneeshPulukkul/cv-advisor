"""Regression tests for the external review findings (2x P1, 3x P2)."""

import numpy as np
import pandas as pd
import pytest

from cv_advisor import CVAdvisor
from cv_advisor.server import app
from cv_advisor.splits import PurgedGroupTimeSeriesSplit

try:
    from fastapi.testclient import TestClient

    client = TestClient(app)
except Exception:  # pragma: no cover
    client = None


def _recurring_groups(n_groups=12, per_group=10, seed=0):
    """Groups that recur across the full time range (worst case for P1-1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        # each group appears early AND late
        times = rng.choice(np.arange(200), size=per_group, replace=False)
        for t in times:
            rows.append((f"g{g}", t, rng.normal()))
    df = pd.DataFrame(rows, columns=["entity", "ts", "x"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffled


def test_p1_purged_no_future_in_train():
    df = _recurring_groups()
    base = pd.date_range("2020-01-01", periods=200, freq="D")
    df = df.assign(ts=base[df["ts"].to_numpy()].to_numpy())
    cv = PurgedGroupTimeSeriesSplit(n_splits=3, group_gap=1, embargo=1, time_col="ts")
    groups = df["entity"].to_numpy()
    folds = list(cv.split(df, df["x"].to_numpy(), groups))
    assert len(folds) >= 2
    t = df["ts"].to_numpy()
    for tr, te in folds:
        assert len(tr) and len(te)
        assert t[tr].max() < t[te].min(), "future row leaked into train"
        assert not set(groups[tr]) & set(groups[te])


def test_p1_purged_backward_compat_without_time():
    df = _recurring_groups()
    cv = PurgedGroupTimeSeriesSplit(n_splits=2)
    folds = list(cv.split(df[["x"]].to_numpy(), None, df["entity"].to_numpy()))
    assert len(folds) == 2
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0


def test_p1_time_snippets_sort():
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2020-01-01", periods=100, freq="D")[::-1], "price": np.arange(100.0)}
    )
    rec = CVAdvisor("price").advise(df, "price")
    assert rec.recommended_splitter == "TimeSeriesSplit"
    assert "sort_values" in rec.code_snippet and "timestamp" in rec.code_snippet


def test_p1_purged_snippet_uses_time_and_groups():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(6):  # staggered waves -> purged path (interleaved would route elsewhere)
        for _ in range(20):
            rows.append((f"g{g}", g * 20 + int(rng.integers(0, 60)), rng.normal()))
    df = pd.DataFrame(rows, columns=["entity", "t", "x"])
    base = pd.date_range("2020-01-01", periods=200, freq="h")
    df = df.assign(ts=base[df["t"].to_numpy()].to_numpy())
    df["label"] = (df["x"] > 0).astype(int)
    adv = CVAdvisor("label")
    prof = adv.profile(df, "label")
    assert prof.has_temporal and prof.has_groups
    rec = adv.advise(df, "label")
    assert rec.recommended_splitter == "PurgedGroupTimeSeriesSplit"
    assert "sort_values" in rec.code_snippet and "group_col" in rec.code_snippet
    splitter = adv.get_splitter(df, rec)
    assert splitter.time_col == prof.time_col


def test_p2_folds_capped_by_minority():
    df = pd.DataFrame({"f": range(200), "label": [0] * 197 + [1] * 3})
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "StratifiedKFold"
    assert rec.parameters["n_splits"] <= 3
    rec_r = CVAdvisor("label").advise(df, "label", prefer_repeated=True)
    assert rec_r.parameters["n_splits"] <= 3


def test_p2_single_minority_falls_back_to_kfold():
    df = pd.DataFrame({"f": range(100), "label": [0] * 99 + [1]})
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "KFold"  # stratification impossible


def test_p2_upload_limit_enforced_early(monkeypatch):
    if client is None:
        pytest.skip("fastapi test client unavailable")
    import cv_advisor.server as srv

    monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 100)
    big = pd.DataFrame({"a": range(1000)}).to_csv(index=False).encode()
    assert len(big) > 100
    r = client.post("/api/advise", files={"file": ("d.csv", big, "text/csv")}, data={"target": "a"})
    assert r.status_code == 413


def test_p2_spatial_follows_reordered_index():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"lat": rng.uniform(32, 42, 120), "lon": rng.uniform(-124, -114, 120), "v": rng.normal(size=120)}
    )
    adv = CVAdvisor("v")
    rec = adv.advise(df, "v")
    assert rec.recommended_splitter == "GroupKFold" and rec.lat_col
    splitter = adv.get_splitter(df, rec)
    X = df[["v"]]
    ref = [(sorted(tr.tolist()), sorted(te.tolist())) for tr, te in splitter.split(X, df["v"].to_numpy())]
    shuff = X.sample(frac=1.0, random_state=7)
    got = [(sorted(tr.tolist()), sorted(te.tolist())) for tr, te in splitter.split(shuff, df.loc[shuff.index, "v"].to_numpy())]
    # positional folds over the shuffled frame must carry the same ROW LABELS per fold
    ref_labels = [set(X.index[t]) for t, _ in ref]
    got_labels = [set(shuff.index[t]) for t, _ in got]
    assert sorted(map(sorted, ref_labels)) == sorted(map(sorted, got_labels))


def test_p2_spatial_length_mismatch_raises():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {"lat": rng.uniform(32, 42, 60), "lon": rng.uniform(-124, -114, 60), "v": rng.normal(size=60)}
    )
    adv = CVAdvisor("v")
    rec = adv.advise(df, "v")
    splitter = adv.get_splitter(df, rec)
    with pytest.raises(ValueError, match="groups= explicitly"):
        list(splitter.split(np.zeros((10, 1)), np.zeros(10)))
