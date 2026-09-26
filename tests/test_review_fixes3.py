"""Regression tests for review round 3 (1x P1, 1x P2)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from cv_advisor import CVAdvisor
from cv_advisor.splits import PurgedGroupTimeSeriesSplit


def _frame(n_groups=6, per_group=8, seed=0, stagger=20):
    """Staggered wave groups: roughly time-ordered, so the purged path applies.

    (Fully time-overlapping entities now route to TimeSeriesSplit — see fixes5.)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        start = g * stagger
        for _ in range(per_group):
            rows.append((f"g{g}", rng.integers(start, start + 60), rng.normal()))
    df = pd.DataFrame(rows, columns=["entity", "t", "x"])
    base = pd.date_range("2021-01-01", periods=200, freq="D")
    return df.assign(ts=base[df["t"].to_numpy()].to_numpy())


# ---- P1: no silent fallback when time_col is set ----


def test_p1_nat_timestamps_raise():
    df = _frame()
    df.loc[df.sample(frac=0.1, random_state=0).index, "ts"] = pd.NaT
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, time_col="ts")
    with pytest.raises(ValueError, match="missing or unparseable"):
        list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))


def test_p1_unparseable_timestamps_raise():
    df = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # expected: garbage is unparseable
        df["ts"] = ["not-a-date"] * len(df)
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, time_col="ts")
    with pytest.raises(ValueError, match="[Mm]issing or unparseable|parsed"):
        list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))


def test_p1_missing_time_col_raises():
    df = _frame().drop(columns=["ts"])
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, time_col="ts")
    with pytest.raises(ValueError, match="not found in X"):
        list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))


def test_p1_legacy_no_time_col_still_works():
    df = _frame()
    cv = PurgedGroupTimeSeriesSplit(n_splits=2)  # no time_col: positional fallback
    folds = list(cv.split(df[["x"]].to_numpy(), None, df["entity"].to_numpy()))
    assert len(folds) == 2


# ---- P2: purged splits must satisfy n_splits < n_groups ----


def test_p2_six_groups_end_to_end():
    df = _frame(n_groups=6, per_group=10)
    df["label"] = (df["x"] > 0).astype(int)
    adv = CVAdvisor("label")
    rec = adv.advise(df, "label")
    assert rec.recommended_splitter == "PurgedGroupTimeSeriesSplit"
    assert 2 <= rec.parameters["n_splits"] <= 6 - 1 - 1  # n - gap - 1
    splitter = adv.get_splitter(df, rec)
    folds = list(splitter.split(df, df["label"].to_numpy(), df["entity"].to_numpy()))
    assert len(folds) == rec.parameters["n_splits"]  # no silent fold drops
    t = df["ts"].to_numpy().astype("int64")
    for tr, te in folds:
        assert t[tr].max() < t[te].min()


def test_p2_splitter_rejects_infeasible_request():
    df = _frame(n_groups=4, per_group=10)
    cv = PurgedGroupTimeSeriesSplit(n_splits=3, group_gap=1, time_col="ts")
    with pytest.raises(ValueError, match="at most 2 splits"):
        list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))


def test_p2_two_groups_falls_back_to_groupkfold():
    df = _frame(n_groups=2, per_group=20, stagger=80)  # separate eras, overlap ~0
    df["label"] = (df["x"] > 0).astype(int)
    adv = CVAdvisor("label")
    prof = adv.profile(df, "label")
    assert prof.has_temporal and prof.has_groups and prof.n_groups == 2
    rec = adv.advise(df, "label")
    assert rec.recommended_splitter == "GroupKFold"
    assert any(">=4 groups" in r for r in rec.leakage_risks)
    splitter = adv.get_splitter(df, rec)
    folds = list(splitter.split(df[["x"]].to_numpy(), df["label"].to_numpy(), df["entity"].to_numpy()))
    assert len(folds) == 2
