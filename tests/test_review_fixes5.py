"""Overlap-aware temporal+group routing + empty-yield backstop."""

import numpy as np
import pandas as pd
import pytest

from cv_advisor import CVAdvisor
from cv_advisor.splits import PurgedGroupTimeSeriesSplit


def _interleaved(n_groups=4, per_group=30, seed=0):
    """Every entity spans the whole timeline (panel-data shape)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for _ in range(per_group):
            rows.append((f"g{g}", rng.integers(0, 200), rng.normal()))
    df = pd.DataFrame(rows, columns=["entity", "t", "x"])
    base = pd.date_range("2021-01-01", periods=200, freq="D")
    return df.assign(ts=base[df["t"].to_numpy()].to_numpy())


def test_profiler_overlap_interleaved():
    df = _interleaved()
    prof = CVAdvisor("x").profile(df, "x")
    assert prof.has_temporal and prof.has_groups
    assert prof.group_time_overlap is not None and prof.group_time_overlap > 0.7


def test_profiler_overlap_waves():
    rows = []
    for wave, span in ((0, range(60)), (1, range(40, 100)), (2, range(80, 140))):
        for s in range(2):
            for d in span:
                rows.append((f"w{wave}s{s}", d, float(d)))
    df = pd.DataFrame(rows, columns=["device_id", "t", "x"])
    base = pd.date_range("2022-01-01", periods=140, freq="D")
    df = df.assign(ts=base[df["t"].to_numpy()].to_numpy())
    prof = CVAdvisor("x").profile(df, "x")
    assert prof.group_time_overlap is not None and prof.group_time_overlap < 0.7


def test_engine_interleaved_falls_back_to_timeseries():
    df = _interleaved().assign(label=lambda d: (d["x"] > 0).astype(int))
    adv = CVAdvisor("label")
    rec = adv.advise(df, "label")
    assert rec.recommended_splitter == "TimeSeriesSplit"
    assert any("interleaved" in r for r in rec.leakage_risks)
    splitter = adv.get_splitter(df, rec)
    folds = list(splitter.split(df[["x"]].to_numpy(), df["label"].to_numpy()))
    assert len(folds) == rec.parameters["n_splits"]


def test_splitter_empty_yield_raises_not_silent():
    # identical timelines per group: every chronological split purges its train
    rows = [(f"g{g}", d, float(d)) for g in range(4) for d in range(10)]
    df = pd.DataFrame(rows, columns=["entity", "t", "x"])
    base = pd.date_range("2021-01-01", periods=10, freq="D")
    df = df.assign(ts=base[df["t"].to_numpy()].to_numpy())
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, group_gap=1, time_col="ts")
    with pytest.raises(ValueError, match="no usable folds"):
        list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))
