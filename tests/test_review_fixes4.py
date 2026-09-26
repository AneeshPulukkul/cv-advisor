"""Regression tests for review round 4 (2x P2)."""

import numpy as np
import pandas as pd

from cv_advisor import CVAdvisor
from cv_advisor.advisor import _build_code_snippet
from cv_advisor.splits import PurgedGroupTimeSeriesSplit


def _tz_frame(n_groups=6, per_group=8, seed=0):  # even-sized groups: median must average
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for _ in range(per_group):
            rows.append((f"g{g}", rng.integers(0, 200), rng.normal()))
    df = pd.DataFrame(rows, columns=["entity", "t", "x"])
    base = pd.date_range("2021-01-01", periods=200, freq="D", tz="US/Eastern")
    return df.assign(ts=base[df["t"].to_numpy()].to_numpy())


def test_p2_tz_aware_purged_splits():
    df = _tz_frame()
    assert isinstance(df["ts"].dtype, pd.DatetimeTZDtype)  # object-array median path
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, group_gap=1, embargo=1, time_col="ts")
    groups = df["entity"].to_numpy()
    folds = list(cv.split(df, df["x"].to_numpy(), groups))  # used to raise TypeError
    assert len(folds) == 2
    t = pd.to_datetime(df["ts"], utc=True).astype("int64").to_numpy()
    for tr, te in folds:
        assert t[tr].max() < t[te].min()
        assert not set(groups[tr]) & set(groups[te])


def test_p2_tz_aware_string_mixed_still_ordered():
    df = _tz_frame().astype({"ts": str})
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, time_col="ts")
    folds = list(cv.split(df, df["x"].to_numpy(), df["entity"].to_numpy()))
    assert len(folds) == 2


def test_p2_spatial_snippet_matches_downgraded_folds():
    locs = [(34.0, -118.0), (37.7, -122.4), (40.7, -74.0)]
    rows = [locs[i % 3] for i in range(120)]
    df = pd.DataFrame(rows, columns=["lat", "lon"]).assign(v=np.arange(120.0))
    adv = CVAdvisor("v")
    rec = adv.advise(df, "v")
    assert rec.recommended_splitter == "GroupKFold"
    assert rec.target_col == "v"  # target preserved for snippet regeneration
    assert "n_splits=5" in rec.code_snippet  # pre-downgrade rendering
    splitter = adv.get_splitter(df, rec)  # common call: no target_col
    assert rec.parameters["n_splits"] <= 3
    assert rec.target_col == "v"  # downgrade must not clobber it
    assert "n_splits=5" not in rec.code_snippet  # snippet resynced ...
    assert f"n_splits={rec.parameters['n_splits']}" in rec.code_snippet  # ... to effective count
    assert "''" not in rec.code_snippet and '""' not in rec.code_snippet
    # Omitted-arg rebuild must equal explicit-target rebuild: the regenerated
    # snippet may never depend on get_splitter's optional target_col.
    assert rec.code_snippet == _build_code_snippet(rec, "v")
    folds = list(splitter.split(df[["v"]].to_numpy(), df["v"].to_numpy()))
    assert len(folds) == rec.parameters["n_splits"]
