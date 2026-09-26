"""Regression tests for review round 2 (1x P1, 2x P2)."""

import numpy as np
import pandas as pd
import pytest

from cv_advisor import CVAdvisor
from cv_advisor.spatial import make_spatial_blocks


def _splits_work(df, target, rec):
    adv = CVAdvisor(target)
    splitter = adv.get_splitter(df, rec)
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    X = X.to_numpy() if X.shape[1] else np.zeros((len(df), 1))
    kw = {}
    if rec.group_col and rec.recommended_splitter in ("GroupKFold", "StratifiedGroupKFold"):
        kw["groups"] = df[rec.group_col].to_numpy()
    folds = list(splitter.split(X, y, **kw) if kw else splitter.split(X, y))
    assert len(folds) >= 2
    return splitter


# ---- P1: target must never be the time key ----


def test_p1_datetime_target_not_time():
    df = pd.DataFrame(
        {"feat": np.arange(100.0), "event_time": pd.date_range("2020-01-01", periods=100, freq="D")}
    )
    prof = CVAdvisor("event_time").profile(df, "event_time")
    assert not prof.has_temporal
    assert prof.time_col is None


def test_p1_timelike_numeric_target_not_time():
    # monotonic 'year' target: previously detected as the time index
    df = pd.DataFrame({"feat": np.random.default_rng(0).normal(size=60), "year": np.arange(2000, 2060)})
    prof = CVAdvisor("year").profile(df, "year")
    assert not prof.has_temporal


def test_p1_real_time_col_still_detected():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=80, freq="D"),
            "feat": np.random.default_rng(1).normal(size=80),
            "label": [0] * 60 + [1] * 20,
        }
    )
    prof = CVAdvisor("label").profile(df, "label")
    assert prof.has_temporal and prof.time_col == "timestamp"


# ---- P2: feasibility on every stratified path ----


def test_p2_balanced_rare_class_default_capped():
    # 76/4: ratio 0.0526 -> NOT severe imbalance -> default path, but only 4 minority
    df = pd.DataFrame({"f": range(80), "label": [0] * 76 + [1] * 4})
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "StratifiedKFold"
    assert rec.parameters["n_splits"] <= 4
    _splits_work(df, "label", rec)


def test_p2_tiny_20_rows_no_default_five_fold():
    df = pd.DataFrame({"f": range(20), "label": [0] * 14 + [1] * 6})
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.parameters["n_splits"] <= 6  # minority bound, not hardcoded 5
    assert rec.parameters["n_splits"] <= len(df)
    _splits_work(df, "label", rec)


def test_p2_grouped_single_entity_class_falls_back():
    # minority class lives in ONE patient -> StratifiedGroupKFold infeasible
    groups = [f"p{i}" for i in range(16)] * 4  # 16 groups x4 rows = 64
    y = [0] * 61 + [1] * 3
    assert 3 / 61 < 0.05  # severe-imbalance branch
    # put all minority rows into group p0
    df = pd.DataFrame({"patient_id": groups, "x": np.arange(64), "label": y})
    df.loc[df["label"] == 1, "patient_id"] = "p0"
    prof = CVAdvisor("label").profile(df, "label")
    assert prof.has_groups and prof.min_class_group_count == 1
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "GroupKFold"  # safe fallback, not stratified-group


def test_p2_grouped_few_minority_groups_capped():
    # minority spread over exactly 2 groups -> folds capped at 2
    gids, labels = [], []
    for i in range(12):
        gids += [f"p{i}"] * 4
        labels += [0, 0, 0, 1] if i < 2 else [0, 0, 0, 0]
    df = pd.DataFrame({"patient_id": gids, "x": np.arange(48), "label": labels})
    assert sum(labels) / (len(labels) - sum(labels)) < 0.05  # severe-imbalance branch
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "StratifiedGroupKFold"
    assert rec.parameters["n_splits"] <= 2


# ---- P2: spatial hardening ----


def test_p2_spatial_with_infinities():
    rng = np.random.default_rng(5)
    lat = rng.uniform(32, 42, 100)
    lon = rng.uniform(-124, -114, 100)
    lat[::10] = np.inf
    lon[5::20] = np.nan
    df = pd.DataFrame({"lat": lat, "lon": lon, "v": rng.normal(size=100)})
    labels = make_spatial_blocks(df, "lat", "lon", n_splits=5)
    assert len(labels) == len(df) and np.isfinite(labels).all()
    rec = CVAdvisor("v").advise(df, "v")
    _splits_work(df, "v", rec)


def test_p2_spatial_duplicate_coords_downgrade():
    # only 3 distinct locations but engine asks 5 -> downgrade, parameters truthful
    locs = [(34.0, -118.0), (37.7, -122.4), (40.7, -74.0)]
    rows = [locs[i % 3] for i in range(120)]
    df = pd.DataFrame(rows, columns=["lat", "lon"]).assign(v=np.arange(120.0))
    adv = CVAdvisor("v")
    rec = adv.advise(df, "v")
    assert rec.recommended_splitter == "GroupKFold"
    splitter = adv.get_splitter(df, rec)
    assert rec.parameters["n_splits"] <= 3
    folds = list(splitter.split(df[["v"]].to_numpy(), df["v"].to_numpy()))
    assert len(folds) == rec.parameters["n_splits"]


def test_p2_spatial_single_location_raises_clearly():
    df = pd.DataFrame({"lat": [34.0] * 50, "lon": [-118.0] * 50, "v": np.arange(50.0)})
    with pytest.raises(ValueError, match="distinct coordinate"):
        make_spatial_blocks(df, "lat", "lon", n_splits=5)
