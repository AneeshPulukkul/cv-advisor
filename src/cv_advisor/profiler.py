"""Stage 1: dataset statistical and structural profiler.

Defensive: sampling for N > 500k, NaN-tolerant, <2s target on large frames.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cv_advisor.schema import DataProfile

PROFILE_SAMPLE_LIMIT = 500_000
SMALL_N = 200
TINY_N = 50
LARGE_N = 100_000
IMBALANCE_THRESHOLD = 0.05
SKEW_THRESHOLD = 1.0

TIME_NAME_PATTERN = re.compile(r"time|timestamp|date|year|month|day|datetime", re.IGNORECASE)
GROUP_NAME_HINTS = (
    "user_id", "patient_id", "store_id", "client_id", "device_id", "customer_id",
    "subject", "entity", "site", "hospital", "school", "store", "patient",
    "user", "client", "device", "member", "account", "ticker", "symbol",
    "id", "key", "group",
)
COORD_PAIRS = [
    ({"latitude", "longitude"}, ("latitude", "longitude")),
    ({"lat", "lon"}, ("lat", "lon")),
    ({"lat", "lng"}, ("lat", "lng")),
    ({"latitude", "lon"}, ("latitude", "lon")),
    ({"y_lat", "x_lon"}, ("y_lat", "x_lon")),
]


def _gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p**2))


def _detect_coords(
    columns_lower: dict[str, str], target_cols: list[str]
) -> tuple[str | None, str | None]:
    targets = set(target_cols)
    cols = set(columns_lower.keys())
    for names, _ in COORD_PAIRS:
        if names <= cols:
            ordered = sorted(names)
            # return in (lat-like, lon-like) order heuristically
            lat_candidates = [c for c in ordered if "lat" in c]
            lon_candidates = [c for c in ordered if ("lon" in c or "lng" in c)]
            if lat_candidates and lon_candidates:
                lat_col, lon_col = columns_lower[lat_candidates[0]], columns_lower[lon_candidates[0]]
                if lat_col in targets or lon_col in targets:
                    continue  # coordinates that ARE the target can't be blocking keys
                return lat_col, lon_col
    # x/y fallback only when both literally present and neither is the target
    if {"x", "y"} <= cols and columns_lower["x"] not in targets and columns_lower["y"] not in targets:
        return columns_lower["x"], columns_lower["y"]
    return None, None


def _detect_time_col(df: pd.DataFrame, target_cols: list[str]) -> str | None:
    targets = set(target_cols)
    # 1) explicit datetime dtype with a time-like name (strongest signal)
    for col in df.columns:
        if col in targets:
            continue  # the outcome itself must never be the split key
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # 2) parseable object columns with time-like names
    for col in df.columns:
        if col in targets:
            continue
        if TIME_NAME_PATTERN.search(str(col)):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            if pd.api.types.is_numeric_dtype(s):
                # strictly monotonic ascending integer -> time index
                vals = s.to_numpy()
                if len(vals) > 2 and np.all(np.diff(vals) > 0):
                    return col
            else:
                try:
                    parsed = pd.to_datetime(s.head(100), errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        return col
                except Exception:
                    continue
    # 3) any strictly monotonic integer column (weak signal, name-agnostic fallback off
    #    to avoid false positives -> require time-like name)
    return None


def _detect_group_col(df: pd.DataFrame, target_cols: list[str]) -> str | None:
    n = len(df)
    best: str | None = None
    best_score = -1.0
    for col in df.columns:
        if col in target_cols:
            continue
        name = str(col).lower()
        if not any(h in name for h in GROUP_NAME_HINTS):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_unique = int(s.nunique())
        if n_unique < 2 or n_unique >= n:
            continue  # must repeat but not be a row-id
        counts = s.value_counts()
        max_size = int(counts.max())
        if max_size < 2:
            continue
        # prefer columns with many repeats and reasonable group count (>= n_splits)
        score = (1 - n_unique / n) * np.log1p(max_size)
        if score > best_score:
            best_score = score
            best = col
    return best


def profile_dataframe(
    df: pd.DataFrame,
    target_col: str | list[str],
    sample_limit: int = PROFILE_SAMPLE_LIMIT,
) -> DataProfile:
    """Profile a DataFrame and return structured facts for the decision engine."""
    if isinstance(target_col, str):
        target_cols = [target_col]
    else:
        target_cols = list(target_col)
    for t in target_cols:
        if t not in df.columns:
            raise KeyError(f"target column {t!r} not found in DataFrame")

    n_rows_full = int(len(df))
    n_cols = int(df.shape[1])
    memory_mb = float(df.memory_usage(deep=True, index=True).sum() / (1024**2))

    sampled = False
    work = df
    if n_rows_full > sample_limit:
        work = df.sample(n=sample_limit, random_state=42)
        sampled = True

    # ---- target type ----
    multilabel = len(target_cols) > 1
    if multilabel:
        task_type = "multilabel"  # type: ignore[assignment]
        n_unique_target = None
        minority_ratio = None
        minority_count = None
        gini = None
        is_imbalanced = False
        skew = None
        kurt = None
    else:
        t = target_cols[0]
        s = work[t].dropna()
        n_unique_target = int(s.nunique())
        if pd.api.types.is_bool_dtype(s) or n_unique_target == 2:
            task_type = "binary"
        elif pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or (
            pd.api.types.is_integer_dtype(s) and n_unique_target <= 20
        ):
            task_type = "multiclass"
        elif pd.api.types.is_numeric_dtype(s) and n_unique_target > 20:
            task_type = "regression"
        else:
            # low-cardinality fallback -> classification, else regression
            task_type = "multiclass" if n_unique_target <= 20 else "regression"  # type: ignore[assignment]

        minority_ratio = None
        minority_count = None
        gini = None
        is_imbalanced = False
        skew = None
        kurt = None
        if task_type in ("binary", "multiclass"):
            vc = s.value_counts()
            if len(vc) >= 2:
                minority_ratio = float(vc.min() / vc.max())
                minority_count = int(vc.min())
                gini = _gini(vc.to_numpy())
                is_imbalanced = bool(minority_ratio < IMBALANCE_THRESHOLD)
        else:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) > 2:
                skew = float(s_num.skew())
                kurt = float(s_num.kurtosis())

    is_skewed_regression = bool(
        task_type == "regression" and skew is not None and abs(skew) > SKEW_THRESHOLD
    )

    # ---- temporal ----
    time_col = _detect_time_col(work, target_cols)
    has_temporal = time_col is not None
    autocorr: float | None = None
    if has_temporal and not multilabel and task_type == "regression":
        try:
            s_num = pd.to_numeric(work[target_cols[0]], errors="coerce").dropna()
            if len(s_num) > 3:
                autocorr = float(s_num.autocorr(lag=1))
        except Exception:
            autocorr = None

    # ---- spatial ----
    lower = {str(c).lower(): c for c in work.columns}
    lat_col, lon_col = _detect_coords(lower, target_cols)
    has_spatial = lat_col is not None and lon_col is not None
    spatial_clustered = False
    if has_spatial:
        try:
            coords = work[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce")
            coords = coords.replace([np.inf, -np.inf], np.nan).dropna()
            if len(coords) >= 20:
                k = int(min(5, max(2, len(coords) // 50)))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(coords.to_numpy())
                _, counts = np.unique(labels, return_counts=True)
                cv = float(counts.std() / (counts.mean() + 1e-9))
                spatial_clustered = bool(cv > 0.5)
            else:
                spatial_clustered = True  # too few points to tell; assume risky
        except Exception:
            spatial_clustered = False

    # ---- groups ----
    group_col = _detect_group_col(work, target_cols)
    has_groups = group_col is not None
    n_groups: int | None = None
    max_group_size: int | None = None
    min_class_group_count: int | None = None
    if has_groups:
        assert group_col is not None
        vc = work[group_col].value_counts(dropna=True)
        n_groups = int(vc.shape[0])
        max_group_size = int(vc.max())
        if not multilabel and task_type in ("binary", "multiclass"):
            try:
                ct = work[[group_col, target_cols[0]]].dropna()
                gpc = ct.groupby(target_cols[0], observed=True)[group_col].nunique()
                if len(gpc):
                    min_class_group_count = int(gpc.min())
            except Exception:
                min_class_group_count = None

    # ---- group x time overlap (are entities time-contiguous or interleaved?) ----
    group_time_overlap: float | None = None
    if has_temporal and has_groups and time_col is not None and group_col is not None:
        try:
            gt = work[[time_col, group_col]].dropna()
            tnum = pd.to_datetime(gt[time_col], errors="coerce", utc=True)
            gt = gt.assign(_t=tnum).dropna(subset=["_t"])
            if len(gt) >= 10 and gt["_t"].nunique() >= 3:
                tvals = gt["_t"].astype("int64").to_numpy()
                glob_med = float(np.median(tvals))
                lo = gt.groupby(group_col, observed=True)["_t"].quantile(0.05).astype("int64")
                hi = gt.groupby(group_col, observed=True)["_t"].quantile(0.95).astype("int64")
                covers = int(((lo <= glob_med) & (glob_med <= hi)).sum())
                group_time_overlap = float(covers / len(lo))
        except Exception:
            group_time_overlap = None

    return DataProfile(
        n_rows=n_rows_full,
        n_cols=n_cols,
        memory_mb=memory_mb,
        target_cols=target_cols,
        task_type=task_type,  # type: ignore[arg-type]
        n_unique_target=n_unique_target,
        minority_ratio=minority_ratio,
        minority_count=minority_count,
        gini=gini,
        is_imbalanced=is_imbalanced,
        target_skew=skew,
        target_kurtosis=kurt,
        is_skewed_regression=is_skewed_regression,
        time_col=time_col,
        has_temporal=has_temporal,
        temporal_autocorr=autocorr,
        lat_col=lat_col,
        lon_col=lon_col,
        has_spatial=has_spatial,
        spatial_clustered=spatial_clustered,
        group_col=group_col,
        has_groups=has_groups,
        n_groups=n_groups,
        max_group_size=max_group_size,
        min_class_group_count=min_class_group_count,
        group_time_overlap=group_time_overlap,
        is_small=n_rows_full < SMALL_N,
        is_tiny=n_rows_full < TINY_N,
        is_large=n_rows_full > LARGE_N,
        sampled_for_profiling=sampled,
    )
