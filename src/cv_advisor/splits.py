"""Differentiator splitters (sklearn-compatible) + re-exports.

- PurgedGroupTimeSeriesSplit: expanding-window time split over groups with
  purge gap + embargo (prevents lookahead leakage; sklearn has no equivalent).
- BinnedStratifiedKFold: quantile-bin a continuous target, then StratifiedKFold
  on the bins (stratified regression; sklearn has no equivalent).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """Expanding-window split over time-ordered groups with purge and embargo.

    Chronology is enforced at the *row* level, not just the group level: every
    train row must be strictly earlier than every test row. A group that recurs
    across time therefore contributes only its pre-test rows to train — late
    rows of an "early" group are purged, so future information cannot leak.

    Times are resolved from ``X`` at split time: pass a DataFrame containing
    ``time_col`` (datetime or numeric). If ``time_col`` is set but any timestamp
    is missing or unparseable, splitting FAILS LOUDLY instead of silently
    degrading to unordered group folds. Without ``time_col`` it falls back to
    first-appearance group ordering (documented limitation — prefer the
    time-aware path; the advisor always supplies it).

    Parameters
    ----------
    n_splits: number of test folds.
    group_gap: groups dropped between train-end and test-start (purge).
    embargo: groups skipped after each test fold before next train window.
    time_col: column of X holding timestamps (required for the purge guarantee).
    """

    def __init__(
        self,
        n_splits: int = 5,
        group_gap: int = 1,
        embargo: int = 0,
        time_col: str | None = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.group_gap = max(0, int(group_gap))
        self.embargo = max(0, int(embargo))
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _resolve_times(self, X, groups: np.ndarray) -> np.ndarray | None:
        """Resolve timestamps to numeric (int64 ns for datetimes).

        Datetimes — naive, timezone-aware, or string — are normalized to UTC
        epoch nanoseconds so downstream medians/comparisons never see object
        arrays of Timestamps (whose even-sized median raises TypeError).
        """
        if self.time_col is None:
            return None
        if not hasattr(X, "columns") or self.time_col not in X.columns:
            raise ValueError(
                f"time_col={self.time_col!r} not found in X — pass the DataFrame "
                "carrying the timestamp column (see the advisor's code snippet)."
            )
        col = X[self.time_col]
        if isinstance(col.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(col):
            t = pd.to_datetime(col, utc=True)
        elif pd.api.types.is_timedelta64_dtype(col):
            t = col
        else:
            try:
                with warnings.catch_warnings():
                    # Failure is handled below via coerce->NaN->ValueError;
                    # don't leak pandas' format-guessing noise to callers.
                    warnings.simplefilter("ignore", UserWarning)
                    t = pd.to_datetime(col, errors="coerce", utc=True)
                if t.isna().all():
                    t = pd.to_numeric(col, errors="coerce")
            except Exception as exc:
                raise ValueError(
                    f"time_col={self.time_col!r} could not be parsed as timestamps."
                ) from exc
        bad = int(pd.isna(t).sum())
        if bad:
            raise ValueError(
                f"time_col={self.time_col!r} has {bad}/{len(t)} missing or unparseable "
                "timestamps — drop or impute those rows before splitting. Refusing to "
                "fall back to unordered folds, which would void the purge guarantee."
            )
        if t.dtype == object:
            # Mixed-offset Timestamps/strings parse to object dtype (no single tz):
            # coerce everything to UTC. Input is validated parseable above, so no
            # new NaNs may appear — re-check defensively.
            t = pd.to_datetime(t, utc=True, errors="coerce")
            if int(pd.isna(t).sum()):
                raise ValueError(
                    f"time_col={self.time_col!r} contains values that cannot be "
                    "compared on a single timeline (mixed timezones)."
                )
        if isinstance(t.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(t):
            t = pd.to_datetime(t, utc=True).astype("int64")
        elif pd.api.types.is_timedelta64_dtype(t):
            t = t.astype("int64")
        return np.asarray(t)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required for PurgedGroupTimeSeriesSplit")
        groups = np.asarray(groups)
        times = self._resolve_times(X, groups)
        if times is not None and np.asarray(times).dtype.kind in "mM":
            times = np.asarray(times).astype("int64")  # np.median can't reduce M8/m8
        if times is None:
            # Only reachable when time_col is None (legacy mode): positional order.
            ordered = np.asarray(
                sorted(np.unique(groups), key=lambda g: np.min(np.where(groups == g)[0]))
            )
        else:
            # order groups by median timestamp (chronological, not positional)
            medians = {g: np.median(times[groups == g]) for g in np.unique(groups)}
            ordered = np.asarray(sorted(medians, key=lambda g: medians[g]))
        n_groups = len(ordered)
        # Structural feasibility: the first test chunk must start after
        # >=1 train group + the purge gap, and every test chunk needs >=1 group.
        min_start = self.group_gap + 1
        remaining = n_groups - min_start
        if self.n_splits > remaining:
            raise ValueError(
                f"n_splits={self.n_splits} infeasible with {n_groups} groups and "
                f"group_gap={self.group_gap} (at most {max(0, remaining)} splits); "
                "use fewer splits or collect more groups."
            )
        # pack test chunks back-to-back from min_start so every fold has train
        base, rem = divmod(remaining, self.n_splits)
        fold_sizes = np.array([base + 1] * rem + [base] * (self.n_splits - rem))
        start = min_start
        yielded = 0
        for i in range(self.n_splits):
            test_start = start + int(fold_sizes[:i].sum())
            test_end = test_start + int(fold_sizes[i])
            test_groups = set(ordered[test_start:test_end])
            train_end = max(0, test_start - self.group_gap)
            train_groups = set(ordered[:train_end])
            if self.embargo and i > 0:
                skip = set(ordered[test_start - self.embargo : test_start])
                train_groups -= skip
            train_idx = np.where(np.isin(groups, list(train_groups)))[0]
            test_idx = np.where(np.isin(groups, list(test_groups)))[0]
            if times is not None and len(test_idx):
                # ROW-LEVEL PURGE: no train row may be contemporaneous with or
                # later than the earliest test row (kills recurring-group leaks).
                t0 = times[test_idx].min()
                train_idx = train_idx[times[train_idx] < t0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                # Pathological overlap only (layout guarantees non-empty groups;
                # the row purge below can still wipe train if every train-group
                # row postdates the test start). Skip rather than emit empties.
                continue
            yielded += 1
            yield train_idx, test_idx
        if yielded == 0:
            raise ValueError(
                "PurgedGroupTimeSeriesSplit produced no usable folds: groups overlap "
                "in time so fully that every chronological split purges its own "
                "training rows. Use TimeSeriesSplit (interleaved entities) or "
                "GroupKFold (time-contiguous entities, accepts lookahead)."
            )


class BinnedStratifiedKFold(BaseCrossValidator):
    """Stratified KFold for regression via quantile binning of y."""

    def __init__(self, n_splits: int = 5, n_bins: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _bin(self, y: np.ndarray) -> np.ndarray:
        s = pd.Series(np.asarray(y).ravel())
        try:
            bins = pd.qcut(s, q=self.n_bins, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.cut(s, bins=self.n_bins, labels=False)
        bins = pd.Series(bins).fillna(-1).astype(int).to_numpy()
        # guard: StratifiedKFold needs >= n_splits per bin -> merge rare bins
        _, counts = np.unique(bins, return_counts=True)
        if counts.min() < self.n_splits:
            # collapse to fewer bins
            return np.zeros_like(bins)
        return bins

    def split(self, X, y, groups=None):
        if y is None:
            raise ValueError("y is required for BinnedStratifiedKFold")
        bins = self._bin(np.asarray(y))
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        yield from skf.split(X, bins, groups)
