"""Spatial blocking helper: turn coordinates into GroupKFold-ready block labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def make_spatial_blocks(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    n_splits: int = 5,
    method: str = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """Return integer block labels, one per row.

    - ``kmeans``: KMeans(n_clusters=n_splits) on [lat, lon] (differentiator over grids;
      adapts to density, keeps folds balanced-ish).
    - ``grid``: quantile grid (sqrt(n_splits) x sqrt(n_splits)) fallback, deterministic.
    """
    coords = df[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce")
    coords = coords.replace([np.inf, -np.inf], np.nan).dropna()
    if len(coords) == 0:
        raise ValueError("No valid coordinates found for spatial blocking")
    X = coords.to_numpy(dtype=float)
    n_distinct = np.unique(X, axis=0).shape[0]
    if n_distinct < 2:
        raise ValueError(
            "Spatial blocking needs at least 2 distinct coordinate pairs "
            f"(found {n_distinct}); use KFold instead."
        )

    if method == "grid":
        k = int(np.ceil(np.sqrt(n_splits)))
        lat_q = pd.qcut(coords.iloc[:, 0], q=k, labels=False, duplicates="drop").to_numpy()
        lon_q = pd.qcut(coords.iloc[:, 1], q=k, labels=False, duplicates="drop").to_numpy()
        labels = lat_q * k + lon_q
        _, labels = np.unique(labels, return_inverse=True)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Spatial grid collapsed to a single block (duplicate coordinates); use KFold instead."
            )
        return labels.astype(int)

    if method != "kmeans":
        raise ValueError(f"Unknown spatial method {method!r}; use 'kmeans' or 'grid'")
    # Never request more clusters than distinct points (duplicates collapse).
    n_clusters = int(max(2, min(n_splits, n_distinct, len(X))))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    full = df[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # rows with NaN coords -> nearest centroid assignment fallback (-1 then map to largest block)
    nan_mask = ~np.isfinite(full).all(axis=1)
    labels = np.empty(len(df), dtype=int)
    valid = ~nan_mask
    labels[valid] = km.fit_predict(X)
    if nan_mask.any():
        biggest = int(np.bincount(km.labels_).argmax())
        labels[nan_mask] = biggest
    return labels
