"""Pydantic output contracts for cv-advisor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DataProfile(BaseModel):
    """Structured facts extracted by the profiler (Stage 1)."""

    n_rows: int
    n_cols: int
    memory_mb: float
    target_cols: list[str]
    task_type: Literal["binary", "multiclass", "multilabel", "regression"]
    n_unique_target: int | None = None
    minority_ratio: float | None = None
    minority_count: int | None = None
    gini: float | None = None
    is_imbalanced: bool = False
    target_skew: float | None = None
    target_kurtosis: float | None = None
    is_skewed_regression: bool = False
    time_col: str | None = None
    has_temporal: bool = False
    temporal_autocorr: float | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    has_spatial: bool = False
    spatial_clustered: bool = False
    group_col: str | None = None
    has_groups: bool = False
    n_groups: int | None = None
    max_group_size: int | None = None
    min_class_group_count: int | None = None
    group_time_overlap: float | None = None
    is_small: bool = False  # N < 200
    is_tiny: bool = False  # N < 50
    is_large: bool = False  # N > 100_000
    sampled_for_profiling: bool = False


class AlternativeOption(BaseModel):
    splitter: str
    when_to_use: str


class CVRecommendation(BaseModel):
    """Human-readable + executable CV recommendation (Stage 3)."""

    recommended_splitter: str = Field(description='e.g. "StratifiedGroupKFold"')
    splitter_import: str = Field(description='e.g. "from sklearn.model_selection import StratifiedGroupKFold"')
    parameters: dict[str, Any] = Field(description='e.g. {"n_splits": 5}')
    detected_patterns: list[str] = Field(default_factory=list)
    leakage_risks: list[str] = Field(default_factory=list)
    rationale: str = Field(description="Comprehensive explanation of the choice")
    code_snippet: str = Field(description="Copy-pasteable boilerplate")
    alternative_options: list[AlternativeOption] = Field(default_factory=list)
    # Machine-actionable hints consumed by CVAdvisor.get_splitter()
    group_col: str | None = None
    time_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    n_bins: int | None = None  # for binned-stratified regression
    # Display target captured at advise() time so snippet regeneration inside
    # get_splitter() never depends on its optional target_col argument.
    target_col: str | None = None
