"""Stage 2: deterministic rule-based decision engine.

Precedence (approved): Time > Group > Spatial > Imbalance > Skewed-reg > Tiny > Default.
Large-N (>100k) penalizes high-iteration splitters (LOO / repeated KFold).
"""

from __future__ import annotations

from cv_advisor.schema import AlternativeOption, CVRecommendation, DataProfile


def _adaptive_splits(n: int, minority_count: int | None, n_groups: int | None, cap: int = 5) -> int:
    k = cap
    if n_groups is not None:
        k = min(k, n_groups)
    if minority_count is not None and minority_count > 1:
        k = min(k, minority_count)
    if n < 100:
        k = min(k, 3)
    return max(2, int(k))


def _cap_by_minority(k: int, profile: DataProfile, risks: list[str]) -> int | None:
    """Cap fold count by minority-class size.

    Returns None when stratification is impossible (minority < 2 examples) so
    callers can fall back to unstratified KFold instead of emitting a splitter
    that warns and yields minority-free folds.
    """
    mc = profile.minority_count
    if mc is None:
        return k
    if mc < 2:
        risks.append(
            "Minority class has <2 examples — stratification impossible, using plain KFold"
        )
        return None
    capped = max(2, min(k, mc))
    if capped < k:
        risks.append(f"Capped folds at {capped}: minority class has only {mc} examples")
    return capped


def decide(profile: DataProfile, prefer_repeated: bool = False) -> CVRecommendation:
    patterns: list[str] = []
    risks: list[str] = []

    if profile.is_imbalanced and profile.minority_ratio is not None:
        patterns.append(f"Severe class imbalance (minority/majority={profile.minority_ratio:.3f})")
    if profile.has_temporal:
        patterns.append(f"Temporal dependency ('{profile.time_col}')")
    if profile.has_groups:
        patterns.append(f"Group entity ('{profile.group_col}', {profile.n_groups} groups)")
    if profile.has_spatial:
        patterns.append(f"Spatial coordinates ('{profile.lat_col}', '{profile.lon_col}')")
    if profile.is_skewed_regression and profile.target_skew is not None:
        patterns.append(f"Skewed regression target (skew={profile.target_skew:.2f})")
    if profile.is_tiny:
        patterns.append(f"Tiny sample (N={profile.n_rows})")
    elif profile.is_small:
        patterns.append(f"Small sample (N={profile.n_rows})")
    if profile.is_large:
        patterns.append(f"Large dataset (N={profile.n_rows:,}) — capped iterations")

    is_classification = profile.task_type in ("binary", "multiclass")

    # ---- 1. Temporal ----
    if profile.has_temporal:
        risks.append("Lookahead leakage if rows are shuffled — future must not leak into train")
        risks.append(f"Splitters follow row order: sort by '{profile.time_col}' before splitting")
        if profile.has_groups:
            gap, embargo = 1, 1
            overlap = profile.group_time_overlap
            if overlap is not None and overlap > 0.7:
                # Entities are temporally interleaved (each spans the whole
                # timeline): no split can be both entity-disjoint AND
                # chronological with non-empty folds. Forward-chaining matches
                # deployment (predict future from past, same entities recur).
                risks.append(
                    f"Entities temporally interleaved (overlap={overlap:.2f}): same "
                    f"'{profile.group_col}' values appear in both past and future — "
                    "expect within-entity optimistic bias, do not shuffle"
                )
                n_splits = min(5, max(2, profile.n_rows // 200 + 2)) if not profile.is_large else 3
                return CVRecommendation(
                    recommended_splitter="TimeSeriesSplit",
                    splitter_import="from sklearn.model_selection import TimeSeriesSplit",
                    parameters={"n_splits": int(n_splits)},
                    detected_patterns=patterns,
                    leakage_risks=risks,
                    rationale=(
                        f"Time column '{profile.time_col}' plus entity '{profile.group_col}' "
                        "detected, but entities recur across the whole timeline so a purged "
                        "group split would yield empty folds. TimeSeriesSplit preserves "
                        "deployment order; entity overlap is the documented caveat."
                    ),
                    code_snippet="",
                    alternative_options=[
                        AlternativeOption(splitter="GroupKFold", when_to_use="If unseen-entity generalization matters more than recency (accepts lookahead)"),
                    ],
                    group_col=profile.group_col,
                    time_col=profile.time_col,
                )
            n_groups = profile.n_groups or 0
            # Purged expanding window needs >=1 train group + gap + >=2 test
            # chunks, i.e. n_groups >= gap + 3 (splitter enforces the same).
            if n_groups < gap + 3:
                # Surface the infeasibility and keep entity-disjointness via GroupKFold.
                risks.append(
                    f"Only {n_groups} entities — purged time split needs >={gap + 3} groups; "
                    "falling back to GroupKFold (no chronological guarantee, "
                    "collect more entities/time span for full protection)"
                )
                return CVRecommendation(
                    recommended_splitter="GroupKFold",
                    splitter_import="from sklearn.model_selection import GroupKFold",
                    parameters={"n_splits": max(2, min(2, n_groups))},
                    detected_patterns=patterns,
                    leakage_risks=risks,
                    rationale=(
                        f"Time + entity structure exists but {n_groups} entities cannot "
                        "support a purged expanding window. GroupKFold at least keeps "
                        "entities disjoint across folds."
                    ),
                    code_snippet="",
                    alternative_options=[
                        AlternativeOption(splitter="TimeSeriesSplit", when_to_use="If entity leakage matters less than recency"),
                    ],
                    group_col=profile.group_col,
                    time_col=profile.time_col,
                )
            n_splits = max(
                2, min(_adaptive_splits(profile.n_rows, None, profile.n_groups), n_groups - gap - 1)
            )
            return CVRecommendation(
                recommended_splitter="PurgedGroupTimeSeriesSplit",
                splitter_import="from cv_advisor.splits import PurgedGroupTimeSeriesSplit",
                parameters={"n_splits": n_splits, "group_gap": gap, "embargo": embargo},
                detected_patterns=patterns,
                leakage_risks=risks,
                rationale=(
                    f"Time column '{profile.time_col}' plus entity '{profile.group_col}' detected. "
                    "Expanding-window time split with purge gap + embargo prevents both lookahead "
                    "and identity leakage. Standard shuffle/group-shuffle is forbidden here."
                ),
                code_snippet="",
                alternative_options=[
                    AlternativeOption(splitter="GroupTimeSeriesSplit", when_to_use="If you need pure sklearn with no purge/embargo"),
                    AlternativeOption(splitter="TimeSeriesSplit", when_to_use="If group leakage is negligible"),
                ],
                group_col=profile.group_col,
                time_col=profile.time_col,
            )
        n_splits = min(5, max(2, profile.n_rows // 200 + 2)) if not profile.is_large else 3
        return CVRecommendation(
            recommended_splitter="TimeSeriesSplit",
            splitter_import="from sklearn.model_selection import TimeSeriesSplit",
            parameters={"n_splits": int(n_splits)},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale=(
                f"Temporal signal in '{profile.time_col}' requires forward-chaining validation. "
                "TimeSeriesSplit trains on the past and tests on the future, matching deployment."
            ),
            code_snippet="",
            alternative_options=[
                AlternativeOption(splitter="PurgedGroupTimeSeriesSplit", when_to_use="If samples overlap in time / need embargo"),
            ],
            time_col=profile.time_col,
        )

    # ---- 2. Groups ----
    if profile.has_groups:
        risks.append("Identity leakage: same entity must not appear in both train and test")
        if is_classification and profile.is_imbalanced:
            mcg = profile.min_class_group_count
            if mcg is not None and mcg < 2:
                risks.append(
                    "A class lives in a single entity — stratified grouping impossible, "
                    "falling back to GroupKFold"
                )
            else:
                n_splits = _adaptive_splits(profile.n_rows, profile.minority_count, profile.n_groups)
                if mcg is not None and mcg < n_splits:
                    risks.append(
                        f"Capped folds at {mcg}: fewest groups in any class is {mcg}"
                    )
                    n_splits = max(2, mcg)
                return CVRecommendation(
                    recommended_splitter="StratifiedGroupKFold",
                    splitter_import="from sklearn.model_selection import StratifiedGroupKFold",
                    parameters={"n_splits": n_splits},
                    detected_patterns=patterns,
                    leakage_risks=risks,
                    rationale=(
                        f"Repeated entity '{profile.group_col}' with imbalanced classes. "
                        "StratifiedGroupKFold keeps entities disjoint across folds while preserving "
                        "class ratios — prevents identity leakage without destroying the minority signal."
                    ),
                    code_snippet="",
                    alternative_options=[
                        AlternativeOption(splitter="GroupKFold", when_to_use="If stratification fails (too few minority groups)"),
                    ],
                    group_col=profile.group_col,
                )
        n_splits = _adaptive_splits(profile.n_rows, None, profile.n_groups)
        return CVRecommendation(
            recommended_splitter="GroupKFold",
            splitter_import="from sklearn.model_selection import GroupKFold",
            parameters={"n_splits": n_splits},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale=(
                f"Repeated entity '{profile.group_col}' detected. GroupKFold guarantees "
                "entity-disjoint folds so the model is evaluated on unseen entities."
            ),
            code_snippet="",
            alternative_options=[
                AlternativeOption(splitter="StratifiedGroupKFold", when_to_use="Classification with enough per-class groups"),
            ],
            group_col=profile.group_col,
        )

    # ---- 3. Spatial ----
    if profile.has_spatial:
        risks.append("Spatial autocorrelation leakage: nearby points in train+test inflate scores")
        n_spatial = max(2, min(5, profile.n_rows // 10))
        return CVRecommendation(
            recommended_splitter="GroupKFold",
            splitter_import="from sklearn.model_selection import GroupKFold  # groups = spatial blocks from cv_advisor.spatial.make_spatial_blocks",
            parameters={"n_splits": n_spatial},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale=(
                f"Coordinates ('{profile.lat_col}', '{profile.lon_col}') detected. Rows are clustered "
                "into spatial blocks (KMeans) and GroupKFold splits on blocks, so validation measures "
                "generalization to new regions, not memorized neighborhoods."
            ),
            code_snippet="",
            alternative_options=[
                AlternativeOption(splitter="KFold(shuffle=True)", when_to_use="Only if you prove no spatial autocorrelation"),
            ],
            lat_col=profile.lat_col,
            lon_col=profile.lon_col,
        )

    # ---- 4. Extreme imbalance ----
    if is_classification and profile.is_imbalanced:
        risks.append("Random KFold may produce folds with zero minority samples")
        k = 10 if (profile.n_rows > 10_000 and not profile.is_large) else 5
        k_capped = _cap_by_minority(k, profile, risks)
        if k_capped is None:
            return CVRecommendation(
                recommended_splitter="KFold",
                splitter_import="from sklearn.model_selection import KFold",
                parameters={"n_splits": 5, "shuffle": True, "random_state": 42},
                detected_patterns=patterns,
                leakage_risks=risks,
                rationale="Single-example minority class: stratification is mathematically impossible.",
                code_snippet="",
                alternative_options=[],
            )
        k = k_capped
        if prefer_repeated and not profile.is_large and profile.task_type != "multilabel":
            return CVRecommendation(
                recommended_splitter="RepeatedStratifiedKFold",
                splitter_import="from sklearn.model_selection import RepeatedStratifiedKFold",
                parameters={"n_splits": k, "n_repeats": 5, "random_state": 42},
                detected_patterns=patterns,
                leakage_risks=risks,
                rationale=(
                    "Severe imbalance with no time/group/spatial structure. Repeated stratified "
                    "folds average out split noise so minority-class metrics are stable. "
                    "Use only at small/medium scale — repeats multiply fit cost."
                ),
                code_snippet="",
                alternative_options=[
                    AlternativeOption(splitter="StratifiedKFold", when_to_use="Large-N or tight compute budget"),
                ],
            )
        return CVRecommendation(
            recommended_splitter="StratifiedKFold",
            splitter_import="from sklearn.model_selection import StratifiedKFold",
            parameters={"n_splits": k, "shuffle": True, "random_state": 42},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale=(
                "Severe imbalance with no time/group/spatial structure. StratifiedKFold preserves "
                "class ratios in every fold so recall/precision estimates are stable."
            ),
            code_snippet="",
            alternative_options=[
                AlternativeOption(splitter="RepeatedStratifiedKFold", when_to_use="Small-N: stabilize variance (N<~2k only)"),
            ],
        )

    # ---- 5. Skewed regression ----
    if profile.task_type == "regression" and profile.is_skewed_regression:
        return CVRecommendation(
            recommended_splitter="BinnedStratifiedKFold",
            splitter_import="from cv_advisor.splits import BinnedStratifiedKFold",
            parameters={"n_splits": 5, "n_bins": 5, "shuffle": True, "random_state": 42},
            detected_patterns=patterns,
            leakage_risks=risks + ["Random folds may miss the heavy tail entirely"],
            rationale=(
                "Continuous target is heavily skewed. Quantile-binning + stratified splitting "
                "ensures tail values appear in every fold; plain KFold would give unstable RMSE."
            ),
            code_snippet="",
            alternative_options=[
                AlternativeOption(splitter="KFold(shuffle=True)", when_to_use="If target transform (log/Box-Cox) removes skew"),
            ],
            n_bins=5,
        )

    # ---- 6. Tiny sample ----
    # NOTE: every path here returns — nothing may fall through to the default
    # 5-fold below (a 20-row dataset must never receive StratifiedKFold(5)).
    if profile.is_tiny:
        if is_classification:
            risks.append("Tiny-N: fold estimates have very high variance")
            k_tiny = _cap_by_minority(
                _adaptive_splits(profile.n_rows, profile.minority_count, None, cap=5),
                profile,
                risks,
            )
            if k_tiny is None:
                return CVRecommendation(
                    recommended_splitter="KFold",
                    splitter_import="from sklearn.model_selection import KFold",
                    parameters={"n_splits": max(2, min(5, profile.n_rows)), "shuffle": True, "random_state": 42},
                    detected_patterns=patterns,
                    leakage_risks=risks,
                    rationale="Single-example minority class: stratification is mathematically impossible.",
                    code_snippet="",
                    alternative_options=[],
                )
            if prefer_repeated and not profile.is_large and profile.task_type != "multilabel":
                return CVRecommendation(
                    recommended_splitter="RepeatedStratifiedKFold",
                    splitter_import="from sklearn.model_selection import RepeatedStratifiedKFold",
                    parameters={
                        "n_splits": k_tiny,
                        "n_repeats": 10,
                        "random_state": 42,
                    },
                    detected_patterns=patterns,
                    leakage_risks=risks,
                    rationale="Tiny sample: repeats squeeze maximum signal out of few rows at acceptable cost.",
                    code_snippet="",
                    alternative_options=[AlternativeOption(splitter="LeaveOneOut", when_to_use="Unbiased error, high variance")],
                )
            return CVRecommendation(
                recommended_splitter="StratifiedKFold",
                splitter_import="from sklearn.model_selection import StratifiedKFold",
                parameters={"n_splits": k_tiny, "shuffle": True, "random_state": 42},
                detected_patterns=patterns,
                leakage_risks=risks,
                rationale="Tiny sample: repeated stratified folds balance bias/variance better than LOO here.",
                code_snippet="",
                alternative_options=[AlternativeOption(splitter="LeaveOneOut", when_to_use=f"N<={profile.n_rows}: max data use, costly + high variance")],
            )
        else:
            return CVRecommendation(
                recommended_splitter="LeaveOneOut",
                splitter_import="from sklearn.model_selection import LeaveOneOut",
                parameters={},
                detected_patterns=patterns,
                leakage_risks=["LOO has high variance; use only for tiny-N unbiased error"],
                rationale="Tiny regression sample: LeaveOneOut maximizes training data per fold.",
                code_snippet="",
                alternative_options=[AlternativeOption(splitter="KFold", when_to_use="If LOO variance is too high")],
            )

    # ---- 7. Default i.i.d. (feasibility-capped: never more folds than rows
    # or minority examples — the tiny block above returns, but small-N and
    # rare-class data can still reach here) ----
    def _kfold_fallback(reason: str) -> CVRecommendation:
        return CVRecommendation(
            recommended_splitter="KFold",
            splitter_import="from sklearn.model_selection import KFold",
            parameters={"n_splits": max(2, min(5, profile.n_rows)), "shuffle": True, "random_state": 42},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale=reason,
            code_snippet="",
            alternative_options=[],
        )

    if is_classification:
        if profile.task_type == "multilabel":
            risks.append("Multilabel: sklearn StratifiedKFold unsupported — falling back to KFold")
            return _kfold_fallback(
                "Multilabel target: plain shuffled KFold (use iterative-stratification for strict stratification)."
            )
        k_def = _cap_by_minority(
            min(5, _adaptive_splits(profile.n_rows, profile.minority_count, None)),
            profile,
            risks,
        )
        if k_def is None:
            return _kfold_fallback(
                "Single-example minority class: stratification is mathematically impossible."
            )
        return CVRecommendation(
            recommended_splitter="StratifiedKFold",
            splitter_import="from sklearn.model_selection import StratifiedKFold",
            parameters={"n_splits": k_def, "shuffle": True, "random_state": 42},
            detected_patterns=patterns,
            leakage_risks=risks,
            rationale="Standard i.i.d. classification: StratifiedKFold preserves class ratios with shuffling.",
            code_snippet="",
            alternative_options=[AlternativeOption(splitter="KFold", when_to_use="If stratification is unnecessary")],
        )
    return _kfold_fallback("Standard i.i.d. regression: shuffled KFold gives unbiased error estimates.")
