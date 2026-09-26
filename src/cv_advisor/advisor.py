"""Stage 3 orchestrator: profile -> decide -> executable splitter + snippet."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from cv_advisor import engine as engine_mod
from cv_advisor.profiler import profile_dataframe
from cv_advisor.schema import CVRecommendation, DataProfile
from cv_advisor.spatial import make_spatial_blocks
from cv_advisor.splits import BinnedStratifiedKFold, PurgedGroupTimeSeriesSplit


def _build_code_snippet(rec: CVRecommendation, target: str) -> str:
    name = rec.recommended_splitter
    p = rec.parameters
    if name == "PurgedGroupTimeSeriesSplit":
        return (
            "import pandas as pd\n"
            "from cv_advisor import CVAdvisor\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            f"advisor = CVAdvisor(target_col={target!r})\n"
            f"rec = advisor.advise(df, target_col={target!r})\n"
            "# Sort by time FIRST — splits are chronological, not positional.\n"
            "df_sorted = df.sort_values(rec.time_col).reset_index(drop=True)\n"
            f"X = df_sorted.drop(columns=[{target!r}])\n"
            f"y = df_sorted[{target!r}]\n"
            "# Pass the frame (it carries the time col) so the splitter can purge\n"
            "# late rows of early groups; groups keep entities disjoint.\n"
            "splitter = advisor.get_splitter(df_sorted, rec)\n"
            "cv = splitter.split(df_sorted, y, groups=df_sorted[rec.group_col])\n"
            "scores = cross_val_score(model, X, y, cv=cv)"
        )
    if name == "GroupKFold" and rec.lat_col:
        return (
            "from sklearn.model_selection import GroupKFold, cross_val_score\n"
            "from cv_advisor.spatial import make_spatial_blocks\n\n"
            f"blocks = make_spatial_blocks(df, lat_col={rec.lat_col!r}, lon_col={rec.lon_col!r}, "
            f"n_splits={p.get('n_splits', 5)})\n"
            f"cv = GroupKFold(n_splits={p.get('n_splits', 5)})\n"
            "scores = cross_val_score(model, X, y, cv=cv.split(X, y, groups=blocks))"
        )
    if name in ("GroupKFold", "StratifiedGroupKFold"):
        return (
            f"{rec.splitter_import}\nfrom sklearn.model_selection import cross_val_score\n\n"
            f"cv = {name}(n_splits={p.get('n_splits', 5)})\n"
            f"scores = cross_val_score(model, X, y, cv=cv.split(X, y, groups=df[{rec.group_col!r}]))"
        )
    if name == "BinnedStratifiedKFold":
        return (
            "from cv_advisor.splits import BinnedStratifiedKFold\n"
            "from sklearn.model_selection import cross_val_score\n\n"
            f"cv = BinnedStratifiedKFold(n_splits={p.get('n_splits', 5)}, n_bins={p.get('n_bins', 5)}, "
            "shuffle=True, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv.split(X, y))"
        )
    if name == "TimeSeriesSplit":
        sort_block = ""
        if rec.time_col:
            sort_block = (
                "# Sort by time FIRST — TimeSeriesSplit follows row order, it does not sort.\n"
                f"df_sorted = df.sort_values({rec.time_col!r}).reset_index(drop=True)\n"
                f"X = df_sorted.drop(columns=[{target!r}])\n"
                f"y = df_sorted[{target!r}]\n"
            )
        return (
            "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n\n"
            f"{sort_block}"
            f"cv = TimeSeriesSplit(n_splits={p.get('n_splits', 5)})\n"
            "scores = cross_val_score(model, X, y, cv=cv.split(X))  # NO shuffle"
        )
    if name == "LeaveOneOut":
        return (
            "from sklearn.model_selection import LeaveOneOut, cross_val_score\n\n"
            "cv = LeaveOneOut()\n"
            "scores = cross_val_score(model, X, y, cv=cv.split(X))"
        )
    # StratifiedKFold / KFold / RepeatedStratifiedKFold default
    params = ", ".join(f"{k}={v!r}" for k, v in p.items())
    return (
        f"{rec.splitter_import}\nfrom sklearn.model_selection import cross_val_score\n\n"
        f"cv = {name}({params})\n"
        "scores = cross_val_score(model, X, y, cv=cv.split(X, y))"
    )


class CVAdvisor:
    """Main entry point: advise() + get_splitter()."""

    def __init__(self, target_col: str | list[str] | None = None):
        self.default_target = target_col

    def advise(
        self,
        data: pd.DataFrame,
        target_col: str | list[str] | None = None,
        prefer_repeated: bool = False,
    ) -> CVRecommendation:
        target = target_col or self.default_target
        if target is None:
            raise ValueError("target_col is required")
        if len(data) < 2:
            raise ValueError("cv-advisor needs at least 2 rows to recommend a splitter")
        profile: DataProfile = profile_dataframe(data, target)
        rec = engine_mod.decide(profile, prefer_repeated=prefer_repeated)
        if isinstance(target, list):
            t_repr = target[0]
        else:
            t_repr = target
        rec.target_col = str(t_repr)
        rec.code_snippet = _build_code_snippet(rec, str(t_repr))
        return rec

    def get_splitter(self, data: pd.DataFrame, rec: CVRecommendation, target_col: str | None = None):
        """Return a configured sklearn-compatible CV splitter.

        For spatial GroupKFold the returned splitter yields block-aware splits directly
        (no need to pass groups manually). Blocks are bound to ``data``'s index: pass a
        DataFrame with a matching index and labels follow the rows even if filtered or
        reordered; otherwise X must have exactly the original row count and order, else
        a clear ValueError is raised (pass ``groups=`` explicitly in that case).
        """
        name = rec.recommended_splitter
        p = rec.parameters
        if name == "PurgedGroupTimeSeriesSplit":
            return PurgedGroupTimeSeriesSplit(
                n_splits=int(p.get("n_splits", 5)),
                group_gap=int(p.get("group_gap", 1)),
                embargo=int(p.get("embargo", 0)),
                time_col=rec.time_col,
            )
        if name == "BinnedStratifiedKFold":
            return BinnedStratifiedKFold(
                n_splits=int(p.get("n_splits", 5)),
                n_bins=int(rec.n_bins or p.get("n_bins", 5)),
                shuffle=bool(p.get("shuffle", True)),
                random_state=int(p.get("random_state", 42)),
            )
        if name == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=int(p.get("n_splits", 5)))
        if name == "StratifiedGroupKFold":
            return StratifiedGroupKFold(n_splits=int(p.get("n_splits", 5)))
        if name == "GroupKFold":
            if rec.lat_col and rec.lon_col:
                requested = int(p.get("n_splits", 5))
                labels = make_spatial_blocks(data, rec.lat_col, rec.lon_col, n_splits=requested)
                n_blocks = len(set(labels.tolist()))
                if n_blocks < requested:
                    # Duplicate coordinates collapsed the clustering: downgrade the
                    # fold count rather than letting GroupKFold fail opaquely.
                    effective = max(2, n_blocks)
                    labels = make_spatial_blocks(data, rec.lat_col, rec.lon_col, n_splits=effective)
                    n_blocks = len(set(labels.tolist()))
                    if n_blocks < 2:
                        raise ValueError(
                            "Spatial coordinates collapse to a single block; "
                            "spatial CV is infeasible — use KFold instead."
                        )
                    rec.parameters["n_splits"] = effective
                    rec.detected_patterns.append(
                        f"Spatial blocks collapsed to {n_blocks} distinct clusters — folds reduced to {effective}"
                    )
                    # The snippet was rendered at advise() time with the original
                    # count — rebuild it so displayed code matches the splitter.
                    # Prefer the rec's own target (captured at advise time) over
                    # the optional argument, which the common call omits.
                    rec.code_snippet = _build_code_snippet(rec, target_col or rec.target_col or "")
                    requested = effective
                blocks = pd.Series(labels, index=data.index)
                base = GroupKFold(n_splits=requested)

                class _SpatialBlockCV(base.__class__):  # thin wrapper binding blocks
                    def split(self, X, y=None, groups=None):
                        if groups is not None:
                            yield from base.split(X, y, groups=groups)
                            return
                        if hasattr(X, "index") and not blocks.index.equals(X.index):
                            try:
                                aligned = blocks.reindex(X.index)
                            except Exception:
                                aligned = None
                            if aligned is None or aligned.isna().any():
                                raise ValueError(
                                    "X's index does not match the profiled DataFrame: pass "
                                    "groups= explicitly (e.g. make_spatial_blocks on your "
                                    "filtered frame) or keep the original rows/order."
                                )
                            yield from base.split(X, y, groups=aligned.to_numpy())
                            return
                        n = len(X) if hasattr(X, "__len__") else len(blocks)
                        if n != len(blocks):
                            raise ValueError(
                                f"X has {n} rows but spatial blocks were built for "
                                f"{len(blocks)}: pass groups= explicitly or keep the "
                                "original rows/order."
                            )
                        yield from base.split(X, y, groups=blocks.to_numpy())

                    def get_n_splits(self, X=None, y=None, groups=None) -> int:
                        return requested

                splitter = _SpatialBlockCV(n_splits=requested)
                return splitter
            return GroupKFold(n_splits=int(p.get("n_splits", 5)))
        if name == "StratifiedKFold":
            return StratifiedKFold(
                n_splits=int(p.get("n_splits", 5)),
                shuffle=bool(p.get("shuffle", True)),
                random_state=p.get("random_state", 42),
            )
        if name == "RepeatedStratifiedKFold":
            return RepeatedStratifiedKFold(
                n_splits=int(p.get("n_splits", 5)),
                n_repeats=int(p.get("n_repeats", 5)),
                random_state=p.get("random_state", 42),
            )
        if name == "LeaveOneOut":
            return LeaveOneOut()
        # KFold default (incl. multilabel fallback)
        return KFold(
            n_splits=int(p.get("n_splits", 5)),
            shuffle=bool(p.get("shuffle", True)),
            random_state=p.get("random_state", 42),
        )

    def profile(self, data: pd.DataFrame, target_col: str | list[str] | None = None) -> DataProfile:
        target = target_col or self.default_target
        if target is None:
            raise ValueError("target_col is required")
        return profile_dataframe(data, target)
