from cv_advisor.engine import decide
from cv_advisor.schema import DataProfile


def _base(**kw):
    d = {
        "n_rows": 1000, "n_cols": 5, "memory_mb": 1.0, "target_cols": ["y"], "task_type": "binary",
        "n_unique_target": 2, "minority_ratio": 0.4, "gini": 0.48, "is_imbalanced": False,
    }
    d.update(kw)
    return DataProfile(**d)


def test_precedence_time_over_group():
    p = _base(has_temporal=True, time_col="ts", has_groups=True, group_col="user_id", n_groups=50)
    assert decide(p).recommended_splitter == "PurgedGroupTimeSeriesSplit"


def test_group_imbalanced():
    p = _base(has_groups=True, group_col="patient_id", n_groups=50, is_imbalanced=True, minority_ratio=0.01)
    assert decide(p).recommended_splitter == "StratifiedGroupKFold"


def test_group_balanced():
    p = _base(has_groups=True, group_col="store_id", n_groups=50)
    assert decide(p).recommended_splitter == "GroupKFold"


def test_spatial():
    p = _base(has_spatial=True, lat_col="lat", lon_col="lon", task_type="regression")
    assert decide(p).recommended_splitter == "GroupKFold"


def test_imbalanced_default():
    p = _base(is_imbalanced=True, minority_ratio=0.01)
    assert decide(p).recommended_splitter == "StratifiedKFold"


def test_default_classification():
    assert decide(_base()).recommended_splitter == "StratifiedKFold"


def test_default_regression():
    assert decide(_base(task_type="regression")).recommended_splitter == "KFold"
