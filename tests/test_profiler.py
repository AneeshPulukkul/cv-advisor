import numpy as np
import pandas as pd

from cv_advisor.profiler import profile_dataframe


def test_binary_imbalance_flag():
    y = np.array([0] * 995 + [1] * 5)
    df = pd.DataFrame({"x": np.random.randn(1000), "y": y})
    p = profile_dataframe(df, "y")
    assert p.task_type == "binary"
    assert p.is_imbalanced
    assert p.minority_ratio is not None and p.minority_ratio < 0.05


def test_regression_skew_flag():
    y = np.random.exponential(scale=2.0, size=500)
    df = pd.DataFrame({"x": np.random.randn(500), "price": y})
    p = profile_dataframe(df, "price")
    assert p.task_type == "regression"
    assert p.is_skewed_regression


def test_time_detection():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="D"), "y": np.random.randn(100)})
    p = profile_dataframe(df, "y")
    assert p.has_temporal and p.time_col == "timestamp"


def test_group_detection():
    df = pd.DataFrame({"patient_id": [f"p{i // 5}" for i in range(100)], "y": np.random.randint(0, 2, 100)})
    p = profile_dataframe(df, "y")
    assert p.has_groups and p.group_col == "patient_id"
    assert p.n_groups == 20


def test_spatial_detection():
    df = pd.DataFrame({"lat": np.random.uniform(32, 42, 100), "lon": np.random.uniform(-124, -114, 100), "y": np.random.randn(100)})
    p = profile_dataframe(df, "y")
    assert p.has_spatial


def test_multilabel():
    df = pd.DataFrame({"x": range(10), "a": [0, 1] * 5, "b": [1, 0] * 5})
    p = profile_dataframe(df, ["a", "b"])
    assert p.task_type == "multilabel"
