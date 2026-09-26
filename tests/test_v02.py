import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from cv_advisor import CVAdvisor
from cv_advisor.server import app

client = TestClient(app)


def test_repeated_imbalanced():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"x": rng.normal(size=n), "fraud": np.array([0] * 990 + [1] * 10)})
    adv = CVAdvisor("fraud")
    rec = adv.advise(df, "fraud", prefer_repeated=True)
    assert rec.recommended_splitter == "RepeatedStratifiedKFold"
    splitter = adv.get_splitter(df, rec)
    splits = list(splitter.split(np.zeros((n, 1)), df["fraud"].to_numpy()))
    assert len(splits) == 5 * 5  # n_splits x n_repeats


def test_repeated_off_by_default():
    df = pd.DataFrame({"feat_a": range(200), "label": [0] * 190 + [1] * 10})
    rec = CVAdvisor("label").advise(df, "label")
    assert rec.recommended_splitter == "StratifiedKFold"


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_api_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_rules():
    r = client.get("/api/rules")
    assert r.status_code == 200
    assert r.json()["precedence"][0] == "Time"


def test_api_advise_csv():
    df = pd.DataFrame({"feat_a": range(20), "label": [0, 1] * 10})
    r = client.post(
        "/api/advise",
        files={"file": ("d.csv", _csv_bytes(df), "text/csv")},
        data={"target": "label"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommendation"]["recommended_splitter"] == "StratifiedKFold"
    assert "code_snippet" in body["recommendation"]


def test_api_advise_bad_target():
    df = pd.DataFrame({"feat_a": range(10), "label": [0, 1] * 5})
    r = client.post(
        "/api/advise",
        files={"file": ("d.csv", _csv_bytes(df), "text/csv")},
        data={"target": "nope"},
    )
    assert r.status_code == 400
