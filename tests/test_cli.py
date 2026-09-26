import pandas as pd
from typer.testing import CliRunner

from cv_advisor.cli import app

runner = CliRunner()


def test_cli_inspect_csv(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [0, 1, 0, 1, 0, 1]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    r = runner.invoke(app, ["inspect", str(p), "--target", "y"])
    assert r.exit_code == 0, r.output
    assert "StratifiedKFold" in r.output or "KFold" in r.output


def test_cli_json(tmp_path):
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    r = runner.invoke(app, ["inspect", str(p), "--target", "y", "--format", "json"])
    assert r.exit_code == 0, r.output
    assert "recommended_splitter" in r.output
