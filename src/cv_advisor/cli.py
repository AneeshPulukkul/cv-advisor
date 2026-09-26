"""Rich/Typer CLI: cv-advisor inspect data.csv --target y."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cv_advisor.advisor import CVAdvisor

app = typer.Typer(add_completion=False, help="Recommend the right sklearn CV splitter.")
console = Console()


@app.callback()
def _root() -> None:
    """cv-advisor root (keeps `inspect` as an explicit subcommand)."""


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise typer.BadParameter(f"Unsupported file type {path.suffix!r} (use .csv or .parquet)")


@app.command(name="inspect")
def inspect_cmd(
    data: Annotated[Path, typer.Argument(help="Path to data.csv / data.parquet")],
    target: Annotated[str, typer.Option("--target", "-t", help="Target column name")],
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich|json")] = "rich",
    n_rows: Annotated[int, typer.Option(help="Preview rows echoed in JSON mode")] = 5,
    repeated: Annotated[bool, typer.Option("--repeated", help="Prefer RepeatedStratifiedKFold for imbalance/small-N")] = False,
) -> None:
    """Profile DATA, print recommendation dashboard + runnable snippet."""
    df = _load(Path(data))
    advisor = CVAdvisor(target_col=target)
    profile = advisor.profile(df, target)
    rec = advisor.advise(df, target, prefer_repeated=repeated)

    if fmt == "json":
        payload = {
            "profile": profile.model_dump(),
            "recommendation": rec.model_dump(),
            "preview": df.head(n_rows).to_dict(orient="records"),
        }
        console.print_json(json.dumps(payload, default=str))
        return
    if fmt != "rich":
        raise typer.BadParameter("--format must be rich|json")

    prof_table = Table(title=f"Data profile — {data.name}", show_header=True)
    prof_table.add_column("Signal")
    prof_table.add_column("Value")
    prof_table.add_row("Rows x Cols", f"{profile.n_rows:,} x {profile.n_cols} ({profile.memory_mb:.1f} MB)")
    prof_table.add_row("Task", f"{profile.task_type} (target={target})")
    if profile.minority_ratio is not None:
        prof_table.add_row("Minority ratio", f"{profile.minority_ratio:.4f} (imbalanced={profile.is_imbalanced})")
    if profile.target_skew is not None:
        prof_table.add_row("Target skew", f"{profile.target_skew:.2f}")
    prof_table.add_row("Time", str(profile.time_col or "—"))
    prof_table.add_row("Groups", str(profile.group_col or "—") + (f" ({profile.n_groups})" if profile.n_groups else ""))
    prof_table.add_row("Spatial", f"{profile.lat_col},{profile.lon_col}" if profile.has_spatial else "—")
    console.print(prof_table)

    console.print(Panel(f"[bold green]{rec.recommended_splitter}[/]  {rec.parameters}", title="Primary recommendation"))
    if rec.detected_patterns:
        console.print(Panel("\n".join(f"• {p}" for p in rec.detected_patterns), title="Detected patterns"))
    if rec.leakage_risks:
        console.print(Panel("\n".join(f"⚠ {r}" for r in rec.leakage_risks), title="Leakage risks", style="yellow"))
    console.print(Panel(rec.rationale, title="Why"))
    console.print(Panel(rec.code_snippet, title="Run it"))


@app.command(name="serve")
def serve_cmd(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8000,
    host: Annotated[str, typer.Option("--host", help="Host to bind")] = "127.0.0.1",
) -> None:
    """Start the FastAPI backend for the web UI."""
    import uvicorn

    uvicorn.run("cv_advisor.server:app", host=host, port=port)


if __name__ == "__main__":
    app()
