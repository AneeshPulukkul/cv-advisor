"""FastAPI backend for the cv-advisor web UI (v0.2).

Run:  cv-advisor serve --port 8000
UI dev server proxies /api -> http://localhost:8000
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cv_advisor import CVAdvisor
from cv_advisor.schema import CVRecommendation, DataProfile

app = FastAPI(title="cv-advisor API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_CHUNK = 1024 * 1024  # 1 MB streaming reads so oversized bodies fail fast


async def _read_limited(upload: UploadFile) -> bytes:
    """Read the upload in chunks, aborting with 413 before buffering too much."""
    declared = upload.headers.get("content-length") if upload.headers else None
    if declared is not None:
        try:
            if int(declared) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
        except ValueError:
            pass
    chunks: list[bytes] = []
    total = 0
    while True:
        piece = await upload.read(_CHUNK)
        if not piece:
            break
        total += len(piece)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
        chunks.append(piece)
    return b"".join(chunks)


def _parse_upload(filename: str, raw: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw))
        if name.endswith((".parquet", ".pq")):
            return pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc
    raise HTTPException(status_code=400, detail="Unsupported file type (use .csv or .parquet)")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.get("/api/rules")
def rules() -> dict[str, object]:
    """Decision-tree precedence + thresholds, consumed by the UI Rules page."""
    return {
        "precedence": ["Time", "Group", "Spatial", "Imbalance", "Skewed-regression", "Tiny-N", "Default i.i.d."],
        "thresholds": {
            "imbalance_minority_ratio": 0.05,
            "skew": 1.0,
            "tiny_n": 50,
            "small_n": 200,
            "large_n": 100000,
            "profile_sample_limit": 500000,
        },
        "splitters": [
            "TimeSeriesSplit",
            "PurgedGroupTimeSeriesSplit",
            "GroupKFold",
            "StratifiedGroupKFold",
            "StratifiedKFold",
            "RepeatedStratifiedKFold",
            "BinnedStratifiedKFold",
            "KFold",
            "LeaveOneOut",
        ],
    }


@app.post("/api/advise")
async def advise(
    file: UploadFile = File(...),
    target: str = Form(...),
    prefer_repeated: bool = Form(False),
) -> dict[str, object]:
    raw = await _read_limited(file)
    df = _parse_upload(file.filename or "", raw)
    advisor = CVAdvisor(target_col=target)
    try:
        profile: DataProfile = advisor.profile(df, target)
        rec: CVRecommendation = advisor.advise(df, target, prefer_repeated=prefer_repeated)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "profile": profile.model_dump(),
        "recommendation": rec.model_dump(),
        "columns": list(df.columns),
    }
