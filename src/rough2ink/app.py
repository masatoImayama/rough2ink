"""FastAPI アプリのエントリポイント。

`uv run uvicorn rough2ink.app:app` で起動する。
`web/` を静的配信し、API ルータはここに `include_router` で登録する
（後続タスクの受け口。現時点では `/api/health` `/api/params/defaults` のみ）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rough2ink.api import routes_ingest
from rough2ink.core.params import AnalysisParams

# src/rough2ink/app.py から見て 2 階層上がプロジェクトルート
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="rough2ink 前処理検証ツール", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    """死活監視用エンドポイント。"""
    return {"status": "ok"}


@app.get("/api/params/defaults")
def params_defaults() -> AnalysisParams:
    """`AnalysisParams` の既定値を返す。"""
    return AnalysisParams()


app.include_router(routes_ingest.router)

# 残りのルータは後続タスクで追加していく。
# from rough2ink.api import routes_analyze, routes_gt, routes_presets, routes_batch
# app.include_router(routes_analyze.router)
# app.include_router(routes_gt.router)
# app.include_router(routes_presets.router)
# app.include_router(routes_batch.router)

# 静的配信は API ルートより後にマウントする（先に登録した具体的なパスが優先される）。
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
