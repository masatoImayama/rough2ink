"""バッチ処理 API（Epic 仕様書 8節 / T11）。

`POST /api/batch` はフォルダ一括処理をバックグラウンドスレッドで開始し、即座に `job_id` を返す。
バッチ処理は数十ページ規模で長時間かかりうるため、進捗は `GET /api/batch/{job_id}` の
ポーリングで取得する（`core.batch.run_batch` の `progress_callback` をジョブ状態へ橋渡しする）。

ジョブ状態はプロセス内メモリに保持する（PoC のため永続化しない。プロセス再起動で失われる）。
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rough2ink.core.batch import BatchProgress, run_batch
from rough2ink.core.config import get_out_dir
from rough2ink.core.params import AnalysisParams
from rough2ink.core.presets import load_preset

router = APIRouter(prefix="/api", tags=["batch"])

JobStatus = Literal["running", "done", "error"]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class BatchStartRequest(BaseModel):
    """バッチ処理開始リクエスト。"""

    input_dir: str
    out_dir: str | None = None
    preset: str | None = None


class BatchJobStatus(BaseModel):
    """バッチジョブの現在状態（`GET /api/batch/{job_id}` の応答）。"""

    job_id: str
    status: JobStatus
    processed: int
    total: int
    current_page_id: str | None = None
    report: dict | None = None
    error: str | None = None


def _new_job_state() -> dict:
    return {
        "status": "running",
        "processed": 0,
        "total": 0,
        "current_page_id": None,
        "report": None,
        "error": None,
    }


def _run_job(job_id: str, input_dir: Path, out_dir: Path, params: AnalysisParams) -> None:
    def on_progress(progress: BatchProgress) -> None:
        with _jobs_lock:
            _jobs[job_id]["processed"] = progress.index
            _jobs[job_id]["total"] = progress.total
            _jobs[job_id]["current_page_id"] = progress.page_id

    try:
        report = run_batch(input_dir, out_dir, params, progress_callback=on_progress)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["report"] = report
    except Exception as exc:  # noqa: BLE001 -- バックグラウンドジョブの失敗はジョブ状態に残す
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


@router.post("/batch", response_model=BatchJobStatus)
def start_batch(body: BatchStartRequest) -> BatchJobStatus:
    """バッチ処理をバックグラウンドで開始する。"""
    input_dir = Path(body.input_dir)
    if not input_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"input_dir not found: {body.input_dir!r}")

    params = AnalysisParams()
    if body.preset is not None:
        try:
            params = load_preset(body.preset)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    out_dir = Path(body.out_dir) if body.out_dir else get_out_dir()

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = _new_job_state()

    thread = threading.Thread(target=_run_job, args=(job_id, input_dir, out_dir, params), daemon=True)
    thread.start()

    with _jobs_lock:
        return BatchJobStatus(job_id=job_id, **_jobs[job_id])


@router.get("/batch/{job_id}", response_model=BatchJobStatus)
def get_batch_status(job_id: str) -> BatchJobStatus:
    """バッチジョブの進捗・完了レポートを取得する。"""
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"batch job not found: {job_id!r}")
        return BatchJobStatus(job_id=job_id, **state)
