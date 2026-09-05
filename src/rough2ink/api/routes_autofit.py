"""GT に対するパラメータ自動フィッティング API。

設計書 3章はレイヤー付き原稿を「分解器の教師データ兼検証データ」と位置づけている。
正解があるのだから閾値は目視で合わせるのではなく GT に対して当てにいく、というのが
本来の運用であり、その入口がここ。

探索には数十秒〜数分かかるため、バッチ処理（`routes_batch.py`）と同じく
バックグラウンドで走らせて進捗をポーリングする。
"""

from __future__ import annotations

import threading
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rough2ink.core import gt
from rough2ink.core.autofit import FitProgress, build_samples, fit_params
from rough2ink.core.balloons import detect_balloons
from rough2ink.core.config import resolve_page_dir
from rough2ink.core.params import AnalysisParams

router = APIRouter(prefix="/api", tags=["autofit"])

JobStatus = Literal["running", "done", "error"]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class AutofitRequest(BaseModel):
    """自動フィッティングの開始リクエスト。"""

    params: AnalysisParams | None = None
    crop_size: int = 640
    max_crops: int = 4
    passes: int = 2


class AutofitJobStatus(BaseModel):
    """探索ジョブの現在状態。"""

    job_id: str
    status: JobStatus
    evaluated: int = 0
    total: int = 0
    current_parameter: str | None = None
    best_score: float | None = None
    baseline_score: float | None = None
    params: AnalysisParams | None = None
    changed: dict[str, list[float]] | None = None
    baseline_metrics: dict | None = None
    best_metrics: dict | None = None
    sample_count: int | None = None
    error: str | None = None


def _new_job_state() -> dict:
    return {
        "status": "running",
        "evaluated": 0,
        "total": 0,
        "current_parameter": None,
        "best_score": None,
        "baseline_score": None,
        "params": None,
        "changed": None,
        "baseline_metrics": None,
        "best_metrics": None,
        "sample_count": None,
        "error": None,
    }


def _run_job(job_id: str, page_id: str, body: AutofitRequest) -> None:
    def on_progress(progress: FitProgress) -> None:
        with _jobs_lock:
            state = _jobs[job_id]
            state["evaluated"] = progress.evaluated
            state["total"] = progress.total
            state["current_parameter"] = progress.parameter
            state["best_score"] = progress.best_score

    try:
        page_path = resolve_page_dir(page_id) / "page.png"
        gray = cv2.imdecode(
            np.fromfile(str(page_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if gray is None:
            raise RuntimeError(f"failed to read page image: {page_path}")

        base_params = body.params or AnalysisParams()
        gt_masks = gt.build_gt_masks(page_id)
        balloon = detect_balloons(gray, base_params.balloon)
        exclude = (gt_masks["text"] > 0) | (balloon > 0)

        samples = build_samples(
            gray,
            gt_masks,
            exclude,
            crop_size=body.crop_size,
            max_crops=body.max_crops,
        )
        if not samples:
            raise RuntimeError(
                "GT に中身のあるクロップが見つかりません。"
                "レイヤーの役割割当が空か、割り当てたレイヤーに墨がありません。"
            )

        result = fit_params(
            samples, base_params, passes=body.passes, progress_callback=on_progress
        )
        with _jobs_lock:
            state = _jobs[job_id]
            state["status"] = "done"
            state["params"] = result.params
            state["baseline_score"] = result.baseline_score
            state["best_score"] = result.best_score
            state["changed"] = {key: [before, after] for key, (before, after) in result.changed.items()}
            state["baseline_metrics"] = result.baseline_metrics
            state["best_metrics"] = result.best_metrics
            state["sample_count"] = result.sample_count
    except Exception as exc:  # noqa: BLE001 -- 失敗はジョブ状態に残して UI へ返す
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


@router.post("/pages/{page_id}/autofit", response_model=AutofitJobStatus)
def start_autofit(page_id: str, body: AutofitRequest | None = None) -> AutofitJobStatus:
    """GT に対するパラメータ探索をバックグラウンドで開始する。"""
    try:
        resolve_page_dir(page_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}") from exc

    try:
        mapping = gt.load_mapping(page_id)
    except gt.PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not mapping:
        raise HTTPException(
            status_code=400,
            detail="GT 役割マッピングが未保存です。先にレイヤーの役割を割り当ててください。",
        )

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = _new_job_state()

    thread = threading.Thread(
        target=_run_job, args=(job_id, page_id, body or AutofitRequest()), daemon=True
    )
    thread.start()

    with _jobs_lock:
        return AutofitJobStatus(job_id=job_id, **_jobs[job_id])


@router.get("/autofit/{job_id}", response_model=AutofitJobStatus)
def get_autofit_status(job_id: str) -> AutofitJobStatus:
    """探索ジョブの進捗・結果を取得する。"""
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"autofit job not found: {job_id!r}")
        return AutofitJobStatus(job_id=job_id, **state)
