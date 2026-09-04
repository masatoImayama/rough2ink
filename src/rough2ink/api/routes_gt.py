"""GT（正解）レイヤー役割マッピングの保存・取得 API（Epic 仕様書 5節 / T7）。

レイヤー一覧は既に `GET /api/pages/{page_id}/layers`（T2, `routes_ingest.py`）で提供済みのため
ここでは重複実装しない。ここが持つのは役割マッピングの CRUD のみ。
GT マスクの生成（`core.gt.build_gt_masks`）は、指標算出（T8）側から直接呼ばれる想定であり、
本モジュールはそれを REST では公開しない。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rough2ink.core import gt
from rough2ink.core.config import resolve_page_dir
from rough2ink.core.types import LayerRole

router = APIRouter(prefix="/api", tags=["gt"])


class GTMapping(BaseModel):
    """レイヤーパス（`LayerInfo.path`）→役割の手動マッピング。"""

    mapping: dict[str, LayerRole]


def _validate_page_id(page_id: str) -> None:
    """`page_id` のホワイトリスト検証のみ行う（Review #21: パストラバーサル対策）。

    実際のディレクトリ解決・存在確認は `core.gt`（`_page_dir`/`_load_meta`）が担うため
    ここでは path 解決は行わず、URL から渡された `page_id` に不正な文字（バックスラッシュ等）
    が含まれていないことだけを確認して 404 に変換する。
    """
    try:
        resolve_page_dir(page_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}") from exc


@router.put("/pages/{page_id}/gt", response_model=GTMapping)
def put_gt_mapping(page_id: str, body: GTMapping) -> GTMapping:
    """役割マッピングを `workspace/gt/<page_id>.json` に保存する。"""
    _validate_page_id(page_id)
    try:
        saved = gt.save_mapping(page_id, body.mapping)
    except gt.PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GTMapping(mapping=saved)


@router.get("/pages/{page_id}/gt", response_model=GTMapping)
def get_gt_mapping(page_id: str) -> GTMapping:
    """保存済みの役割マッピングを取得する（未保存なら空 mapping）。"""
    _validate_page_id(page_id)
    try:
        mapping = gt.load_mapping(page_id)
    except gt.PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GTMapping(mapping=mapping)
