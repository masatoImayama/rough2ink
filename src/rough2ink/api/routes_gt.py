"""GT（正解）レイヤー役割マッピングの保存・取得 API（Epic 仕様書 5節 / T7）。

レイヤー一覧は既に `GET /api/pages/{page_id}/layers`（T2, `routes_ingest.py`）で提供済みのため
ここでは重複実装しない。ここが持つのは役割マッピングの CRUD のみ。
GT マスクの生成（`core.gt.build_gt_masks`）は、指標算出（T8）側から直接呼ばれる想定であり、
本モジュールはそれを REST では公開しない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from psd_tools import PSDImage
from pydantic import BaseModel

from rough2ink.core import gt, imageio
from rough2ink.core.config import resolve_page_dir
from rough2ink.core.gt_suggest import RoleSuggestion, suggest_roles
from rough2ink.core.layer_stats import (
    LayerStat,
    compute_layer_stats,
    load_cached_stats,
    thumbnail_path,
)
from rough2ink.core.params import PreviewParams
from rough2ink.core.types import LayerInfo, LayerRole

router = APIRouter(prefix="/api", tags=["gt"])

# `LayerInfo.id` の形式（`lid<数字>` / `idx<数字>`）。ファイルパスに使うため検証する。
_LAYER_ID_RE = re.compile(r"(?:lid|idx)\d+")


class GTMapping(BaseModel):
    """レイヤーキー（`LayerInfo.id`）→役割の手動マッピング。

    `LayerInfo.path` は同名レイヤーがあると一意性を保証しないため（#20）、
    マッピングのキーには使わない。
    """

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


def _load_page_meta(page_id: str) -> dict:
    """`workspace/pages/<page_id>/meta.json` を読む（無ければ 404）。"""
    meta_path = resolve_page_dir(page_id) / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


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


@router.get("/pages/{page_id}/gt/suggest", response_model=list[RoleSuggestion])
def suggest_gt_roles(page_id: str) -> list[RoleSuggestion]:
    """レイヤー名から役割の**初期値**を推定する（半自動 GT 割当）。

    設計書 4章 A の通り命名規則は一貫していない前提であり、**これは正解ではない**。
    実測でも `集中線（円形）` という名前のレイヤーがページの 74% を占める黒地
    （機能的にはベタ）だった例がある。必ず `layers/stats` の墨被覆率・サムネイルと
    突き合わせて人間が確認すること。
    """
    _validate_page_id(page_id)
    meta = _load_page_meta(page_id)
    layers = [LayerInfo.model_validate(item) for item in meta.get("layers", [])]
    return suggest_roles(layers)


@router.get("/pages/{page_id}/layers/stats", response_model=list[LayerStat])
def get_layer_stats(page_id: str, refresh: bool = False) -> list[LayerStat]:
    """レイヤーごとの墨の被覆率などを返す（初回はここで算出しキャッシュする）。

    名前だけでは判断できないレイヤーを目視で切り分けるための情報。
    `refresh=true` でキャッシュを無視して再計算する。
    """
    _validate_page_id(page_id)
    meta = _load_page_meta(page_id)

    if not refresh:
        cached = load_cached_stats(page_id)
        if cached is not None:
            return cached

    source_path = Path(meta.get("source_path", ""))
    if not source_path.is_file():
        raise HTTPException(
            status_code=400, detail=f"source PSD not found for page {page_id!r}"
        )
    return compute_layer_stats(
        page_id,
        meta.get("layers", []),
        source_path,
        (meta["width"], meta["height"]),
    )


@router.get("/pages/{page_id}/layers/{layer_id}/mask")
def get_layer_mask(page_id: str, layer_id: str) -> Response:
    """レイヤー1枚の墨の位置を、プレビュー解像度のマスク PNG で返す。

    サムネイルはレイヤーの bbox しか映さないため「ページのどこを指しているか」が
    分からない。役割を割り当てるとき、そのレイヤーがページ上のどこを占めるかを
    オーバーレイで重ねて確認できるようにする。

    GT マスク生成（`core.gt.build_gt_masks`）と**同じ墨の判定**を使う。ここで見えている
    ものがそのまま GT に入る、という対応を崩さないため。
    """
    _validate_page_id(page_id)
    if not _LAYER_ID_RE.fullmatch(layer_id):
        raise HTTPException(status_code=404, detail=f"layer not found: {layer_id!r}")

    meta = _load_page_meta(page_id)
    source_path = Path(meta.get("source_path", ""))
    if not source_path.is_file():
        raise HTTPException(status_code=400, detail=f"source PSD not found for page {page_id!r}")

    canvas = np.zeros((meta["height"], meta["width"]), dtype=bool)
    psd = PSDImage.open(str(source_path))
    gt._paint_opaque_pixels(psd, layer_id, canvas, ink_threshold=gt._DEFAULT_INK_THRESHOLD)
    if not canvas.any():
        raise HTTPException(status_code=404, detail=f"layer has no ink: {layer_id!r}")

    mask = np.where(canvas, np.uint8(255), np.uint8(0))
    mask = imageio.make_preview(mask, PreviewParams().max_long_side)
    return Response(content=imageio.encode_png_bytes(mask), media_type="image/png")


@router.get("/pages/{page_id}/layers/{layer_id}/thumbnail")
def get_layer_thumbnail(page_id: str, layer_id: str) -> FileResponse:
    """レイヤーの縮小画像を返す（`layers/stats` の算出時に生成される）。"""
    _validate_page_id(page_id)
    if not _LAYER_ID_RE.fullmatch(layer_id):
        # `layer_id` はパスの一部になるため、`page_id` と同様にホワイトリストで検証する
        # （Review #21 と同じパストラバーサル対策）。
        raise HTTPException(status_code=404, detail=f"layer not found: {layer_id!r}")

    path = thumbnail_path(page_id, layer_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"thumbnail not found: {layer_id!r}")
    return FileResponse(path, media_type="image/png")
