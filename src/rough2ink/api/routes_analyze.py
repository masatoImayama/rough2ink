"""単一ページ解析 API（Epic 仕様書 7・9 節）。

`POST /api/pages/{page_id}/analyze` はパラメータ（`AnalysisParams`）を受け取り、
品質ゲート・分解器（線/ベタ/トーン）・コマ分割・フキダシ検出の解析一式を実行して
オーバーレイ表示に必要な情報を返す。

**解析自体は必ず原寸で行う**（Epic 仕様書 9・10 節）。縮小はレスポンス生成の一点のみで、
コマポリゴンの座標は**プレビュー座標系に変換**して返す。**レスポンスに原寸画像は載せない**
（5000px 級のため）。線/ベタ/トーン/フキダシの各マスクは原寸のまま `workspace/pages/<page_id>/masks/`
へ永続化し（後続のバッチ書き出し #12 で再利用する）、ブラウザへは
`GET /api/pages/{page_id}/mask/{kind}` の個別エンドポイントからプレビュー解像度で返す。

GT（#7, #8）が用意されている場合、`workspace/gt/<page_id>.json` に役割マッピングが
保存されていれば `metrics` に IoU/Precision/Recall/F1（役割ごと）を算出して返す
（`core.metrics.compute_page_decompose_metrics` に集約された規則: 評価対象から `text` 領域と
フキダシ損失マスクを除外する。`core.batch` の report.json/report.md も同じ関数を呼ぶため
除外規則が食い違うことはない ── Review #25）。マッピング未保存のページでは常に `None`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from rough2ink.core import imageio
from rough2ink.core.balloons import detect_balloons
from rough2ink.core.config import get_workspace_dir
from rough2ink.core.decompose import decompose
from rough2ink.core.metrics import compute_page_decompose_metrics
from rough2ink.core.panels import detect_panels
from rough2ink.core.params import AnalysisParams, PreviewParams
from rough2ink.core.quality import evaluate_quality
from rough2ink.core.types import BBox, PanelInfo, QualityReport

router = APIRouter(prefix="/api", tags=["analyze"])

_MASK_KINDS: tuple[str, ...] = ("line", "fill", "tone", "balloon")
MaskKind = Literal["line", "fill", "tone", "balloon"]


class MaskUrls(BaseModel):
    """オーバーレイ表示用マスク PNG（プレビュー解像度）を取得する個別エンドポイントの URL。"""

    line: str
    fill: str
    tone: str
    balloon: str


class AnalysisResult(BaseModel):
    """`POST /api/pages/{page_id}/analyze` の応答。

    **原寸画像・原寸マスクは含まない**（5000px 級のため）。マスクは `mask_urls` の
    個別エンドポイントから取得する。`panels` のポリゴン/bbox 座標は
    `preview_width` x `preview_height` の座標系に変換済み。
    """

    page_id: str
    quality: QualityReport
    panels: list[PanelInfo]
    preview_width: int
    preview_height: int
    mask_urls: MaskUrls
    # GT マッピング（`workspace/gt/<page_id>.json`）が保存されている場合のみ算出する。
    # `{role: {"iou": ..., "precision": ..., "recall": ..., "f1": ...}}`。未保存なら None。
    metrics: dict[str, Any] | None = None


def _pages_dir() -> Path:
    return get_workspace_dir() / "pages"


def _page_dir(page_id: str) -> Path:
    return _pages_dir() / page_id


def _masks_dir(page_id: str) -> Path:
    masks_dir = _page_dir(page_id) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    return masks_dir


def _load_page_gray(page_id: str):
    page_png = _page_dir(page_id) / "page.png"
    if not page_png.is_file():
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}")
    return imageio.read_gray(page_png)


def _load_text_rects(page_id: str) -> list[BBox]:
    """PSD の `kind == "type"` レイヤー bbox 一覧を返す（画像/PDF は空リスト）。"""
    meta_path = _page_dir(page_id) / "meta.json"
    if not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [
        tuple(layer["bbox"])
        for layer in meta.get("layers", [])
        if layer.get("kind") == "type" and layer.get("bbox")
    ]


def _scale_polygon(
    polygon: list[tuple[float, float]], scale_x: float, scale_y: float
) -> list[tuple[float, float]]:
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def _scale_bbox(bbox: BBox | None, scale_x: float, scale_y: float) -> BBox | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return (
        round(x * scale_x),
        round(y * scale_y),
        round(w * scale_x),
        round(h * scale_y),
    )


@router.post("/pages/{page_id}/analyze", response_model=AnalysisResult)
def analyze_page(
    page_id: str,
    params: AnalysisParams = Body(default_factory=AnalysisParams),
) -> AnalysisResult:
    """パラメータを受け取り、品質・分解・コマ・フキダシの解析一式を実行する。"""
    gray = _load_page_gray(page_id)
    height, width = gray.shape[:2]

    quality = evaluate_quality(gray, params.quality)
    masks = decompose(gray, params)
    panels = detect_panels(gray, params.panel)
    balloon_mask = detect_balloons(
        gray, params.balloon, text_rects=_load_text_rects(page_id) or None
    )
    masks["balloon"] = balloon_mask
    metrics = compute_page_decompose_metrics(page_id, masks, balloon_mask)

    masks_dir = _masks_dir(page_id)
    for kind in _MASK_KINDS:
        imageio.write_mask_png(masks_dir / f"{kind}.png", masks[kind])

    preview = imageio.make_preview(gray, params.preview.max_long_side)
    preview_height, preview_width = preview.shape[:2]
    scale_x = preview_width / width
    scale_y = preview_height / height

    scaled_panels = [
        panel.model_copy(
            update={
                "polygon": _scale_polygon(panel.polygon, scale_x, scale_y),
                "bbox": _scale_bbox(panel.bbox, scale_x, scale_y),
            }
        )
        for panel in panels
    ]

    mask_urls = MaskUrls(**{kind: f"/api/pages/{page_id}/mask/{kind}" for kind in _MASK_KINDS})

    return AnalysisResult(
        page_id=page_id,
        quality=quality,
        panels=scaled_panels,
        preview_width=preview_width,
        preview_height=preview_height,
        mask_urls=mask_urls,
        metrics=metrics,
    )


@router.get("/pages/{page_id}/mask/{kind}")
def get_mask(page_id: str, kind: MaskKind, preview: bool = Query(True)) -> Response:
    """`analyze` で永続化されたマスク PNG を返す。

    `preview=1`（既定）はオーバーレイ表示用にプレビュー解像度へ縮小して返す。
    `preview=0` は原寸のまま返す（5000px 級になりうるため、サーバ内部用途を想定）。
    """
    mask_path = _page_dir(page_id) / "masks" / f"{kind}.png"
    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"mask not found (call analyze first): page_id={page_id!r} kind={kind!r}",
        )
    mask = imageio.read_mask_png(mask_path)
    if preview:
        mask = imageio.make_preview(mask, PreviewParams().max_long_side)
    return Response(content=imageio.encode_png_bytes(mask), media_type="image/png")
