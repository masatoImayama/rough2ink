"""アップロード / ページ一覧 / プレビュー取得 API。

`POST /api/ingest` でアップロードされたファイルを 3 系統のローダ（画像 / PSD / PDF）で
`PageDocument` に正規化し、`workspace/pages/<page_id>/` に永続化する。

**巨大画像（5000px 級）を扱うため、レスポンスに原寸画像を載せない**（Epic 仕様書 9・10 節）。
原寸はサーバ側のワークスペースに保持し、`page_id` で参照する。ブラウザに返すのは
`GET /api/pages/{page_id}/preview` の縮小プレビュー（長辺 `preview.max_long_side` 以下）のみ。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from rough2ink.core import imageio
from rough2ink.core.config import get_workspace_dir
from rough2ink.core.loaders import load_image, load_pdf, load_psd
from rough2ink.core.params import PreviewParams
from rough2ink.core.types import LayerInfo, PageDocument

router = APIRouter(prefix="/api", tags=["ingest"])

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PSD_SUFFIXES = {".psd"}
_PDF_SUFFIXES = {".pdf"}


class PageSummary(BaseModel):
    """ページ一覧・アップロード応答で使う軽量なページ情報（原寸画像は含まない）。"""

    page_id: str
    filename: str
    source_kind: Literal["image", "psd", "pdf"]
    width: int
    height: int
    layer_count: int


def _pages_dir() -> Path:
    pages_dir = get_workspace_dir() / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    return pages_dir


def _page_dir(page_id: str) -> Path:
    return _pages_dir() / page_id


def _uploads_dir() -> Path:
    uploads_dir = _pages_dir() / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir


def _save_page_document(doc: PageDocument, *, filename: str) -> PageSummary:
    """`PageDocument` を `workspace/pages/<page_id>/` に永続化する。

    - `page.png`: 原寸グレースケール（解析はこの原寸で行う。縮小しない）
    - `preview.png`: ブラウザ表示用の縮小プレビュー（長辺 `preview.max_long_side` 以下）
    - `meta.json`: 元ファイル名・形式・サイズ・レイヤー一覧（PSD のみ）
    """
    page_dir = _page_dir(doc.page_id)
    page_dir.mkdir(parents=True, exist_ok=True)

    gray = doc.as_array()
    imageio.write_gray_png(page_dir / "page.png", gray)

    preview = imageio.make_preview(gray, PreviewParams().max_long_side)
    imageio.write_gray_png(page_dir / "preview.png", preview)

    meta = {
        "page_id": doc.page_id,
        "filename": filename,
        "source_path": str(doc.source_path),
        "source_kind": doc.source_kind,
        "width": doc.width,
        "height": doc.height,
        "layers": [layer.model_dump(mode="json") for layer in doc.layers],
    }
    (page_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return PageSummary(
        page_id=doc.page_id,
        filename=filename,
        source_kind=doc.source_kind,
        width=doc.width,
        height=doc.height,
        layer_count=len(doc.layers),
    )


@router.post("/ingest", response_model=list[PageSummary])
async def ingest(file: UploadFile = File(...)) -> list[PageSummary]:
    """ファイルをアップロードし、`PageDocument` に正規化して `workspace/pages/` に保存する。

    画像 / PSD は 1 ページ、PDF は複数ページに展開されうるため常にリストで返す。
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    # 元ファイル名（日本語・空白を含みうる）の衝突を避けるため uuid を挟むが、
    # 拡張子・元ファイル名自体は保つ（`pathlib.Path` で扱うのでそのまま通る）。
    tmp_path = _uploads_dir() / f"{uuid.uuid4().hex}_{filename}"
    tmp_path.write_bytes(await file.read())

    try:
        if suffix in _IMAGE_SUFFIXES:
            docs = [load_image(tmp_path)]
        elif suffix in _PSD_SUFFIXES:
            docs = [load_psd(tmp_path)]
        elif suffix in _PDF_SUFFIXES:
            docs = load_pdf(tmp_path)
        else:
            raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [_save_page_document(doc, filename=filename) for doc in docs]


@router.get("/pages", response_model=list[PageSummary])
def list_pages() -> list[PageSummary]:
    """永続化済みの全ページを一覧で返す。"""
    summaries: list[PageSummary] = []
    for meta_path in sorted(_pages_dir().glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summaries.append(
            PageSummary(
                page_id=meta["page_id"],
                filename=meta["filename"],
                source_kind=meta["source_kind"],
                width=meta["width"],
                height=meta["height"],
                layer_count=len(meta.get("layers", [])),
            )
        )
    return summaries


@router.get("/pages/{page_id}/preview")
def get_preview(page_id: str) -> Response:
    """ページのプレビュー画像（長辺 `preview.max_long_side` 以下）を PNG で返す。"""
    preview_path = _page_dir(page_id) / "preview.png"
    if not preview_path.is_file():
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}")
    return Response(content=preview_path.read_bytes(), media_type="image/png")


@router.get("/pages/{page_id}/layers", response_model=list[LayerInfo])
def get_layers(page_id: str) -> list[LayerInfo]:
    """PSD のレイヤー一覧を返す（画像 / PDF から取り込んだページは空リスト）。"""
    meta_path = _page_dir(page_id) / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail=f"page not found: {page_id!r}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [LayerInfo.model_validate(layer) for layer in meta.get("layers", [])]
