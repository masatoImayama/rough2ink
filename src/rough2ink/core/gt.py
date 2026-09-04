"""PSD レイヤー役割の手動マッピングと GT（正解）マスク生成（設計書 5節 / Epic 仕様書 5節 / T7）。

設計書 4章 A「レイヤー命名規則は一貫しているか、原稿によってバラバラか」が未確認であり、
「一貫していない前提のため自動判定には頼らない」という方針をユーザーとのヒアリングで
確定済み。そのため役割割当は常に手動マッピング（`workspace/gt/<page_id>.json`）を介する。

優先順位は **ベタ(fill) > トーン(tone) > 線(line)**。`core.decompose`（T4）と同一の規則に
する契約であり、変更しないこと（規則がずれると GT と分解結果の IoU が不当に下がる）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from psd_tools import PSDImage

from rough2ink.core.config import get_gt_dir, get_workspace_dir
from rough2ink.core.loaders.psd_loader import _find_layer  # noqa: SLF001 -- ツリー走査を再利用
from rough2ink.core.types import LayerRole

_MASK_ON = np.uint8(255)
_MASK_OFF = np.uint8(0)

# GT マスクとして合成する3役割。優先順位順（先頭が最優先）。
# `core.decompose.decompose()` の「優先順位: ベタ > トーン > 線」と揃える契約。
_MASK_ROLES: tuple[str, ...] = ("fill", "tone", "line")


class PageNotFoundError(Exception):
    """指定した page_id のページ（`workspace/pages/<page_id>/meta.json`）が存在しない。"""


class GTMappingError(Exception):
    """GT マスク生成に必要な入力（元 PSD ファイル等）が不整合。"""


def _page_dir(page_id: str) -> Path:
    return get_workspace_dir() / "pages" / page_id


def _load_meta(page_id: str) -> dict:
    meta_path = _page_dir(page_id) / "meta.json"
    if not meta_path.is_file():
        raise PageNotFoundError(f"page not found: {page_id!r}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _mapping_path(page_id: str) -> Path:
    return get_gt_dir() / f"{page_id}.json"


def save_mapping(page_id: str, mapping: dict[str, LayerRole]) -> dict[str, LayerRole]:
    """役割マッピング `{layer_path: role}` を `workspace/gt/<page_id>.json` に保存する。

    未知の `page_id`（`workspace/pages/<page_id>/meta.json` が無い）は `PageNotFoundError`。
    """
    _load_meta(page_id)  # ページの存在確認のみ（内容は使わない）
    _mapping_path(page_id).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return mapping


def load_mapping(page_id: str, *, require_page: bool = True) -> dict[str, LayerRole]:
    """保存済みの役割マッピングを取得する（未保存なら空 dict）。

    `require_page=True`（既定）では未知の `page_id` に対し `PageNotFoundError` を送出する。
    """
    if require_page:
        _load_meta(page_id)
    path = _mapping_path(page_id)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_gt_masks(page_id: str) -> dict[str, np.ndarray]:
    """保存済みマッピングから GT マスクを生成する。

    Returns:
        `{"line": mask, "fill": mask, "tone": mask, "text": mask}`。
        いずれも原寸 `(H, W)` の `uint8` 配列で値は 0/255。
        `line`/`fill`/`tone` は相互排他（優先順位: fill > tone > line）。
        `text` は評価から除外する領域として別枠で返す（他マスクとの重なりを許す）。
        `ignore` に割り当てたレイヤーはどのマスクにも含めない。
    """
    meta = _load_meta(page_id)
    mapping = load_mapping(page_id, require_page=False)
    height, width = meta["height"], meta["width"]

    source_path = Path(meta["source_path"])
    if not source_path.is_file():
        raise GTMappingError(f"source PSD not found for page {page_id!r}: {source_path}")

    layer_paths_by_role: dict[str, list[str]] = {}
    for layer_path, role in mapping.items():
        layer_paths_by_role.setdefault(role, []).append(layer_path)

    psd = PSDImage.open(str(source_path))
    canvases: dict[str, np.ndarray] = {
        role: np.zeros((height, width), dtype=bool) for role in (*_MASK_ROLES, "text")
    }

    for role, layer_paths in layer_paths_by_role.items():
        if role not in canvases:  # "ignore" および未知の役割は完全に無視する
            continue
        canvas = canvases[role]
        for layer_path in layer_paths:
            _paint_opaque_pixels(psd, layer_path, canvas)

    fill = canvases["fill"]
    tone = canvases["tone"] & ~fill
    line = canvases["line"] & ~fill & ~tone

    return {
        "line": _to_mask(line),
        "fill": _to_mask(fill),
        "tone": _to_mask(tone),
        "text": _to_mask(canvases["text"]),
    }


def _to_mask(boolean: np.ndarray) -> np.ndarray:
    return np.where(boolean, _MASK_ON, _MASK_OFF)


def _paint_opaque_pixels(psd: PSDImage, layer_path: str, canvas: np.ndarray) -> None:
    """`layer_path` の不透明画素を、ページ原寸の `canvas`（bool, 論理和で更新）へ書き込む。

    レイヤーが見つからない、ピクセルを持たない、または完全にページ外の場合は何もしない。
    """
    layer = _find_layer(psd, "", layer_path)
    if layer is None:
        return

    left, top, right, bottom = layer.bbox
    if right <= left or bottom <= top:
        return

    pil_image = layer.topil()
    if pil_image is None:
        return
    array = np.array(pil_image)
    bands = pil_image.getbands()

    if "A" in bands:
        opaque = array[:, :, bands.index("A")] > 0
    else:
        # アルファチャンネルを持たないレイヤー（背景等）は全画素を不透明として扱う。
        opaque = np.ones(array.shape[:2], dtype=bool)

    page_height, page_width = canvas.shape
    top_c, left_c = max(0, top), max(0, left)
    bottom_c = min(page_height, top + opaque.shape[0])
    right_c = min(page_width, left + opaque.shape[1])
    if top_c >= bottom_c or left_c >= right_c:
        return  # ページ範囲外

    cropped = opaque[top_c - top : bottom_c - top, left_c - left : right_c - left]
    canvas[top_c:bottom_c, left_c:right_c] |= cropped
