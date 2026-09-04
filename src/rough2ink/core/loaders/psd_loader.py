"""PSD ローダ。合成画像（フラット化結果）とレイヤーツリー（階層パス付き）の両方を取得する。

`kind == "type"`（テキストレイヤー）の矩形はフキダシ検出のヒントとして必ず保持する
（Epic 仕様書 4-A・4-E 節）。
"""

from __future__ import annotations

import itertools
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np
from psd_tools import PSDImage

from rough2ink.core.types import BBox, LayerInfo, LayerKind, PageDocument


class _PsdLayerLike(Protocol):
    """`_layer_info` / `_walk_layers` が要求するレイヤーの最小インタフェース。

    psd-tools の `Layer` 系クラス（`PixelLayer` / `TypeLayer` / `Group` など）は
    すべてこれを満たす。テストでは実際に構築できない `TypeLayer` の代わりに、
    この形を持つ軽量スタブを渡してマッピングロジックを検証する。
    """

    name: str
    kind: str
    visible: bool
    opacity: int
    layer_id: int

    @property
    def blend_mode(self) -> object: ...  # `.name` 属性を持つ Enum を想定

    @property
    def bbox(self) -> tuple[int, int, int, int]: ...

    def __iter__(self) -> object: ...  # group/artboard のときのみ呼ばれる


# psd-tools の kind は 'pixel' 'group' 'type' 'shape' 'smartobject' 'artboard'
# 'adjustment' 'fill' を返しうるが、`LayerKind` は設計書 4-A 節の 4 種類
# （pixel/type/group/shape）に絞っている。未定義の kind はもっとも近い分類へ丸める。
_KIND_MAP: dict[str, LayerKind] = {
    "pixel": "pixel",
    "type": "type",
    "group": "group",
    "shape": "shape",
    "artboard": "group",
    "smartobject": "pixel",
    "adjustment": "shape",
    "fill": "shape",
}
_GROUP_LIKE_KINDS = {"group", "artboard"}


def _map_kind(raw_kind: str) -> LayerKind:
    return _KIND_MAP.get(raw_kind, "pixel")


def _layer_bbox(layer: _PsdLayerLike) -> BBox | None:
    left, top, right, bottom = layer.bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    return (left, top, width, height)


def _layer_id(layer: _PsdLayerLike, index: int) -> str:
    raw_id = layer.layer_id
    if raw_id is not None and raw_id != -1:
        return f"lid{raw_id}"
    return f"idx{index}"


def _layer_info(layer: _PsdLayerLike, path: str, index: int) -> LayerInfo:
    return LayerInfo(
        id=_layer_id(layer, index),
        name=layer.name,
        path=path,
        kind=_map_kind(layer.kind),
        visible=layer.visible,
        opacity=layer.opacity / 255.0,
        blend_mode=layer.blend_mode.name.lower(),
        bbox=_layer_bbox(layer),
    )


def _walk_layers(group: object, parent_path: str, counter: itertools.count) -> list[LayerInfo]:
    infos: list[LayerInfo] = []
    for layer in group:
        path = f"{parent_path}/{layer.name}" if parent_path else layer.name
        infos.append(_layer_info(layer, path, next(counter)))
        if layer.kind in _GROUP_LIKE_KINDS:
            infos.extend(_walk_layers(layer, path, counter))
    return infos


def load_psd(path: Path, *, page_id: str | None = None) -> PageDocument:
    """PSD ファイルを 1 件の `PageDocument` に正規化する。

    合成画像（フラット化結果）を原寸グレースケールとして持ち、`layers` にレイヤーツリーを
    階層パス付き（グループも含む）で列挙する。
    """
    psd = PSDImage.open(str(path))

    composite = psd.composite()
    gray = np.array(composite.convert("L"))
    height, width = gray.shape[:2]

    layers = _walk_layers(psd, "", itertools.count())

    return PageDocument(
        page_id=page_id or uuid.uuid4().hex,
        source_path=path,
        source_kind="psd",
        width=width,
        height=height,
        gray=gray,
        layers=layers,
    )


def extract_layer_raster(path: Path, layer_id: str) -> np.ndarray | None:
    """`LayerInfo.id`（`layer_id`）に一致するレイヤーのラスタをグレースケール ndarray で返す。

    GT マッピング（T7）でレイヤーの実ピクセルを参照するための補助関数。
    `LayerInfo.path` は同名レイヤーがあると一意性を保証できない（#20）ため、
    `LayerInfo.id` で解決する。レイヤーが見つからない、またはピクセルを持たない場合は
    `None` を返す。
    """
    psd = PSDImage.open(str(path))
    target = _find_layer(psd, layer_id)
    if target is None:
        return None

    pil_image = target.topil()
    if pil_image is None:
        return None
    return np.array(pil_image.convert("L"))


def _find_layer(
    group: object, target_id: str, counter: itertools.count | None = None
) -> object | None:
    """`target_id`（`LayerInfo.id`）に一致するレイヤーをツリーから探す。

    `_walk_layers` と全く同じ順序・同じ規則（`_layer_id`）で走査することで、
    `layer_id` を持たないレイヤーに割り振られる `idx<n>` フォールバックも
    `_walk_layers` の結果と一致させる。`counter` は再帰全体で共有する
    （最初の呼び出しでは省略してよい）。
    """
    if counter is None:
        counter = itertools.count()
    for layer in group:
        layer_id = _layer_id(layer, next(counter))
        if layer_id == target_id:
            return layer
        if layer.kind in _GROUP_LIKE_KINDS:
            found = _find_layer(layer, target_id, counter)
            if found is not None:
                return found
    return None
