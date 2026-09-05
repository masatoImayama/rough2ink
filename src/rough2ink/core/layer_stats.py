"""PSD レイヤーごとの「墨の量」とサムネイルを算出する（GT 割当を目視で検証可能にする）。

実原稿で GT を作る際、レイヤー名だけでは役割を判断できない。実測で次のような例があった。

- `集中線（円形）` という名前だが中身はページ全体の 74% が墨（黒地に白で放射線を抜いた
  ベタフラッシュ）。名前は「線」だが機能的には「ベタ」
- `描き文字線画` は不透明画素 7.0% に対し墨 0.0%（白抜き文字）
- `線画` という名前のレイヤーが複数あり、その多くは画素を持たない空レイヤー

設計書 4章 A が「レイヤー命名規則は一貫していない前提。自動判定に頼らない」としている通り、
最終判断は人間が行う。この情報はその判断を数秒で済ませるためのもので、
**墨の被覆率と縮小画像を並べれば上記3種の誤りはいずれも一目で分かる**。

結果は `workspace/pages/<page_id>/layer_stats.json` と
`workspace/pages/<page_id>/layer_thumbs/<layer_id>.png` にキャッシュする
（実測: 69レイヤーのラスタライズに約2.5秒）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from pydantic import BaseModel

from rough2ink.core.config import get_workspace_dir
from rough2ink.core.loaders.psd_loader import _find_layer  # noqa: SLF001 -- ツリー走査を再利用

# 墨とみなす輝度の上限。`core.gt` の GT 生成と同じ既定値にする（見た目と GT を一致させる）。
_DEFAULT_INK_THRESHOLD = 128

# サムネイルの長辺（px）。一覧で並べて判断できる最小限の大きさ。
_THUMBNAIL_LONG_SIDE = 120

# サムネイルの背景。白抜き文字のレイヤーも見えるよう、白でも黒でもない中間色にする。
_THUMBNAIL_BACKGROUND = 176


class LayerStat(BaseModel):
    """レイヤー1枚分の統計。"""

    layer_id: str
    path: str
    kind: str
    has_pixels: bool
    # 墨画素数 / ページ全体の画素数。ページのどれだけを占めるか。
    ink_ratio_page: float = 0.0
    # 墨画素数 / レイヤーのバウンディングボックスの画素数。レイヤー内での密度。
    ink_ratio_layer: float = 0.0
    # 不透明画素数 / バウンディングボックス。墨との差が大きいレイヤーは白抜きの疑い。
    opaque_ratio_layer: float = 0.0
    has_thumbnail: bool = False


def _stats_path(page_id: str) -> Path:
    return get_workspace_dir() / "pages" / page_id / "layer_stats.json"


def thumbnail_path(page_id: str, layer_id: str) -> Path:
    return get_workspace_dir() / "pages" / page_id / "layer_thumbs" / f"{layer_id}.png"


def _luminance(array: np.ndarray, bands: tuple[str, ...]) -> np.ndarray:
    color_indices = [index for index, band in enumerate(bands) if band != "A"]
    if not color_indices:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    if array.ndim == 2:
        return array
    return array[:, :, color_indices].mean(axis=2)


def _write_thumbnail(array: np.ndarray, bands: tuple[str, ...], destination: Path) -> None:
    """レイヤーを中間色の背景に合成した縮小画像を書き出す。

    白背景に合成すると白抜き文字のレイヤーが真っ白になって判別できないため、
    背景を中間色にして「白い墨」も「黒い墨」も見えるようにする。
    """
    luminance = _luminance(array, bands).astype(np.float32)
    if "A" in bands:
        alpha = array[:, :, bands.index("A")].astype(np.float32) / 255.0
    else:
        alpha = np.ones(array.shape[:2], dtype=np.float32)

    composited = luminance * alpha + _THUMBNAIL_BACKGROUND * (1.0 - alpha)
    image = Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8), mode="L")
    image.thumbnail((_THUMBNAIL_LONG_SIDE, _THUMBNAIL_LONG_SIDE))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def compute_layer_stats(
    page_id: str,
    layers: list[dict],
    source_path: Path,
    page_size: tuple[int, int],
    *,
    ink_threshold: int = _DEFAULT_INK_THRESHOLD,
) -> list[LayerStat]:
    """全レイヤーの統計とサムネイルを算出し、キャッシュに書き出して返す。

    グループレイヤーは画素を持たないため対象外。
    """
    page_width, page_height = page_size
    page_pixels = max(1, page_width * page_height)

    psd = PSDImage.open(str(source_path))
    stats: list[LayerStat] = []

    for layer_meta in layers:
        if layer_meta.get("kind") == "group":
            continue
        layer_id = layer_meta["id"]
        base = LayerStat(
            layer_id=layer_id,
            path=layer_meta.get("path", ""),
            kind=layer_meta.get("kind", "pixel"),
            has_pixels=False,
        )

        layer = _find_layer(psd, layer_id)
        pil_image = layer.topil() if layer is not None else None
        if pil_image is None:
            stats.append(base)
            continue

        array = np.array(pil_image)
        bands = pil_image.getbands()
        if "A" in bands:
            opaque = array[:, :, bands.index("A")] > 0
        else:
            opaque = np.ones(array.shape[:2], dtype=bool)
        ink = opaque & (_luminance(array, bands) <= ink_threshold)

        layer_pixels = max(1, opaque.size)
        base.has_pixels = True
        base.ink_ratio_page = float(ink.sum()) / page_pixels
        base.ink_ratio_layer = float(ink.sum()) / layer_pixels
        base.opaque_ratio_layer = float(opaque.sum()) / layer_pixels

        try:
            _write_thumbnail(array, bands, thumbnail_path(page_id, layer_id))
            base.has_thumbnail = True
        except (OSError, ValueError):
            # サムネイル生成の失敗で統計そのものを失わない（一覧は数値だけでも役に立つ）。
            base.has_thumbnail = False

        stats.append(base)

    _stats_path(page_id).write_text(
        json.dumps([stat.model_dump(mode="json") for stat in stats], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return stats


def load_cached_stats(page_id: str) -> list[LayerStat] | None:
    """キャッシュ済みの統計を返す（無ければ `None`）。"""
    path = _stats_path(page_id)
    if not path.is_file():
        return None
    return [LayerStat.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
