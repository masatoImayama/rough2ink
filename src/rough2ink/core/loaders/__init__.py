"""入力ローダ（画像 / PSD / PDF）。

3 系統すべてを共通の `PageDocument`（原寸グレースケール + レイヤー情報）に正規化する
（Epic 仕様書 3 章・4-A 節）。
"""

from __future__ import annotations

from rough2ink.core.loaders.image_loader import load_image
from rough2ink.core.loaders.pdf_loader import load_pdf
from rough2ink.core.loaders.psd_loader import extract_layer_raster, load_psd

__all__ = [
    "load_image",
    "load_pdf",
    "load_psd",
    "extract_layer_raster",
]
