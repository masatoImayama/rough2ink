"""画像ローダ（`.png` `.jpg` `.jpeg`）。単一ページ・レイヤー情報なし。"""

from __future__ import annotations

import uuid
from pathlib import Path

from rough2ink.core import imageio
from rough2ink.core.types import PageDocument

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_image(path: Path, *, page_id: str | None = None) -> PageDocument:
    """画像ファイルを 1 件の `PageDocument` に正規化する（原寸グレースケール）。

    レイヤー情報は持たない（画像形式にレイヤー概念が無いため `layers` は空）。
    """
    gray = imageio.read_gray(path)
    height, width = gray.shape[:2]

    return PageDocument(
        page_id=page_id or uuid.uuid4().hex,
        source_path=path,
        source_kind="image",
        width=width,
        height=height,
        gray=gray,
        layers=[],
    )
