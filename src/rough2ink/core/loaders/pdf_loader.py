"""PDF ローダ。複数ページを 1 ページ 1 `PageDocument` に展開する。

原寸維持のため既定 600dpi 相当でラスタライズする（Epic 仕様書 3 章）。
上限は長辺 12000px とし、超える場合は上限に収まるようスケールを落とす
（極端に大きい用紙サイズでメモリを食い潰さないための安全弁）。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from rough2ink.core.types import PageDocument

DEFAULT_DPI = 600.0
POINTS_PER_INCH = 72.0
MAX_LONG_SIDE_PX = 12000


def load_pdf(
    path: Path,
    *,
    dpi: float = DEFAULT_DPI,
    max_long_side_px: int = MAX_LONG_SIDE_PX,
    page_id_prefix: str | None = None,
) -> list[PageDocument]:
    """PDF ファイルをページごとの `PageDocument` のリストに正規化する。"""
    base_id = page_id_prefix or uuid.uuid4().hex

    documents: list[PageDocument] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        page_count = len(pdf)
        for index in range(page_count):
            page = pdf[index]
            try:
                width_pt, height_pt = page.get_size()
                scale = dpi / POINTS_PER_INCH
                long_side_pt = max(width_pt, height_pt)
                if long_side_pt > 0:
                    max_scale = max_long_side_px / long_side_pt
                    scale = min(scale, max_scale)

                bitmap = page.render(scale=scale, grayscale=True)
                try:
                    pil_image = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()

            gray = np.array(pil_image.convert("L"))
            height, width = gray.shape[:2]

            page_id = base_id if page_count == 1 else f"{base_id}_p{index + 1:03d}"
            documents.append(
                PageDocument(
                    page_id=page_id,
                    source_path=path,
                    source_kind="pdf",
                    width=width,
                    height=height,
                    gray=gray,
                    layers=[],
                )
            )
    finally:
        pdf.close()

    return documents
