"""`POST /api/ingest` → `GET /api/pages` → `GET /api/pages/{page_id}/preview` のテスト。

`workspace/` はテストごとに `ROUGH2INK_WORKSPACE_DIR` で隔離する
（`tests/test_config.py` と同じパターン）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from rough2ink.app import app
from rough2ink.core.params import PreviewParams


def _png_bytes(width: int, height: int, fill: int = 128) -> bytes:
    array = np.full((height, width), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


def _decode_png(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def test_ingest_pages_preview_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    # プレビューの縮小(長辺 1600px)を検証できるよう、原稿の長辺をそれより大きくする。
    long_side = PreviewParams().max_long_side + 400
    content = _png_bytes(width=long_side, height=100)

    # 日本語・空白入りファイル名でアップロードできることも合わせて確認する。
    response = client.post(
        "/api/ingest",
        files={"file": ("漫画 原稿 01.png", content, "image/png")},
    )
    assert response.status_code == 200
    ingested = response.json()
    assert len(ingested) == 1
    page = ingested[0]
    assert page["source_kind"] == "image"
    assert page["width"] == long_side
    assert page["height"] == 100
    page_id = page["page_id"]

    pages_response = client.get("/api/pages")
    assert pages_response.status_code == 200
    listed = pages_response.json()
    assert any(p["page_id"] == page_id for p in listed)

    preview_response = client.get(f"/api/pages/{page_id}/preview")
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"] == "image/png"

    preview_array = _decode_png(preview_response.content)
    assert max(preview_array.shape) <= PreviewParams().max_long_side

    # 原寸は縮小されず workspace に保持されている（page.png を直接検証）。
    page_png = tmp_path / "ws" / "pages" / page_id / "page.png"
    original_array = _decode_png(page_png.read_bytes())
    assert original_array.shape == (100, long_side)


def test_get_preview_returns_404_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/pages/does-not-exist/preview")

    assert response.status_code == 404


# --- パストラバーサル対策（Review #21） ------------------------------------------
#
# uvicorn は URL をデコードしてからルーティングするため、`..%5C..%5C..%5CUsers%5Cvictim`
# のようなバックスラッシュ入りの値が単一パスセグメントとして `page_id` に一致しうる
# （Windows 上で `workspace\pages\..\..\..\Users\victim\...` に解決されるとレビュアーが実測）。
#
# サンドボックス(Linux)ではバックスラッシュはパス区切りとして解釈されないため、
# 「該当ディレクトリが存在しないから404」という偶然の一致では検証にならない。
# そのため、あえて `..\\..\\victim` という**リテラルな名前のディレクトリ**を
# ワークスペース配下に作って中身を用意したうえで、それでもホワイトリスト検証
# （`core.config.resolve_page_dir`）によって拒否され 404 になることを確認する。


def test_get_preview_returns_404_for_backslash_traversal_page_id_even_if_literal_dir_exists(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    traversal_page_id = "..\\..\\victim"
    literal_page_dir = tmp_path / "ws" / "pages" / traversal_page_id
    literal_page_dir.mkdir(parents=True)
    (literal_page_dir / "preview.png").write_bytes(b"secret-bytes")

    response = client.get(f"/api/pages/{traversal_page_id}/preview")

    assert response.status_code == 404
    assert response.content != b"secret-bytes"


def test_get_layers_returns_404_for_backslash_traversal_page_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    traversal_page_id = "..\\..\\victim"
    response = client.get(f"/api/pages/{traversal_page_id}/layers")

    assert response.status_code == 404


def test_ingest_rejects_unsupported_file_type(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.post(
        "/api/ingest",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_ingest_pdf_produces_one_page_document_per_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    from PIL import Image

    pdf_path = tmp_path / "multi.pdf"
    images = [Image.new("L", (100, 80), color=255), Image.new("L", (100, 80), color=200)]
    images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=72.0)

    with pdf_path.open("rb") as fh:
        response = client.post(
            "/api/ingest",
            files={"file": ("原稿.pdf", fh, "application/pdf")},
        )

    assert response.status_code == 200
    ingested = response.json()
    assert len(ingested) == 2
    assert {page["source_kind"] for page in ingested} == {"pdf"}
    assert len({page["page_id"] for page in ingested}) == 2
