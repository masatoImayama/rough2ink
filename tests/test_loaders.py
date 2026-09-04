"""入力ローダ（画像 / PSD / PDF）のテスト。

実原稿は手元にしか無いため、合成フィクスチャで決定的にテストする
（Epic 仕様書「テスト方針」）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer

from rough2ink.core.loaders import extract_layer_raster, load_image, load_pdf, load_psd
from rough2ink.core.loaders.psd_loader import _layer_info, _map_kind

# 日本語・空白入りのディレクトリ名／ファイル名（Windows のパス処理を通すことの確認を兼ねる）。
_JP_DIR_NAME = "取り込み テスト"


def _make_png(path: Path, width: int, height: int, fill: int = 200) -> None:
    array = np.full((height, width), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", array)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def _make_pdf(path: Path, page_sizes_px: list[tuple[int, int]]) -> None:
    """`page_sizes_px` の各ページを resolution=72dpi の PDF として書き出す。

    72dpi なら「ページの pt サイズ == 画素サイズ」になるため、テストでの
    期待値計算が単純になる。
    """
    images = [Image.new("L", size, color=255) for size in page_sizes_px]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=72.0)


def _make_psd(path: Path) -> None:
    """グループ 1 つ・ピクセルレイヤー 1 枚を持つ PSD を作る。

    レイヤー名は ASCII にする: psd-tools の低レベルレイヤー構築 API
    (`PixelLayer.frompil` 等)は、書き出し時にレイヤー名を legacy な
    Mac Roman エンコードでも書こうとするため、Mac Roman で表現できない
    文字（日本語等）を含む名前だと書き出しに失敗する（psd-tools 側の
    write パスの制約であり、本ローダの読み込みロジックとは無関係。
    ファイルパスの日本語対応は `test_load_psd_handles_japanese_and_space_in_path`
    で別途検証する）。
    """
    psd = PSDImage.new("RGB", (40, 30), color=255)
    group = Group.new(psd, "Group1")
    black_square = np.zeros((10, 10, 3), dtype=np.uint8)
    PixelLayer.frompil(Image.fromarray(black_square, mode="RGB"), group, "BlackLayer1", top=5, left=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


# --- image_loader -----------------------------------------------------------


def test_load_image_normalizes_to_page_document(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    _make_png(path, width=64, height=48)

    doc = load_image(path)

    assert doc.source_kind == "image"
    assert doc.width == 64
    assert doc.height == 48
    assert doc.layers == []
    gray = doc.as_array()
    assert gray.shape == (48, 64)
    assert gray.dtype == np.uint8


def test_load_image_handles_japanese_and_space_in_path(tmp_path: Path) -> None:
    path = tmp_path / _JP_DIR_NAME / "漫画 原稿 01.png"
    _make_png(path, width=32, height=24)

    doc = load_image(path)

    assert doc.width == 32
    assert doc.height == 24


# --- pdf_loader ---------------------------------------------------------


def test_load_pdf_expands_multi_page_to_one_page_document_each(tmp_path: Path) -> None:
    path = tmp_path / "manuscript.pdf"
    _make_pdf(path, page_sizes_px=[(200, 100), (200, 100)])

    docs = load_pdf(path, dpi=72.0)

    assert len(docs) == 2
    ids = [doc.page_id for doc in docs]
    assert len(set(ids)) == 2  # ページごとに一意な page_id
    for doc in docs:
        assert doc.source_kind == "pdf"
        # 72dpi(=等倍) でラスタライズしたので原寸に近い画素数になる（丸め誤差のみ許容）。
        assert abs(doc.width - 200) <= 2
        assert abs(doc.height - 100) <= 2
        assert doc.as_array().shape == (doc.height, doc.width)


def test_load_pdf_rasterizes_at_higher_dpi_than_page_points(tmp_path: Path) -> None:
    path = tmp_path / "high_res.pdf"
    _make_pdf(path, page_sizes_px=[(200, 100)])

    # 600dpi (既定) は 72dpi の 200x100pt ページを約 8.33 倍に拡大するはず。
    docs = load_pdf(path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.width > 1600  # 200 * (600/72) ≈ 1667
    assert doc.height > 800  # 100 * (600/72) ≈ 833


def test_load_pdf_clamps_to_max_long_side(tmp_path: Path) -> None:
    path = tmp_path / "clamp.pdf"
    _make_pdf(path, page_sizes_px=[(200, 100)])

    docs = load_pdf(path, dpi=600.0, max_long_side_px=150)

    assert len(docs) == 1
    assert max(docs[0].width, docs[0].height) <= 150


def test_load_pdf_handles_japanese_and_space_in_path(tmp_path: Path) -> None:
    path = tmp_path / _JP_DIR_NAME / "冊子 資料.pdf"
    _make_pdf(path, page_sizes_px=[(120, 80)])

    docs = load_pdf(path, dpi=72.0)

    assert len(docs) == 1


# --- psd_loader -----------------------------------------------------------


def test_load_psd_normalizes_composite_and_layer_tree(tmp_path: Path) -> None:
    path = tmp_path / "art.psd"
    _make_psd(path)

    doc = load_psd(path)

    assert doc.source_kind == "psd"
    assert doc.width == 40
    assert doc.height == 30
    gray = doc.as_array()
    assert gray.shape == (30, 40)

    # レイヤーツリーがグループも含めて階層パス付きで列挙されている。
    by_path = {layer.path: layer for layer in doc.layers}
    assert "Group1" in by_path
    assert by_path["Group1"].kind == "group"

    assert "Group1/BlackLayer1" in by_path
    pixel_layer = by_path["Group1/BlackLayer1"]
    assert pixel_layer.kind == "pixel"
    assert pixel_layer.visible is True
    assert pixel_layer.opacity == pytest.approx(1.0)
    assert pixel_layer.blend_mode == "normal"
    # 原寸座標系での bbox (x, y, width, height)。
    assert pixel_layer.bbox == (5, 5, 10, 10)


def test_load_psd_handles_japanese_and_space_in_path(tmp_path: Path) -> None:
    path = tmp_path / _JP_DIR_NAME / "作品 A.psd"
    _make_psd(path)

    doc = load_psd(path)

    assert doc.width == 40
    assert len(doc.layers) == 2


def test_extract_layer_raster_returns_pixel_data_for_layer_id(tmp_path: Path) -> None:
    path = tmp_path / "art.psd"
    _make_psd(path)
    doc = load_psd(path)
    by_path = {layer.path: layer for layer in doc.layers}
    layer_id = by_path["Group1/BlackLayer1"].id

    raster = extract_layer_raster(path, layer_id)

    assert raster is not None
    assert raster.shape == (10, 10)
    # フィクスチャで真っ黒に塗った領域なのでグレースケールは 0 に近い。
    assert raster.max() < 10


def test_extract_layer_raster_returns_none_for_unknown_id(tmp_path: Path) -> None:
    path = tmp_path / "art.psd"
    _make_psd(path)

    assert extract_layer_raster(path, "idx999") is None


# --- psd_loader: 同名レイヤー（#20） -----------------------------------------


def _make_psd_with_duplicate_layer_names(path: Path) -> None:
    """同一グループ内に同名レイヤーを2枚持つ PSD を作る（#20 の回帰）。

    実PSDでは「レイヤー 1」「レイヤー 1 のコピー」のような重複名が日常的に存在する
    （Epic 仕様書 4-A 節: レイヤー命名規則は一貫していない前提）。片方を暗い塗り、
    もう片方を明るい塗りにしておくことで、id 解決の取り違えを画素値で検出できる。
    """
    psd = PSDImage.new("RGB", (40, 30), color=255)
    group = Group.new(psd, "Group1")
    dark = np.zeros((10, 10, 3), dtype=np.uint8)
    light = np.full((10, 10, 3), 200, dtype=np.uint8)
    PixelLayer.frompil(Image.fromarray(dark, mode="RGB"), group, "DupLayer", top=0, left=0)
    PixelLayer.frompil(Image.fromarray(light, mode="RGB"), group, "DupLayer", top=15, left=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


def test_load_psd_assigns_unique_ids_to_layers_with_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "dup.psd"
    _make_psd_with_duplicate_layer_names(path)

    doc = load_psd(path)

    dup_layers = [layer for layer in doc.layers if layer.name == "DupLayer"]
    assert len(dup_layers) == 2
    # path（表示用）は同一になりうるが、id は一意に区別できる。
    assert dup_layers[0].path == dup_layers[1].path == "Group1/DupLayer"
    assert dup_layers[0].id != dup_layers[1].id


def test_extract_layer_raster_resolves_duplicate_named_layers_independently(tmp_path: Path) -> None:
    """同名レイヤーが2枚あっても、id で指定した方だけを正しく解決できること（#20）。"""
    path = tmp_path / "dup.psd"
    _make_psd_with_duplicate_layer_names(path)
    doc = load_psd(path)
    dup_layers = [layer for layer in doc.layers if layer.name == "DupLayer"]

    raster_a = extract_layer_raster(path, dup_layers[0].id)
    raster_b = extract_layer_raster(path, dup_layers[1].id)

    assert raster_a is not None
    assert raster_b is not None
    # path 基準（旧実装）だと常に同一レイヤーが返り raster_a == raster_b になっていたはず。
    assert not np.array_equal(raster_a, raster_b)
    maxima = sorted([raster_a.max(), raster_b.max()])
    assert maxima[0] < 10  # 暗い方のレイヤー
    assert maxima[1] > 190  # 明るい方のレイヤー


def test_map_kind_maps_type_layer_to_type() -> None:
    """`kind == 'type'`（テキストレイヤー）はフキダシ検出のヒントとして必ず保持する。

    psd-tools は Photoshop で作成された実データが無いと `TypeLayer` を構築できないため、
    ここではマッピングロジック（`_map_kind`）とレイヤー情報構築ロジック（`_layer_info`）
    を、実レイヤーと同じ形（duck typing）の軽量スタブで直接検証する。
    """
    assert _map_kind("type") == "type"

    class _TypeLayerStub:
        name = "セリフ"
        kind = "type"
        visible = True
        opacity = 255
        layer_id = 7

        class _BlendMode:
            name = "NORMAL"

        blend_mode = _BlendMode()
        bbox = (10, 20, 110, 60)

    info = _layer_info(_TypeLayerStub(), "セリフ", index=0)

    assert info.kind == "type"
    assert info.name == "セリフ"
    assert info.bbox == (10, 20, 100, 40)
    assert info.id == "lid7"


def test_map_kind_falls_back_unknown_kinds_reasonably() -> None:
    assert _map_kind("artboard") == "group"
    assert _map_kind("smartobject") == "pixel"
    assert _map_kind("adjustment") == "shape"
    assert _map_kind("fill") == "shape"
    assert _map_kind("something-unforeseen") == "pixel"
