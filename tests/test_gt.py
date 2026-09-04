"""GT（正解）レイヤー役割マッピングと GT マスク生成のテスト（T7, #8）。

`core.gt` の単体テストに加え、`PUT/GET /api/pages/{page_id}/gt` の往復も検証する。
実原稿は手元にしか無いため、合成 PSD フィクスチャで決定的にテストする
（Epic 仕様書「テスト方針」）。

優先順位は **ベタ(fill) > トーン(tone) > 線(line)**（`core.decompose` と同一の契約）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer

from rough2ink.app import app
from rough2ink.core import gt

# フィクスチャの座標設計（bbox は [top, top+h) x [left, left+w) の半開区間）:
#   FillLayer   rows[ 1, 6) cols[ 1, 6)
#   ToneLayer   rows[ 4,10) cols[ 4,10)   -- FillLayer と rows/cols[4,6) で重なる
#   LineLayer   rows[ 8,14) cols[ 8,14)   -- ToneLayer と rows/cols[8,10) で重なる
#   TextLayer   rows[ 1, 5) cols[20,24)   -- 他レイヤーと非重複
#   IgnoreLayer rows[20,24) cols[ 1, 5)   -- 他レイヤーと非重複
#   HoleLayer   rows[20,26) cols[20,26)   -- 内側 rows[22,24) cols[22,24) だけ透明な穴
_PAGE_SIZE = (28, 28)  # (width, height)


def _opaque_layer(group: Group, name: str, *, left: int, top: int, width: int, height: int) -> None:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    PixelLayer.frompil(Image.fromarray(rgba, mode="RGBA"), group, name, top=top, left=left)


def _make_gt_psd(path: Path) -> None:
    psd = PSDImage.new("RGBA", _PAGE_SIZE, color=(255, 255, 255, 255))
    group = Group.new(psd, "Group1")

    _opaque_layer(group, "FillLayer", left=1, top=1, width=5, height=5)
    _opaque_layer(group, "ToneLayer", left=4, top=4, width=6, height=6)
    _opaque_layer(group, "LineLayer", left=8, top=8, width=6, height=6)
    _opaque_layer(group, "TextLayer", left=20, top=1, width=4, height=4)
    _opaque_layer(group, "IgnoreLayer", left=1, top=20, width=4, height=4)

    hole = np.zeros((6, 6, 4), dtype=np.uint8)
    hole[:, :, 3] = 255
    hole[2:4, 2:4, 3] = 0  # 内側に透明な穴（不透明画素のみが合成されることの検証用）
    PixelLayer.frompil(Image.fromarray(hole, mode="RGBA"), group, "HoleLayer", top=20, left=20)

    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


_ROLE_MAPPING = {
    "Group1/FillLayer": "fill",
    "Group1/ToneLayer": "tone",
    "Group1/LineLayer": "line",
    "Group1/TextLayer": "text",
    "Group1/IgnoreLayer": "ignore",
    "Group1/HoleLayer": "tone",
}


def _ingest_gt_psd(tmp_path: Path, monkeypatch) -> tuple[TestClient, str]:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    psd_path = tmp_path / "art.psd"
    _make_gt_psd(psd_path)
    with psd_path.open("rb") as fh:
        response = client.post("/api/ingest", files={"file": ("art.psd", fh, "image/vnd.adobe.photoshop")})
    assert response.status_code == 200
    page_id = response.json()[0]["page_id"]
    return client, page_id


# --- core.gt: マッピングの保存・取得 -----------------------------------------


def test_save_and_load_mapping_roundtrip(tmp_path: Path, monkeypatch) -> None:
    client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)
    assert client is not None  # ページを永続化するためだけに使う

    gt.save_mapping(page_id, _ROLE_MAPPING)

    loaded = gt.load_mapping(page_id)
    assert loaded == _ROLE_MAPPING


def test_load_mapping_returns_empty_dict_when_unsaved(tmp_path: Path, monkeypatch) -> None:
    _client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)

    assert gt.load_mapping(page_id) == {}


def test_save_mapping_raises_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(gt.PageNotFoundError):
        gt.save_mapping("does-not-exist", {"L1": "line"})


def test_load_mapping_raises_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(gt.PageNotFoundError):
        gt.load_mapping("does-not-exist")


# --- core.gt: GT マスク生成 ---------------------------------------------------


def test_build_gt_masks_resolves_priority_and_excludes_ignore(tmp_path: Path, monkeypatch) -> None:
    _client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)
    gt.save_mapping(page_id, _ROLE_MAPPING)

    masks = gt.build_gt_masks(page_id)

    assert set(masks) == {"line", "fill", "tone", "text"}
    for mask in masks.values():
        assert mask.shape == (_PAGE_SIZE[1], _PAGE_SIZE[0])  # (height, width)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    fill, tone, line, text = masks["fill"], masks["tone"], masks["line"], masks["text"]

    # fill 単独領域。
    assert fill[2, 2] == 255
    assert tone[2, 2] == 0
    assert line[2, 2] == 0

    # fill/tone の重なりは fill が勝つ（ベタ > トーン）。
    assert fill[5, 5] == 255
    assert tone[5, 5] == 0

    # tone 単独領域（fill にも line にも属さない）。
    assert tone[6, 6] == 255
    assert fill[6, 6] == 0
    assert line[6, 6] == 0

    # tone/line の重なりは tone が勝つ（トーン > 線）。
    assert tone[9, 9] == 255
    assert line[9, 9] == 0

    # line 単独領域。
    assert line[12, 12] == 255
    assert tone[12, 12] == 0
    assert fill[12, 12] == 0

    # text は評価除外領域として別枠で返る。
    assert text[2, 21] == 255
    assert fill[2, 21] == 0
    assert tone[2, 21] == 0
    assert line[2, 21] == 0

    # ignore は完全に無視される（どのマスクにも現れない）。
    assert fill[21, 2] == 0
    assert tone[21, 2] == 0
    assert line[21, 2] == 0
    assert text[21, 2] == 0

    # HoleLayer(role=tone): 不透明画素のみが合成される。外周は tone、内側の穴は 0。
    assert tone[21, 21] == 255
    assert tone[23, 23] == 0

    # 3マスクは相互排他。
    assert not np.any((fill > 0) & (tone > 0))
    assert not np.any((fill > 0) & (line > 0))
    assert not np.any((tone > 0) & (line > 0))


def test_build_gt_masks_with_no_mapping_returns_all_empty(tmp_path: Path, monkeypatch) -> None:
    _client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)

    masks = gt.build_gt_masks(page_id)

    for mask in masks.values():
        assert not np.any(mask)


def test_build_gt_masks_raises_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(gt.PageNotFoundError):
        gt.build_gt_masks("does-not-exist")


# --- routes_gt: PUT/GET /api/pages/{page_id}/gt ------------------------------


def test_put_and_get_gt_mapping_roundtrip(tmp_path: Path, monkeypatch) -> None:
    client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)

    put_response = client.put(f"/api/pages/{page_id}/gt", json={"mapping": _ROLE_MAPPING})
    assert put_response.status_code == 200
    assert put_response.json()["mapping"] == _ROLE_MAPPING

    get_response = client.get(f"/api/pages/{page_id}/gt")
    assert get_response.status_code == 200
    assert get_response.json()["mapping"] == _ROLE_MAPPING

    # ディスクに永続化され、再読み込みでも復元できる。
    assert gt.load_mapping(page_id) == _ROLE_MAPPING


def test_get_gt_mapping_returns_empty_mapping_when_unsaved(tmp_path: Path, monkeypatch) -> None:
    client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)

    response = client.get(f"/api/pages/{page_id}/gt")

    assert response.status_code == 200
    assert response.json()["mapping"] == {}


def test_get_gt_mapping_returns_404_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/pages/does-not-exist/gt")

    assert response.status_code == 404


def test_put_gt_mapping_returns_404_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.put("/api/pages/does-not-exist/gt", json={"mapping": {"L1": "line"}})

    assert response.status_code == 404


def test_put_gt_mapping_rejects_unknown_role(tmp_path: Path, monkeypatch) -> None:
    client, page_id = _ingest_gt_psd(tmp_path, monkeypatch)

    response = client.put(f"/api/pages/{page_id}/gt", json={"mapping": {"Group1/FillLayer": "not-a-role"}})

    assert response.status_code == 422


# --- パストラバーサル対策（Review #21） ------------------------------------------
#
# `routes_gt.py` は `page_id` を自分ではパス結合しない（`core.gt` の内部関数に委ねる）が、
# レビューで同種の欠落が指摘されている。ここでは `core.config.resolve_page_dir` による
# ホワイトリスト検証が `core.gt` を呼ぶ前段で 404 に変換することだけを確認する
# （マッピングの中身・キー形式には触れない）。


def test_get_gt_mapping_returns_404_for_backslash_traversal_page_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/pages/..\\..\\victim/gt")

    assert response.status_code == 404


def test_put_gt_mapping_returns_404_for_backslash_traversal_page_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.put("/api/pages/..\\..\\victim/gt", json={"mapping": {}})

    assert response.status_code == 404
