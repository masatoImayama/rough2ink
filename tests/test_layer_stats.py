"""レイヤー統計（墨の被覆率・サムネイル）と、それを返す API のテスト。

GT 割当をレイヤー名だけで判断すると誤る（設計書 4章 A）。実原稿で実際に起きた
次の3パターンを、統計だけで見分けられることを固定する。

- 名前は「線」だが中身はページの大半が墨（実例: `集中線（円形）` が 74%）
- 不透明だが墨が無い白抜きレイヤー（実例: `描き文字線画` が墨 0.0%）
- 画素を持たない空レイヤー（実例: `線画` という名前の複数レイヤー）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer

from rough2ink.app import app

_PAGE_SIZE = (40, 40)  # (width, height)


def _layer(group: Group, name: str, *, left: int, top: int, size: int, luminance: int) -> None:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = luminance
    rgba[:, :, 3] = 255
    PixelLayer.frompil(Image.fromarray(rgba, mode="RGBA"), group, name, top=top, left=left)


def _make_psd(path: Path) -> None:
    psd = PSDImage.new("RGBA", _PAGE_SIZE, color=(255, 255, 255, 255))
    group = Group.new(psd, "Group1")
    # ページ(40x40=1600px)の 1/4 を占める真っ黒なレイヤー = 400px -> 墨/ページ 0.25
    _layer(group, "HeavyInk", left=0, top=0, size=20, luminance=0)
    # 不透明だが白い = 墨ゼロ（白抜きレイヤー相当）
    _layer(group, "WhiteOnly", left=20, top=20, size=10, luminance=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


def _ingest(client: TestClient, psd_path: Path) -> str:
    with psd_path.open("rb") as fh:
        response = client.post(
            "/api/ingest", files={"file": (psd_path.name, fh, "image/vnd.adobe.photoshop")}
        )
    assert response.status_code == 200
    return response.json()[0]["page_id"]


def test_layer_stats_separates_ink_from_opacity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)
    psd_path = tmp_path / "stats.psd"
    _make_psd(psd_path)
    page_id = _ingest(client, psd_path)

    response = client.get(f"/api/pages/{page_id}/layers/stats")
    assert response.status_code == 200
    stats = {item["path"].split("/")[-1]: item for item in response.json()}

    heavy = stats["HeavyInk"]
    assert heavy["has_pixels"] is True
    assert heavy["ink_ratio_page"] == 0.25, heavy
    assert heavy["ink_ratio_layer"] == 1.0

    # 白抜きレイヤー: 不透明だが墨は無い。この差が名前では分からない誤りを可視化する。
    white = stats["WhiteOnly"]
    assert white["opaque_ratio_layer"] == 1.0
    assert white["ink_ratio_layer"] == 0.0
    assert white["ink_ratio_page"] == 0.0


def test_layer_stats_are_cached_and_groups_excluded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)
    psd_path = tmp_path / "stats.psd"
    _make_psd(psd_path)
    page_id = _ingest(client, psd_path)

    first = client.get(f"/api/pages/{page_id}/layers/stats").json()
    # グループは画素を持たないので対象外。
    assert all(item["kind"] != "group" for item in first)

    cache = tmp_path / "ws" / "pages" / page_id / "layer_stats.json"
    assert cache.is_file(), "2回目以降のために結果をキャッシュするべき"

    second = client.get(f"/api/pages/{page_id}/layers/stats").json()
    assert second == first


def test_layer_thumbnail_is_served(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)
    psd_path = tmp_path / "stats.psd"
    _make_psd(psd_path)
    page_id = _ingest(client, psd_path)

    stats = client.get(f"/api/pages/{page_id}/layers/stats").json()
    heavy = next(item for item in stats if item["path"].endswith("HeavyInk"))
    assert heavy["has_thumbnail"] is True

    response = client.get(f"/api/pages/{page_id}/layers/{heavy['layer_id']}/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_layer_thumbnail_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    """`layer_id` もパスの一部になるため、`page_id` と同様に検証する（Review #21 と同種）。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)
    psd_path = tmp_path / "stats.psd"
    _make_psd(psd_path)
    page_id = _ingest(client, psd_path)

    for bad in ("..%5C..%5Cevil", "..", "lid1%2F..%2F..%2Fevil"):
        response = client.get(f"/api/pages/{page_id}/layers/{bad}/thumbnail")
        assert response.status_code == 404, bad


def test_role_suggestion_endpoint_returns_initial_values(tmp_path: Path, monkeypatch) -> None:
    """推定エンドポイントが各レイヤーぶんの結果を返し、判定できない名前は未判定にすること。

    キーワードは日本語だが、psd-tools は PSD 保存時にレイヤー名を macroman で
    エンコードするため**日本語のレイヤー名を持つ PSD をテスト内で生成できない**
    （`core.loaders.psd_loader` のライブラリ制約。読み込み側には影響しない）。
    そのためキーワード照合そのものは `tests/test_gt_suggest.py` で `LayerInfo` を
    直接組み立てて検証し、ここでは API の形と「未判定を勝手に埋めない」ことだけを見る。
    """
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)
    psd_path = tmp_path / "suggest.psd"
    _make_psd(psd_path)
    page_id = _ingest(client, psd_path)

    response = client.get(f"/api/pages/{page_id}/gt/suggest")
    assert response.status_code == 200
    suggestions = response.json()

    # グループを除く全レイヤーぶん返る。
    assert {item["path"].split("/")[-1] for item in suggestions} == {"HeavyInk", "WhiteOnly"}
    # ASCII の任意名は日本語キーワードに当たらないので未判定のまま。
    assert all(item["role"] is None for item in suggestions)
    assert all(item["matched_keyword"] is None for item in suggestions)
