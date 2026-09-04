"""`POST /api/pages/{page_id}/analyze` と `GET /api/pages/{page_id}/mask/{kind}` のテスト。

実原稿は手元にないため、枠線を描いた合成ページで検証する（Epic 本文「テスト方針」）。
`workspace/` はテストごとに `ROUGH2INK_WORKSPACE_DIR` で隔離する
（`tests/test_routes_ingest.py` と同じパターン）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer

from rough2ink.app import app
from rough2ink.core import gt
from rough2ink.core.panels import detect_panels as core_detect_panels
from rough2ink.core.params import AnalysisParams, PreviewParams, QualityParams


def _draw_border(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, thickness: int = 3) -> None:
    # `tests/test_panels.py` と同じく辺ごとに `cv2.line` で描く。`cv2.rectangle` は角の
    # 描画が僅かに斜めになり `oblique` フラグを誤って立てることがあるため使わない。
    cv2.line(img, (x1, y1), (x2, y1), 0, thickness)
    cv2.line(img, (x1, y2), (x2, y2), 0, thickness)
    cv2.line(img, (x1, y1), (x1, y2), 0, thickness)
    cv2.line(img, (x2, y1), (x2, y2), 0, thickness)


def _panel_grid_page(height: int = 500, width: int = 500) -> np.ndarray:
    """500x500 の 2x2 コマ割りページ（各コマとも例外フラグ無しで検出できる）。"""
    page = np.full((height, width), 255, dtype=np.uint8)
    cells = [(40, 40, 220, 220), (280, 40, 460, 220), (40, 280, 220, 460), (280, 280, 460, 460)]
    for x1, y1, x2, y2 in cells:
        _draw_border(page, x1, y1, x2, y2)
    return page


def _png_bytes(array: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


def _decode_png(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _ingest_page(client: TestClient, array: np.ndarray, filename: str = "page.png") -> str:
    response = client.post(
        "/api/ingest", files={"file": (filename, _png_bytes(array), "image/png")}
    )
    assert response.status_code == 200
    return response.json()[0]["page_id"]


def _analyze(client: TestClient, page_id: str, params: AnalysisParams):
    return client.post(f"/api/pages/{page_id}/analyze", json=params.model_dump(mode="json"))


def test_analyze_returns_quality_panels_and_mask_urls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    page = _panel_grid_page()
    page_id = _ingest_page(client, page)

    response = _analyze(client, page_id, AnalysisParams())

    assert response.status_code == 200
    body = response.json()

    assert body["page_id"] == page_id
    assert body["quality"]["short_side"] == 500
    assert len(body["panels"]) == 4
    for panel in body["panels"]:
        assert panel["flags"] == []
        assert len(panel["polygon"]) >= 3

    # 500 <= preview.max_long_side(1600) なので縮小されない。
    assert body["preview_width"] == 500
    assert body["preview_height"] == 500

    for kind in ("line", "fill", "tone", "balloon"):
        assert body["mask_urls"][kind] == f"/api/pages/{page_id}/mask/{kind}"

    # GT マッピング未保存のページでは metrics は None。
    assert body["metrics"] is None


def test_analyze_response_never_contains_full_resolution_image(
    tmp_path: Path, monkeypatch
) -> None:
    """レスポンスに原寸画像が含まれないことを確認する（Epic 仕様書 9・10 節）。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    long_side = PreviewParams().max_long_side + 400
    page = np.full((long_side, long_side), 255, dtype=np.uint8)
    page_id = _ingest_page(client, page)

    response = _analyze(client, page_id, AnalysisParams())

    assert response.status_code == 200
    body = response.json()

    # 応答は JSON のみで、原寸(5000px級)を表現しうる巨大なバイナリ/配列は含まない。
    assert set(body.keys()) == {
        "page_id",
        "quality",
        "panels",
        "preview_width",
        "preview_height",
        "mask_urls",
        "metrics",
    }
    assert all(isinstance(url, str) for url in body["mask_urls"].values())
    assert body["preview_width"] <= PreviewParams().max_long_side
    assert body["preview_height"] <= PreviewParams().max_long_side
    # 応答本文自体も画素データを含まないため十分小さい。
    assert len(response.content) < 10_000


def test_analyze_scales_panel_polygons_to_preview_coordinates(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    height, width = 2000, 800
    page = np.full((height, width), 255, dtype=np.uint8)
    _draw_border(page, 40, 40, width - 40, height - 40)
    page_id = _ingest_page(client, page)

    params = AnalysisParams()
    response = _analyze(client, page_id, params)

    assert response.status_code == 200
    body = response.json()
    assert body["preview_height"] <= PreviewParams().max_long_side
    assert len(body["panels"]) == 1

    scale_x = body["preview_width"] / width
    scale_y = body["preview_height"] / height

    full_res_panels = core_detect_panels(page, params.panel)
    assert len(full_res_panels) == 1
    expected_polygon = [
        (round(x * scale_x, 3), round(y * scale_y, 3)) for x, y in full_res_panels[0].polygon
    ]
    actual_polygon = [(round(x, 3), round(y, 3)) for x, y in body["panels"][0]["polygon"]]
    assert actual_polygon == expected_polygon


def test_analyze_reflects_changed_params(tmp_path: Path, monkeypatch) -> None:
    """パラメータを変えて再解析すると結果が変わること。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    page = _panel_grid_page()
    page_id = _ingest_page(client, page)

    default_response = _analyze(client, page_id, AnalysisParams())
    assert default_response.status_code == 200
    assert len(default_response.json()["panels"]) == 4

    strict_params = AnalysisParams()
    # 各コマの面積比は約 0.13。0.5 まで引き上げるとすべて弾かれる。
    strict_params.panel.min_panel_area_ratio = 0.5
    strict_response = _analyze(client, page_id, strict_params)
    assert strict_response.status_code == 200
    assert strict_response.json()["panels"] == []


def test_analyze_persists_full_resolution_masks_to_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    page = _panel_grid_page()
    page_id = _ingest_page(client, page)

    response = _analyze(client, page_id, AnalysisParams())
    assert response.status_code == 200

    for kind in ("line", "fill", "tone", "balloon"):
        mask_path = tmp_path / "ws" / "pages" / page_id / "masks" / f"{kind}.png"
        assert mask_path.is_file()
        mask_array = _decode_png(mask_path.read_bytes())
        assert mask_array.shape == page.shape


def test_get_mask_returns_preview_and_full_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    long_side = PreviewParams().max_long_side + 400
    page = np.full((long_side, long_side), 255, dtype=np.uint8)
    page[100:600, 100:600] = 0  # ベタマスク検出用の黒領域
    page_id = _ingest_page(client, page)

    assert _analyze(client, page_id, AnalysisParams()).status_code == 200

    preview_response = client.get(f"/api/pages/{page_id}/mask/fill")
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"] == "image/png"
    preview_mask = _decode_png(preview_response.content)
    assert max(preview_mask.shape) <= PreviewParams().max_long_side

    full_response = client.get(f"/api/pages/{page_id}/mask/fill", params={"preview": 0})
    assert full_response.status_code == 200
    full_mask = _decode_png(full_response.content)
    assert full_mask.shape == (long_side, long_side)


def test_get_mask_returns_404_before_analyze(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    page_id = _ingest_page(client, np.full((300, 300), 255, dtype=np.uint8))

    response = client.get(f"/api/pages/{page_id}/mask/line")

    assert response.status_code == 404


def test_get_mask_returns_404_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/pages/does-not-exist/mask/line")

    assert response.status_code == 404


def test_analyze_returns_404_for_unknown_page(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = _analyze(client, "does-not-exist", AnalysisParams())

    assert response.status_code == 404


# --- metrics: GT マッピングがある場合に IoU/F1 を返す（#8 の gt/metrics との接続） -------


def _make_metrics_psd(path: Path) -> None:
    """全面が不透明黒の Fill レイヤー 1 枚だけを持つ PSD（GT 指標算出の疎通確認用）。

    `tests/test_batch.py::_make_metrics_psd` と同じ構成
    （`core.metrics.compute_page_decompose_metrics` を両者が共通で呼ぶ契約の検証のため。
    Review #25: 除外規則の二重実装を無くしたので、この関数は routes_analyze と
    batch のどちらの経路でも同一の計算になる）。
    """
    psd = PSDImage.new("RGBA", (60, 60), color=(255, 255, 255, 255))
    group = Group.new(psd, "Group1")
    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # 不透明黒
    PixelLayer.frompil(Image.fromarray(rgba, mode="RGBA"), group, "FillLayer", top=0, left=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


def test_analyze_returns_metrics_when_gt_mapping_exists(tmp_path: Path, monkeypatch) -> None:
    """GT マッピング保存済みの PSD ページでは `metrics` に IoU/F1 が入ること（routes_analyze <-> core.gt/core.metrics の接続確認）。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    psd_path = tmp_path / "art.psd"
    _make_metrics_psd(psd_path)

    with psd_path.open("rb") as f:
        response = client.post(
            "/api/ingest", files={"file": ("art.psd", f, "application/octet-stream")}
        )
    assert response.status_code == 200
    page_id = response.json()[0]["page_id"]

    # マッピング未保存の間は metrics は None（データが繋がっているだけで無条件に埋まらないこと）。
    # フィクスチャは 60x60 と小さいため、専用の緩い品質閾値を使う（tests/test_batch.py と同じ）。
    params = AnalysisParams(quality=QualityParams(min_short_side=1))
    before = _analyze(client, page_id, params)
    assert before.status_code == 200
    assert before.json()["metrics"] is None

    gt.save_mapping(page_id, {"Group1/FillLayer": "fill"})

    after = _analyze(client, page_id, params)
    assert after.status_code == 200
    metrics = after.json()["metrics"]
    assert metrics is not None
    assert "fill" in metrics
    assert metrics["fill"]["iou"] == pytest.approx(1.0)
    assert metrics["fill"]["precision"] == pytest.approx(1.0)
    assert metrics["fill"]["recall"] == pytest.approx(1.0)
    assert metrics["fill"]["f1"] == pytest.approx(1.0)
    assert metrics["fill"]["support"] > 0

    # line/tone: GT にも予測にも該当画素が無いロール（Review #18）。1.0（満点）で
    # 水増しせず None（値なし）を返す。tests/test_batch.py の同名シナリオと揃える。
    for role in ("line", "tone"):
        assert metrics[role]["iou"] is None
        assert metrics[role]["precision"] is None
        assert metrics[role]["recall"] is None
        assert metrics[role]["f1"] is None
        assert metrics[role]["support"] == 0
