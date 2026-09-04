"""バッチ処理（`core.batch` / `api.routes_batch`）のテスト（T11, #12）。

実原稿は手元に無いため、`tests/test_api_analyze.py` と同じ手法で合成フィクスチャを使う
（Epic 本文「テスト方針」）。`workspace/` はテストごとに `ROUGH2INK_WORKSPACE_DIR` で隔離する
（`tests/test_routes_ingest.py` と同じパターン）。
"""

from __future__ import annotations

import json
import time
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
from rough2ink.core.batch import BatchProgress, discover_input_files, run_batch
from rough2ink.core.loaders import load_psd
from rough2ink.core.params import AnalysisParams, QualityParams
from rough2ink.core.presets import save_preset

# --- 合成フィクスチャ（tests/test_api_analyze.py と同じ描き方） -----------------


def _draw_border(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, thickness: int = 3) -> None:
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


def _write_png(path: Path, array: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", array)
    assert ok
    path.write_bytes(buf.tobytes())


def _decode_png(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


# 合成ページは 500px 級のため、既定の min_short_side(2000) では常に fail してしまう。
# fail 判定そのものを検証するテスト以外は、この閾値を使って意図的に pass させる。
_PASSABLE_QUALITY = QualityParams(min_short_side=100)


def _make_metrics_psd(path: Path) -> None:
    """全面が不透明黒の Fill レイヤー 1 枚だけを持つ PSD（GT 指標算出の疎通確認用）。"""
    psd = PSDImage.new("RGBA", (60, 60), color=(255, 255, 255, 255))
    group = Group.new(psd, "Group1")
    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # 不透明黒
    PixelLayer.frompil(Image.fromarray(rgba, mode="RGBA"), group, "FillLayer", top=0, left=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


# --- discover_input_files ------------------------------------------------------


def test_discover_input_files_lists_supported_suffixes_sorted(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_png(input_dir / "b.png", _panel_grid_page())
    _write_png(input_dir / "a.jpg", _panel_grid_page())
    (input_dir / "ignore.txt").write_text("not an image")

    files = discover_input_files(input_dir)

    assert [p.name for p in files] == ["a.jpg", "b.png"]


# --- run_batch: 正常系のディレクトリレイアウト ----------------------------------


def test_run_batch_writes_expected_output_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"

    page = _panel_grid_page()
    _write_png(input_dir / "page_a.png", page)

    params = AnalysisParams(quality=_PASSABLE_QUALITY)
    report = run_batch(input_dir, out_dir, params)

    assert report["total_pages"] == 1
    assert report["included_count"] == 1
    assert report["excluded_count"] == 0

    page_dir = out_dir / "page_a"
    assert (page_dir / "page.png").is_file()
    page_png = _decode_png((page_dir / "page.png").read_bytes())
    assert page_png.shape == page.shape  # 原寸のまま

    quality = json.loads((page_dir / "quality.json").read_text(encoding="utf-8"))
    assert quality["status"] == "pass"

    for kind in ("line", "fill", "tone", "balloon"):
        mask_path = page_dir / "masks" / f"{kind}.png"
        assert mask_path.is_file()
        mask = _decode_png(mask_path.read_bytes())
        assert mask.shape == page.shape  # マスクも原寸
        assert set(np.unique(mask)).issubset({0, 255})  # 0/255 二値

    panels = json.loads((page_dir / "panels.json").read_text(encoding="utf-8"))
    assert len(panels) == 4
    for panel in panels:
        assert panel["flags"] == []

    panels_dir = page_dir / "panels"
    for panel in panels:
        crop_path = panels_dir / f"{panel['panel_id']}.png"
        mask_crop_path = panels_dir / f"{panel['panel_id']}_mask.png"
        assert crop_path.is_file()
        assert mask_crop_path.is_file()
        x, y, w, h = panel["bbox"]
        crop = _decode_png(crop_path.read_bytes())
        assert crop.shape == (h, w)

    # GT 未割当のため metrics.json は書き出されない。
    assert not (page_dir / "metrics.json").is_file()

    assert (out_dir / "report.json").is_file()
    assert (out_dir / "report.md").is_file()
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# バッチ処理レポート" in report_md
    assert "## 分解器マクロ平均 IoU/F1" in report_md
    assert "## コマ分割" in report_md


def test_run_batch_multiple_pages_aggregate_panel_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"

    _write_png(input_dir / "page_a.png", _panel_grid_page())
    _write_png(input_dir / "page_b.png", _panel_grid_page())

    params = AnalysisParams(quality=_PASSABLE_QUALITY)
    report = run_batch(input_dir, out_dir, params)

    assert report["included_count"] == 2
    assert report["panel_metrics"]["total_panel_count"] == 8  # 4 コマ x 2 ページ
    assert report["panel_metrics"]["success_rate"] == pytest.approx(1.0)
    assert report["panel_metrics"]["panel_count_by_page"] == {"page_a": 4, "page_b": 4}


# --- run_batch: 品質ゲート fail の除外 ------------------------------------------


def test_run_batch_excludes_fail_quality_pages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"

    _write_png(input_dir / "good.png", _panel_grid_page())
    # 短辺 50px の小さい画像は quality.min_short_side=100 未満で fail する。
    _write_png(input_dir / "tiny.png", np.full((50, 50), 255, dtype=np.uint8))

    params = AnalysisParams(quality=_PASSABLE_QUALITY)
    report = run_batch(input_dir, out_dir, params)

    assert report["included_count"] == 1
    assert report["excluded_count"] == 1

    tiny_page = next(p for p in report["pages"] if p["page_id"] == "tiny")
    assert tiny_page["status"] == "excluded"
    assert tiny_page["quality_status"] == "fail"
    assert any("short_side" in reason for reason in tiny_page["reasons"])

    # 除外ページは quality.json のみ。masks/panels/page.png は作られない。
    tiny_dir = out_dir / "tiny"
    assert (tiny_dir / "quality.json").is_file()
    assert not (tiny_dir / "page.png").is_file()
    assert not (tiny_dir / "masks").exists()
    assert not (tiny_dir / "panels.json").is_file()

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## 除外ページ（品質ゲート fail）" in report_md
    assert "tiny" in report_md


# --- run_batch: 進捗コールバック ------------------------------------------------


def test_run_batch_reports_progress_for_each_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"

    _write_png(input_dir / "a.png", _panel_grid_page())
    _write_png(input_dir / "b.png", _panel_grid_page())

    events: list[BatchProgress] = []
    params = AnalysisParams(quality=_PASSABLE_QUALITY)
    run_batch(input_dir, out_dir, params, progress_callback=events.append)

    assert [e.page_id for e in events] == ["a", "a", "b", "b"]
    assert [e.status for e in events] == ["processing", "included", "processing", "included"]
    assert all(e.total == 2 for e in events)
    assert [e.index for e in events] == [1, 1, 2, 2]


# --- run_batch: GT がある場合の metrics.json ------------------------------------


def test_run_batch_computes_metrics_when_gt_mapping_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"

    psd_path = input_dir / "art.psd"
    _make_metrics_psd(psd_path)

    # フィクスチャは 60x60 と小さいため、専用の緩い閾値を使う。
    params = AnalysisParams(quality=QualityParams(min_short_side=1))

    # 1回目: GT マッピング未設定なので metrics.json は書き出されない
    # （バッチ処理自体が workspace/pages/art/meta.json を永続化し、GT 設定を可能にする）。
    first_report = run_batch(input_dir, out_dir, params)
    assert first_report["pages_with_metrics"] == 0
    assert not (out_dir / "art" / "metrics.json").is_file()

    # GT マッピングのキーは `LayerInfo.id`（同名レイヤーがあっても衝突しない、#20）。
    fill_layer_id = next(
        layer.id for layer in load_psd(psd_path).layers if layer.path == "Group1/FillLayer"
    )
    gt.save_mapping("art", {fill_layer_id: "fill"})

    # 2回目: 同じ page_id ("art") に GT マッピングが設定済みなので metrics.json が書かれる。
    second_report = run_batch(input_dir, out_dir, params)
    assert second_report["pages_with_metrics"] == 1

    metrics = json.loads((out_dir / "art" / "metrics.json").read_text(encoding="utf-8"))
    assert "fill" in metrics
    assert metrics["fill"]["iou"] == pytest.approx(1.0)

    assert "fill" in second_report["decompose_macro_average"]
    assert second_report["decompose_macro_average"]["fill"]["iou"] == pytest.approx(1.0)

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "| fill |" in report_md


def test_run_batch_raises_for_missing_input_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(NotADirectoryError):
        run_batch(tmp_path / "does-not-exist", tmp_path / "out")


# --- API: POST /api/batch, GET /api/batch/{job_id} -----------------------------


def _wait_for_job(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/batch/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    pytest.fail(f"batch job {job_id!r} did not finish within {timeout}s")


def test_start_batch_and_poll_until_done(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_png(input_dir / "page_a.png", _panel_grid_page())

    start_response = client.post(
        "/api/batch",
        json={
            "input_dir": str(input_dir),
            "out_dir": str(out_dir),
            "preset": None,
        },
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    final_status = _wait_for_job(client, job_id)

    assert final_status["status"] == "done"
    assert final_status["processed"] == final_status["total"] == 1
    assert final_status["report"] is not None
    assert (out_dir / "report.json").is_file()


def test_start_batch_uses_named_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    save_preset("low-threshold", AnalysisParams(quality=_PASSABLE_QUALITY))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_png(input_dir / "page_a.png", _panel_grid_page())

    start_response = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": str(out_dir), "preset": "low-threshold"},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    final_status = _wait_for_job(client, job_id)

    assert final_status["status"] == "done"
    assert final_status["report"]["included_count"] == 1


def test_start_batch_returns_400_for_missing_input_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.post(
        "/api/batch", json={"input_dir": str(tmp_path / "does-not-exist"), "out_dir": None, "preset": None}
    )

    assert response.status_code == 400


def test_start_batch_returns_404_for_unknown_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    input_dir = tmp_path / "in"
    input_dir.mkdir()

    response = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": None, "preset": "does-not-exist"},
    )

    assert response.status_code == 404


def test_get_batch_status_returns_404_for_unknown_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/batch/does-not-exist")

    assert response.status_code == 404
