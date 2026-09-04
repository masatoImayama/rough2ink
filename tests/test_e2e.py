"""E2E テスト（T12, #13）: 3 系統の入力それぞれで通しの経路を検証する。

Epic 本文「テスト方針」のとおり実原稿は手元に無いため合成フィクスチャを使うが、
本ファイルは各モジュールの単体テストではなく、**API を通じた経路のつながり**を検証する
（`tests/test_api_analyze.py` / `tests/test_batch.py` / `tests/test_gt.py` 等の単体テストと
役割が違う）。`workspace/` はテストごとに `ROUGH2INK_WORKSPACE_DIR` で隔離する。

1. 画像入力 → 品質ゲート → 分解 → コマ分割 → フキダシ → 解析API → バッチ書き出し
2. PSD 入力 → レイヤー一覧 → 役割マッピング → GT 生成 → 指標（IoU/F1）→ バッチ書き出し
   （routes_analyze.py の `metrics` 接続漏れ修正の検証を兼ねる）
3. PDF 入力（複数ページ）→ ページ展開 → 一括解析（バッチAPI）→ レポート
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
from rough2ink.core.params import AnalysisParams, QualityParams
from rough2ink.core.presets import save_preset

# --- 合成フィクスチャ（tests/test_api_analyze.py, tests/test_batch.py と同じ描き方） --------


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


def _png_bytes(array: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


def _make_metrics_psd(path: Path) -> None:
    """全面が不透明黒の Fill レイヤー 1 枚だけを持つ PSD（GT 指標算出の疎通確認用）。

    `tests/test_batch.py::_make_metrics_psd` と同じ構成。
    """
    psd = PSDImage.new("RGBA", (60, 60), color=(255, 255, 255, 255))
    group = Group.new(psd, "Group1")
    rgba = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # 不透明黒
    PixelLayer.frompil(Image.fromarray(rgba, mode="RGBA"), group, "FillLayer", top=0, left=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(path))


def _make_pdf(path: Path, page_sizes_px: list[tuple[int, int]]) -> None:
    """`tests/test_loaders.py::_make_pdf` と同じ 72dpi 書き出し（px == pt になる）。"""
    images = [Image.new("L", size, color=255) for size in page_sizes_px]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=72.0)


# --- 経路1: 画像入力 → 品質ゲート → 分解 → コマ分割 → フキダシ → 解析API → バッチ書き出し ----


def test_e2e_image_path_quality_decompose_panels_balloons_analyze_batch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    page = _panel_grid_page()

    # 取り込み
    ingest_response = client.post(
        "/api/ingest", files={"file": ("page.png", _png_bytes(page), "image/png")}
    )
    assert ingest_response.status_code == 200
    summary = ingest_response.json()[0]
    page_id = summary["page_id"]
    assert summary["source_kind"] == "image"
    assert summary["width"] == 500 and summary["height"] == 500

    # 単一ページ解析API: 品質ゲート・分解・コマ分割・フキダシが一式で返る
    analyze_response = client.post(
        f"/api/pages/{page_id}/analyze", json=AnalysisParams().model_dump(mode="json")
    )
    assert analyze_response.status_code == 200
    body = analyze_response.json()
    assert body["quality"]["short_side"] == 500
    assert len(body["panels"]) == 4
    for panel in body["panels"]:
        assert panel["flags"] == []
    # 画像入力には GT マッピングが無いため metrics は None のまま。
    assert body["metrics"] is None

    # マスク（線/ベタ/トーン/フキダシ）の個別エンドポイントが取得できる。
    for kind in ("line", "fill", "tone", "balloon"):
        mask_response = client.get(f"/api/pages/{page_id}/mask/{kind}")
        assert mask_response.status_code == 200
        assert mask_response.headers["content-type"] == "image/png"

    # バッチ書き出し: 同じ画像をフォルダ一括処理し、中間成果物一式を確認する。
    # 合成ページは 500px 級のため、既定の min_short_side(2000) では fail してしまう。
    # 品質ゲート自体は tests/test_quality.py で個別に検証済みのため、ここでは緩めた
    # プリセットで通す（tests/test_batch.py::_PASSABLE_QUALITY と同じ考え方）。
    input_dir = tmp_path / "batch_in"
    input_dir.mkdir()
    (input_dir / "page_a.png").write_bytes(_png_bytes(page))
    out_dir = tmp_path / "batch_out"

    save_preset("image-e2e", AnalysisParams(quality=QualityParams(min_short_side=100)))
    start_response = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": str(out_dir), "preset": "image-e2e"},
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]
    final_status = _wait_for_job(client, job_id)
    assert final_status["status"] == "done"
    assert final_status["report"]["included_count"] == 1

    page_out_dir = out_dir / "page_a"
    assert (page_out_dir / "page.png").is_file()
    assert (page_out_dir / "quality.json").is_file()
    for kind in ("line", "fill", "tone", "balloon"):
        assert (page_out_dir / "masks" / f"{kind}.png").is_file()
    panels = json.loads((page_out_dir / "panels.json").read_text(encoding="utf-8"))
    assert len(panels) == 4
    for panel in panels:
        assert (page_out_dir / "panels" / f"{panel['panel_id']}.png").is_file()
        assert (page_out_dir / "panels" / f"{panel['panel_id']}_mask.png").is_file()
    # GT 未割当のため metrics.json は書き出されない。
    assert not (page_out_dir / "metrics.json").is_file()
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "report.md").is_file()


# --- 経路2: PSD 入力 → レイヤー一覧 → 役割マッピング → GT 生成 → 指標 → バッチ書き出し -------


def test_e2e_psd_path_layer_mapping_gt_metrics_analyze_and_batch(
    tmp_path: Path, monkeypatch
) -> None:
    """routes_analyze.py の `metrics` 接続漏れ修正の検証を兼ねる。

    UI (`web/js/app.js::renderGTMetrics`) は `metrics` が非 null なら IoU/F1 テーブルを
    表示する実装を持つ。ここではバックエンド (`POST /api/pages/{page_id}/analyze`) が
    実際に `metrics` を埋めて返すことを、GT マッピング保存済みの PSD ページで確認する。
    """
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    psd_path = tmp_path / "art.psd"
    _make_metrics_psd(psd_path)

    with psd_path.open("rb") as fh:
        ingest_response = client.post(
            "/api/ingest", files={"file": ("art.psd", fh, "application/octet-stream")}
        )
    assert ingest_response.status_code == 200
    summary = ingest_response.json()[0]
    page_id = summary["page_id"]
    assert summary["source_kind"] == "psd"

    # レイヤー一覧（UI のレイヤーマッピング画面が使う入力）。
    layers_response = client.get(f"/api/pages/{page_id}/layers")
    assert layers_response.status_code == 200
    layers = layers_response.json()
    layer_paths = [layer["path"] for layer in layers]
    assert "Group1/FillLayer" in layer_paths
    # GT マッピングのキーは `LayerInfo.id`（同名レイヤーがあっても衝突しない、#20）。
    fill_layer_id = next(layer["id"] for layer in layers if layer["path"] == "Group1/FillLayer")

    # 60x60 と小さいフィクスチャのため、専用の緩い品質閾値を使う。
    params = AnalysisParams(quality=QualityParams(min_short_side=1))

    # マッピング未保存の間は metrics は None（GT が無ければ算出しないことの確認）。
    before_response = client.post(
        f"/api/pages/{page_id}/analyze", json=params.model_dump(mode="json")
    )
    assert before_response.status_code == 200
    assert before_response.json()["metrics"] is None

    # 役割マッピングを保存（UI「3. PSD レイヤー役割マッピング」画面が呼ぶ経路）。
    put_response = client.put(
        f"/api/pages/{page_id}/gt", json={"mapping": {fill_layer_id: "fill"}}
    )
    assert put_response.status_code == 200

    # 単一ページ解析API: マッピング保存後は IoU/F1 が metrics に入る。
    after_response = client.post(
        f"/api/pages/{page_id}/analyze", json=params.model_dump(mode="json")
    )
    assert after_response.status_code == 200
    metrics = after_response.json()["metrics"]
    assert metrics is not None
    assert metrics["fill"]["iou"] == pytest.approx(1.0)
    assert metrics["fill"]["precision"] == pytest.approx(1.0)
    assert metrics["fill"]["recall"] == pytest.approx(1.0)
    assert metrics["fill"]["f1"] == pytest.approx(1.0)

    # バッチ書き出し: バッチは入力ファイル名の stem ("art") から page_id を決めるため、
    # 上で保存したマッピング（アップロード時の uuid page_id 向け）はそのままでは使えない。
    # `core.batch` の契約どおり、1回目のバッチ実行で `workspace/pages/art/meta.json` を
    # 永続化してから、その page_id ("art") に対して改めてマッピングを保存する
    # （`tests/test_batch.py::test_run_batch_computes_metrics_when_gt_mapping_exists` と同じ手順）。
    input_dir = tmp_path / "batch_in"
    input_dir.mkdir()
    _make_metrics_psd(input_dir / "art.psd")
    out_dir = tmp_path / "batch_out"

    save_preset("psd-e2e", params)
    first_start = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": str(out_dir), "preset": "psd-e2e"},
    )
    assert first_start.status_code == 200
    first_status = _wait_for_job(client, first_start.json()["job_id"])
    assert first_status["status"] == "done"
    assert first_status["report"]["pages_with_metrics"] == 0
    assert not (out_dir / "art" / "metrics.json").is_file()

    # バッチ側の page_id ("art") で改めてレイヤー id を解決する（ページごとに独立した id 空間）。
    batch_layers_response = client.get("/api/pages/art/layers")
    assert batch_layers_response.status_code == 200
    batch_fill_layer_id = next(
        layer["id"] for layer in batch_layers_response.json() if layer["path"] == "Group1/FillLayer"
    )
    batch_gt_response = client.put(
        "/api/pages/art/gt", json={"mapping": {batch_fill_layer_id: "fill"}}
    )
    assert batch_gt_response.status_code == 200

    second_start = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": str(out_dir), "preset": "psd-e2e"},
    )
    assert second_start.status_code == 200
    second_status = _wait_for_job(client, second_start.json()["job_id"])
    assert second_status["status"] == "done"
    assert second_status["report"]["pages_with_metrics"] == 1

    batch_metrics = json.loads((out_dir / "art" / "metrics.json").read_text(encoding="utf-8"))
    assert batch_metrics["fill"]["iou"] == pytest.approx(1.0)


# --- 経路3: PDF 入力（複数ページ）→ ページ展開 → 一括解析（バッチAPI）→ レポート -------------


def test_e2e_pdf_path_multi_page_expansion_batch_and_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    pdf_path = tmp_path / "manuscript.pdf"
    _make_pdf(pdf_path, page_sizes_px=[(200, 100), (200, 100), (200, 100)])

    # 取り込み: PDF は複数ページに展開され、それぞれ独立した page_id を持つ。
    with pdf_path.open("rb") as fh:
        ingest_response = client.post(
            "/api/ingest", files={"file": ("manuscript.pdf", fh, "application/pdf")}
        )
    assert ingest_response.status_code == 200
    summaries = ingest_response.json()
    assert len(summaries) == 3
    ids = [s["page_id"] for s in summaries]
    assert len(set(ids)) == 3  # ページごとに一意
    for summary in summaries:
        assert summary["source_kind"] == "pdf"
        # 既定 600dpi でラスタライズされる（72dpi の 200x100pt page なので約 8.33 倍）。
        assert summary["width"] > 1600

    # 各ページのプレビューが取得できる（オーバーレイ表示の前段）。
    for page_id in ids:
        preview_response = client.get(f"/api/pages/{page_id}/preview")
        assert preview_response.status_code == 200

    # 一括解析（バッチAPI）: 同じ PDF をフォルダに置いて処理し、複数ページ分のレポートを得る。
    input_dir = tmp_path / "batch_in"
    input_dir.mkdir()
    (input_dir / "manuscript.pdf").write_bytes(pdf_path.read_bytes())
    out_dir = tmp_path / "batch_out"

    # 合成ページは白紙（品質ゲート自体は独立にテスト済みのため、ここでは通す前提で緩める）。
    save_preset("pdf-e2e", AnalysisParams(quality=QualityParams(min_short_side=100)))
    start_response = client.post(
        "/api/batch",
        json={"input_dir": str(input_dir), "out_dir": str(out_dir), "preset": "pdf-e2e"},
    )
    assert start_response.status_code == 200
    final_status = _wait_for_job(client, start_response.json()["job_id"], timeout=30.0)

    assert final_status["status"] == "done"
    report = final_status["report"]
    assert report["total_pages"] == 3
    assert report["included_count"] == 3

    page_ids_in_report = {p["page_id"] for p in report["pages"]}
    assert page_ids_in_report == {"manuscript_p001", "manuscript_p002", "manuscript_p003"}

    for suffix in ("p001", "p002", "p003"):
        assert (out_dir / f"manuscript_{suffix}" / "page.png").is_file()
        assert (out_dir / f"manuscript_{suffix}" / "quality.json").is_file()

    assert (out_dir / "report.json").is_file()
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# バッチ処理レポート" in report_md
    assert "総ページ数: 3" in report_md


# --- ヘルパー -------------------------------------------------------------------


def _wait_for_job(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict:
    """`tests/test_batch.py::_wait_for_job` と同じポーリング実装。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/batch/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    pytest.fail(f"batch job {job_id!r} did not finish within {timeout}s")
