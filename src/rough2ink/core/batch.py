"""バッチ処理: フォルダ一括解析と中間成果物・レポート書き出し（Epic 仕様書 8節 / T11）。

`run_batch()` は入力フォルダ内の画像 / PSD / PDF を全て読み込み、品質ゲート→分解→
コマ分割→フキダシ検出の解析一式を行い、**後続の学習パイプラインにそのまま繋げる形式**で
`out/` へ書き出す（Epic 仕様書 5章 Phase 1「パッチ抽出」がこの出力を入力として受け取る）。

```
out/
  <page_id>/
    page.png                   原寸グレースケール
    quality.json               品質ゲートの判定と各指標
    masks/line.png              0/255 二値・原寸
    masks/fill.png
    masks/tone.png
    masks/balloon.png          損失マスク（255 = 無視）
    panels.json                 ポリゴン頂点列・例外フラグ・面積
    panels/panel_000.png        ポリゴン外を白で埋めた bbox 切り出し
    panels/panel_000_mask.png   ポリゴンマスク
    metrics.json                GT がある場合のみ
  report.json                   全ページの集計（機械可読）
  report.md                     同内容の人間可読サマリー
```

品質ゲートで `fail` と判定されたページは分解以降の処理を行わず除外する
（`quality.json` のみを書き出し、`report.json` の `pages[].reasons` に理由を残す）。

`page_id` は入力ファイル名の stem から決定的に生成する（PDF は複数ページに展開されるため
`<stem>_p001` のように連番を付ける、`core.loaders.load_pdf` と同じ規則）。処理対象ページは
`workspace/pages/<page_id>/meta.json` としても永続化する。これにより、GT UI（#7, #8）で
同じ page_id に対して事前に役割マッピング（`workspace/gt/<page_id>.json`）を設定していれば、
`core.gt.build_gt_masks()` をそのまま呼び出して `metrics.json` を算出できる
（マッピングが無いページは `metrics.json` を書き出さない）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from rough2ink.core import gt as gt_module
from rough2ink.core import imageio
from rough2ink.core.balloons import detect_balloons
from rough2ink.core.config import get_gt_dir, get_out_dir, get_workspace_dir
from rough2ink.core.decompose import decompose
from rough2ink.core.loaders import load_image, load_pdf, load_psd
from rough2ink.core.metrics import (
    decompose_metrics,
    macro_average_decompose_metrics,
    panel_flag_metrics,
)
from rough2ink.core.panels import detect_panels, polygon_mask
from rough2ink.core.params import AnalysisParams
from rough2ink.core.quality import evaluate_quality
from rough2ink.core.types import PageDocument, PanelInfo

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PSD_SUFFIXES = {".psd"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = _IMAGE_SUFFIXES | _PSD_SUFFIXES | _PDF_SUFFIXES

_MASK_KINDS: tuple[str, ...] = ("line", "fill", "tone", "balloon")

BatchPageStatus = Literal["processing", "included", "excluded"]


@dataclass
class BatchProgress:
    """進捗通知1件分（`run_batch(..., progress_callback=...)` に渡される）。

    入力ファイル単位の進捗（`index`/`total`）。PDF は複数ページに展開されるため、
    1 ファイルの処理中に同じ `index` で複数回（page ごとに）通知されうる。
    """

    index: int  # 1始まりの入力ファイル番号
    total: int  # 入力ファイル総数
    page_id: str
    status: BatchPageStatus


ProgressCallback = Callable[[BatchProgress], None]


@dataclass
class PageResult:
    """1ページ分の処理結果（`report.json` 生成の中間表現）。"""

    page_id: str
    status: Literal["included", "excluded"]
    quality_status: str
    reasons: list[str] = field(default_factory=list)
    panels: list[PanelInfo] = field(default_factory=list)
    decompose_metrics: dict | None = None


def discover_input_files(input_dir: Path) -> list[Path]:
    """対応拡張子（png/jpg/jpeg/psd/pdf）のファイルをファイル名順（決定的）で列挙する。"""
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES),
        key=lambda p: p.name,
    )


def _load_documents(path: Path) -> list[PageDocument]:
    """入力ファイル 1 件を `PageDocument` のリストへ正規化する（page_id はファイル名 stem 由来）。"""
    suffix = path.suffix.lower()
    stem = path.stem
    if suffix in _IMAGE_SUFFIXES:
        return [load_image(path, page_id=stem)]
    if suffix in _PSD_SUFFIXES:
        return [load_psd(path, page_id=stem)]
    if suffix in _PDF_SUFFIXES:
        return load_pdf(path, page_id_prefix=stem)
    raise ValueError(f"unsupported file type: {suffix!r}")


def run_batch(
    input_dir: Path,
    out_dir: Path | None = None,
    params: AnalysisParams | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """`input_dir` 配下の対応ファイルを一括解析し、`out_dir` へ中間成果物とレポートを書き出す。

    Args:
        input_dir: 画像 / PSD / PDF を含む入力フォルダ。
        out_dir: 出力先。省略時は `core.config.get_out_dir()`。
        params: 解析パラメータ（プリセット由来）。省略時は既定値。
        progress_callback: 各ページの処理前後に呼ばれる進捗通知（省略可）。

    Returns:
        `report.json` と同内容の dict。
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir not found: {input_dir}")

    params = params or AnalysisParams()
    out_dir = out_dir or get_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_input_files(input_dir)
    total_files = len(files)

    page_results: list[PageResult] = []
    for file_index, path in enumerate(files, start=1):
        for doc in _load_documents(path):
            if progress_callback is not None:
                progress_callback(
                    BatchProgress(index=file_index, total=total_files, page_id=doc.page_id, status="processing")
                )
            result = _process_page(doc, out_dir, params)
            page_results.append(result)
            if progress_callback is not None:
                progress_callback(
                    BatchProgress(index=file_index, total=total_files, page_id=doc.page_id, status=result.status)
                )

    report = _build_report(page_results)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_render_report_md(report), encoding="utf-8")
    return report


def _process_page(doc: PageDocument, out_dir: Path, params: AnalysisParams) -> PageResult:
    """1ページ分を解析し、`out_dir/<page_id>/` へ中間成果物を書き出す。"""
    gray = doc.as_array()
    page_out_dir = out_dir / doc.page_id
    page_out_dir.mkdir(parents=True, exist_ok=True)

    quality = evaluate_quality(gray, params.quality)
    (page_out_dir / "quality.json").write_text(
        json.dumps(quality.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if quality.status == "fail":
        return PageResult(
            page_id=doc.page_id,
            status="excluded",
            quality_status=quality.status,
            reasons=quality.reasons,
        )

    imageio.write_gray_png(page_out_dir / "page.png", gray)

    masks = decompose(gray, params)
    panels = detect_panels(gray, params.panel)
    text_rects = [layer.bbox for layer in doc.layers if layer.kind == "type" and layer.bbox] or None
    balloon_mask = detect_balloons(gray, params.balloon, text_rects=text_rects)
    masks["balloon"] = balloon_mask

    masks_dir = page_out_dir / "masks"
    for kind in _MASK_KINDS:
        imageio.write_mask_png(masks_dir / f"{kind}.png", masks[kind])

    (page_out_dir / "panels.json").write_text(
        json.dumps([panel.model_dump(mode="json") for panel in panels], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_panel_crops(gray, panels, page_out_dir / "panels")

    _persist_workspace_meta(doc)
    metrics = _compute_decompose_metrics(doc, masks, balloon_mask)
    if metrics is not None:
        (page_out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return PageResult(
        page_id=doc.page_id,
        status="included",
        quality_status=quality.status,
        reasons=quality.reasons,
        panels=panels,
        decompose_metrics=metrics,
    )


def _write_panel_crops(gray: np.ndarray, panels: list[PanelInfo], panels_dir: Path) -> None:
    """各コマの `panel_NNN.png`（ポリゴン外を白で埋めた bbox 切り出し）と `panel_NNN_mask.png` を書き出す。"""
    if not panels:
        return
    for panel in panels:
        if panel.bbox is None:
            continue
        x, y, w, h = panel.bbox
        if w <= 0 or h <= 0:
            continue
        full_mask = polygon_mask(gray.shape, panel.polygon)
        crop_mask = full_mask[y : y + h, x : x + w]
        crop = gray[y : y + h, x : x + w].copy()
        crop[crop_mask == 0] = 255  # ポリゴン外を白で埋める
        imageio.write_gray_png(panels_dir / f"{panel.panel_id}.png", crop)
        imageio.write_mask_png(panels_dir / f"{panel.panel_id}_mask.png", crop_mask)


def _persist_workspace_meta(doc: PageDocument) -> None:
    """`workspace/pages/<page_id>/meta.json` を永続化する（GT マッピング参照を可能にするため）。

    `routes_ingest._save_page_document` のメタ情報部分と同じスキーマで書き出す
    （`core.gt.build_gt_masks` はこのファイルの `source_path`/`width`/`height` を読む）。
    """
    page_dir = get_workspace_dir() / "pages" / doc.page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "page_id": doc.page_id,
        "filename": doc.source_path.name,
        "source_path": str(doc.source_path),
        "source_kind": doc.source_kind,
        "width": doc.width,
        "height": doc.height,
        "layers": [layer.model_dump(mode="json") for layer in doc.layers],
    }
    (page_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _compute_decompose_metrics(
    doc: PageDocument, masks: dict[str, np.ndarray], balloon_mask: np.ndarray
) -> dict | None:
    """GT マッピング（`workspace/gt/<page_id>.json`）が存在する場合のみ分解器指標を算出する。"""
    gt_path = get_gt_dir(create=False) / f"{doc.page_id}.json"
    if not gt_path.is_file():
        return None
    try:
        gt_masks = gt_module.build_gt_masks(doc.page_id)
    except (gt_module.PageNotFoundError, gt_module.GTMappingError):
        return None
    exclude_mask = (((gt_masks["text"] > 0) | (balloon_mask > 0)).astype(np.uint8)) * 255
    return decompose_metrics(masks, gt_masks, exclude_mask=exclude_mask)


def _build_report(page_results: list[PageResult]) -> dict:
    """全ページの `PageResult` から `report.json` 相当の集計 dict を組み立てる。"""
    included = [r for r in page_results if r.status == "included"]
    excluded = [r for r in page_results if r.status == "excluded"]

    panels_by_page = {r.page_id: r.panels for r in included}
    panel_metrics = panel_flag_metrics(panels_by_page)

    metrics_list = [r.decompose_metrics for r in included if r.decompose_metrics]
    decompose_macro_average = macro_average_decompose_metrics(metrics_list)

    pages_summary = [
        {
            "page_id": r.page_id,
            "status": r.status,
            "quality_status": r.quality_status,
            "reasons": r.reasons,
            "panel_count": len(r.panels),
            "has_metrics": r.decompose_metrics is not None,
        }
        for r in page_results
    ]

    return {
        "total_pages": len(page_results),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "pages_with_metrics": sum(1 for r in included if r.decompose_metrics),
        "pages": pages_summary,
        "decompose_macro_average": decompose_macro_average,
        "panel_metrics": panel_metrics,
    }


def _render_report_md(report: dict) -> str:
    """`report.json` 相当の dict から人間可読な Markdown サマリーを生成する。"""
    lines: list[str] = ["# バッチ処理レポート", ""]
    lines.append(f"- 総ページ数: {report['total_pages']}")
    lines.append(f"- 解析対象（品質ゲート通過）: {report['included_count']}")
    lines.append(f"- 品質ゲート除外: {report['excluded_count']}")
    lines.append(f"- GT 指標算出済み: {report['pages_with_metrics']}")
    lines.append("")

    lines.append("## 分解器マクロ平均 IoU/F1")
    lines.append("")
    macro = report["decompose_macro_average"]
    if macro:
        lines.append("| role | iou | precision | recall | f1 |")
        lines.append("|---|---|---|---|---|")
        for role in ("line", "fill", "tone"):
            if role not in macro:
                continue
            m = macro[role]
            lines.append(
                f"| {role} | {m['iou']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
            )
    else:
        lines.append("GT が割り当てられたページが無いため算出できません。")
    lines.append("")

    lines.append("## コマ分割")
    lines.append("")
    panel_metrics = report["panel_metrics"]
    lines.append(f"- 検出コマ総数: {panel_metrics['total_panel_count']}")
    lines.append(f"- 成功率（例外フラグなしで検出できたコマの割合）: {panel_metrics['success_rate']:.1%}")
    lines.append("")
    lines.append("| flag | count | page_ratio |")
    lines.append("|---|---|---|")
    for flag, stats in panel_metrics["flags"].items():
        lines.append(f"| {flag} | {stats['count']} | {stats['page_ratio']:.1%} |")
    lines.append("")

    if report["excluded_count"]:
        lines.append("## 除外ページ（品質ゲート fail）")
        lines.append("")
        lines.append("| page_id | reasons |")
        lines.append("|---|---|")
        for page in report["pages"]:
            if page["status"] == "excluded":
                reasons = "; ".join(page["reasons"])
                lines.append(f"| {page['page_id']} | {reasons} |")
        lines.append("")

    return "\n".join(lines)
