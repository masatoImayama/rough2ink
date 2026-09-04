"""定量指標（`rough2ink.core.metrics`）のテスト（T8, #9）。

分解器の IoU/Precision/Recall/F1 が、text/フキダシ除外・境界ケースのゼロ除算・
マクロ平均のいずれについても仕様どおりに動くことを確認する。加えてコマ分割の
成功率とフラグ別例外発生率の集計を検証する（Epic 仕様書 6節）。
"""

from __future__ import annotations

import numpy as np
import pytest

from rough2ink.core.metrics import (
    decompose_metrics,
    macro_average_decompose_metrics,
    panel_flag_metrics,
)
from rough2ink.core.types import PanelInfo

_SHAPE = (10, 10)


def _mask(rows: slice, cols: slice, shape: tuple[int, int] = _SHAPE) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    m[rows, cols] = 255
    return m


def _empty(shape: tuple[int, int] = _SHAPE) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


# --- decompose_metrics: 完全一致 / 完全不一致 --------------------------------


def test_decompose_metrics_perfect_match_gives_iou_one() -> None:
    line = _mask(slice(0, 5), slice(0, 5))
    pred = {"line": line.copy(), "fill": _empty(), "tone": _empty()}
    gt = {"line": line.copy(), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    # line: 25px 完全一致（support=tp+fn=25、valid_pixels=画像全体 100px、除外なし）。
    assert result["line"] == {
        "iou": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 25,
        "valid_pixels": 100,
    }


def test_decompose_metrics_complete_mismatch_gives_iou_zero() -> None:
    pred = {"line": _mask(slice(0, 5), slice(0, 5)), "fill": _empty(), "tone": _empty()}
    gt = {"line": _mask(slice(5, 10), slice(5, 10)), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    assert result["line"]["iou"] == 0.0
    assert result["line"]["precision"] == 0.0
    assert result["line"]["recall"] == 0.0
    assert result["line"]["f1"] == 0.0


def test_decompose_metrics_partial_overlap_computes_confusion_counts() -> None:
    # pred: rows[0,6) cols[0,5) (30px), gt: rows[0,5) cols[0,5) (25px)
    # 重なり(TP)=25, pred のみ(FP)=5(rows[5,6)), gt のみ(FN)=0
    pred = {"line": _mask(slice(0, 6), slice(0, 5)), "fill": _empty(), "tone": _empty()}
    gt = {"line": _mask(slice(0, 5), slice(0, 5)), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    metrics = result["line"]
    assert metrics["precision"] == pytest.approx(25 / 30)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(25 / 30)


# --- decompose_metrics: text / フキダシ除外 -----------------------------------


def test_decompose_metrics_excludes_text_and_balloon_regions() -> None:
    # pred と gt は rows[5,10) cols[5,10) だけ食い違う。この領域を exclude_mask で除外すれば
    # 完全一致として扱われる（除外しなければ不一致が混入する）ことを確認する。
    common = _mask(slice(0, 5), slice(0, 5))
    pred_line = common.copy()
    gt_line = common.copy()
    pred_line[5:10, 5:10] = 255  # pred だけの偽陽性領域（ノイズ）
    gt_line[5:10, 5:10] = 0

    pred = {"line": pred_line, "fill": _empty(), "tone": _empty()}
    gt = {"line": gt_line, "fill": _empty(), "tone": _empty(), "text": _empty()}

    # 除外しない場合は不一致が混ざり IoU < 1.0 になる。
    without_exclude = decompose_metrics(pred, gt)
    assert without_exclude["line"]["iou"] < 1.0

    exclude_mask = _mask(slice(5, 10), slice(5, 10))  # text 領域 or フキダシ損失マスク相当
    with_exclude = decompose_metrics(pred, gt, exclude_mask=exclude_mask)
    # 除外後: 一致部分 25px（support=tp+fn=25）、valid_pixels=100px中 exclude_mask の25px を除いた75px。
    assert with_exclude["line"] == {
        "iou": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 25,
        "valid_pixels": 75,
    }


# --- decompose_metrics: 境界ケースのゼロ除算 ----------------------------------


def test_decompose_metrics_both_empty_is_safe_and_returns_no_data() -> None:
    """GT にも予測にも一切存在しないロール（該当画素が pred/gt どちらにも無い）は、
    「比較対象が無いので満点」ではなく「値なし（None）」を返す（Review #18）。1.0 で
    水増しするとマクロ平均が実力より高く出るため、空虚な 1.0 と真の 1.0 を区別する。
    """
    pred = {"line": _empty(), "fill": _empty(), "tone": _empty()}
    gt = {"line": _empty(), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    for metrics in result.values():
        assert metrics == {
            "iou": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "support": 0,
            "valid_pixels": 100,
        }


def test_decompose_metrics_pred_empty_gt_nonempty_no_exception() -> None:
    pred = {"line": _empty(), "fill": _empty(), "tone": _empty()}
    gt = {"line": _mask(slice(0, 3), slice(0, 3)), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    assert result["line"]["iou"] == 0.0
    assert result["line"]["recall"] == 0.0
    # 何も予測していないので precision は「比較対象が無い」として 1.0 扱い。
    assert result["line"]["precision"] == 1.0


def test_decompose_metrics_gt_empty_pred_nonempty_no_exception() -> None:
    pred = {"line": _mask(slice(0, 3), slice(0, 3)), "fill": _empty(), "tone": _empty()}
    gt = {"line": _empty(), "fill": _empty(), "tone": _empty()}

    result = decompose_metrics(pred, gt)

    assert result["line"]["iou"] == 0.0
    assert result["line"]["precision"] == 0.0
    # GT に陽性が無いので recall は「比較対象が無い」として 1.0 扱い。
    assert result["line"]["recall"] == 1.0


def test_decompose_metrics_all_excluded_is_safe() -> None:
    """全画素が除外され評価対象が無い場合も「値なし（None）」を返す（Review #18）。
    従来は 1.0（満点）で安全とみなしていたが、これは #15（フキダシ損失マスクがページ全面を
    覆う）と組み合わさると report.json 全体が偽の満点になる欠陥だった。
    valid_pixels=0 が「評価対象が無かった」ことを明示する。
    """
    pred = {"line": _mask(slice(0, 5), slice(0, 5)), "fill": _empty(), "tone": _empty()}
    gt = {"line": _mask(slice(5, 10), slice(5, 10)), "fill": _empty(), "tone": _empty()}
    exclude_mask = np.full(_SHAPE, 255, dtype=np.uint8)  # 全画素を除外

    result = decompose_metrics(pred, gt, exclude_mask=exclude_mask)

    for metrics in result.values():
        assert metrics == {
            "iou": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "support": 0,
            "valid_pixels": 0,
        }


def test_decompose_metrics_only_computes_roles_present_in_both() -> None:
    pred = {"line": _mask(slice(0, 3), slice(0, 3))}
    gt = {"line": _mask(slice(0, 3), slice(0, 3)), "fill": _empty(), "text": _empty()}

    result = decompose_metrics(pred, gt)

    assert set(result) == {"line"}


def test_decompose_metrics_empty_input_returns_empty_dict() -> None:
    assert decompose_metrics({}, {}) == {}
    assert decompose_metrics({"line": _empty()}, {}) == {}


# --- macro_average_decompose_metrics -----------------------------------------


def test_macro_average_decompose_metrics_averages_across_pages() -> None:
    page1 = {"line": {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}
    page2 = {"line": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}}

    averaged = macro_average_decompose_metrics([page1, page2])

    # 入力に support/valid_pixels が無い場合は 0 として集計される（.get 既定値）。
    assert averaged["line"] == {
        "iou": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 0,
        "valid_pixels": 0,
    }


def test_macro_average_decompose_metrics_excludes_no_data_pages_from_average() -> None:
    """あるページのロールが「値なし」（None。GT にも予測にも存在しないロール）だった場合、
    そのページを 0 として平均に混入させず、平均対象から除外する（Review #18 の核心）。
    値ありのページだけで平均するため、macro 平均は「値なし」ページに影響されない。
    """
    page_with_data = {
        "line": {"iou": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10, "valid_pixels": 20},
    }
    page_without_data = {
        "line": {
            "iou": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "support": 0,
            "valid_pixels": 15,
        },
    }

    averaged = macro_average_decompose_metrics([page_with_data, page_without_data])

    # 値ありページ（page_with_data）だけの値がそのまま平均になる（0 で薄まらない）。
    assert averaged["line"]["iou"] == pytest.approx(0.8)
    assert averaged["line"]["precision"] == pytest.approx(0.8)
    assert averaged["line"]["recall"] == pytest.approx(0.8)
    assert averaged["line"]["f1"] == pytest.approx(0.8)
    # support/valid_pixels は「値なし」ページも含めて合計する（集計規模を可視化するため）。
    assert averaged["line"]["support"] == 10
    assert averaged["line"]["valid_pixels"] == 35


def test_macro_average_decompose_metrics_all_pages_no_data_returns_none() -> None:
    """全ページで「値なし」だったロールは、マクロ平均も None（値なし）になる。
    0.0 でも 1.0 でもない「算出不能」を維持する。
    """
    page1 = {"line": {"iou": None, "precision": None, "recall": None, "f1": None, "support": 0, "valid_pixels": 10}}
    page2 = {"line": {"iou": None, "precision": None, "recall": None, "f1": None, "support": 0, "valid_pixels": 5}}

    averaged = macro_average_decompose_metrics([page1, page2])

    assert averaged["line"]["iou"] is None
    assert averaged["line"]["precision"] is None
    assert averaged["line"]["recall"] is None
    assert averaged["line"]["f1"] is None
    assert averaged["line"]["support"] == 0
    assert averaged["line"]["valid_pixels"] == 15


def test_macro_average_decompose_metrics_handles_missing_roles_per_page() -> None:
    # page1 は line/fill 両方、page2 は line のみ検出（ロールが出現しないページがある）。
    page1 = {
        "line": {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "fill": {"iou": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
    }
    page2 = {"line": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}}

    averaged = macro_average_decompose_metrics([page1, page2])

    assert averaged["line"]["iou"] == pytest.approx(0.5)
    assert averaged["fill"]["iou"] == pytest.approx(0.5)  # page1 のみで平均


def test_macro_average_decompose_metrics_empty_input_returns_empty_dict() -> None:
    assert macro_average_decompose_metrics([]) == {}


# --- panel_flag_metrics -------------------------------------------------------


def _panel(panel_id: str, flags: list[str]) -> PanelInfo:
    return PanelInfo(panel_id=panel_id, polygon=[(0, 0), (1, 0), (1, 1)], flags=flags)


def test_panel_flag_metrics_success_rate_and_flag_counts() -> None:
    panels_by_page = {
        "page1": [
            _panel("p1", []),  # クリーン
            _panel("p2", ["cut_off"]),
            _panel("p3", ["cut_off", "oblique"]),
        ],
        "page2": [
            _panel("p4", []),  # クリーン
            _panel("p5", ["oblique"]),
        ],
    }

    result = panel_flag_metrics(panels_by_page)

    assert result["panel_count_by_page"] == {"page1": 3, "page2": 2}
    assert result["total_panel_count"] == 5
    # クリーン: p1, p4 の 2件 / 5件
    assert result["success_rate"] == pytest.approx(2 / 5)

    assert result["flags"]["cut_off"]["count"] == 2  # p2, p3
    assert result["flags"]["cut_off"]["page_ratio"] == pytest.approx(1 / 2)  # page1 のみ

    assert result["flags"]["oblique"]["count"] == 2  # p3, p5
    assert result["flags"]["oblique"]["page_ratio"] == pytest.approx(2 / 2)  # page1, page2 両方

    # 発生しなかったフラグも 0 として含まれる。
    assert result["flags"]["unclosed"] == {"count": 0, "page_ratio": 0.0}
    assert result["flags"]["overflow"] == {"count": 0, "page_ratio": 0.0}
    assert result["flags"]["spread"] == {"count": 0, "page_ratio": 0.0}
    assert result["flags"]["effect_lines"] == {"count": 0, "page_ratio": 0.0}


def test_panel_flag_metrics_no_panels_is_safe() -> None:
    result = panel_flag_metrics({"page1": []})

    assert result["total_panel_count"] == 0
    assert result["success_rate"] == 0.0
    for flag_stats in result["flags"].values():
        assert flag_stats == {"count": 0, "page_ratio": 0.0}


def test_panel_flag_metrics_no_pages_is_safe() -> None:
    result = panel_flag_metrics({})

    assert result["panel_count_by_page"] == {}
    assert result["total_panel_count"] == 0
    assert result["success_rate"] == 0.0
    for flag_stats in result["flags"].values():
        assert flag_stats == {"count": 0, "page_ratio": 0.0}
