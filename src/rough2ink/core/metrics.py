"""定量指標: 分解器の IoU/Precision/Recall/F1 とコマ分割の成功率・例外発生率
（設計書 7章「未決事項」/ Epic 仕様書 6節 / T8）。

本 PoC の第一のゴールに実測値で答えるための集計処理。学習・推論は一切行わず、
既存の分解結果（`core.decompose`, #5）・GT マスク（`core.gt`, #8）・
コマ検出結果（`core.panels`, #6）・フキダシ損失マスク（`core.balloons`, #7）を
突き合わせるだけの純粋な集計関数群。

分解器の評価は `text` 領域とフキダシ損失マスクをノイズ源として評価対象から除外する
契約（`decompose_metrics()` の `exclude_mask` 引数）。呼び出し側が
`gt["text"]` とフキダシ損失マスクの和集合を渡す想定。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import get_args

import numpy as np

from rough2ink.core.types import PanelFlag, PanelInfo

# 分解器の評価対象ロール（GT/予測ともに相互排他の3マスク）。
# `text` は評価対象外（ノイズ源として除外する側の領域）。
_DECOMPOSE_ROLES: tuple[str, ...] = ("line", "fill", "tone")

RoleMetrics = dict[str, float]
DecomposeMetrics = dict[str, RoleMetrics]

# `core.panels.PanelInfo.flags` に立ちうる全フラグ。発生 0 件のフラグも
# 集計結果に 0 として出すため、`types.PanelFlag` から機械的に取得する。
_ALL_PANEL_FLAGS: tuple[PanelFlag, ...] = get_args(PanelFlag)


def decompose_metrics(
    pred: Mapping[str, np.ndarray],
    gt: Mapping[str, np.ndarray],
    exclude_mask: np.ndarray | None = None,
) -> DecomposeMetrics:
    """分解結果 `pred` と GT マスク `gt` から役割ごとの IoU/Precision/Recall/F1 を算出する。

    Args:
        pred: `core.decompose.decompose()` の戻り値相当。`{"line": mask, "fill": mask, "tone": mask}`。
            各マスクは同一形状 `(H, W)` の配列で、0 より大きい画素を陽性として扱う。
        gt: `core.gt.build_gt_masks()` の戻り値相当。`text` キーを含んでいてもよいが無視する
            （`text` は評価から除外する側の領域であり、役割の一つとしては評価しない）。
        exclude_mask: 評価から除外する画素（0 より大きい画素 = 除外）。
            `gt["text"]` とフキダシ損失マスク（`core.balloons.detect_balloons()`）の
            和集合を渡す想定。`None` なら除外なし。

    Returns:
        `pred`/`gt` 双方に存在するロールについて
        `{role: {"iou": ..., "precision": ..., "recall": ..., "f1": ...}}` を返す。
        どちらのロールも無ければ空 dict。
    """
    roles = [role for role in _DECOMPOSE_ROLES if role in pred and role in gt]
    if not roles:
        return {}

    sample_shape = pred[roles[0]].shape
    valid = np.ones(sample_shape, dtype=bool)
    if exclude_mask is not None:
        valid &= exclude_mask == 0

    result: DecomposeMetrics = {}
    for role in roles:
        pred_bool = (pred[role] > 0) & valid
        gt_bool = (gt[role] > 0) & valid
        result[role] = _confusion_metrics(pred_bool, gt_bool)
    return result


def _confusion_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray) -> RoleMetrics:
    """真陽性/偽陽性/偽陰性から IoU/Precision/Recall/F1 を算出する（ゼロ除算安全）。

    分母が 0 になる境界ケース（該当画素が予測にも GT にも存在しない等）は、
    「比較対象が無いので一致とみなす（1.0）」という慣例に沿って安全な値を返す。
    F1 は precision/recall の調和平均。両方が 0 の場合のみ 0.0 とする。
    """
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fp = int(np.count_nonzero(pred_bool & ~gt_bool))
    fn = int(np.count_nonzero(~pred_bool & gt_bool))

    iou_denom = tp + fp + fn
    iou = 1.0 if iou_denom == 0 else tp / iou_denom

    precision_denom = tp + fp
    precision = 1.0 if precision_denom == 0 else tp / precision_denom

    recall_denom = tp + fn
    recall = 1.0 if recall_denom == 0 else tp / recall_denom

    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def macro_average_decompose_metrics(per_page: Sequence[DecomposeMetrics]) -> DecomposeMetrics:
    """複数ページ分の `decompose_metrics()` 結果から、ロールごとのマクロ平均を算出する。

    ページによって出現するロールが異なっていてもよい（そのロールの値があったページだけで
    平均する）。入力が空、またはどのページにもロールが1つも無ければ空 dict を返す
    （ゼロ除算は起きない）。
    """
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for page_metrics in per_page:
        for role, metrics in page_metrics.items():
            role_sums = sums.setdefault(role, {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            for key, value in metrics.items():
                role_sums[key] += value
            counts[role] = counts.get(role, 0) + 1

    return {
        role: {key: value / counts[role] for key, value in role_sums.items()}
        for role, role_sums in sums.items()
    }


def panel_flag_metrics(panels_by_page: Mapping[str, Sequence[PanelInfo]]) -> dict:
    """コマ分割の成功率とフラグ別例外発生率を、渡された全ページ分まとめて集計する。

    Args:
        panels_by_page: `{page_id: detect_panels() の戻り値}`。

    Returns:
        ``{
            "panel_count_by_page": {page_id: 検出コマ数},
            "total_panel_count": 検出コマ総数（全ページ合計）,
            "success_rate": 例外フラグが1つも立たずに閉領域が取れたコマ数 / 検出コマ総数,
            "flags": {
                flag: {
                    "count": そのフラグが立ったコマの件数,
                    "page_ratio": そのフラグが1件以上発生したページの割合,
                }
                for flag in 全 PanelFlag（未発生のフラグも 0 で含む）
            },
        }``

        検出コマ総数が 0、または対象ページ数が 0 の場合も例外を出さない
        （該当する率は 0.0 として扱う。「比較対象が無いので成功」ではなく
        「何も検出できなかった」ことを表すため、`decompose_metrics` とは逆に 0.0 を採用する）。
    """
    panel_count_by_page = {page_id: len(panels) for page_id, panels in panels_by_page.items()}
    all_panels = [panel for panels in panels_by_page.values() for panel in panels]
    total_panel_count = len(all_panels)

    success_rate = (
        0.0
        if total_panel_count == 0
        else sum(1 for panel in all_panels if panel.is_clean) / total_panel_count
    )

    total_pages = len(panels_by_page)
    flags: dict[str, dict[str, float]] = {}
    for flag in _ALL_PANEL_FLAGS:
        count = sum(1 for panel in all_panels if flag in panel.flags)
        pages_with_flag = sum(
            1 for panels in panels_by_page.values() if any(flag in panel.flags for panel in panels)
        )
        page_ratio = 0.0 if total_pages == 0 else pages_with_flag / total_pages
        flags[flag] = {"count": count, "page_ratio": page_ratio}

    return {
        "panel_count_by_page": panel_count_by_page,
        "total_panel_count": total_panel_count,
        "success_rate": success_rate,
        "flags": flags,
    }
