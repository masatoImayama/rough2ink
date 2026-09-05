"""GT に対するパラメータ自動フィッティングのテスト。

探索そのものより、**目的関数が抜け穴を持たないこと**を固定するのが主眼。
最初の実測では「GT が空のロールを予測しなくなる」ことでマクロ平均が
0.208 → 0.279 に上がっていた（分解が良くなったわけではない）。
"""

from __future__ import annotations

import numpy as np

from rough2ink.core.autofit import _Sample, build_samples, fit_params, scored_roles
from rough2ink.core.params import AnalysisParams


def _mask(shape: tuple[int, int], region: tuple[slice, slice] | None = None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if region is not None:
        mask[region] = 255
    return mask


def test_scored_roles_only_includes_roles_present_in_gt() -> None:
    """GT に陽性画素が無いロールは目的関数に含めない（抜け穴を塞ぐ）。"""
    shape = (32, 32)
    sample = _Sample(
        gray=np.full(shape, 255, dtype=np.uint8),
        gt={
            "line": _mask(shape, (slice(4, 8), slice(4, 28))),
            "fill": _mask(shape),  # GT に存在しない
            "tone": _mask(shape),  # GT に存在しない
        },
        exclude=None,
    )

    assert scored_roles([sample]) == {"line"}


def test_scored_roles_ignores_gt_inside_excluded_region() -> None:
    """除外領域（フキダシ等）の中にしか GT が無いロールは評価対象にしない。"""
    shape = (32, 32)
    region = (slice(4, 8), slice(4, 28))
    sample = _Sample(
        gray=np.full(shape, 255, dtype=np.uint8),
        gt={"line": _mask(shape, region)},
        exclude=_mask(shape, region),  # GT と完全に重なる除外領域
    )

    assert scored_roles([sample]) == set()


def test_predicting_nothing_for_a_gt_role_scores_zero_not_excluded() -> None:
    """GT にあるロールを一切予測しなくても、平均から外れず 0 点として数えること。

    外してしまうと「予測をやめる」ほどスコアが上がる（最初の実測で実際に起きた）。
    """
    shape = (32, 32)
    # GT には fill があるが、入力は真っ白なので分解器は何も出力しない。
    sample = _Sample(
        gray=np.full(shape, 255, dtype=np.uint8),
        gt={"fill": _mask(shape, (slice(8, 24), slice(8, 24)))},
        exclude=None,
    )

    result = fit_params([sample], AnalysisParams(), passes=1)

    assert result.baseline_score == 0.0
    assert result.best_score == 0.0
    assert result.best_metrics["fill"]["f1"] == 0.0


def test_fit_improves_score_on_a_case_the_defaults_get_wrong() -> None:
    """既定値では取りこぼす入力に対し、探索がスコアを改善すること。

    薄い（輝度の高い）線は既定の `line.black_threshold=128` では拾えない。
    閾値を上げる候補が探索空間にあるので、改善できるはず。
    """
    shape = (64, 64)
    gray = np.full(shape, 255, dtype=np.uint8)
    gray[30:33, 4:60] = 170  # 既定閾値(128)より明るいグレーの線
    gt = {"line": _mask(shape, (slice(30, 33), slice(4, 60)))}

    sample = _Sample(gray=gray, gt=gt, exclude=None)
    result = fit_params([sample], AnalysisParams(), passes=1)

    assert result.baseline_score == 0.0, "既定値では拾えない前提が崩れている"
    assert result.best_score > 0.8, f"探索で改善するべき, got {result.best_score}"
    assert result.params.line.black_threshold > 128


def test_build_samples_picks_crops_with_gt_content() -> None:
    """GT に中身のある位置からクロップを選ぶ（白紙だけのクロップは探索の役に立たない）。"""
    shape = (200, 200)
    gray = np.full(shape, 255, dtype=np.uint8)
    gt_masks = {
        "line": _mask(shape, (slice(150, 190), slice(150, 190))),
        "fill": _mask(shape),
        "tone": _mask(shape),
    }

    samples = build_samples(gray, gt_masks, None, crop_size=64, max_crops=3)

    assert samples, "GT に中身があるのにクロップが選ばれていない"
    for sample in samples:
        assert (sample.gt["line"] > 0).any(), "GT が空のクロップを選んでいる"


def test_build_samples_returns_empty_when_gt_is_empty() -> None:
    """GT が空なら評価のしようがないので空リストを返す（呼び出し側でエラーにする）。"""
    shape = (200, 200)
    gray = np.full(shape, 255, dtype=np.uint8)
    gt_masks = {role: _mask(shape) for role in ("line", "fill", "tone")}

    assert build_samples(gray, gt_masks, None, crop_size=64, max_crops=3) == []
