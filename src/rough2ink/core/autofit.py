"""GT に対して分解器のパラメータを自動フィッティングする。

設計書 3章はレイヤー付き原稿を「**分解器の教師データ兼検証データ**」と位置づけている。
正解が分かっているのだから、閾値は目視で合わせるのではなく **GT に対して当てにいく**のが
本来の使い方であり、ここがその実装にあたる。

同時にこの探索は、設計書 7章「未決事項」の筆頭である
「**分解器の精度が実用に足るか**」への答えを出すためのものでもある。最良のパラメータで
到達できる F1 が低いのであれば、それは調整不足ではなく**古典的画像処理の上限**であり、
分解器そのものを学習させる方向へ切り替える判断材料になる。

探索は座標降下法（1パラメータずつ候補値を総当たりし、良ければ採用）。追加依存を持たず、
同じ入力に対して常に同じ結果を返す（乱数はクロップ位置の選択のみで、シードを固定する）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from rough2ink.core.decompose import decompose
from rough2ink.core.metrics import DecomposeMetrics, decompose_metrics
from rough2ink.core.params import AnalysisParams

# 評価対象のロール。`text` は除外側の領域であり役割としては評価しない。
_ROLES = ("line", "fill", "tone")

# 探索するパラメータと候補値。`tone.window` / `tone.stride` は評価コストそのものを
# 変えてしまい探索の公平性を損なうため、また画素スケール固定の方針（設計書「縦横比の扱い」）
# と結び付いているため、探索対象に含めない。
_SEARCH_SPACE: tuple[tuple[str, tuple[float | int, ...]], ...] = (
    ("line.black_threshold", (96, 112, 128, 144, 160, 176)),
    ("fill.black_threshold", (32, 48, 64, 80, 96)),
    ("fill.erosion_radius", (2, 4, 6, 8, 12, 16, 24)),
    ("fill.min_area_ratio", (0.0002, 0.0005, 0.0008, 0.002, 0.005)),
    ("tone.energy_threshold", (0.10, 0.15, 0.20, 0.30, 0.40)),
    ("tone.sharpness_threshold", (200.0, 500.0, 1500.0, 5000.0, 20000.0)),
    ("tone.min_block_std", (2.0, 6.0, 12.0, 20.0)),
    ("tone.min_direction_ratio", (0.05, 0.20, 0.35, 0.50)),
    ("tone.bandpass_low", (0.04, 0.08, 0.12)),
    ("tone.bandpass_high", (0.35, 0.45, 0.49)),
)


class FitProgress(BaseModel):
    """探索の進捗（1回の評価ごとに通知する）。"""

    evaluated: int
    total: int
    parameter: str
    value: float
    score: float
    best_score: float


@dataclass
class _Sample:
    """評価に使う1クロップ（原寸のまま切り出す。縮小はしない）。"""

    gray: np.ndarray
    gt: dict[str, np.ndarray]
    exclude: np.ndarray | None


@dataclass
class FitResult:
    """フィッティングの結果。"""

    params: AnalysisParams
    baseline_score: float
    best_score: float
    baseline_metrics: DecomposeMetrics = field(default_factory=dict)
    best_metrics: DecomposeMetrics = field(default_factory=dict)
    changed: dict[str, tuple[float, float]] = field(default_factory=dict)
    sample_count: int = 0
    evaluations: int = 0


def _get(params: AnalysisParams, path: str) -> float:
    group, name = path.split(".")
    return getattr(getattr(params, group), name)


def _set(params: AnalysisParams, path: str, value: float) -> AnalysisParams:
    """`params` を書き換えず、指定パラメータだけ差し替えた複製を返す。"""
    group, name = path.split(".")
    updated = params.model_copy(deep=True)
    setattr(getattr(updated, group), name, value)
    return updated


def build_samples(
    gray: np.ndarray,
    gt_masks: dict[str, np.ndarray],
    exclude_mask: np.ndarray | None,
    *,
    crop_size: int = 640,
    max_crops: int = 4,
    seed: int = 0,
) -> list[_Sample]:
    """評価用のクロップを切り出す。

    **原寸のまま切り出す**（縮小するとトーンの網点が壊れ、評価が無意味になる）。
    ページ全体で評価すると1回の評価に数秒かかり座標降下が現実的な時間で終わらないため、
    GT に中身のある位置を選んで数枚のクロップで代表させる。

    GT の陽性画素が多い位置を優先して選ぶ（白紙だけのクロップは、どのパラメータでも
    同じ結果になり探索の役に立たない）。
    """
    height, width = gray.shape
    size = min(crop_size, height, width)
    if size <= 0:
        return []

    content = np.zeros(gray.shape, dtype=bool)
    for role in _ROLES:
        mask = gt_masks.get(role)
        if mask is not None:
            content |= mask > 0
    if exclude_mask is not None:
        content &= ~(exclude_mask > 0)

    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, int, int]] = []
    # 位置をランダムに振って、GT の密度が高い順に採用する。
    for _ in range(max_crops * 12):
        top = int(rng.integers(0, max(1, height - size + 1)))
        left = int(rng.integers(0, max(1, width - size + 1)))
        density = float(content[top : top + size, left : left + size].mean())
        candidates.append((density, top, left))

    candidates.sort(reverse=True)
    samples: list[_Sample] = []
    for density, top, left in candidates:
        if density <= 0.0:
            continue
        rows = slice(top, top + size)
        cols = slice(left, left + size)
        samples.append(
            _Sample(
                gray=gray[rows, cols],
                gt={role: gt_masks[role][rows, cols] for role in _ROLES if role in gt_masks},
                exclude=None if exclude_mask is None else exclude_mask[rows, cols],
            )
        )
        if len(samples) >= max_crops:
            break
    return samples


def _macro_f1(metrics: DecomposeMetrics) -> float:
    """算出できたロールの F1 の平均。算出不能（None）のロールは平均から外す。"""
    values = [
        role_metrics["f1"]
        for role_metrics in metrics.values()
        if role_metrics.get("f1") is not None
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def scored_roles(samples: Sequence[_Sample]) -> set[str]:
    """目的関数に含めるロール（GT に実際に陽性画素があるロール）を返す。

    **GT が空のロールを目的関数に入れてはならない。** 入れると「そのロールを一切
    予測しない」ことでスコアが上がる抜け穴になる。実際、最初の実測では tone の
    GT が空（トーンレイヤーが未割当）だったため、探索が tone を予測しなくなる方向へ
    進み、F1 が 0.000 から「算出不能」に変わってマクロ平均だけが 0.208→0.279 に
    上がっていた（分解が良くなったわけではない）。

    GT にあるロールについては、予測が空でも F1 = 0 として必ず数える。
    """
    roles: set[str] = set()
    for sample in samples:
        for role in _ROLES:
            mask = sample.gt.get(role)
            if mask is None:
                continue
            positive = mask > 0
            if sample.exclude is not None:
                positive &= ~(sample.exclude > 0)
            if positive.any():
                roles.add(role)
    return roles


def _aggregate(
    samples: Sequence[_Sample], params: AnalysisParams, roles: set[str]
) -> tuple[float, DecomposeMetrics]:
    """全クロップの分解結果をまとめ、マクロ平均 F1 と役割別指標を返す。

    `roles` に含まれるロールだけを対象にする（`scored_roles` 参照）。対象ロールで
    指標が算出不能（GT・予測ともに空）だったクロップは、そのクロップに関しては
    比較対象が無いということなので平均に含めない。
    """
    totals: dict[str, dict[str, float]] = {}
    for sample in samples:
        pred = decompose(sample.gray, params)
        metrics = decompose_metrics(pred, sample.gt, sample.exclude)
        for role, role_metrics in metrics.items():
            if role not in roles or role_metrics.get("f1") is None:
                continue
            accumulator = totals.setdefault(role, {"f1": 0.0, "iou": 0.0, "count": 0.0})
            accumulator["f1"] += float(role_metrics["f1"])
            accumulator["iou"] += float(role_metrics["iou"])
            accumulator["count"] += 1.0

    averaged: DecomposeMetrics = {}
    for role in roles:
        accumulator = totals.get(role)
        if accumulator is None or accumulator["count"] == 0:
            # GT にはあるのに全クロップで算出不能＝予測が一切出ていない。0 点として数える。
            averaged[role] = {
                "f1": 0.0,
                "iou": 0.0,
                "support": None,
                "valid_pixels": None,
                "precision": None,
                "recall": None,
            }
            continue
        count = accumulator["count"]
        averaged[role] = {
            "f1": accumulator["f1"] / count,
            "iou": accumulator["iou"] / count,
            "support": None,
            "valid_pixels": None,
            "precision": None,
            "recall": None,
        }
    return _macro_f1(averaged), averaged


def fit_params(
    samples: Sequence[_Sample],
    base_params: AnalysisParams | None = None,
    *,
    passes: int = 2,
    progress_callback: Callable[[FitProgress], None] | None = None,
) -> FitResult:
    """座標降下法でマクロ平均 F1 を最大化するパラメータを探す。

    1パラメータずつ候補値を総当たりし、現在の最良より良ければ採用して次のパラメータへ進む。
    これを `passes` 回繰り返す（前のパスで変わった値が他のパラメータの最適値を動かすため）。

    局所最適に落ちうる素朴な手法だが、パラメータ間の相互作用が強くない前提では十分で、
    追加依存を持たず結果が決定的である利点が大きい。
    """
    params = (base_params or AnalysisParams()).model_copy(deep=True)
    initial = params.model_copy(deep=True)

    # GT に実際に存在するロールだけを目的関数に含める（`scored_roles` の説明を参照）。
    roles = scored_roles(samples)
    baseline_score, baseline_metrics = _aggregate(samples, params, roles)
    best_score = baseline_score
    evaluated = 0
    total = passes * sum(len(values) for _, values in _SEARCH_SPACE)

    for _ in range(passes):
        for path, values in _SEARCH_SPACE:
            current = _get(params, path)
            for value in values:
                evaluated += 1
                if value == current:
                    if progress_callback is not None:
                        progress_callback(
                            FitProgress(
                                evaluated=evaluated,
                                total=total,
                                parameter=path,
                                value=float(value),
                                score=best_score,
                                best_score=best_score,
                            )
                        )
                    continue

                candidate = _set(params, path, value)
                score, _ = _aggregate(samples, candidate, roles)
                if score > best_score:
                    best_score = score
                    params = candidate
                    current = value
                if progress_callback is not None:
                    progress_callback(
                        FitProgress(
                            evaluated=evaluated,
                            total=total,
                            parameter=path,
                            value=float(value),
                            score=score,
                            best_score=best_score,
                        )
                    )

    _, best_metrics = _aggregate(samples, params, roles)
    changed = {
        path: (float(_get(initial, path)), float(_get(params, path)))
        for path, _ in _SEARCH_SPACE
        if _get(initial, path) != _get(params, path)
    }
    return FitResult(
        params=params,
        baseline_score=baseline_score,
        best_score=best_score,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        changed=changed,
        sample_count=len(samples),
        evaluations=evaluated,
    )
