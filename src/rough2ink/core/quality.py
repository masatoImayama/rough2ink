"""取り込み品質ゲート（設計書 4章 D / Epic 仕様書 4-D 節）。

原寸グレースケール画像に対して次の 3 指標を算出し、`pass` / `warn` / `fail` を判定する。

- `short_side`: 原寸の短辺画素数。web 用に縮小された画像を弾く
- `tone_depth`: 値 0 と 255 に集中する画素の割合から `binary` / `gray` を判別する（情報として記録するのみ）
- `jpeg_block_score`: 8px 格子境界での勾配平均 / 非境界での勾配平均 − 1。
  配信用 PDF に多い低解像度 + JPEG 圧縮由来のブロックノイズを検出する

判定は必ず**原寸**の画像に対して行う（縮小した画像で JPEG ブロック検出をしても意味がない）。
"""

from __future__ import annotations

import numpy as np

from rough2ink.core.params import QualityParams
from rough2ink.core.types import QualityReport

_JPEG_BLOCK_SIZE = 8

# `_compute_jpeg_block_score` を行方向にチャンクして走査する際の目安バイト数。
# チャンク 1 つあたりの int16 差分配列がこの程度に収まる行数を選ぶ
# （5000x7000px でもピークメモリを入力の数倍程度に抑えるため。Review #19）。
_MEMORY_CHUNK_TARGET_BYTES = 8 * 1024 * 1024

# fail 閾値の手前で早期に注意喚起するための warn マージン。
# UI で閾値調整の参考にできるよう、境界に近い素材を pass と fail の間で区別する
# （Epic 仕様書「各指標の生の数値も併せて返す（閾値調整のため UI に出す）」）。
_SHORT_SIDE_WARN_MARGIN = 1.1  # min_short_side の 110% 未満なら warn
_JPEG_BLOCK_WARN_MARGIN = 0.7  # jpeg_block_threshold の 70% 超なら warn


def _compute_tone_depth(gray: np.ndarray, binary_ratio_threshold: float) -> tuple[str, float]:
    """値 0 と 255 に集中する画素の割合を算出し `binary` / `gray` を返す。"""
    total = gray.size
    binary_pixel_count = int(np.count_nonzero((gray == 0) | (gray == 255)))
    binary_ratio = binary_pixel_count / total if total > 0 else 0.0
    tone_depth = "binary" if binary_ratio >= binary_ratio_threshold else "gray"
    return tone_depth, binary_ratio


def _compute_jpeg_block_score(gray: np.ndarray, block_size: int = _JPEG_BLOCK_SIZE) -> float:
    """8px 格子境界での勾配平均 / 非境界での勾配平均 − 1 を算出する。

    JPEG は `block_size` 画素単位で DCT 量子化するため、圧縮が強いほど
    ブロック境界に不自然な段差（勾配の跳ね上がり）が現れる。
    境界位置の勾配が非境界位置に比べて有意に大きいほどスコアが上がる。

    メモリ方針（Review #19）: 入力を float64 に昇格させず int16 のまま差分を取り、
    境界／非境界の平均は配列を連結せずに総和と件数を累積して求める。さらに行方向に
    チャンクして走査することで、ピークメモリを入力サイズの数倍程度に抑える
    （5000x7000px で peak RSS 増分 1164MB だったものを大幅に削減する）。
    """
    height, width = gray.shape

    col_positions = np.arange(1, width)
    boundary_cols = col_positions % block_size == 0
    nonboundary_cols = ~boundary_cols

    row_positions = np.arange(1, height)
    boundary_rows = row_positions % block_size == 0

    boundary_sum = 0
    boundary_count = 0
    nonboundary_sum = 0
    nonboundary_count = 0

    chunk_height = max(1, _MEMORY_CHUNK_TARGET_BYTES // max(width * 2, 1))

    for start in range(0, height, chunk_height):
        end = min(start + chunk_height, height)
        chunk = gray[start:end].astype(np.int16)  # このチャンク分のみコピーする

        # 横方向: 列 i-1, i 間の勾配。チャンク内で完結するので追加の行は不要。
        dx = np.abs(np.diff(chunk, axis=1))  # shape (end-start, W-1)
        boundary_sum += int(dx[:, boundary_cols].sum())
        boundary_count += int(dx[:, boundary_cols].size)
        nonboundary_sum += int(dx[:, nonboundary_cols].sum())
        nonboundary_count += int(dx[:, nonboundary_cols].size)
        del dx

        # 縦方向: 行 i-1, i 間の勾配。チャンク末尾の行は次の行との差分が必要なので、
        # 最終チャンクでなければ 1 行分余分に読み込む。
        if end < height:
            dy_source = gray[start : end + 1].astype(np.int16)
            row_mask = boundary_rows[start:end]
        else:
            dy_source = chunk
            row_mask = boundary_rows[start : max(end - 1, start)]

        if row_mask.size > 0:
            dy = np.abs(np.diff(dy_source, axis=0))  # shape (len(row_mask), W)
            boundary_sum += int(dy[row_mask, :].sum())
            boundary_count += int(dy[row_mask, :].size)
            boundary_nonrows = ~row_mask
            nonboundary_sum += int(dy[boundary_nonrows, :].sum())
            nonboundary_count += int(dy[boundary_nonrows, :].size)
            del dy

    if boundary_count == 0 or nonboundary_count == 0:
        return 0.0

    boundary_mean = boundary_sum / boundary_count
    nonboundary_mean = nonboundary_sum / nonboundary_count

    # 非境界勾配が 0（ブロック内部が完全に平坦）になるのは強い JPEG 量子化の典型的な帰結であり、
    # 境界勾配がわずかでも残っていればブロックノイズの明確な証拠になる。
    # 0 除算を避けるためだけに両者へ同じ微小値 eps を加える（分母を単純に底上げすると
    # nonboundary_mean == 0 の場合にスコアが不当に潰れてしまうため、比を保つ形にする）。
    eps = 1e-6
    return (boundary_mean + eps) / (nonboundary_mean + eps) - 1.0


def evaluate_quality(gray: np.ndarray, params: QualityParams | None = None) -> QualityReport:
    """原寸グレースケール画像を品質ゲートに掛け `QualityReport` を返す。

    Args:
        gray: 原寸グレースケール画像 (H, W) の配列（dtype は問わず内部で数値として扱う）。
        params: 品質ゲートの閾値。省略時は既定値（`QualityParams()`）を使う。

    Returns:
        `short_side` / `tone_depth` / `jpeg_block_score` の生値と、
        それらに基づく `status`（`pass` / `warn` / `fail`）、失格・注意理由の一覧を含む `QualityReport`。
    """
    if params is None:
        params = QualityParams()

    if gray.ndim != 2:
        raise ValueError(f"gray must be a 2D array (H, W), got shape {gray.shape}")

    height, width = gray.shape
    short_side = int(min(height, width))

    tone_depth, _binary_ratio = _compute_tone_depth(gray, params.binary_ratio_threshold)
    jpeg_block_score = _compute_jpeg_block_score(gray)

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if short_side < params.min_short_side:
        fail_reasons.append(
            f"short_side={short_side} は quality.min_short_side={params.min_short_side} 未満です"
        )
    elif short_side < params.min_short_side * _SHORT_SIDE_WARN_MARGIN:
        warn_reasons.append(
            f"short_side={short_side} は quality.min_short_side={params.min_short_side} に近く、"
            "縮小画像の疑いがあります"
        )

    if jpeg_block_score > params.jpeg_block_threshold:
        fail_reasons.append(
            f"jpeg_block_score={jpeg_block_score:.4f} は "
            f"quality.jpeg_block_threshold={params.jpeg_block_threshold} を超えています"
        )
    elif jpeg_block_score > params.jpeg_block_threshold * _JPEG_BLOCK_WARN_MARGIN:
        warn_reasons.append(
            f"jpeg_block_score={jpeg_block_score:.4f} は "
            f"quality.jpeg_block_threshold={params.jpeg_block_threshold} に近く、"
            "JPEG 圧縮由来のブロックノイズの疑いがあります"
        )

    if fail_reasons:
        status = "fail"
        reasons = fail_reasons + warn_reasons
    elif warn_reasons:
        status = "warn"
        reasons = warn_reasons
    else:
        status = "pass"
        reasons = []

    return QualityReport(
        short_side=short_side,
        tone_depth=tone_depth,
        jpeg_block_score=jpeg_block_score,
        status=status,
        reasons=reasons,
    )
