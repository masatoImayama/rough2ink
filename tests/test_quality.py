"""取り込み品質ゲート（設計書 4章 D）のテスト。

`short_side` / `tone_depth` / `jpeg_block_score` の 3 指標と、
それらに基づく `pass` / `warn` / `fail` 判定を検証する。
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from rough2ink.core.params import QualityParams
from rough2ink.core.quality import _compute_jpeg_block_score, evaluate_quality


def _synthetic_photo(size: int = 256, seed: int = 0) -> np.ndarray:
    """JPEG 圧縮の効果を測るための、滑らかな濃淡と細部を持つ合成画像を作る。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    base = np.sin(xx) * np.cos(yy) * 100 + 128
    noise = rng.normal(0, 8, size=(size, size))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(gray: np.ndarray, quality: int) -> np.ndarray:
    """`gray` を JPEG エンコード → デコードし直したグレースケール配列を返す。"""
    image = Image.fromarray(gray, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("L")
    return np.array(decoded)


def test_short_side_below_threshold_fails() -> None:
    """短辺が `min_short_side` 未満の画像は `fail` になる（完了条件）。"""
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(1200, 3000), dtype=np.uint8)  # 短辺 1200 < 2000

    report = evaluate_quality(gray, QualityParams(min_short_side=2000))

    assert report.short_side == 1200
    assert report.status == "fail"
    assert any("short_side" in reason for reason in report.reasons)


def test_short_side_well_above_threshold_has_no_short_side_reason() -> None:
    """短辺が閾値を十分に上回れば `short_side` を理由とした fail/warn は発生しない。

    閾値ちょうど付近は warn 対象になり得るため、warn マージン（既定 110%）の外側の
    サイズで検証する。
    """
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, size=(2400, 2400), dtype=np.uint8)  # min_short_side * 1.1 = 2200 超

    report = evaluate_quality(gray, QualityParams(min_short_side=2000))

    assert report.short_side == 2400
    assert not any("short_side" in reason for reason in report.reasons)


def test_short_side_just_above_threshold_warns() -> None:
    """短辺が閾値ちょうど（warn マージン内）の画像は `warn` として理由付きで記録される。"""
    rng = np.random.default_rng(20)
    gray = rng.integers(0, 256, size=(2000, 2000), dtype=np.uint8)

    report = evaluate_quality(gray, QualityParams(min_short_side=2000))

    assert report.short_side == 2000
    assert report.status in ("warn", "fail")
    assert any("short_side" in reason for reason in report.reasons)


def test_binary_image_is_classified_as_binary() -> None:
    """値が 0 と 255 に集中する画像は `tone_depth == "binary"` と記録される（完了条件）。"""
    rng = np.random.default_rng(3)
    gray = (rng.integers(0, 2, size=(500, 500)) * 255).astype(np.uint8)

    report = evaluate_quality(gray, QualityParams(binary_ratio_threshold=0.98))

    assert report.tone_depth == "binary"


def test_grayscale_image_is_classified_as_gray() -> None:
    """連続階調の画像は `tone_depth == "gray"` と記録される（完了条件）。"""
    gradient_row = np.linspace(0, 255, 500, dtype=np.uint8)
    gray = np.tile(gradient_row, (500, 1))

    report = evaluate_quality(gray, QualityParams(binary_ratio_threshold=0.98))

    assert report.tone_depth == "gray"


def test_jpeg_block_score_rises_with_heavier_compression() -> None:
    """JPEG 品質を落として保存した合成画像で `jpeg_block_score` が有意に上がる（完了条件）。"""
    source = _synthetic_photo(size=256, seed=42)

    high_quality = _jpeg_roundtrip(source, quality=95)
    low_quality = _jpeg_roundtrip(source, quality=5)

    # short_side の失格を混ぜないよう、この検証では短辺閾値を無効化する
    params = QualityParams(min_short_side=1)
    report_high = evaluate_quality(high_quality, params)
    report_low = evaluate_quality(low_quality, params)

    assert report_low.jpeg_block_score > report_high.jpeg_block_score
    # 既定閾値 (0.15) を明確に超えるレベルの劣化であることも確認する
    assert report_low.jpeg_block_score > QualityParams().jpeg_block_threshold


def test_heavy_jpeg_compression_fails_quality_gate() -> None:
    """強い JPEG 圧縮由来のブロックノイズは `fail` として理由付きで記録される。"""
    source = _synthetic_photo(size=2048, seed=7)
    low_quality = _jpeg_roundtrip(source, quality=5)

    report = evaluate_quality(low_quality, QualityParams())

    assert report.status == "fail"
    assert any("jpeg_block_score" in reason for reason in report.reasons)


def test_clean_large_image_passes() -> None:
    """短辺が十分大きく（warn マージンの外側）JPEG 劣化の無い画像は `pass` になる。"""
    source = _synthetic_photo(size=2400, seed=99)

    report = evaluate_quality(source, QualityParams())

    assert report.short_side == 2400
    assert report.status == "pass"
    assert report.reasons == []


def test_default_params_used_when_omitted() -> None:
    """`params` を省略した場合は `QualityParams()` の既定値で判定される。"""
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, size=(1000, 1000), dtype=np.uint8)

    report = evaluate_quality(gray)

    assert report.status == "fail"
    assert any("short_side" in reason for reason in report.reasons)


def test_jpeg_block_score_flat_blocks_with_boundary_steps_is_not_zero() -> None:
    """ブロック内部が完全に平坦（非境界勾配ゼロ）でも、境界に段差があればスコアが0へ潰れない。

    強い JPEG 量子化の典型的な帰結（ブロック内部が完全に平坦）を市松模様で再現する。
    #19 でメモリ使用量を削減する際に、境界／非境界の集計方法（連結コピー -> 総和・件数の
    累積）を変えてもこの eps ガードの挙動を壊していないことを確認する回帰テスト。
    """
    block = 8
    n_blocks = 20
    row_idx = np.arange(n_blocks)[:, None]
    col_idx = np.arange(n_blocks)[None, :]
    block_values = ((row_idx + col_idx) % 2 * 200 + 20).astype(np.uint8)
    gray = np.repeat(np.repeat(block_values, block, axis=0), block, axis=1)

    score = _compute_jpeg_block_score(gray, block_size=block)

    assert score > 0.0


def test_jpeg_block_score_memory_stays_bounded_for_large_image() -> None:
    """5000x7000px の巨大画像でも peak RSS の増分が入力サイズの数倍程度に収まる（Review #19）。

    修正前は float64 昇格 + 境界/非境界の連結コピーにより、レビュー時点の実測で
    入力の約33倍（増分1164MB）まで膨らんでいた。int16 での差分計算とチャンク処理により、
    これを大きく下回ることを確認する。
    """
    resource = pytest.importorskip("resource", reason="peak RSS 計測は POSIX の resource モジュールに依存する")

    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, size=(5000, 7000), dtype=np.uint8)
    input_bytes = gray.nbytes

    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    score = _compute_jpeg_block_score(gray)
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    increment_bytes = max(0, (after_kb - before_kb) * 1024)

    assert isinstance(score, float)
    # 修正前の実測（増分1164MB、入力の約33倍）を大きく下回ることを確認する。
    # 環境差によるブレを許容しつつ、明確な改善を検証するため入力サイズの10倍を上限とする。
    assert increment_bytes < input_bytes * 10, (
        f"increment={increment_bytes / 1024 / 1024:.1f}MB "
        f"input={input_bytes / 1024 / 1024:.1f}MB"
    )
