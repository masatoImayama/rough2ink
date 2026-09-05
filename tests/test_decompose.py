"""`decompose()` のテスト（Epic 仕様書 4-B 節 / T4, #5）。

合成フィクスチャで、網点パターン領域が `tone` に、黒塗り矩形が `fill` に、
細線が `line` に分類されることを確認する。グラデーション網点（周波数が空間的に
変化する網点）でも `tone` として検出できることと、3マスクが相互排他であること、
5000x7000px でメモリエラーにならないことも検証する。

トーン判定は帯域エネルギー比だけでなく帯域内スペクトルの尖鋭度（周期性）も見る（#16）。
曲線ストローク（同心楕円）・カケアミ（クロスハッチング）はエネルギーが帯域内に広く
分散するため tone に分類されず、真の網点（周期的ドット）だけが尖鋭なピークを持つため
tone に分類されることを検証する。
"""

from __future__ import annotations

import cv2
import numpy as np

from rough2ink.core.decompose import decompose
from rough2ink.core.params import AnalysisParams


def _make_halftone(
    shape: tuple[int, int],
    period: int,
    radius: float,
    *,
    region: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """白背景に規則的な網点（黒丸）を敷き詰めた画像を作る。

    `region` を指定すると、その領域内だけに網点を敷き詰める（他は白のまま）。
    """
    height, width = shape
    canvas = np.full((height, width), 255, dtype=np.uint8)
    row_region, col_region = region if region is not None else (slice(0, height), slice(0, width))

    yy, xx = np.indices((height, width))
    mod_y = yy % period - period / 2.0
    mod_x = xx % period - period / 2.0
    dots = (mod_y**2 + mod_x**2) <= radius**2

    mask = np.zeros((height, width), dtype=bool)
    mask[row_region, col_region] = True
    canvas[dots & mask] = 0
    return canvas


def _make_gradient_halftone(shape: tuple[int, int], periods: list[int], radius_ratio: float = 0.35) -> np.ndarray:
    """ドット径（周期）が列方向に段階的に変化するグラデーション網点画像を作る。"""
    height, width = shape
    canvas = np.full((height, width), 255, dtype=np.uint8)
    band_width = width // len(periods)

    for band_index, period in enumerate(periods):
        col_start = band_index * band_width
        col_end = width if band_index == len(periods) - 1 else col_start + band_width
        radius = max(1.0, period * radius_ratio)
        band = _make_halftone(
            (height, col_end - col_start),
            period=period,
            radius=radius,
        )
        canvas[:, col_start:col_end] = band
    return canvas


def _make_fill_rect(shape: tuple[int, int], rect: tuple[slice, slice]) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    canvas[rect] = 0
    return canvas


def _make_thin_line(shape: tuple[int, int], row: int, thickness: int = 2) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    canvas[row : row + thickness, :] = 0
    return canvas


def _make_concentric_curves(shape: tuple[int, int], spacing: int = 16, thickness: int = 1) -> np.ndarray:
    """同心楕円を並べた曲線ストローク画像を作る（周波数がブロック内で連続的に変化する）。

    手描きの服の皺・キャラクターの輪郭に近い、細い（1px）曲線ストロークを想定する。
    間隔を狭く（例: 6px）・太く（例: 2px）すると局所的にほぼ等間隔な平行線と化し、
    網点と区別できない正当な周期構造になってしまうため、標準的な線幅・間隔を選ぶ。
    """
    height, width = shape
    canvas = np.full(shape, 255, dtype=np.uint8)
    center = (width // 2 + 15, height // 2 - 10)
    max_radius = max(height, width)
    for radius in range(4, max_radius, spacing):
        cv2.ellipse(canvas, center, (radius, int(radius * 0.7)), 15, 0, 360, 0, thickness)
    return canvas


def _make_cross_hatch(
    shape: tuple[int, int],
    spacing: int = 6,
    thickness: int = 1,
    jitter: float = 1.2,
    angle_jitter_deg: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """手描き風のばらつき（間隔・角度のジッター）を持つカケアミ画像を作る。

    完全に規則正しいグリッドは網点と同じく周波数的に孤立ピークを持ってしまうため、
    実際の手描きカケアミに近いばらつきを与える。
    """
    height, width = shape
    canvas = np.full(shape, 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    max_dim = max(height, width)

    for base_angle in (45.0, -45.0):
        for offset in range(-2 * max_dim, 2 * max_dim, spacing):
            jittered_offset = offset + rng.uniform(-jitter, jitter)
            angle = base_angle + rng.uniform(-angle_jitter_deg, angle_jitter_deg)
            rad = np.deg2rad(angle)
            dx, dy = np.cos(rad), np.sin(rad)
            cx = width / 2 + jittered_offset * (-dy)
            cy = height / 2 + jittered_offset * dx
            length = max_dim * 1.5
            x1, y1 = cx - dx * length, cy - dy * length
            x2, y2 = cx + dx * length, cy + dy * length
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 0, thickness)
    return canvas


def _inner(region_slice: slice, margin: int) -> slice:
    """境界付近（ブロック重なりの影響を受けやすい領域）を除いた内側スライスを返す。"""
    start = region_slice.start + margin
    stop = region_slice.stop - margin
    return slice(start, stop)


def test_halftone_region_is_classified_as_tone() -> None:
    shape = (256, 256)
    region = (slice(64, 192), slice(64, 192))
    gray = _make_halftone(shape, period=8, radius=2.5, region=region)
    params = AnalysisParams()

    result = decompose(gray, params)

    inner_region = (_inner(region[0], 32), _inner(region[1], 32))
    tone_ratio = result["tone"][inner_region].mean() / 255.0
    assert tone_ratio > 0.8, f"halftone core should be classified as tone, got ratio={tone_ratio}"

    # 網点の外側（余白）はトーンとして検出されないこと
    background = result["tone"][10:40, 10:40]
    assert background.max() == 0


def test_black_rectangle_is_classified_as_fill() -> None:
    shape = (200, 200)
    rect = (slice(50, 130), slice(60, 150))
    gray = _make_fill_rect(shape, rect)
    params = AnalysisParams()

    result = decompose(gray, params)

    inner_rect = (_inner(rect[0], 6), _inner(rect[1], 6))
    fill_ratio = result["fill"][inner_rect].mean() / 255.0
    assert fill_ratio > 0.95, f"solid rectangle core should be fill, got ratio={fill_ratio}"
    assert result["tone"][inner_rect].max() == 0
    assert result["line"][inner_rect].max() == 0


def test_thin_lines_touching_a_solid_area_stay_line() -> None:
    """実原稿で報告された「線がほぼベタ判定される」問題の回帰テスト。

    以前の実装は「収縮後も残る画素を含む**連結成分まるごと**」をベタにしていた。実原稿では
    ペン入れ・コマ枠・ベタが1つの巨大な連結成分に繋がる（実測: B4原稿1枚で最大成分が
    ページ全体を覆う 5,132,079px）ため、ベタが1箇所あるだけでそこに繋がる線が全て
    ベタへ流れ込み、線の被覆率が 1.8% まで落ちていた。

    ここではベタ矩形に細線を**接続した**状態を作り、細線が線のまま残ることを確認する。
    線がベタから独立していると、連結成分を辿る旧実装でも通ってしまうため、
    「接続している」ことがこのテストの要点。
    """
    shape = (300, 300)
    gray = np.full(shape, 255, dtype=np.uint8)
    # ベタ（十分に太い塊）
    gray[40:120, 40:200] = 0
    # そのベタから伸びる細線（太さ 2px）。旧実装ではこの線もベタ扱いになっていた。
    gray[160, 40:260] = 0
    gray[161, 40:260] = 0
    gray[120:200, 100] = 0  # ベタと細線を縦に繋ぐ（同一連結成分にする）
    gray[120:200, 101] = 0

    params = AnalysisParams()
    result = decompose(gray, params)

    # 前提: 意図どおり1つの連結成分になっていること（そうでないとこのテストは無意味）。
    dark = (gray <= params.fill.black_threshold).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(dark, connectivity=8)
    assert num_labels == 2, "ベタと細線が1つの連結成分になっている前提が崩れている"

    # ベタ矩形の芯はベタ。
    assert result["fill"][60:100, 60:180].mean() / 255.0 > 0.95

    # ベタに接続していても、細線は線のまま残る。
    horizontal_line = result["line"][160:162, 210:260]
    assert horizontal_line.mean() / 255.0 > 0.95, (
        "ベタに接続した細線がベタへ流れ込んでいる（連結成分まるごとの判定に戻っていないか）"
    )
    assert result["fill"][160:162, 210:260].max() == 0


def test_thin_line_is_classified_as_line() -> None:
    shape = (200, 200)
    gray = _make_thin_line(shape, row=100, thickness=2)
    params = AnalysisParams()

    result = decompose(gray, params)

    line_pixels = result["line"][100:102, 20:180]
    assert line_pixels.mean() / 255.0 > 0.9, "thin line should be classified as line"
    assert result["fill"][100:102, 20:180].max() == 0, "thin line must not survive erosion into fill"
    assert result["tone"][100:102, 20:180].max() == 0


def test_concentric_curve_strokes_are_not_classified_as_tone() -> None:
    """曲線ストローク（同心楕円）は帯域エネルギー比だけを見ると tone と誤判定されうるが、
    エネルギーが帯域内に広く分散し孤立ピークを持たないため tone に分類されないこと（#16）。
    """
    shape = (220, 220)
    gray = _make_concentric_curves(shape, spacing=16, thickness=1)
    params = AnalysisParams()

    result = decompose(gray, params)

    inner = (slice(40, shape[0] - 40), slice(40, shape[1] - 40))
    tone_ratio = result["tone"][inner].mean() / 255.0
    assert tone_ratio < 0.05, (
        f"curved strokes must not be misclassified as tone, got tone_ratio={tone_ratio}"
    )


def test_uniform_solid_area_is_not_classified_as_tone() -> None:
    """一様な黒面（ベタ）がトーン判定されないこと。

    尖鋭度（帯域内の最大ビン / 中央値）だけを見ていた頃は、一様面でスペクトルが退化して
    この比が無意味な値を取り、**純粋なベタ矩形の内側の 96% がトーン判定されていた**。
    ベタ優先の排他化に隠れて表面化していなかったが、トーンを先に差し引く設計にすると
    ベタの取りこぼしとして現れる。ブロック内の標準偏差の下限で落とす。
    """
    from rough2ink.core.decompose import _detect_tone

    gray = np.full((300, 300), 255, dtype=np.uint8)
    gray[40:200, 40:260] = 0
    params = AnalysisParams()

    tone = _detect_tone(gray, params.tone)

    # ベタの内側（エッジから十分離れた領域）はトーンにならない。
    interior_ratio = float(tone[80:160, 80:220].mean())
    assert interior_ratio < 0.05, f"uniform solid area must not be tone, got {interior_ratio}"


def test_straight_edge_is_not_classified_as_tone() -> None:
    """まっすぐで鋭いエッジがトーン判定されないこと。

    直線エッジは2次元周波数空間で原点を通る**1本の直線上**にエネルギーが集中するため、
    帯域内の大半のビンがほぼ 0 になり、「最大 / 中央値」が網点と同程度に跳ね上がる。
    網点は2次元格子なので基本ベクトル2本ぶんの**独立した2方向**にピークを持つ点で
    区別できる（`min_direction_ratio`）。
    """
    from rough2ink.core.decompose import _detect_tone

    gray = np.full((256, 256), 255, dtype=np.uint8)
    gray[:, 128:] = 0  # 縦一直線のエッジ
    params = AnalysisParams()

    tone = _detect_tone(gray, params.tone)

    edge_ratio = float(tone[64:192, 96:160].mean())
    assert edge_ratio < 0.05, f"straight edge must not be tone, got {edge_ratio}"


def test_parallel_lines_are_not_classified_as_tone() -> None:
    """等間隔の平行線（1方向のみの周期構造）がトーン判定されないこと。

    完全に等間隔なので尖鋭度は高いが、方向が1つしかないため網点の格子とは区別できる。
    """
    from rough2ink.core.decompose import _detect_tone

    gray = np.full((256, 256), 255, dtype=np.uint8)
    gray[::6, :] = 0
    params = AnalysisParams()

    tone = _detect_tone(gray, params.tone)

    ratio = float(tone[64:192, 64:192].mean())
    assert ratio < 0.05, f"parallel lines must not be tone, got {ratio}"


def test_cross_hatching_is_not_classified_as_tone() -> None:
    """カケアミ（クロスハッチング）は帯域エネルギー比だけを見ると tone と誤判定されうるが、
    網点のような孤立した周波数ピークを持たないため tone に分類されないこと（#16）。
    """
    shape = (220, 220)
    gray = _make_cross_hatch(shape, spacing=6, thickness=1)
    params = AnalysisParams()

    result = decompose(gray, params)

    inner = (slice(40, shape[0] - 40), slice(40, shape[1] - 40))
    tone_ratio = result["tone"][inner].mean() / 255.0
    assert tone_ratio < 0.05, (
        f"cross-hatching must not be misclassified as tone, got tone_ratio={tone_ratio}"
    )


def test_true_halftone_dots_are_still_classified_as_tone() -> None:
    """真の網点（周期的なドット）は尖鋭度判定を追加した後も引き続き tone に分類されること（#16）。"""
    shape = (220, 220)
    gray = _make_halftone(shape, period=8, radius=2.5)
    params = AnalysisParams()

    result = decompose(gray, params)

    inner = (slice(40, shape[0] - 40), slice(40, shape[1] - 40))
    tone_ratio = result["tone"][inner].mean() / 255.0
    assert tone_ratio > 0.9, f"true halftone dots should still be tone, got tone_ratio={tone_ratio}"


def test_gradient_halftone_is_classified_as_tone() -> None:
    shape = (200, 320)
    gray = _make_gradient_halftone(shape, periods=[4, 6, 8, 10, 12, 14, 16])
    params = AnalysisParams()

    result = decompose(gray, params)

    inner = (slice(24, shape[0] - 24), slice(24, shape[1] - 24))
    tone_ratio = result["tone"][inner].mean() / 255.0
    assert tone_ratio > 0.6, (
        f"gradient halftone (varying dot period) should still be classified as tone overall, "
        f"got ratio={tone_ratio}"
    )


def test_masks_are_mutually_exclusive() -> None:
    shape = (220, 300)
    canvas = np.full(shape, 255, dtype=np.uint8)

    # 網点
    canvas[10:90, 10:130] = _make_halftone((80, 120), period=8, radius=2.5)
    # ベタ
    canvas[120:200, 20:120] = 0
    # 線
    canvas[30:32, 150:290] = 0
    canvas[:, 200:202] = 0

    params = AnalysisParams()
    result = decompose(canvas, params)

    line_bool = result["line"] > 0
    fill_bool = result["fill"] > 0
    tone_bool = result["tone"] > 0

    assert not np.any(line_bool & fill_bool)
    assert not np.any(line_bool & tone_bool)
    assert not np.any(fill_bool & tone_bool)


def test_output_contract() -> None:
    shape = (128, 96)
    gray = np.full(shape, 200, dtype=np.uint8)
    gray[40:60, 30:50] = 0
    params = AnalysisParams()

    result = decompose(gray, params)

    assert set(result.keys()) == {"line", "fill", "tone"}
    for name, mask in result.items():
        assert mask.shape == shape, name
        assert mask.dtype == np.uint8, name
        assert set(np.unique(mask).tolist()) <= {0, 255}, name


def test_large_image_does_not_raise_memory_error() -> None:
    """5000x7000px の画像をメモリエラーなく処理できること。"""
    shape = (5000, 7000)
    gray = np.full(shape, 255, dtype=np.uint8)
    gray[1000:1200, 1500:1700] = 0  # ベタ領域
    gray[3000:3002, 500:6500] = 0  # 線

    params = AnalysisParams()

    result = decompose(gray, params)

    for mask in result.values():
        assert mask.shape == shape
        assert mask.dtype == np.uint8
