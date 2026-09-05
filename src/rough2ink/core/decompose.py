"""B. 分解器プロトタイプ: 線 (line) / ベタ (fill) / トーン (tone) の相互排他マスク生成。

設計書 4章 B「分解器の精度がこの設計全体を規定する（最重要）」に対応する部品。
学習は一切行わず、古典的画像処理（周波数解析・連結成分ラベリング・二値化）のみで、
原寸グレースケール画像 1 枚から 3 枚の相互排他な二値マスクを作る。

処理の指針（Epic 仕様書 4-B 節 / 設計書 3章「分解器」）:

1. トーン: 網点は周期構造なので `tone.window` x `tone.window` の重なり合うブロックに
   窓関数を掛けて 2D FFT する。帯域エネルギー比（DC 成分を除いた全エネルギーに対する、
   `tone.bandpass_low`〜`tone.bandpass_high` 帯域のエネルギーの比）だけでは「帯域に構造が
   あるか」しか測れず、通常の線画やカケアミもエネルギーが帯域内に広く分散するため高い
   比を示してしまう（#16）。網点は周期構造なのでスペクトルに孤立した鋭いピークを持つのに
   対し、線画・カケアミはエネルギーが帯域内に広く分散する。そこで帯域内の最大ビン
   エネルギーと中央値の比（尖鋭度）を主判定に使い、`tone.sharpness_threshold` 以上の
   ブロックだけをトーン候補とする。帯域エネルギー比 (`tone.energy_threshold`) は
   「そもそも帯域に有意なエネルギーがあるか」を落とす前段フィルタとして併用する
   （両方の閾値を満たしたブロックのみがトーン候補になる）。ブロック単位の判定であること
   自体が、周波数が空間的に変化するグラデーショントーンへの対処になる。
2. ベタ: `fill.black_threshold` 以下で二値化した連結成分のうち、面積が
   ページ面積の `fill.min_area_ratio` 以上、かつ `fill.erosion_radius` の収縮後も
   画素が残る成分を採用する（細線は収縮で消えるため線と自然に分離できる）。
3. 線: `line.black_threshold` 以下の暗画素のうち、ベタにもトーンにも属さないもの。
4. 優先順位は **ベタ > トーン > 線**。重なった画素は上位に割り当て、3 マスクを
   相互排他にする。この優先順位は GT マスク生成（#8, Epic 仕様書 5節）と同一の規則に
   する契約であり、変更しないこと。

5000px 級の画像でもメモリエラーにならないよう、トーン検出は画像全体を一度に FFT せず、
小ブロックを走査しながら (H, W) サイズの累積バッファへ加算するタイル処理で行う
（`_TONE_ROW_CHUNK_BUDGET_BYTES` でチャンクあたりのメモリ上限を抑える）。
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rough2ink.core.params import AnalysisParams, FillParams, ToneParams

_MASK_ON = np.uint8(255)
_MASK_OFF = np.uint8(0)

# トーン検出のブロックFFTを行方向にまとめてバッチ処理する際、1チャンクあたりに許す
# ブロックデータの概算メモリ量（バイト）。大きすぎるとメモリを圧迫し、
# 小さすぎるとPythonループのオーバーヘッドが増える。
_TONE_ROW_CHUNK_BUDGET_BYTES = 64 * 1024 * 1024

# 帯域内エネルギーを方向別に集計するときの角度ビン数（0〜π を等分する）。
_DIRECTION_BINS = 12
# 「独立した方向」と認めるための角度ビンの最小距離。12ビン（15°刻み）で 2 なら 30° 以上。
# 網点格子の基本ベクトル2本は通常これより大きく離れる一方、直線エッジのわずかな
# 角度のブレは同じ方向として扱われる。
_DIRECTION_MIN_SEPARATION_BINS = 2

# トーンマスクの整形（ブロック格子状の輪郭を滑らかにする）に使う構造要素の大きさ。
_TONE_MORPH_KERNEL_SIZE = 5


def decompose(gray: np.ndarray, params: AnalysisParams) -> dict[str, np.ndarray]:
    """原寸グレースケール画像を line / fill / tone の3マスク（相互排他）に分解する。

    Args:
        gray: (H, W) の uint8 グレースケール画像（原寸）。
        params: `AnalysisParams`。`tone` / `fill` / `line` の各パラメータを使う。

    Returns:
        `{"line": mask, "fill": mask, "tone": mask}`。各マスクは gray と同じ (H, W) の
        uint8 配列で、値は 0 または 255。3マスクの論理積は常に空（相互排他）。
    """
    if gray.ndim != 2:
        raise ValueError(f"gray must be a 2D array, got shape {gray.shape}")

    tone_mask = _detect_tone(gray, params.tone)
    fill_mask = _detect_fill(gray, params.fill)
    line_candidate = gray <= params.line.black_threshold

    # 優先順位: ベタ > トーン > 線。重なった画素は上位に割り当てて相互排他にする。
    final_fill = fill_mask
    final_tone = tone_mask & ~final_fill
    final_line = line_candidate & ~final_fill & ~final_tone

    return {
        "line": _to_mask(final_line),
        "fill": _to_mask(final_fill),
        "tone": _to_mask(final_tone),
    }


def _to_mask(boolean: np.ndarray) -> np.ndarray:
    return np.where(boolean, _MASK_ON, _MASK_OFF)


def _detect_fill(gray: np.ndarray, params: FillParams) -> np.ndarray:
    """面積閾値以上の「太い黒の塊」を「ベタ」として検出する。

    判定は**画素の局所的な太さ**で行う。距離変換で各黒画素から白までの距離を測り、
    半径 `erosion_radius` の円が内側に収まる画素（＝塊の芯）を種にして、同じ半径で
    膨張し直した領域をベタとする（モルフォロジーのオープニングと等価）。

    以前は「収縮後も残る画素を含む**連結成分まるごと**」をベタにしていたが、実原稿では
    ペン入れ・コマ枠・ベタ・網点がすべて1つの巨大な連結成分に繋がるため（実測: B4原稿1枚で
    最大成分がページ全体を覆う 5,132,079px）、芯が1箇所でもあればページの大半がベタに
    流れ込んでいた。連結成分の同一性ではなく局所的な太さで判定することでこれを断つ。

    優先順位（ベタ > トーン > 線）は呼び出し側の排他化で保つため、ここではトーンを
    考慮しない。二値原稿では高濃度の網点が繋がって太い塊になりベタへ流れ込みうるが、
    トーン検出を信頼して差し引くとトーン側の誤検出がそのままベタの取りこぼしになるため、
    依存させない。
    """
    dark = gray <= params.black_threshold
    if not dark.any():
        return np.zeros(gray.shape, dtype=bool)

    radius = max(1, params.erosion_radius)

    # 各黒画素が白まで何 px 離れているか＝その位置に収まる円の半径。
    distance = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    core = distance >= radius
    if not core.any():
        return np.zeros(gray.shape, dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grown = cv2.dilate(core.astype(np.uint8) * 255, kernel) > 0
    grown &= dark  # 膨張が黒画素の外へはみ出さないように戻す

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        grown.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros(gray.shape, dtype=bool)

    min_area = params.min_area_ratio * gray.size
    fill_ids = [
        label_id
        for label_id in range(1, num_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area
    ]
    if not fill_ids:
        return np.zeros(gray.shape, dtype=bool)
    return np.isin(labels, fill_ids)


def _detect_tone(gray: np.ndarray, params: ToneParams) -> np.ndarray:
    """ブロック単位の FFT で、帯域内スペクトルの尖鋭度（周期性）からトーン（網点）領域を検出する。

    網点のような周期構造は帯域内の1〜数箇所に孤立したピークを持つため、
    「帯域内最大ビンエネルギー / 帯域内中央値」（尖鋭度）が高くなる。一方、通常の
    線画・カケアミは帯域エネルギー比自体は高くてもエネルギーが帯域内に広く分散し、
    尖鋭度は低い（#16）。判定は帯域エネルギー比（前段フィルタ）と尖鋭度（主判定）の
    両方が閾値以上であることを要求する。
    """
    height, width = gray.shape
    window = max(8, min(params.window, height, width))
    stride = max(1, min(params.stride, window))

    row_starts = _block_starts(height, window, stride)
    col_starts = _block_starts(width, window, stride)

    win2d = np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)
    band_mask, total_mask = _band_masks(window, params.bandpass_low, params.bandpass_high)
    direction_onehot, direction_independent = _direction_bins(window, band_mask)

    gray_f = gray.astype(np.float32)
    # ウィンドウ抽出はビュー（コピーなし）。実データのコピーはチャンク単位に限定して
    # メモリ使用量を抑える。
    view = sliding_window_view(gray_f, (window, window))

    score_sum = np.zeros((height, width), dtype=np.float32)
    score_count = np.zeros((height, width), dtype=np.float32)

    bytes_per_row = max(1, len(col_starts)) * window * window * 4
    row_chunk = max(1, _TONE_ROW_CHUNK_BUDGET_BYTES // bytes_per_row)

    for chunk_start in range(0, len(row_starts), row_chunk):
        chunk_rows = row_starts[chunk_start : chunk_start + row_chunk]
        blocks = view[np.ix_(chunk_rows, col_starts)]  # (nrows, ncols, window, window)
        windowed = blocks * win2d
        spectrum = np.fft.fft2(windowed, axes=(-2, -1))
        power = np.abs(spectrum) ** 2

        band_values = power[:, :, band_mask]  # (nrows, ncols, n_band_bins)
        band_energy = band_values.sum(axis=-1)
        total_energy = (power * total_mask).sum(axis=(-2, -1))
        ratio = np.divide(
            band_energy,
            total_energy,
            out=np.zeros_like(band_energy),
            where=total_energy > 0,
        )

        # 尖鋭度（周期性）: 帯域内で1〜数箇所に孤立したピークがあるほど、
        # 最大ビンエネルギーが中央値を大きく上回る。網点はこの値が高く、
        # エネルギーが帯域内に広く分散する線画・カケアミは低い（#16）。
        median_band = np.median(band_values, axis=-1)
        max_band = band_values.max(axis=-1)
        sharpness = np.divide(
            max_band,
            median_band,
            out=np.zeros_like(max_band),
            where=median_band > 0,
        )

        # 一様な面（ベタの内側・白紙）はスペクトルが退化して尖鋭度が無意味な値を取るため、
        # ブロック内の画素値のばらつきで落とす。実測でこの guard が無いと、純粋なベタ矩形の
        # 96% がトーン判定されていた。
        block_std = blocks.std(axis=(-2, -1))

        # 方向の独立性: 網点は2次元格子なので基本ベクトル2本ぶんの方向にエネルギーを持つ。
        # 一方、直線エッジ・平行線は原点を通る1本の直線上にしか energy を持たないため、
        # 「最大方向」と「そこから十分離れた方向」の比が小さくなる。
        direction_energy = band_values @ direction_onehot  # (nrows, ncols, _DIRECTION_BINS)
        primary_index = np.argmax(direction_energy, axis=-1)
        primary_energy = np.take_along_axis(
            direction_energy, primary_index[..., None], axis=-1
        ).squeeze(-1)
        # 最大方向から十分離れた方向だけを候補にして、その中の最大を second とする。
        allowed = direction_independent[primary_index]  # (nrows, ncols, _DIRECTION_BINS)
        secondary_energy = np.where(allowed, direction_energy, 0.0).max(axis=-1)
        direction_ratio = np.divide(
            secondary_energy,
            primary_energy,
            out=np.zeros_like(secondary_energy),
            where=primary_energy > 0,
        )

        is_tone = (
            (ratio >= params.energy_threshold)
            & (sharpness >= params.sharpness_threshold)
            & (block_std >= params.min_block_std)
            & (direction_ratio >= params.min_direction_ratio)
        ).astype(np.float32)

        for row_index, top in enumerate(chunk_rows):
            row_slice = slice(top, top + window)
            for col_index, left in enumerate(col_starts):
                col_slice = slice(left, left + window)
                score_sum[row_slice, col_slice] += is_tone[row_index, col_index]
                score_count[row_slice, col_slice] += 1.0

    score = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count > 0)
    tone_bool = score >= 0.5

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_TONE_MORPH_KERNEL_SIZE, _TONE_MORPH_KERNEL_SIZE)
    )
    tone_u8 = tone_bool.astype(np.uint8) * 255
    tone_u8 = cv2.morphologyEx(tone_u8, cv2.MORPH_CLOSE, kernel)
    tone_u8 = cv2.morphologyEx(tone_u8, cv2.MORPH_OPEN, kernel)
    return tone_u8 > 0


def _band_masks(window: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """正規化周波数の帯域マスクと、DC成分を除いた全体マスクを返す。"""
    freqs = np.fft.fftfreq(window)
    fx, fy = np.meshgrid(freqs, freqs)
    radius = np.sqrt(fx**2 + fy**2)
    band_mask = (radius >= low) & (radius <= high)
    total_mask = radius > 0.0  # DC成分（ブロック内の平均輝度）は正規化対象から除く
    return band_mask, total_mask


def _direction_bins(window: int, band_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """帯域内の各ビンを角度ビンに割り当てる行列と、角度ビン同士の「独立」判定表を返す。

    パワースペクトルは原点対称なので、角度は 0〜π の範囲だけ見れば足りる
    （`angle` と `angle + π` は同じ方向を指す）。

    Returns:
        (onehot, independent):
            onehot: (帯域ビン数, 角度ビン数) の 0/1 行列。`band_values @ onehot` で
                角度ごとのエネルギーが得られる。
            independent[i, j]: 角度ビン i と j が十分に離れているか（円環距離で判定）。
                同じ方向のわずかなブレを「別方向」と数えないためのもの。
    """
    freqs = np.fft.fftfreq(window)
    fx, fy = np.meshgrid(freqs, freqs)
    angles = np.mod(np.arctan2(fy, fx), np.pi)  # 0〜π

    bin_index = np.minimum(
        (angles[band_mask] / np.pi * _DIRECTION_BINS).astype(np.int32), _DIRECTION_BINS - 1
    )
    onehot = np.zeros((bin_index.size, _DIRECTION_BINS), dtype=np.float32)
    onehot[np.arange(bin_index.size), bin_index] = 1.0

    indices = np.arange(_DIRECTION_BINS)
    circular_distance = np.abs(indices[:, None] - indices[None, :])
    circular_distance = np.minimum(circular_distance, _DIRECTION_BINS - circular_distance)
    independent = circular_distance >= _DIRECTION_MIN_SEPARATION_BINS
    return onehot, independent


def _block_starts(total: int, window: int, stride: int) -> list[int]:
    """`stride` 刻みのブロック開始位置一覧。末尾は必ず `total - window` に揃える。"""
    if total <= window:
        return [0]
    starts = list(range(0, total - window + 1, stride))
    last = total - window
    if starts[-1] != last:
        starts.append(last)
    return starts
