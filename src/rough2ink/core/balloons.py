"""E. フキダシ検出 → 損失マスク生成（設計書 3章「フキダシのノイズ対策」/ Epic 仕様書 4-E 節）。

フキダシは作画の一部として描く作家が多く機械的除去は困難。**除去せず、損失マスクで無視する**
（穴埋めは偽の教師信号になるが、マスクなら何も学習しない）。**マスクは過剰に広く取ってよい**
（誤検出のコストが非対称。失うのは学習サンプルの一部のみ）。

検出の手がかりは単独では不十分なため、次の 3 つを重ねて使う。

1. テキストレイヤーの矩形（`text_rects`）を膨張させた領域 — PSD 入力時のみ利用可能
2. 一定面積以上で内部がほぼ白の連結領域 — フチなしフキダシも拾う
3. 楕円・角丸の凸形状（凸包面積比 = solidity）

3 つの和集合をさらに膨張したものを最終的な損失マスクとする（255 = 無視）。
"""

from __future__ import annotations

import cv2
import numpy as np

from rough2ink.core.params import BalloonParams
from rough2ink.core.types import BBox

# cv2.connectedComponentsWithStats の戻り値そのままの形。ラベル 0 は背景。
_Components = tuple[int, np.ndarray, np.ndarray]

# --- コマ枠バリア（#28）検出用の内部定数 ---
#
# `panels.py` の `_extract_frame_lines` 相当（長い直線をハフ変換で検出する処理）だが、
# フキダシがコマ枠線を塗り潰して生じる断裂（フキダシの幅ぶん）を橋渡しできるよう
# `maxLineGap` を大きく取る点が異なる。`panels.py` 側の値をそのまま流用すると、
# 断ち切りコマ検出用に調整された小さい `maxLineGap`（15px）では典型的なフキダシの
# 幅を橋渡しできないため、この用途専用の値を持つ（`panels.py` は変更しない）。
# 投票閾値・最小長は `panels.py` の値をそのまま踏襲する。閾値を緩めると、フキダシの
# 縁取り（楕円の弧）の一部が局所的に直線とみなされ、コマ枠を跨いでいない通常の
# フキダシにまで誤ったバリアを引いてしまう（実装時に既存テストの回帰で確認済み）。
_BARRIER_LINE_MIN_LENGTH_RATIO = 0.2
_BARRIER_LINE_MIN_LENGTH_FLOOR = 30
_BARRIER_HOUGH_VOTE_THRESHOLD_RATIO = 0.5
_BARRIER_HOUGH_VOTE_THRESHOLD_FLOOR = 15
_BARRIER_HOUGH_MAX_LINE_GAP = 80
_BARRIER_LINE_THICKNESS_PX = 6


def detect_balloons(
    gray: np.ndarray,
    params: BalloonParams,
    text_rects: list[BBox] | None = None,
    *,
    use_text_rects: bool = True,
    use_white_fill: bool = True,
    use_solidity: bool = True,
) -> np.ndarray:
    """フキダシの損失マスクを検出する。

    Args:
        gray: 原寸グレースケール画像 (H, W) の uint8 配列。
        params: `balloon.*`（`AnalysisParams.balloon`）。
        text_rects: PSD の `kind == "type"` レイヤー bbox 一覧（原寸座標）。
            PSD 入力時のみ渡される。画像 / PDF 入力では渡さなくてよく、
            その場合は手がかり 2・3（白い連結領域・凸形状）のみで動作する。
        use_text_rects: 手がかり 1（テキスト矩形）を有効にするか。
        use_white_fill: 手がかり 2（白い連結領域）を有効にするか。
        use_solidity: 手がかり 3（凸形状）を有効にするか。
            3 つとも既定で有効。各手がかりの寄与を個別に確認するテスト用に
            無効化できる。

    Returns:
        入力と同じ形状 (H, W) の uint8 マスク。255 = 学習時に無視する画素、0 = 通常どおり使う画素。
    """
    if gray.ndim != 2:
        raise ValueError(f"gray must be a 2D array, got shape {gray.shape}")

    union = np.zeros(gray.shape, dtype=np.uint8)

    if use_text_rects:
        union |= _text_rect_mask(gray.shape, text_rects, params.text_rect_dilate)

    if use_white_fill or use_solidity:
        barrier = _panel_border_barrier(gray)
        components = _white_connected_components(gray, params.white_threshold, barrier)

        # テキスト矩形が使えるときは、白領域由来の候補をそれで絞る（フキダシには
        # ほぼ必ずセリフが入るため、セリフと重ならない白領域はフキダシではない）。
        # 絵の中の白い領域（モニタ画面・白い床・明るい背景）の誤検出を落とすのが目的。
        allowed_labels = None
        if params.require_text_overlap and use_text_rects and text_rects:
            allowed_labels = _labels_overlapping_text(
                gray.shape, components, text_rects, params.text_rect_dilate
            )

        if use_white_fill:
            union |= _white_fill_mask(
                gray,
                components,
                min_area_ratio=params.min_area_ratio,
                max_area_ratio=params.max_area_ratio,
                white_threshold=params.white_threshold,
                white_fill_ratio=params.white_fill_ratio,
                allowed_labels=allowed_labels,
            )
        if use_solidity:
            union |= _solidity_mask(
                gray.shape,
                components,
                min_area_ratio=params.min_area_ratio,
                max_area_ratio=params.max_area_ratio,
                min_solidity=params.min_solidity,
                allowed_labels=allowed_labels,
            )

    return _dilate(union, params.dilate_radius)


def _labels_overlapping_text(
    shape: tuple[int, int],
    components: _Components,
    text_rects: list[BBox],
    dilate_px: int,
) -> set[int]:
    """膨張させたテキスト矩形に重なる白連結成分のラベル ID を返す。

    **PSD 入力（テキストレイヤーがある）でのみ使える絞り込み。** 画像 / PDF 入力では
    テキスト矩形が得られないため呼ばれず、従来どおり白領域の形状だけで判定する。
    セリフをラスタライズしている原稿でもテキストレイヤーが取れないため、
    `text_rects` が空のときは呼び出し側でこの絞り込み自体を行わない
    （空集合を返すと全てのフキダシが落ちてしまう）。
    """
    _num_labels, labels, _stats = components
    text_zone = _dilate(_text_rect_mask(shape, text_rects, 0), dilate_px) > 0
    overlapping = np.unique(labels[text_zone])
    return {int(label) for label in overlapping if label != 0}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """円形カーネルで `mask` を膨張させる（`radius` <= 0 なら何もしない）。"""
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


def _text_rect_mask(
    shape: tuple[int, int], text_rects: list[BBox] | None, dilate_px: int
) -> np.ndarray:
    """手がかり 1: テキストレイヤー矩形を膨張させたマスク（`text_rects` が None/空なら全 0）。"""
    mask = np.zeros(shape, dtype=np.uint8)
    if not text_rects:
        return mask
    height, width = shape
    for x, y, w, h in text_rects:
        x0 = max(int(x), 0)
        y0 = max(int(y), 0)
        x1 = min(int(x) + int(w), width)
        y1 = min(int(y) + int(h), height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return _dilate(mask, dilate_px)


def _white_connected_components(
    gray: np.ndarray, white_threshold: int, panel_barrier: np.ndarray | None = None
) -> _Components:
    """`gray > white_threshold` の連結成分を返す（手がかり 2・3 で共有する下準備）。

    `panel_barrier` を渡すと、その画素位置は白から除外してから連結成分を求める。
    コマ枠を跨いではみ出したフキダシは、跨いだ箇所の枠線インクごとフキダシに塗り
    潰されるため、素の白領域だけではコマ内側とコマ外（余白）が地続きの 1 成分に
    なってしまう（#28）。`_panel_border_barrier` が再構成したコマ枠線を分離線として
    渡すことで、内側と外側を別成分に切り離す。
    """
    white = (gray > white_threshold).astype(np.uint8) * 255
    if panel_barrier is not None:
        white = cv2.bitwise_and(white, cv2.bitwise_not(panel_barrier))
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        white, connectivity=8
    )
    return num_labels, labels, stats


def _panel_border_barrier(gray: np.ndarray) -> np.ndarray:
    """コマ枠線を分離線として再構成したバリアマスクを返す（#28）。

    フキダシがコマ枠を跨いで余白（gutter）へはみ出すと、跨いだ箇所の枠線インクごと
    フキダシに塗り潰されてしまう。素の白領域（`gray > white_threshold`）だけを見ると
    その箇所で文字どおり枠線が途切れているため、コマ内側と余白が地続きの 1 連結成分に
    なり、`_touches_border` による背景除外に丸ごと巻き込まれて脱落する。

    ここではモルフォロジー勾配で得たインクの輪郭にハフ変換を掛けて長い直線（コマ枠線）
    を検出するが、`maxLineGap` を大きく取ることで、フキダシに塗り潰された断裂（フキダシ
    の幅ぶん）があっても直線として橋渡しして復元する。復元した直線を分離線として描く
    ことで、コマ内側のフキダシ部分と余白側を別の連結成分に切り離し、コマ内側の部分だけ
    でも損失マスクの候補として残せるようにする。

    コマ枠線に相当する長い直線が画像中に無い場合（コマ枠を持たない画像等）は空マスクを
    返し、既存挙動をそのまま保つ。
    """
    height, width = gray.shape
    gray_u8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    _, ink = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.morphologyEx(ink, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    min_length = max(_BARRIER_LINE_MIN_LENGTH_FLOOR, int(min(height, width) * _BARRIER_LINE_MIN_LENGTH_RATIO))
    vote_threshold = max(
        _BARRIER_HOUGH_VOTE_THRESHOLD_FLOOR, int(min_length * _BARRIER_HOUGH_VOTE_THRESHOLD_RATIO)
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=vote_threshold,
        minLineLength=min_length,
        maxLineGap=_BARRIER_HOUGH_MAX_LINE_GAP,
    )

    barrier = np.zeros(gray.shape, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(barrier, (int(x1), int(y1)), (int(x2), int(y2)), 255, _BARRIER_LINE_THICKNESS_PX)
    return barrier


def _touches_border(x: int, y: int, w: int, h: int, shape: tuple[int, int]) -> bool:
    """連結成分の外接矩形が画像の外周に接しているかを判定する。

    実原稿の白ページ背景（コマ枠の外側を取り巻く白領域）は必ず画像の外周まで達する。
    この判定でそのような成分を候補から除外することで、ページ全体がフキダシの損失
    マスクとして誤って塗られるのを防ぐ（#15）。
    """
    height, width = shape
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


def _white_fill_mask(
    gray: np.ndarray,
    components: _Components,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    white_threshold: int,
    white_fill_ratio: float,
    allowed_labels: set[int] | None = None,
) -> np.ndarray:
    """手がかり 2: 一定面積以上で内部がほぼ白の連結領域を検出する（フチなしフキダシも拾う）。

    連結成分そのもの（定義上 100% 白）ではなく、その **外接矩形** 内での白画素割合を見る。
    こうすることで、フチや文字などわずかに白でない画素が混じっていても許容しつつ、
    外接矩形をそのままマスクとして塗る（過剰に広く取ってよい方針に沿う）。

    ただし、ページ背景（コマ枠の外側を取り巻く白領域）はこの手がかりで拾ってはならない
    ため、外接矩形が画像の外周に接する成分・`max_area_ratio` を超える成分は候補から除外
    する（#15）。
    """
    num_labels, labels, stats = components
    total_area = gray.shape[0] * gray.shape[1]
    min_area = max(1.0, total_area * min_area_ratio)
    max_area = total_area * max_area_ratio

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if allowed_labels is not None and label_id not in allowed_labels:
            continue
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        if _touches_border(x, y, w, h, gray.shape):
            continue
        bbox_region = gray[y : y + h, x : x + w]
        white_ratio = float(np.count_nonzero(bbox_region > white_threshold)) / bbox_region.size
        if white_ratio >= white_fill_ratio:
            mask[y : y + h, x : x + w] = 255
    return mask


def _solidity_mask(
    shape: tuple[int, int],
    components: _Components,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_solidity: float,
    allowed_labels: set[int] | None = None,
) -> np.ndarray:
    """手がかり 3: 凸包面積比（solidity）が高い（楕円・角丸の凸形状）連結領域を検出する。

    塗るのは連結成分そのものではなく **凸包**。フキダシの縁取り線（連結成分には含まれない
    暗画素）まで含めて覆えるため、白い連結領域だけを塗るより広く取れる。

    solidity は「連結成分の実画素数（`stats` の `CC_STAT_AREA`、穴は含まない）/ 凸包面積」
    で計算する。`cv2.contourArea(contour)` は RETR_EXTERNAL の外側輪郭が囲む面積であり
    内部の穴を埋めた値になるため、コマ枠に囲まれた環状のページ背景のような穴だらけの
    成分でも 1.0 近くと誤判定してしまう（#15）。加えて、外接矩形が画像の外周に接する
    成分・`max_area_ratio` を超える成分も候補から除外する。
    """
    num_labels, labels, stats = components
    total_area = shape[0] * shape[1]
    min_area = max(1.0, total_area * min_area_ratio)
    max_area = total_area * max_area_ratio

    mask = np.zeros(shape, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if allowed_labels is not None and label_id not in allowed_labels:
            continue
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        if _touches_border(x, y, w, h, shape):
            continue
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity >= min_solidity:
            cv2.fillConvexPoly(mask, hull, 255)
    return mask
