"""C. コマ分割（設計書 4章 C）。

**矩形クロップは不可。ポリゴンマスクで切る。** キャラのはみ出しで隣コマを巻き込むため
（Epic 仕様書 3節）、コマの形状は輪郭近似から得たポリゴン頂点列として保持する。

設計書 4章 C は「コマ分割自体は古典的画像処理で対応可能だが、例外処理が工数の大半を占める」
としている。本モジュールの目的の半分は例外検出そのもの（`PanelFlag` 6 種）にある。

処理の流れ（Epic 仕様書 4-C 節）:

1. 二値化 → 長い直線成分を抽出（本実装ではハフ変換）して枠線候補を得る
2. `panel.close_kernel` でクローズして枠線を連結し、閉領域（穴）を輪郭抽出する
3. ページ面積の `panel.min_panel_area_ratio` 未満の領域は破棄する
4. 輪郭を `panel.approx_epsilon_ratio` で多角形近似し、ポリゴン頂点列（原寸座標）として保持する
5. 例外ケースを検出してフラグを立てる

`unclosed`（枠線が検出できても閉領域にならない）は、通常のクロージングでは閉じない
領域を、より大きいカーネルでのクロージング（本実装では固定倍率のフォールバックパス）で
救済して検出する。通常パスで既に見つかっているコマと重なるものは重複として除外する。

`cut_off`（断ち切りコマ）は、枠線がページ端の `panel.virtual_frame_margin` 以内まで
迫っている場合に、ページ外へ仮想的にキャンバスを延長してそこへ枠を閉じる壁を立てることで
補完する。閉じた結果のポリゴン頂点はページ範囲へクリップして保持する（原寸座標系のまま）。
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from rough2ink.core.params import PanelParams
from rough2ink.core.types import PanelFlag, PanelInfo

# --- 内部定数（AnalysisParams には含まれない、アルゴリズム実装上の調整値） ---

# ハフ変換で枠線候補とみなす最小長は、ページ短辺に対する比率で決める。
_LINE_MIN_LENGTH_RATIO = 0.2
_LINE_MIN_LENGTH_FLOOR = 30
_HOUGH_VOTE_THRESHOLD = 30
_HOUGH_MAX_LINE_GAP = 15
_HOUGH_LINE_THICKNESS = 3

# unclosed 救済パスのクロージングカーネルは close_kernel の倍数（+ 下駄）で決める。
_LOOSE_CLOSE_MULTIPLIER = 4
_LOOSE_CLOSE_EXTRA = 20

# overflow 判定: ポリゴン境界からこの深さ（px）を超えて内外にまたがる暗画素塊があれば
# 「はみ出し」とみなす（枠線自体は太さが小さいのでこの深さには届かない）。
_OVERFLOW_DEPTH_PX = 10

# oblique 判定で無視する短辺（近似誤差のノイズ除去用）。
_OBLIQUE_MIN_EDGE_LEN_PX = 6.0

# effect_lines 判定でのハフ変換の投票閾値。
_EFFECT_LINE_HOUGH_THRESHOLD = 15
_EFFECT_LINE_MAX_GAP = 3
_EFFECT_LINE_THICKNESS = 2

_VIRTUAL_FRAME_SEAL_THICKNESS = 2


def detect_panels(gray: np.ndarray, params: PanelParams | None = None) -> list[PanelInfo]:
    """原寸グレースケール画像からコマポリゴンを検出する。

    戻り値の `PanelInfo.polygon` は原寸座標系の頂点列。
    """
    if params is None:
        params = PanelParams()
    if gray.ndim != 2:
        raise ValueError("gray must be a 2D single-channel array")

    h, w = gray.shape
    page_area = float(h * w)
    if page_area <= 0:
        return []

    ink = _binarize_ink(gray)
    frame_lines = _extract_frame_lines(ink, h, w)

    strict_mask = _close(frame_lines, params.close_kernel)
    strict_sealed, strict_offset = _seal_edges_with_virtual_frame(strict_mask, params.virtual_frame_margin)
    strict_holes = [_offset_contour(c, strict_offset) for c in _hole_contours(strict_sealed)]

    loose_kernel = max(int(params.close_kernel) * _LOOSE_CLOSE_MULTIPLIER, int(params.close_kernel) + _LOOSE_CLOSE_EXTRA)
    loose_mask = _close(frame_lines, loose_kernel)
    loose_sealed, loose_offset = _seal_edges_with_virtual_frame(loose_mask, params.virtual_frame_margin)
    loose_holes = [_offset_contour(c, loose_offset) for c in _hole_contours(loose_sealed)]

    min_area = params.min_panel_area_ratio * page_area

    candidates: list[tuple[np.ndarray, bool]] = []
    for contour in strict_holes:
        if cv2.contourArea(contour) >= min_area:
            candidates.append((contour, False))

    for contour in loose_holes:
        if cv2.contourArea(contour) < min_area:
            continue
        if _matches_any(contour, [c for c, _ in candidates]):
            continue
        candidates.append((contour, True))

    is_spread = (max(w, h) / max(1.0, float(min(w, h)))) > params.spread_aspect_ratio

    num_labels, labels = cv2.connectedComponentsWithStats(ink, connectivity=8)[:2]
    depth_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * _OVERFLOW_DEPTH_PX + 1, 2 * _OVERFLOW_DEPTH_PX + 1)
    )

    panels: list[PanelInfo] = []
    for contour, is_unclosed in candidates:
        epsilon = params.approx_epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [
            (_clip(float(pt[0][0]), 0.0, float(w - 1)), _clip(float(pt[0][1]), 0.0, float(h - 1)))
            for pt in approx
        ]
        if len(polygon) < 3:
            continue

        clipped_pts = np.array([[round(x), round(y)] for x, y in polygon], dtype=np.int32)
        area = abs(cv2.contourArea(clipped_pts))
        if area < min_area:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs) - min(xs))),
            int(round(max(ys) - min(ys))),
        )

        flags: list[PanelFlag] = []
        if is_unclosed:
            flags.append("unclosed")
        if _touches_edge(polygon, w, h, params.virtual_frame_margin):
            flags.append("cut_off")
        if _has_oblique_edge(polygon, params.oblique_angle_deg):
            flags.append("oblique")
        if is_spread:
            flags.append("spread")

        panel_mask = polygon_mask((h, w), polygon)
        if _has_overflow(labels, panel_mask, depth_kernel):
            flags.append("overflow")
        if _effect_line_density(ink, panel_mask, area) > params.effect_line_density:
            flags.append("effect_lines")

        panels.append(
            PanelInfo(
                panel_id="",
                polygon=polygon,
                bbox=bbox,
                area_ratio=area / page_area,
                flags=flags,
            )
        )

    panels.sort(key=lambda p: (p.bbox[1], p.bbox[0]) if p.bbox is not None else (0, 0))
    for idx, panel in enumerate(panels):
        panel.panel_id = f"panel_{idx:03d}"

    return panels


def polygon_mask(shape: tuple[int, int], polygon: Sequence[tuple[float, float]]) -> np.ndarray:
    """`shape`（高さ, 幅）と同じ大きさの、ポリゴン内部を 255 で塗ったマスクを返す。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    pts = np.array([[round(x), round(y)] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    """暗画素（インク: 枠線・線画・ベタ等）を 255、それ以外を 0 とする二値画像を返す。"""
    gray_u8 = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    _, ink = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _extract_frame_lines(ink: np.ndarray, h: int, w: int) -> np.ndarray:
    """長い直線成分をハフ変換で抽出し、枠線候補マスクを作る。

    塗りつぶされた塊（フキダシはみ出しの塊など）の輪郭は曲線であり長い直線を
    生じないため、モルフォロジー勾配で輪郭線化してからハフ変換に掛けることで
    誤検出を抑える。傾いたコマ枠（`oblique`）も検出できるよう、水平・垂直に
    限定しない汎用のハフ変換を用いる（Epic 仕様書 4-C 節の「ハフ変換」経路）。
    """
    edges = cv2.morphologyEx(ink, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    min_length = max(_LINE_MIN_LENGTH_FLOOR, int(min(h, w) * _LINE_MIN_LENGTH_RATIO))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=_HOUGH_VOTE_THRESHOLD,
        minLineLength=min_length,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )
    canvas = np.zeros_like(ink)
    if lines is None:
        return canvas
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, _HOUGH_LINE_THICKNESS)
    return canvas


def _close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """枠線を連結するためのクロージング（`panel.close_kernel` 等）。"""
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _seal_edges_with_virtual_frame(mask: np.ndarray, margin: int) -> tuple[np.ndarray, int]:
    """`margin` 以内でページ端に迫る枠線を、ページ外へ延長した仮想の壁で閉じる。

    枠線候補がページ端の `margin` px 以内まで来ている列・行だけを対象に、
    その列・行をページ外の拡張領域（`margin` px 分パディング）へまっすぐ延長し、
    拡張領域の外周に壁（`seal`）を立てて閉じる。これにより、枠線がページ端の
    近くで途切れている（＝断ち切りコマ）場合だけが選択的に閉領域になり、
    ページ端から十分離れた場所にある別のコマ（枠線がページ端に無関係）は
    影響を受けない。戻り値は (拡張後マスク, 拡張オフセット)。
    オフセットは輪郭座標を原寸へ戻す際に使う。
    """
    margin = int(margin)
    if margin <= 0:
        return mask.copy(), 0

    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape
    seal = _VIRTUAL_FRAME_SEAL_THICKNESS

    top_cols = np.any(mask[:margin, :] > 0, axis=0)
    if np.any(top_cols):
        top_pad = padded[0:margin, margin : margin + w]
        top_pad[:, top_cols] = 255
        padded[0:seal, :] = 255

    bottom_cols = np.any(mask[h - margin :, :] > 0, axis=0)
    if np.any(bottom_cols):
        bottom_pad = padded[margin + h : ph, margin : margin + w]
        bottom_pad[:, bottom_cols] = 255
        padded[ph - seal : ph, :] = 255

    left_rows = np.any(mask[:, :margin] > 0, axis=1)
    if np.any(left_rows):
        left_pad = padded[margin : margin + h, 0:margin]
        left_pad[left_rows, :] = 255
        padded[:, 0:seal] = 255

    right_rows = np.any(mask[:, w - margin :] > 0, axis=1)
    if np.any(right_rows):
        right_pad = padded[margin : margin + h, margin + w : pw]
        right_pad[right_rows, :] = 255
        padded[:, pw - seal : pw] = 255

    return padded, margin


def _hole_contours(mask: np.ndarray) -> list[np.ndarray]:
    """枠線マスクの「穴」（枠に囲まれた背景領域 = コマ内部）の輪郭一覧を返す。"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    holes = []
    for contour, h_row in zip(contours, hierarchy[0]):
        parent = h_row[3]
        if parent != -1:
            holes.append(contour)
    return holes


def _offset_contour(contour: np.ndarray, offset: int) -> np.ndarray:
    """`_seal_edges_with_virtual_frame` の拡張ぶんを差し引き、原寸座標系へ戻す。"""
    if offset == 0:
        return contour
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= offset
    shifted[:, 0, 1] -= offset
    return shifted


def _matches_any(contour: np.ndarray, others: list[np.ndarray]) -> bool:
    """`contour` の重心が `others` のいずれかの内部にあるか（重複判定）。"""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return False
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    for other in others:
        if cv2.pointPolygonTest(other, (cx, cy), False) >= 0:
            return True
    return False


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _touches_edge(polygon: list[tuple[float, float]], w: int, h: int, margin: int) -> bool:
    """ポリゴンの頂点がページ端の `margin` px 以内にあるか（`cut_off` 判定）。"""
    margin = max(0, int(margin))
    for x, y in polygon:
        if x <= margin or x >= (w - 1 - margin) or y <= margin or y >= (h - 1 - margin):
            return True
    return False


def _has_oblique_edge(polygon: list[tuple[float, float]], threshold_deg: float) -> bool:
    """水平・垂直から `threshold_deg` を超えて傾いた辺があるか（`oblique` 判定）。"""
    n = len(polygon)
    if n < 2:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < _OBLIQUE_MIN_EDGE_LEN_PX:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 90.0
        deviation = min(angle, 90.0 - angle)
        if deviation > threshold_deg:
            return True
    return False


def _has_overflow(labels: np.ndarray, panel_mask: np.ndarray, depth_kernel: np.ndarray) -> bool:
    """ポリゴン境界を跨いで内外に深く広がる連続した暗画素塊があるか（`overflow` 判定）。

    枠線自体は太さが小さく `_OVERFLOW_DEPTH_PX` ほど深く内外どちらにも入り込まないため、
    「境界から一定深さより内側」と「境界から一定深さより外側」の両方に画素を持つ
    連結成分があれば、それは枠線ではなくキャラクター等の絵柄がはみ出したものとみなす。
    """
    deep_inside = cv2.erode(panel_mask, depth_kernel)
    deep_outside = cv2.bitwise_not(cv2.dilate(panel_mask, depth_kernel))

    inside_labels = set(np.unique(labels[deep_inside > 0]).tolist()) - {0}
    if not inside_labels:
        return False
    outside_labels = set(np.unique(labels[deep_outside > 0]).tolist()) - {0}
    return len(inside_labels & outside_labels) > 0


def _effect_line_density(ink: np.ndarray, panel_mask: np.ndarray, area: float) -> float:
    """コマ領域内の直線密度（ハフ変換で検出した直線が占める画素比率）を返す。"""
    if area <= 0:
        return 0.0
    region = cv2.bitwise_and(ink, panel_mask)
    min_len = max(10, int(0.15 * math.sqrt(area)))
    lines = cv2.HoughLinesP(
        region,
        1,
        np.pi / 180,
        threshold=_EFFECT_LINE_HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=_EFFECT_LINE_MAX_GAP,
    )
    if lines is None:
        return 0.0
    canvas = np.zeros_like(region)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, _EFFECT_LINE_THICKNESS)
    canvas = cv2.bitwise_and(canvas, panel_mask)
    line_pixels = cv2.countNonZero(canvas)
    return line_pixels / area
