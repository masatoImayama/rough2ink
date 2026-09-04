"""コマ分割（`rough2ink.core.panels`）のテスト（Epic 仕様書 4-C 節）。

実原稿は手元にないため、枠線を描画した合成ページで検証する
（Epic 本文「テスト方針」）。矩形クロップではなく **ポリゴン** で切ることと、
6 種の例外フラグそれぞれが対応する合成ケースで立つことを重点的に確認する。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from rough2ink.core.panels import detect_panels, polygon_mask
from rough2ink.core.params import PanelParams


def _blank_page(height: int, width: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _draw_border(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    sides: tuple[str, ...] = ("top", "bottom", "left", "right"),
    thickness: int = 3,
) -> None:
    """矩形の枠線を、指定した辺だけ描画する（断ち切り・枠線途切れの合成用）。"""
    if "top" in sides:
        cv2.line(img, (x1, y1), (x2, y1), 0, thickness)
    if "bottom" in sides:
        cv2.line(img, (x1, y2), (x2, y2), 0, thickness)
    if "left" in sides:
        cv2.line(img, (x1, y1), (x1, y2), 0, thickness)
    if "right" in sides:
        cv2.line(img, (x2, y1), (x2, y2), 0, thickness)


def _draw_border_with_gap(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    gap_side: str,
    gap_len: int,
    thickness: int = 3,
) -> None:
    """指定した 1 辺の中央に `gap_len` px の隙間を空けて枠線を描画する（unclosed 合成用）。"""
    _draw_border(img, x1, y1, x2, y2, thickness=thickness)
    if gap_side == "top":
        cx = (x1 + x2) // 2
        cv2.line(img, (cx - gap_len // 2, y1), (cx + gap_len // 2, y1), 255, thickness + 2)
    elif gap_side == "bottom":
        cx = (x1 + x2) // 2
        cv2.line(img, (cx - gap_len // 2, y2), (cx + gap_len // 2, y2), 255, thickness + 2)
    elif gap_side == "left":
        cy = (y1 + y2) // 2
        cv2.line(img, (x1, cy - gap_len // 2), (x1, cy + gap_len // 2), 255, thickness + 2)
    elif gap_side == "right":
        cy = (y1 + y2) // 2
        cv2.line(img, (x2, cy - gap_len // 2), (x2, cy + gap_len // 2), 255, thickness + 2)
    else:
        raise ValueError(f"unknown gap_side: {gap_side}")


def test_detect_panels_counts_synthetic_grid() -> None:
    """2x2 のコマ割りページで 4 コマが検出され、いずれも例外フラグを持たないこと。"""
    page = _blank_page(500, 500)
    cells = [(40, 40, 220, 220), (280, 40, 460, 220), (40, 280, 220, 460), (280, 280, 460, 460)]
    for x1, y1, x2, y2 in cells:
        _draw_border(page, x1, y1, x2, y2)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 4
    for panel in panels:
        assert panel.flags == []
        assert panel.is_clean
        assert panel.bbox is not None
        assert panel.area_ratio > 0
        # 矩形クロップではなくポリゴン頂点列で保持されていること
        assert len(panel.polygon) >= 3
        for x, y in panel.polygon:
            assert 0 <= x <= 500
            assert 0 <= y <= 500


def test_cut_off_panel_is_closed_via_virtual_frame() -> None:
    """ページ端で断ち切られたコマが、仮想枠補完で閉領域として検出され `cut_off` が立つこと。"""
    page = _blank_page(300, 300)
    # 下辺を描かず、ページ下端（y=299）のごく近く（virtual_frame_margin=8 以内）で
    # 左右の枠線を止める。下辺は仮想枠補完で閉じられることを期待する。
    _draw_border(page, 40, 40, 260, 296, sides=("top", "left", "right"))

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 1
    panel = panels[0]
    assert "cut_off" in panel.flags
    # 閉領域として検出できている（穴として輪郭抽出できている）こと
    assert len(panel.polygon) >= 3
    assert panel.area_ratio > 0.1
    # 仮想枠補完でページ下端まで閉じられていること
    assert max(y for _, y in panel.polygon) >= 299 - 1


def test_unclosed_panel_is_flagged_but_still_recovered() -> None:
    """枠線の途中に大きな隙間があるコマは `unclosed` が立ち、それでも救済されて検出されること。"""
    page = _blank_page(400, 400)
    # ページ端から十分離して描画し、cut_off 等の他フラグを誘発しないようにする。
    _draw_border_with_gap(page, 80, 80, 320, 320, gap_side="top", gap_len=26)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 1
    panel = panels[0]
    assert "unclosed" in panel.flags
    assert "cut_off" not in panel.flags


def test_oblique_panel_is_flagged() -> None:
    """水平・垂直から大きく傾いたコマ枠に `oblique` が立つこと。"""
    page = _blank_page(300, 300)
    center = (150, 150)
    half = 80
    angle_deg = 20.0
    angle = math.radians(angle_deg)
    corners = []
    for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        rx = dx * math.cos(angle) - dy * math.sin(angle)
        ry = dx * math.sin(angle) + dy * math.cos(angle)
        corners.append((int(center[0] + rx), int(center[1] + ry)))
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(page, [pts], isClosed=True, color=0, thickness=3)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 1
    assert "oblique" in panels[0].flags


def test_overflow_panel_is_flagged() -> None:
    """コマ境界をまたいで内外に広がる暗画素塊（キャラのはみ出し想定）に `overflow` が立つこと。"""
    page = _blank_page(320, 320)
    _draw_border(page, 60, 60, 260, 260)
    # 右辺の中央をまたぐ位置に、境界の内外へ十分深くはみ出す塗りつぶし円を描く。
    cv2.circle(page, (260, 160), 18, 0, thickness=-1)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 1
    assert "overflow" in panels[0].flags


def test_spread_page_flags_all_panels() -> None:
    """見開き相当（アスペクト比が大きい）ページ上の全コマに `spread` が立つこと。"""
    page = _blank_page(280, 620)  # 620/280 ≈ 2.21 > spread_aspect_ratio(1.2)
    _draw_border(page, 40, 40, 280, 240)
    _draw_border(page, 340, 40, 580, 240)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 2
    for panel in panels:
        assert "spread" in panel.flags


def test_spread_flag_uses_orientation_not_max_min_ratio() -> None:
    """真の見開き（横長）ページでは実寸相当でも `spread` が立つこと（Review #14）。

    `is_spread` を max(w,h)/min(w,h) で読むと縦長・横長を区別できず、
    このテストの横長ページと直後の縦長ページ（同じ aspect 1.414）が
    どちらも spread 扱いになってしまう。向き（w/h）で判定することを確認する。
    """
    page = _blank_page(1000, 1414)  # h=1000, w=1414 -> w/h = 1.414 (真の見開き相当)
    _draw_border(page, 40, 40, 660, 960)
    _draw_border(page, 754, 40, 1374, 960)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 2
    for panel in panels:
        assert "spread" in panel.flags


def test_portrait_page_at_realistic_scale_does_not_flag_spread() -> None:
    """A4/B4 相当（aspect 1.414）の縦長ページで `spread` が立たないこと（Review #14）。

    旧実装の `max(w,h)/min(w,h) > 1.2` は向きを捨てるため、この縦長ページも
    誤って見開き判定になっていた（実原稿に対して成功率が常に0になる原因）。
    """
    page = _blank_page(1414, 1000)  # h=1414, w=1000 -> w/h ≈ 0.707 (縦長・単ページ相当)
    cells = [(40, 40, 480, 680), (520, 40, 960, 680), (40, 720, 480, 1360), (520, 720, 960, 1360)]
    for x1, y1, x2, y2 in cells:
        _draw_border(page, x1, y1, x2, y2)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 4
    for panel in panels:
        assert "spread" not in panel.flags


@pytest.mark.parametrize("thickness", [3, 5, 8, 12])
def test_detect_panels_counts_synthetic_grid_at_realistic_page_scale(thickness: int) -> None:
    """1414x1000（A4/B4相当）の縦長ページに描いた 2x2 グリッドで、4コマ全てが検出されること
    （Review #17 / Review #27）。

    旧実装はモルフォロジー勾配が太い枠線を2本の細いレールに分解してしまい、ハフ変換の
    投票が不足して辺を取りこぼす（4コマ中3コマ、太さによっては2コマしか検出できない）
    ことがあった。しかも `unclosed` フラグも立たず無言で欠落するため、この規模のテストを
    追加しないと検出不能な状態のまま気づけない。

    枠線太さは 3px（既定値）だけでは旧実装でも通ってしまい欠陥を再現しないため
    （Review #27）、B4原稿の標準的な枠線太さである 5px 以上を含む複数の太さで検証する。
    """
    page = _blank_page(1414, 1000)  # h=1414, w=1000 (A4/B4 相当の縦長ページ)
    cells = [(40, 40, 480, 680), (520, 40, 960, 680), (40, 720, 480, 1360), (520, 720, 960, 1360)]
    for x1, y1, x2, y2 in cells:
        _draw_border(page, x1, y1, x2, y2, thickness=thickness)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 4
    for panel in panels:
        assert panel.flags == []
        assert panel.is_clean


def test_effect_lines_panel_is_flagged() -> None:
    """コマ内部の直線密度が高い（集中線・効果線を想定）場合に `effect_lines` が立つこと。"""
    page = _blank_page(320, 320)
    x1, y1, x2, y2 = 60, 60, 260, 260
    _draw_border(page, x1, y1, x2, y2)

    # 集中線を大量に描画する。1 本 1 本はコマ枠として誤検出されないよう短く抑えつつ
    # （枠線抽出のハフ変換が要求する最小長より短くする）、本数で直線密度の閾値超過を狙う。
    rng = np.random.default_rng(0)
    origin = ((x1 + x2) // 2, (y1 + y2) // 2)
    for _ in range(90):
        angle = rng.uniform(0, 2 * math.pi)
        length = 45
        end = (int(origin[0] + length * math.cos(angle)), int(origin[1] + length * math.sin(angle)))
        cv2.line(page, origin, end, 0, 3)

    panels = detect_panels(page, PanelParams())

    assert len(panels) == 1
    assert "effect_lines" in panels[0].flags


def test_polygon_mask_generates_full_res_mask() -> None:
    """ポリゴンマスク画像（コマ形状を 255 で塗る）が原寸で生成できること。"""
    page = _blank_page(200, 200)
    _draw_border(page, 30, 30, 170, 170)

    panels = detect_panels(page, PanelParams())
    assert len(panels) == 1

    mask = polygon_mask(page.shape, panels[0].polygon)

    assert mask.shape == page.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    # ポリゴン内部（コマ中心付近）は 255 で塗られている
    assert mask[100, 100] == 255
    # ページの隅（コマ外）は塗られていない
    assert mask[5, 5] == 0


def test_detect_panels_rejects_non_2d_input() -> None:
    color_image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_panels(color_image, PanelParams())
