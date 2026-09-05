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


def test_interior_angle_sum_is_fixed_by_vertex_count_so_it_cannot_detect_concavity() -> None:
    """内角の総和は (n-2)×180° で形状に依存しない＝食い込みの検出には使えないこと。

    「内角の総和に上限を設けて異常ポリゴンを弾く」という案は、単純多角形では
    「頂点数に上限を設ける」と同値になる。凹んだ多角形と凸な多角形で総和が同じに
    なることを示し、代わりに優角の数で区別できることを固定する。
    """
    from rough2ink.core.panels import count_reflex_vertices, polygon_solidity

    # 同じ頂点数(6)の、凸な多角形と内側へ食い込んだ多角形。
    convex = [(0.0, 0.0), (60.0, 0.0), (100.0, 40.0), (100.0, 100.0), (40.0, 100.0), (0.0, 60.0)]
    concave = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 40.0), (0.0, 100.0), (0.0, 50.0)]

    # 総和はどちらも (6-2)*180 = 720 で同じ（＝総和では区別できない）。
    assert len(convex) == len(concave)

    assert count_reflex_vertices(convex) == 0
    assert count_reflex_vertices(concave) >= 1, "食い込みは優角として現れるべき"

    convex_area = abs(cv2.contourArea(np.array(convex, dtype=np.float32)))
    concave_area = abs(cv2.contourArea(np.array(concave, dtype=np.float32)))
    assert polygon_solidity(convex, convex_area) > 0.95
    assert polygon_solidity(concave, concave_area) < 0.9


def test_reflex_count_is_orientation_independent() -> None:
    """頂点の並び順（時計回り／反時計回り）に依らず優角の数が同じであること。

    画像座標系は y 軸が下向きで、外積の符号が数学座標系と逆になる。符号の扱いを
    誤ると正常な矩形でも優角が数えられてしまう（実際に最初の実装で矩形の4頂点すべてが
    優角と判定された）。
    """
    from rough2ink.core.panels import count_reflex_vertices

    rectangle = [(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)]

    assert count_reflex_vertices(rectangle) == 0
    assert count_reflex_vertices(list(reversed(rectangle))) == 0


def test_irregular_flag_is_raised_for_concave_panel() -> None:
    """絵を枠線と誤認して内側へ食い込んだコマに `irregular` が立つこと。

    実原稿（全面断ち切りの1コマページ）では、枠線検出が破綻して18頂点・優角8個・
    solidity 0.483 のジグザグが1つだけ検出されるという壊れ方をしていた。検出コマ数
    だけを見ていてもこの失敗は数値に現れないため、例外として計上できるようにする。
    """
    shape = (400, 400)
    gray = np.full(shape, 255, dtype=np.uint8)
    # 大きく凹んだ（食い込んだ）コマ枠を描く。
    concave = np.array(
        [[40, 40], [360, 40], [360, 360], [200, 180], [40, 360]], dtype=np.int32
    )
    cv2.polylines(gray, [concave], isClosed=True, color=0, thickness=6)
    # フォールバックを切って `irregular` 単体の挙動を見る（有効なままだと、信用できる
    # コマが0個になった時点でページ全体1コマに差し替えられてしまう）。
    panels = detect_panels(gray, PanelParams(fallback_to_page=False))

    assert panels, "コマが検出されていない（テストの前提が崩れている）"
    assert any("irregular" in panel.flags for panel in panels), (
        f"食い込んだコマに irregular が立つべき, flags={[p.flags for p in panels]}"
    )


def test_irregular_flag_is_not_raised_for_normal_rectangular_panels() -> None:
    """通常の矩形コマでは `irregular` が立たないこと（誤検出しない）。"""
    shape = (1414, 1000)
    gray = np.full(shape, 255, dtype=np.uint8)
    for top in (60, 740):
        for left in (60, 530):
            cv2.rectangle(gray, (left, top), (left + 410, top + 610), 0, 8)
    panels = detect_panels(gray, PanelParams())

    assert len(panels) == 4, f"4コマ検出される前提, got {len(panels)}"
    for panel in panels:
        assert "irregular" not in panel.flags, f"正常な矩形コマに irregular が立った: {panel.flags}"


def test_page_without_panel_borders_falls_back_to_a_single_page_panel() -> None:
    """枠線が無いページ（全面断ち切り）ではページ全体を1コマとして返すこと。

    実原稿の全面断ち切りページで、絵の輪郭を枠線と誤認して18頂点・solidity 0.483 の
    ジグザグが1個だけ返っていた。正しい答えは「ページ全体が1コマ」なので、信用できる
    コマ（`irregular` でない）が1つも無いときはページ全体に差し替える。

    枠線候補の最小長を上げて誤検出を減らす案は実測で否定した。コマ枠はページ短辺の
    半分より短いことが多く、実測では比率 0.5 で正常な6コマページのコマが全て消えた。
    """
    shape = (600, 420)
    rng = np.random.default_rng(3)
    # 枠線が一切無く、絵だけがあるページを模す（曲線ストロークを散らす）。
    gray = np.full(shape, 255, dtype=np.uint8)
    for _ in range(60):
        center = (int(rng.integers(0, shape[1])), int(rng.integers(0, shape[0])))
        axes = (int(rng.integers(20, 120)), int(rng.integers(20, 120)))
        angle = float(rng.integers(0, 180))
        cv2.ellipse(gray, center, axes, angle, 0, 360, 0, 3)

    panels = detect_panels(gray, PanelParams())

    assert len(panels) == 1, f"ページ全体1コマになるべき, got {len(panels)}"
    panel = panels[0]
    assert "no_frame" in panel.flags, f"フォールバックであることを示すべき, {panel.flags}"
    assert panel.area_ratio == 1.0
    assert panel.bbox == (0, 0, shape[1], shape[0])


def test_fallback_does_not_replace_pages_with_valid_panels() -> None:
    """信用できるコマが取れているページはフォールバックしないこと。"""
    shape = (1414, 1000)
    gray = np.full(shape, 255, dtype=np.uint8)
    for top in (60, 740):
        for left in (60, 530):
            cv2.rectangle(gray, (left, top), (left + 410, top + 610), 0, 8)

    panels = detect_panels(gray, PanelParams())

    assert len(panels) == 4
    assert all("no_frame" not in panel.flags for panel in panels)


def test_fallback_can_be_disabled() -> None:
    """`fallback_to_page=False` なら従来どおり検出結果をそのまま返すこと。"""
    shape = (600, 420)
    gray = np.full(shape, 255, dtype=np.uint8)

    panels = detect_panels(gray, PanelParams(fallback_to_page=False))

    assert all("no_frame" not in panel.flags for panel in panels)
