"""`detect_balloons`（E. フキダシ検出 → 損失マスク生成）のテスト。

実原稿は手元にないため合成フィクスチャで検証する（Epic 実装計画書「テスト方針」）。
白楕円（フチあり）・フチなし白領域・テキスト矩形の 3 パターンで検出されること、
3 つの手がかりが個別に有効/無効を切り替えられ、それぞれの寄与が確認できることを見る。

キャンバスは実原稿と同じ極性（ページ背景が白 = 255）で構成する。ページ背景は画像の
外周まで達する巨大な白領域になるが、フキダシ検出の対象ではないため、コマ枠を模した
暗い矩形（`_PANEL_FILL`）で作画領域を作り、その内側にフキダシ形状を描く。これにより
「ページ背景全体が損失マスクとして誤検出される」という回帰（#15）も自然にテストされる
（背景が誤検出されれば、各手がかりの独立性テストの範囲外まで mask が広がってしまう）。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rough2ink.core.balloons import detect_balloons
from rough2ink.core.params import AnalysisParams, BalloonParams

_SIZE = 200
_PAGE_BACKGROUND = 255  # 実原稿と同じ白いページ背景（画像の外周まで達する）
_PANEL_FILL = 20  # コマ枠内側の作画を模した暗い色（ページ背景と区別するため）
_PANEL_RECT = (30, 30, 170, 170)  # コマ枠内側の作画領域 (x0, y0, x1, y1)
_ELLIPSE_CENTER = (100, 100)
_ELLIPSE_AXES = (40, 25)  # (半長軸, 半短軸)


def _blank_canvas() -> np.ndarray:
    """白いページ背景 + コマ枠内側の暗い作画領域を持つキャンバス。

    ページ背景をそのまま白一色にすると、フキダシ形状も白なので区別が付かなくなる
    （実原稿でもフチなしフキダシがページ背景と地続きなら区別不能なのは同じ）。
    そのためコマ枠内側だけを暗くし、その中にフキダシ形状を描く。
    """
    canvas = np.full((_SIZE, _SIZE), _PAGE_BACKGROUND, dtype=np.uint8)
    x0, y0, x1, y1 = _PANEL_RECT
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _PANEL_FILL, -1)
    return canvas


def _draw_ellipse_with_border(canvas: np.ndarray) -> np.ndarray:
    """フチありの白楕円を描いた画像を返す（元画像は書き換えない）。"""
    out = canvas.copy()
    cv2.ellipse(out, _ELLIPSE_CENTER, _ELLIPSE_AXES, 0, 0, 360, 255, -1)
    cv2.ellipse(out, _ELLIPSE_CENTER, _ELLIPSE_AXES, 0, 0, 360, 0, 3)
    return out


def _draw_borderless_rectangle(canvas: np.ndarray) -> np.ndarray:
    """フチなしの白い矩形領域を描いた画像を返す（元画像は書き換えない）。"""
    out = canvas.copy()
    cv2.rectangle(out, (60, 70), (140, 130), 255, -1)
    return out


def test_ellipse_with_border_is_detected() -> None:
    """白楕円（フチあり）が既定パラメータ（全手がかり有効）で検出される。"""
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(gray, params)

    assert mask[_ELLIPSE_CENTER[1], _ELLIPSE_CENTER[0]] == 255


def test_borderless_white_region_is_detected() -> None:
    """フチなし白領域が既定パラメータ（全手がかり有効）で検出される。"""
    gray = _draw_borderless_rectangle(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(gray, params)

    assert mask[100, 100] == 255


def test_text_rect_is_detected() -> None:
    """テキスト矩形（PSD 由来）が既定パラメータで検出される。

    視覚的な手がかり（白領域・凸形状）が一切無い画像でも、`text_rects` だけで検出される。
    """
    gray = _blank_canvas()
    params = BalloonParams()
    text_rects = [(50, 50, 30, 20)]

    mask = detect_balloons(gray, params, text_rects=text_rects)

    assert mask[60, 65] == 255  # 矩形の中心付近


def test_text_rects_none_does_not_raise() -> None:
    """`text_rects=None`（画像 / PDF 入力想定）でもエラーにならず動作する。"""
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(gray, params)  # text_rects を渡さない

    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    # 手がかり 1 が使えなくても、手がかり 2・3 だけで楕円は検出される
    assert mask[_ELLIPSE_CENTER[1], _ELLIPSE_CENTER[0]] == 255


def test_white_fill_clue_contributes_independently() -> None:
    """手がかり 2（白い連結領域）単独で有効/無効を切り替えられ、寄与が確認できる。"""
    gray = _draw_borderless_rectangle(_blank_canvas())
    params = BalloonParams()

    enabled = detect_balloons(
        gray, params, use_text_rects=False, use_white_fill=True, use_solidity=False
    )
    disabled = detect_balloons(
        gray, params, use_text_rects=False, use_white_fill=False, use_solidity=False
    )

    assert enabled[100, 100] == 255
    assert np.count_nonzero(disabled) == 0


def test_solidity_clue_contributes_independently() -> None:
    """手がかり 3（凸形状 solidity）単独で有効/無効を切り替えられ、寄与が確認できる。"""
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    enabled = detect_balloons(
        gray, params, use_text_rects=False, use_white_fill=False, use_solidity=True
    )
    disabled = detect_balloons(
        gray, params, use_text_rects=False, use_white_fill=False, use_solidity=False
    )

    assert enabled[_ELLIPSE_CENTER[1], _ELLIPSE_CENTER[0]] == 255
    assert np.count_nonzero(disabled) == 0


def test_text_rect_clue_contributes_independently() -> None:
    """手がかり 1（テキスト矩形）単独で有効/無効を切り替えられ、寄与が確認できる。"""
    gray = _blank_canvas()  # 視覚的な手がかりが一切無い画像
    params = BalloonParams()
    text_rects = [(50, 50, 30, 20)]

    enabled = detect_balloons(
        gray, params, text_rects=text_rects,
        use_text_rects=True, use_white_fill=False, use_solidity=False,
    )
    disabled = detect_balloons(
        gray, params, text_rects=text_rects,
        use_text_rects=False, use_white_fill=False, use_solidity=False,
    )

    assert enabled[60, 65] == 255
    assert np.count_nonzero(disabled) == 0


def test_output_mask_is_wider_than_raw_white_region() -> None:
    """出力マスクが入力の白領域そのものより広い（最終膨張が効いている）ことを確認する。

    比較の基準はコマ枠内側（フキダシそのもの）の生の白画素数に限定する。ページ背景の
    生の白画素はそもそも損失マスクの対象外（#15）であり、そのまま比較に含めると
    「膨張したのに raw より狭くなる」という無関係な理由で失敗してしまうため。
    """
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    x0, y0, x1, y1 = _PANEL_RECT
    panel_gray = gray[y0:y1, x0:x1]
    raw_white_count = int(np.count_nonzero(panel_gray > params.white_threshold))

    mask = detect_balloons(gray, params)
    mask_count = int(np.count_nonzero(mask))
    panel_mask = mask[y0:y1, x0:x1]

    assert mask_count > raw_white_count
    # コマ枠内側の白領域そのものが縮んではいけない（膨張のみで縮小はしていない）
    assert np.all(panel_mask[panel_gray > params.white_threshold] == 255)

    # 楕円境界の少し外側（膨張半径 12px 以内）は無視領域に含まれる
    near_x = _ELLIPSE_CENTER[0] + _ELLIPSE_AXES[0] + 8
    assert mask[_ELLIPSE_CENTER[1], near_x] == 255
    # 十分遠い外側（膨張半径を超える）は無視領域に含まれない
    far_x = _ELLIPSE_CENTER[0] + _ELLIPSE_AXES[0] + 30
    assert mask[_ELLIPSE_CENTER[1], far_x] == 0


def test_output_mask_shape_dtype_and_values() -> None:
    """出力は原寸・0/255 の uint8 であること。"""
    gray = _draw_borderless_rectangle(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(gray, params)

    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_all_clues_disabled_produces_empty_mask() -> None:
    """3 つの手がかりをすべて無効にすると、何も検出されない。"""
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(
        gray, params, use_text_rects=False, use_white_fill=False, use_solidity=False
    )

    assert np.count_nonzero(mask) == 0


def test_rejects_non_2d_input() -> None:
    """グレースケール以外（例: 3 チャンネル画像）を渡すとエラーになる。"""
    color = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)
    params = BalloonParams()

    with pytest.raises(ValueError):
        detect_balloons(color, params)


def test_page_background_is_excluded_from_loss_mask() -> None:
    """ページ背景（画像外周に接する白い連結成分）は損失マスクに含まれない（#15 回帰）。

    コマ枠の外側を取り巻く白いページ背景は、外側輪郭がページ矩形そのものになるため
    solidity が高く誤判定されやすく、外接矩形も画像全体になるため白充填率も高くなりやすい。
    ページ端に接する成分を候補から除外する修正により、ページ背景自体は損失マスクに
    含まれないことを確認する。
    """
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    mask = detect_balloons(gray, params)

    # ページ四隅（コマ枠の外側 = ページ背景）は損失マスクに含まれない
    assert mask[0, 0] == 0
    assert mask[0, _SIZE - 1] == 0
    assert mask[_SIZE - 1, 0] == 0
    assert mask[_SIZE - 1, _SIZE - 1] == 0


def test_realistic_page_loss_mask_does_not_cover_whole_page() -> None:
    """白背景・コマ枠2つ・フキダシ1つ・ベタ1つを含む現実的なページで、損失マスクが
    ページ全面を覆わないこと（#15 回帰）。

    レビュー時点の実装では、ページ背景（コマ枠の外側を取り巻く白領域）の外側輪郭が
    ページ矩形そのものになるため solidity・白充填率のいずれでも 1.0 と誤判定され、
    損失マスクがページ全面（被覆率 1.0000）を覆っていた。
    """
    height, width = 400, 300
    page = np.full((height, width), _PAGE_BACKGROUND, dtype=np.uint8)

    # コマ枠 1（左上）: 枠線 + 内側の作画領域
    cv2.rectangle(page, (20, 20), (140, 180), 0, 3)
    cv2.rectangle(page, (23, 23), (137, 177), _PANEL_FILL, -1)

    # コマ枠 2（右下）: 枠線 + 内側の作画領域
    cv2.rectangle(page, (160, 200), (280, 380), 0, 3)
    cv2.rectangle(page, (163, 203), (277, 377), _PANEL_FILL, -1)

    # フキダシ: コマ枠 1 の内側に白楕円（フチあり）
    cv2.ellipse(page, (80, 90), _ELLIPSE_AXES, 0, 0, 360, 255, -1)
    cv2.ellipse(page, (80, 90), _ELLIPSE_AXES, 0, 0, 360, 0, 3)

    # ベタ: コマ枠 2 の内側に黒塗り領域
    cv2.rectangle(page, (180, 300), (260, 360), 0, -1)

    params = BalloonParams()
    mask = detect_balloons(page, params)

    coverage = float(np.count_nonzero(mask)) / mask.size
    assert coverage < 0.3, f"loss mask covers {coverage:.4f} of page (expected < 0.3)"


# --- #28 回帰: コマ枠を跨いではみ出したフキダシ ---
#
# `_touches_border` は外接矩形が画像外周に接する白連結成分を一律除外するが、フキダシが
# コマ枠を跨いで余白（gutter）へはみ出すと、跨いだ箇所の枠線インクごとフキダシに塗り
# 潰され、コマ内側と余白が地続きの 1 連結成分になるため、丸ごと除外されてしまっていた
# （レビュー実測: コマ内部配置は被覆率 0.0454、コマ枠を跨ぐ配置は被覆率 0.0000）。

_CROSS_SIZE = (400, 300)  # (height, width)。レビュー実測と同じページサイズ
_CROSS_PANEL_RECT = (20, 20, 280, 380)  # (x0, y0, x1, y1)。レビュー実測と同じコマ枠
_CROSS_ELLIPSE_AXES = (40, 25)
_CROSS_INTERIOR_CENTER = (150, 150)  # コマ内部（比較対象のベースライン）
_CROSS_BORDER_CROSSING_CENTER = (30, 150)  # コマ枠を跨ぐ配置。レビュー実測と同じ


def _cross_page(center: tuple[int, int], *, balloon_has_border: bool) -> np.ndarray:
    """レビュー実測と同じ、実際に枠線を描いたコマを 1 つ持つページを作る。

    他のテストが使う `_blank_canvas`（塗りつぶしのみで枠線を描かない）では、コマ枠線
    そのもの（`cv2.rectangle(..., 0, 3)`）が無いためこの回帰を再現できない。分離線
    バリア（`_panel_border_barrier`）の検出対象は実際に描かれた枠線インクである。
    """
    height, width = _CROSS_SIZE
    page = np.full((height, width), _PAGE_BACKGROUND, dtype=np.uint8)
    x0, y0, x1, y1 = _CROSS_PANEL_RECT
    cv2.rectangle(page, (x0, y0), (x1, y1), 0, 3)
    cv2.rectangle(page, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), _PANEL_FILL, -1)
    if balloon_has_border:
        cv2.ellipse(page, center, _CROSS_ELLIPSE_AXES, 0, 0, 360, 255, -1)
        cv2.ellipse(page, center, _CROSS_ELLIPSE_AXES, 0, 0, 360, 0, 3)
    else:
        cv2.ellipse(page, center, _CROSS_ELLIPSE_AXES, 0, 0, 360, 255, -1)
    return page


@pytest.mark.parametrize("balloon_has_border", [True, False], ids=["with-border", "borderless"])
def test_balloon_crossing_panel_border_is_not_dropped(balloon_has_border: bool) -> None:
    """コマ枠を跨いではみ出したフキダシが損失マスクから丸ごと脱落しない（#28 回帰）。

    レビュー実測（フチあり）: コマ内部配置は被覆率 0.0454（中心が覆われる）に対し、
    コマ枠を跨ぐ配置は被覆率 0.0000（中心が覆われない）だった。これはコマ枠を跨いだ
    箇所の枠線インクがフキダシに塗り潰され、コマ内側と余白が地続きの 1 連結成分に
    なり `_touches_border` による背景除外へ丸ごと巻き込まれるため。フチなしでも
    同じ現象が起きる（レビューコメント参照）。
    """
    params = BalloonParams()

    interior_page = _cross_page(_CROSS_INTERIOR_CENTER, balloon_has_border=balloon_has_border)
    crossing_page = _cross_page(_CROSS_BORDER_CROSSING_CENTER, balloon_has_border=balloon_has_border)

    interior_mask = detect_balloons(interior_page, params)
    crossing_mask = detect_balloons(crossing_page, params)

    # ベースライン: コマ内部配置は修正前後を通じて検出される（回帰していないことの確認）。
    assert interior_mask[_CROSS_INTERIOR_CENTER[1], _CROSS_INTERIOR_CENTER[0]] == 255

    # 修正前はここが 0（被覆率 0.0000）だった。
    assert crossing_mask[_CROSS_BORDER_CROSSING_CENTER[1], _CROSS_BORDER_CROSSING_CENTER[0]] == 255
    crossing_coverage = float(np.count_nonzero(crossing_mask)) / crossing_mask.size
    assert crossing_coverage > 0.0, f"crossing coverage is {crossing_coverage:.4f} (expected > 0)"


def test_panel_border_barrier_does_not_reintroduce_whole_page_coverage() -> None:
    """コマ枠バリア導入後も、ページ全面が損失マスクで覆われることはない（#15 の非退行）。

    `test_realistic_page_loss_mask_does_not_cover_whole_page` と同じ被覆率
    アサーションを、コマ枠を跨ぐフキダシを含むページに対しても確認する。
    """
    page = _cross_page(_CROSS_BORDER_CROSSING_CENTER, balloon_has_border=True)
    params = BalloonParams()

    mask = detect_balloons(page, params)

    coverage = float(np.count_nonzero(mask)) / mask.size
    assert coverage < 0.3, f"loss mask covers {coverage:.4f} of page (expected < 0.3)"


def test_white_artwork_is_dropped_when_text_rects_are_available() -> None:
    """テキスト矩形が使えるとき、セリフと重ならない白領域を候補から外すこと。

    実原稿の PSD で、モニタ画面・白い床・明るい背景といった**絵の中の白い領域**が
    フキダシと誤検出され、損失マスクがページの 27.6% を覆い、全インクの 14.5% が
    学習・評価から外れていた。フキダシにはほぼ必ずセリフが入るので、セリフと
    重ならない白領域はフキダシではないとみなせる。
    """
    shape = (400, 600)
    gray = np.full(shape, 0, dtype=np.uint8)  # 黒地（コマの中身）
    # 左: セリフ入りのフキダシ（白）
    cv2.ellipse(gray, (150, 200), (90, 60), 0, 0, 360, _PAGE_BACKGROUND, -1)
    # 右: セリフの無い白い領域（モニタ画面のつもり）
    cv2.rectangle(gray, (380, 140), (540, 260), _PAGE_BACKGROUND, -1)

    params = AnalysisParams().balloon
    # フキダシの中にだけテキスト矩形がある。
    text_rects = [(120, 180, 60, 40)]

    with_gate = detect_balloons(gray, params, text_rects) > 0
    without_gate = (
        detect_balloons(
            gray, params.model_copy(update={"require_text_overlap": False}), text_rects
        )
        > 0
    )

    # フキダシ側はどちらでも覆われる。
    assert with_gate[200, 150], "セリフ入りのフキダシは覆われるべき"
    # セリフの無い白領域は、ゲートありでは覆われない。
    assert not with_gate[200, 460], "セリフと重ならない白領域が覆われている"
    assert without_gate[200, 460], "ゲート無効時は従来どおり覆う前提が崩れている"
    assert with_gate.mean() < without_gate.mean()


def test_gate_is_not_applied_without_text_rects() -> None:
    """テキスト矩形が無い入力（画像 / PDF、セリフをラスタライズした原稿）では
    絞り込みを行わず、従来どおり白領域の形状だけで判定すること。

    ここで空集合として扱うと、フキダシが全て脱落してマスクが空になってしまう。
    """
    shape = (400, 600)
    gray = np.full(shape, 0, dtype=np.uint8)
    cv2.ellipse(gray, (150, 200), (90, 60), 0, 0, 360, _PAGE_BACKGROUND, -1)

    params = AnalysisParams().balloon

    for text_rects in (None, []):
        mask = detect_balloons(gray, params, text_rects) > 0
        assert mask[200, 150], f"text_rects={text_rects!r} でフキダシが検出されない"
