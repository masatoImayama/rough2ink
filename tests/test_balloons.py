"""`detect_balloons`（E. フキダシ検出 → 損失マスク生成）のテスト。

実原稿は手元にないため合成フィクスチャで検証する（Epic 実装計画書「テスト方針」）。
白楕円（フチあり）・フチなし白領域・テキスト矩形の 3 パターンで検出されること、
3 つの手がかりが個別に有効/無効を切り替えられ、それぞれの寄与が確認できることを見る。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rough2ink.core.balloons import detect_balloons
from rough2ink.core.params import BalloonParams

_SIZE = 200
_BACKGROUND = 20  # 暗い背景（フキダシの白領域と区別できる値）
_ELLIPSE_CENTER = (100, 100)
_ELLIPSE_AXES = (40, 25)  # (半長軸, 半短軸)


def _blank_canvas() -> np.ndarray:
    return np.full((_SIZE, _SIZE), _BACKGROUND, dtype=np.uint8)


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
    """出力マスクが入力の白領域そのものより広い（最終膨張が効いている）ことを確認する。"""
    gray = _draw_ellipse_with_border(_blank_canvas())
    params = BalloonParams()

    raw_white_count = int(np.count_nonzero(gray > params.white_threshold))
    mask = detect_balloons(gray, params)
    mask_count = int(np.count_nonzero(mask))

    assert mask_count > raw_white_count
    # 白領域そのものが縮んではいけない（膨張のみで縮小はしていない）
    assert np.all(mask[gray > params.white_threshold] == 255)

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
