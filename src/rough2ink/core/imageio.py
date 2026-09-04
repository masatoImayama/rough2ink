"""画像の読み書き・プレビュー生成・マスク PNG 入出力。

日本語ファイル名・空白入りパスを通すため、`cv2.imread` / `cv2.imwrite` は使わない
（Windows では非 ASCII パスで読み書きに失敗することがある）。代わりに
`np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `Path.write_bytes` を経由する。

**原寸解析の原則（Epic 仕様書 9 節・4-A 節）**: ここでの読み込みは常に原寸のまま返す。
縮小するのは `make_preview` の一点のみ。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    """画像ファイルを原寸グレースケール `uint8` ndarray として読み込む。

    日本語・空白入りパスに対応するため `cv2.imread` は使わない。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"failed to read file (empty or missing): {path}")
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def read_bgr(path: Path) -> np.ndarray:
    """画像ファイルを原寸カラー(BGR) `uint8` ndarray として読み込む。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"failed to read file (empty or missing): {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def write_gray_png(path: Path, array: np.ndarray) -> None:
    """グレースケール `uint8` ndarray を PNG として書き出す（原寸のまま）。"""
    _write_png(path, array)


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """0/255 の二値マスクを PNG として書き出す（原寸のまま）。"""
    _write_png(path, mask)


def read_mask_png(path: Path) -> np.ndarray:
    """0/255 の二値マスク PNG を読み込む。"""
    return read_gray(path)


def _write_png(path: Path, array: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"failed to encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def encode_png_bytes(array: np.ndarray) -> bytes:
    """ndarray を PNG バイト列にエンコードする（レスポンス送出用）。"""
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("failed to encode PNG")
    return buf.tobytes()


def make_preview(gray: np.ndarray, max_long_side: int) -> np.ndarray:
    """ブラウザ表示用のプレビューを生成する（長辺 `max_long_side` 以下に縮小）。

    原寸の長辺がすでに `max_long_side` 以下の場合は拡大せずそのまま返す
    （プレビューは表示専用であり、解析用の原寸データを汚さない）。
    """
    height, width = gray.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return gray.copy()

    scale = max_long_side / long_side
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
