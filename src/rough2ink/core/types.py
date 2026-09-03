"""パイプライン全体で共有する共通型定義。

`PageDocument` が入力ローダ（T2）の出力であり、以降の品質ゲート (D) / 分解器 (B) /
コマ分割 (C) / フキダシ検出 (E) / GT 割当 (5節) がすべてこれを起点に処理する
（Epic 本文「データフロー」参照）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

LayerKind = Literal["pixel", "type", "group", "shape"]
LayerRole = Literal["line", "fill", "tone", "text", "ignore"]
PanelFlag = Literal[
    "cut_off",
    "unclosed",
    "oblique",
    "overflow",
    "spread",
    "effect_lines",
]
ToneDepth = Literal["binary", "gray"]
QualityStatus = Literal["pass", "warn", "fail"]

# 原寸座標系での bbox。(x, y, width, height)
BBox = tuple[int, int, int, int]


class LayerInfo(BaseModel):
    """PSD レイヤー1枚分の情報（設計書 4章 A）。"""

    id: str
    name: str
    path: str  # グループ階層をスラッシュ区切りにした一意なパス
    kind: LayerKind
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    bbox: BBox | None = None
    role: LayerRole | None = None  # GT 割当（5節）で手動設定される


class PanelInfo(BaseModel):
    """コマ 1 枠分の情報（設計書 4章 C）。"""

    panel_id: str
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    bbox: BBox | None = None
    area_ratio: float = 0.0
    flags: list[PanelFlag] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """例外フラグが一つも立っていないか（コマ分割の成功率算出に使う）。"""
        return len(self.flags) == 0


class QualityReport(BaseModel):
    """取り込み品質ゲートの判定結果（設計書 4章 D）。"""

    short_side: int
    tone_depth: ToneDepth
    jpeg_block_score: float
    status: QualityStatus
    reasons: list[str] = Field(default_factory=list)


class PageDocument(BaseModel):
    """1 ページ分の入力を表す共通ドキュメント（入力ローダの出力）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    source_path: Path
    source_kind: Literal["image", "psd", "pdf"]
    width: int
    height: int
    # 原寸グレースケール画像 (H, W) の uint8 配列。解析はすべてこの原寸で行う。
    gray: Any = None  # numpy.ndarray | None（arbitrary_types_allowed で受け付ける）
    layers: list[LayerInfo] = Field(default_factory=list)

    def as_array(self) -> np.ndarray:
        """`gray` を numpy 配列として返す（未設定なら ValueError）。"""
        if self.gray is None:
            raise ValueError(f"page {self.page_id!r} has no loaded grayscale image")
        return self.gray
