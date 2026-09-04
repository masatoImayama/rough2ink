"""解析パラメータの単一スキーマ。

UI のスライダー・プリセット保存・バッチ実行の 3 経路が同じ設定を共有できるよう、
全パラメータをこの `AnalysisParams` 一箇所に集約する（Epic 仕様書 9 節）。
既定値は Epic 仕様書 9 節の表をそのまま実装したもの。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QualityParams(BaseModel):
    """D. 取り込み品質ゲート用パラメータ。"""

    min_short_side: int = 2000
    jpeg_block_threshold: float = 0.15
    binary_ratio_threshold: float = 0.98


class ToneParams(BaseModel):
    """B. 分解器: トーン検出（周波数解析）用パラメータ。

    `energy_threshold` は帯域に有意なエネルギーがあるかを見る前段フィルタ、
    `sharpness_threshold` は帯域内スペクトルの尖鋭度（周期性）を見る主判定で、
    網点のような孤立ピークを持つブロックだけをトーンとして採用する（#16）。
    """

    window: int = 64
    stride: int = 32
    bandpass_low: float = 0.08
    bandpass_high: float = 0.45
    energy_threshold: float = 0.20
    sharpness_threshold: float = 1500.0


class FillParams(BaseModel):
    """B. 分解器: ベタ検出用パラメータ。"""

    black_threshold: int = 64
    min_area_ratio: float = 0.0008
    erosion_radius: int = 2


class LineParams(BaseModel):
    """B. 分解器: 線検出用パラメータ。"""

    black_threshold: int = 128


class PanelParams(BaseModel):
    """C. コマ分割用パラメータ。"""

    close_kernel: int = 9
    min_panel_area_ratio: float = 0.01
    approx_epsilon_ratio: float = 0.01
    virtual_frame_margin: int = 8
    oblique_angle_deg: float = 5.0
    spread_aspect_ratio: float = 1.2
    effect_line_density: float = 0.15


class BalloonParams(BaseModel):
    """E. フキダシ検出 → 損失マスク用パラメータ。"""

    min_area_ratio: float = 0.002
    white_threshold: int = 235
    white_fill_ratio: float = 0.90
    min_solidity: float = 0.90
    dilate_radius: int = 12
    text_rect_dilate: int = 24


class PreviewParams(BaseModel):
    """ブラウザ表示用プレビュー画像のパラメータ。"""

    max_long_side: int = 1600


class AnalysisParams(BaseModel):
    """全パラメータを集約した単一スキーマ。

    UI のスライダー・プリセット保存 (`workspace/presets/<name>.json`)・
    バッチ実行のいずれもこのモデルの JSON シリアライズ/デシリアライズを介して
    パラメータをやり取りする。
    """

    quality: QualityParams = Field(default_factory=QualityParams)
    tone: ToneParams = Field(default_factory=ToneParams)
    fill: FillParams = Field(default_factory=FillParams)
    line: LineParams = Field(default_factory=LineParams)
    panel: PanelParams = Field(default_factory=PanelParams)
    balloon: BalloonParams = Field(default_factory=BalloonParams)
    preview: PreviewParams = Field(default_factory=PreviewParams)
