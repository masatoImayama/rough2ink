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

    ただし尖鋭度だけでは**まっすぐで鋭いエッジ**を網点と区別できない。直線エッジは
    2次元周波数空間で原点を通る1本の直線上にエネルギーが集中するため、帯域内の大半の
    ビンがほぼ 0 になり、最大/中央値が網点と同程度に跳ね上がるためである。そこで
    次の2つを追加で要求する。

    - `min_block_std`: ブロック内の画素値の標準偏差の下限。一様な面（ベタの内側・白紙）は
      スペクトルが退化して判定が無意味になるため、ここで落とす
    - `min_direction_ratio`: エネルギーが**2つ以上の独立した方向**に存在することを要求する。
      網点は格子（2次元周期構造）なので基本ベクトル2本ぶんの方向にピークを持つ一方、
      直線エッジ・平行線は1方向しか持たない
    """

    window: int = 64
    stride: int = 32
    bandpass_low: float = 0.08
    bandpass_high: float = 0.45
    energy_threshold: float = 0.20
    sharpness_threshold: float = 1500.0
    min_block_std: float = 6.0
    min_direction_ratio: float = 0.20


class FillParams(BaseModel):
    """B. 分解器: ベタ検出用パラメータ。

    `erosion_radius` は「ベタと認めるために内側に収まらなければならない円の半径(px)」。
    幅 `2*erosion_radius` 以下のストロークはベタにならず線として残るため、**原稿の解像度と
    ペン入れの線幅に合わせて調整する**。既定の 8px は 600dpi・B4（短辺3000px級、約12px/mm）で
    線幅 1.3mm までを線として残す値。低解像度の原稿では小さくする。
    """

    black_threshold: int = 64
    min_area_ratio: float = 0.0008
    erosion_radius: int = 8


class LineParams(BaseModel):
    """B. 分解器: 線検出用パラメータ。"""

    black_threshold: int = 128


class PanelParams(BaseModel):
    """C. コマ分割用パラメータ。

    `min_solidity` は「絵の一部を枠線と誤認して内側へ食い込んだポリゴン」を検出する
    ための下限（面積 ÷ 凸包面積）。コマ枠は基本的に凸なので、正常なコマはほぼ 1.0 になる。
    実測では、破綻したポリゴン（18頂点・優角8個）が 0.483、正常な矩形が 1.000 だった。
    """

    close_kernel: int = 9
    min_panel_area_ratio: float = 0.01
    approx_epsilon_ratio: float = 0.01
    virtual_frame_margin: int = 8
    oblique_angle_deg: float = 5.0
    spread_aspect_ratio: float = 1.2
    effect_line_density: float = 0.15
    min_solidity: float = 0.8


class BalloonParams(BaseModel):
    """E. フキダシ検出 → 損失マスク用パラメータ。"""

    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.25
    white_threshold: int = 235
    white_fill_ratio: float = 0.90
    min_solidity: float = 0.90
    dilate_radius: int = 12
    text_rect_dilate: int = 24
    # テキスト矩形が使えるとき、白領域由来の候補をセリフと重なるものに絞るか。
    # 絵の中の白い領域（モニタ画面・白い床・明るい背景）の誤検出を落とす。
    # PSD 入力でのみ効く（画像 / PDF はテキストレイヤーを持たない）。
    require_text_overlap: bool = True


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
