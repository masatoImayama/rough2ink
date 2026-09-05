"""`AnalysisParams` の既定値と JSON 往復のテスト（Epic 仕様書 9 節）。"""

from __future__ import annotations

from rough2ink.core.params import AnalysisParams

EXPECTED_DEFAULTS: dict[str, dict[str, float | int]] = {
    "quality": {
        "min_short_side": 2000,
        "jpeg_block_threshold": 0.15,
        "binary_ratio_threshold": 0.98,
    },
    "tone": {
        "window": 64,
        "stride": 32,
        "bandpass_low": 0.08,
        "bandpass_high": 0.45,
        "energy_threshold": 0.20,
        "sharpness_threshold": 1500.0,
        # 尖鋭度だけでは直線エッジを網点と区別できないため追加した2条件。
        "min_block_std": 6.0,
        "min_direction_ratio": 0.20,
    },
    "fill": {
        "black_threshold": 64,
        "min_area_ratio": 0.0008,
        # 実原稿での実測に基づき 2 -> 8 に変更（ベタが線を巻き込む問題の修正）。
        # 幅 2*8=16px 以下のストロークは線として残る（600dpi・B4 で約 1.3mm）。
        "erosion_radius": 8,
    },
    "line": {
        "black_threshold": 128,
    },
    "panel": {
        "close_kernel": 9,
        "min_panel_area_ratio": 0.01,
        "approx_epsilon_ratio": 0.01,
        "virtual_frame_margin": 8,
        "oblique_angle_deg": 5.0,
        "spread_aspect_ratio": 1.2,
        "effect_line_density": 0.15,
        # 絵を枠線と誤認して内側へ食い込んだポリゴンの検出（凸性の下限）。
        "min_solidity": 0.8,
    },
    "balloon": {
        "min_area_ratio": 0.002,
        "max_area_ratio": 0.25,
        "white_threshold": 235,
        "white_fill_ratio": 0.90,
        "min_solidity": 0.90,
        "dilate_radius": 12,
        "text_rect_dilate": 24,
        "require_text_overlap": True,
    },
    "preview": {
        "max_long_side": 1600,
    },
}


def test_defaults_match_epic_spec() -> None:
    """`AnalysisParams()` が Epic 仕様書 9 節の既定値表と一致すること。"""
    params = AnalysisParams()
    dumped = params.model_dump()

    for group_name, expected_fields in EXPECTED_DEFAULTS.items():
        assert group_name in dumped, f"missing group: {group_name}"
        for field_name, expected_value in expected_fields.items():
            actual_value = dumped[group_name][field_name]
            assert actual_value == expected_value, (
                f"{group_name}.{field_name}: expected {expected_value}, got {actual_value}"
            )

    # 未知のグループ・フィールドが混入していないことも確認する
    assert set(dumped.keys()) == set(EXPECTED_DEFAULTS.keys())
    for group_name, expected_fields in EXPECTED_DEFAULTS.items():
        assert set(dumped[group_name].keys()) == set(expected_fields.keys())


def test_json_roundtrip() -> None:
    """JSON シリアライズ/デシリアライズで往復できること。"""
    params = AnalysisParams()
    json_str = params.model_dump_json()

    restored = AnalysisParams.model_validate_json(json_str)

    assert restored == params
    assert restored.model_dump() == params.model_dump()


def test_json_roundtrip_with_overrides() -> None:
    """一部のパラメータを変更した状態でも往復できること（UI のスライダー調整を想定）。"""
    params = AnalysisParams()
    params.tone.energy_threshold = 0.35
    params.panel.oblique_angle_deg = 10.0

    restored = AnalysisParams.model_validate_json(params.model_dump_json())

    assert restored.tone.energy_threshold == 0.35
    assert restored.panel.oblique_angle_deg == 10.0
    # 変更していないフィールドは既定値のまま
    assert restored.fill.black_threshold == 64
