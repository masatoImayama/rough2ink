"""レイヤー名からの役割推定（半自動 GT 割当）のテスト。

設計書 4章 A の「命名規則は一貫していない前提」に対し、推定は**初期値**であり
正解ではない、という契約を固定する。
"""

from __future__ import annotations

from rough2ink.core.gt_suggest import suggest_roles, suggested_mapping
from rough2ink.core.types import LayerInfo


def _layer(layer_id: str, path: str, kind: str = "pixel") -> LayerInfo:
    return LayerInfo(id=layer_id, name=path.split("/")[-1], path=path, kind=kind)


def test_suggests_roles_from_common_names() -> None:
    layers = [
        _layer("idx0", "本文/線画"),
        _layer("idx1", "本文/ベタ塗り 1"),
        _layer("idx2", "本文/トーン 60線"),
    ]

    by_id = {item.layer_id: item for item in suggest_roles(layers)}

    assert by_id["idx0"].role == "line"
    assert by_id["idx1"].role == "fill"
    assert by_id["idx2"].role == "tone"


def test_hiragana_and_katakana_variants_are_both_matched() -> None:
    """実原稿では同一ページ内に `ベタ塗り` と `べた塗り` が混在していた。"""
    layers = [_layer("idx0", "べた塗り 3"), _layer("idx1", "ベタ塗り 4")]

    roles = {item.layer_id: item.role for item in suggest_roles(layers)}

    assert roles["idx0"] == "fill"
    assert roles["idx1"] == "fill"


def test_text_layers_are_assigned_by_layer_kind_not_by_name() -> None:
    """テキストレイヤーは PSD の種別から機械的に決まるので名前の揺れの影響を受けない。

    実原稿ではテキストレイヤーの名前がセリフ本文そのものになっていた。
    """
    layers = [_layer("idx0", "本文/そんなん どこで手に 入れたんや？", kind="type")]

    suggestion = suggest_roles(layers)[0]

    assert suggestion.role == "text"
    assert suggestion.from_layer_kind is True


def test_draft_layers_are_ignored_before_line_matching() -> None:
    """`下書き線` のように line のキーワードを含む下書きレイヤーが line にならないこと。"""
    layers = [_layer("idx0", "レイヤーテンプレート/下書き/下書き線"), _layer("idx1", "アタリ")]

    roles = {item.layer_id: item.role for item in suggest_roles(layers)}

    assert roles["idx0"] == "ignore"
    assert roles["idx1"] == "ignore"


def test_unknown_names_are_left_unassigned() -> None:
    """判定できない名前は `None`。呼び出し側は「役割なし」ではなく「未判定」として扱う。"""
    layers = [_layer("idx0", "レイヤー 5"), _layer("idx1", "SHIELD")]

    suggestions = suggest_roles(layers)

    assert all(item.role is None for item in suggestions)
    assert all(item.matched_keyword is None for item in suggestions)
    # 未判定はマッピングに含めない（誤った初期値を保存させない）。
    assert suggested_mapping(layers) == {}


def test_groups_are_excluded() -> None:
    layers = [_layer("idx0", "本文", kind="group"), _layer("idx1", "本文/線画")]

    suggestions = suggest_roles(layers)

    assert [item.layer_id for item in suggestions] == ["idx1"]
