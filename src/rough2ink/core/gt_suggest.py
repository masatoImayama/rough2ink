"""PSD レイヤー名から GT の役割を推定する（半自動 GT 割当）。

設計書 4章 A は「**レイヤー命名規則は一貫していない前提**」としており、実際に手元の
実原稿（86レイヤー）でも同一ページ内に `ベタ塗り` と `べた塗り` が混在していた。
そのため名前による自動判定を**正解の決定に使ってはならない**。ここが返すのは
あくまで UI の初期値であり、最終的な割当は人間が確認・修正する（設計書 5節）。

推定できたかどうかを呼び出し側が区別できるよう、当たったキーワードも併せて返す。
"""

from __future__ import annotations

from pydantic import BaseModel

from rough2ink.core.types import LayerInfo, LayerRole

# 役割ごとのキーワード。表記ゆれ（カタカナ/ひらがな）を明示的に並べる。
# 上から順に評価し、最初に当たったものを採用する（`線画` は `画` を含む他語より先に置く）。
_ROLE_KEYWORDS: list[tuple[LayerRole, tuple[str, ...]]] = [
    # 下書き・アタリは学習にも評価にも使わないので最優先で ignore に落とす
    # （`下書き線` のように `線` を含む名前があるため、line より先に判定する）。
    ("ignore", ("下書き", "したがき", "ラフ", "アタリ", "あたり", "ネーム", "ガイド", "テンプレート")),
    ("fill", ("ベタ", "べた", "黒塗", "塗りつぶし")),
    ("tone", ("トーン", "とーん", "網", "スクリーン", "グラデ", "ハーフトーン")),
    # 効果線・集中線は線として扱う（設計書 3章: 効果線はクリスタのツールで足りるため
    # 生成対象ではないが、GT 上は線画の一部である）。
    ("line", ("線画", "主線", "ペン", "ペン入れ", "効果線", "集中線", "フラッシュ", "枠線", "コマ枠")),
    ("text", ("フキダシ", "ふきだし", "吹き出し", "セリフ", "せりふ", "写植", "文字")),
]


class RoleSuggestion(BaseModel):
    """レイヤー1枚に対する役割の推定結果。"""

    layer_id: str
    path: str
    role: LayerRole | None = None
    # 推定の根拠。`None` は推定できなかったことを表す（UI で手動割当を促す）。
    matched_keyword: str | None = None
    # `kind == "type"` から機械的に決まったものか（名前によらず確実）。
    from_layer_kind: bool = False


def suggest_roles(layers: list[LayerInfo]) -> list[RoleSuggestion]:
    """レイヤー一覧から役割の初期値を推定する。

    テキストレイヤー（`kind == "type"`）は名前に依らず `text` にする。PSD の
    レイヤー種別から機械的に決まるため、名前の揺れの影響を受けない。

    グループ（`kind == "group"`）は画素を持たないため推定対象から外す。

    推定できなかったレイヤーは `role=None` で返す。**呼び出し側はこれを
    「役割なし」ではなく「未判定」として扱い、人間に割り当てさせること。**
    """
    suggestions: list[RoleSuggestion] = []
    for layer in layers:
        if layer.kind == "group":
            continue

        if layer.kind == "type":
            suggestions.append(
                RoleSuggestion(
                    layer_id=layer.id, path=layer.path, role="text", from_layer_kind=True
                )
            )
            continue

        name = layer.path.split("/")[-1]
        matched: tuple[LayerRole, str] | None = None
        for role, keywords in _ROLE_KEYWORDS:
            for keyword in keywords:
                if keyword in name:
                    matched = (role, keyword)
                    break
            if matched is not None:
                break

        suggestions.append(
            RoleSuggestion(
                layer_id=layer.id,
                path=layer.path,
                role=matched[0] if matched else None,
                matched_keyword=matched[1] if matched else None,
            )
        )
    return suggestions


def suggested_mapping(layers: list[LayerInfo]) -> dict[str, LayerRole]:
    """`suggest_roles` の結果のうち、推定できたものだけをマッピング形式で返す。"""
    return {
        suggestion.layer_id: suggestion.role
        for suggestion in suggest_roles(layers)
        if suggestion.role is not None
    }
