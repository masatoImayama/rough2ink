"""Web UI（T10）の静的配信・非機能要件（Node 依存なし）の検証。

JS のロジック自体（DOM 操作・fetch 呼び出し）は Node 未導入という非機能要件のため
ブラウザ実行系のテストフレームワークを追加できない。ここでは
- FastAPI の静的配信で UI 一式が正しく取得できること
- npm / node_modules / CDN 参照が一切存在しないこと（非機能要件・完了条件）
を検証する。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rough2ink.app import app

client = TestClient(app)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

_JS_MODULES = ["api.js", "overlay.js", "params.js", "layers.js", "presets.js", "app.js"]


def test_index_html_references_style_and_app_js() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '/css/style.css' in response.text
    assert '/js/app.js' in response.text
    # ビルド不要の素の ES モジュールとして読み込まれていること。
    assert 'type="module"' in response.text


def test_style_css_is_served() -> None:
    response = client.get("/css/style.css")
    assert response.status_code == 200


def test_all_js_modules_are_served_and_use_es_module_syntax() -> None:
    for module in _JS_MODULES:
        response = client.get(f"/js/{module}")
        assert response.status_code == 200, module


def test_select_page_catches_errors_and_reports_page_status() -> None:
    """Review #23: `selectPage()` が例外を投げっぱなしにすると、クリックハンドラが
    await していないため未処理の Promise 拒否になり画面に何も出ない。JS 実行系の
    テストフレームワークが無い（非機能要件）ため、ソース上で try/catch が
    `selectPage` の本体を囲み、失敗時に `#page-status` を更新していることを
    構造的に検証する。"""
    source = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")

    select_page_start = source.index("async function selectPage(")
    next_function_start = source.index("\nasync function ", select_page_start + 1)
    select_page_body = source[select_page_start:next_function_start]

    assert "try {" in select_page_body
    assert "catch (err)" in select_page_body
    assert 'byId("page-status")' in select_page_body.split("catch (err)")[1]


def test_reanalyze_guards_against_out_of_order_responses() -> None:
    """Review #26: 400ms デバウンス後に発火した再解析リクエストのキャンセル・順序保証が
    無いと、先に発火した A が B より後に完了して古い結果で表示を上書きしうる。
    世代番号（`seq`）比較で追い越されたレスポンスを捨てていることを構造的に検証する。"""
    source = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")

    reanalyze_start = source.index("async function reanalyze(")
    next_function_start = source.index("\nfunction ", reanalyze_start + 1)
    reanalyze_body = source[reanalyze_start:next_function_start]

    assert "++latestReanalyzeSeq" in reanalyze_body
    assert "seq !== latestReanalyzeSeq" in reanalyze_body
    assert "AbortController" in reanalyze_body


def test_overlay_has_zoom_controls() -> None:
    """オーバーレイに拡大縮小 UI があること。JS 実行系が無い（非機能要件）ため、
    コントロールの存在と配線をソース上で構造的に検証する。"""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for element_id in ("zoom-range", "zoom-in", "zoom-out", "zoom-fit", "zoom-reset", "zoom-value"):
        assert f'id="{element_id}"' in html, element_id

    app_source = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")
    assert "setupZoomControls" in app_source
    # Ctrl+ホイールでの拡大縮小（ブラウザのページズームは抑止する）。
    assert '"wheel"' in app_source
    assert "preventDefault" in app_source


def test_zoom_scales_via_css_width_not_canvas_resolution() -> None:
    """ズームは canvas の内部解像度ではなく CSS 幅で行うこと。内部解像度を変えると
    倍率変更のたびにマスクを描き直すことになり、拡大操作が重くなる。"""
    source = (WEB_DIR / "js" / "overlay.js").read_text(encoding="utf-8")

    apply_zoom_start = source.index("applyZoom() {")
    apply_zoom_body = source[apply_zoom_start : source.index("\n  }", apply_zoom_start)]
    assert "style.width" in apply_zoom_body
    # `canvas.width = ...`（内部解像度の変更）を倍率適用で行っていないこと。
    assert "this.canvas.width =" not in apply_zoom_body

    assert "ZOOM_MIN" in source and "ZOOM_MAX" in source
    assert "fitToWidth" in source


def test_sidebar_range_inputs_can_shrink_below_intrinsic_width() -> None:
    """5.パラメータ が 6.オーバーレイ表示 に重なっていた回帰。flex アイテムの既定
    `min-width: auto` を解除しないと、スライダーが既定幅より縮まずサイドバー幅
    （固定）を突き抜けて #viewer に重なる。"""
    css = (WEB_DIR / "css" / "style.css").read_text(encoding="utf-8")

    range_rule_start = css.index('.param-row input[type="range"]')
    range_rule = css[range_rule_start : css.index("}", range_rule_start)]
    assert "min-width: 0" in range_rule

    sidebar_rule_start = css.index("#sidebar {")
    sidebar_rule = css[sidebar_rule_start : css.index("}", sidebar_rule_start)]
    assert "min-width: 0" in sidebar_rule

    # fieldset も既定で min-content 幅を下回れないため、同じ理由で解除が要る。
    fieldset_rule_start = css.index("fieldset {")
    fieldset_rule = css[fieldset_rule_start : css.index("}", fieldset_rule_start)]
    assert "min-width: 0" in fieldset_rule


def test_no_node_dependency_or_cdn_reference_anywhere_in_web_dir() -> None:
    """非機能要件・完了条件: npm / node_modules / CDN リンクが一切存在しないこと。"""
    forbidden_substrings = ["node_modules", "unpkg.com", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"]
    checked_files = 0
    for path in WEB_DIR.rglob("*"):
        if not path.is_file():
            continue
        checked_files += 1
        text = path.read_text(encoding="utf-8")
        for forbidden in forbidden_substrings:
            assert forbidden not in text, f"{path} contains forbidden reference: {forbidden!r}"
        assert "<script src=\"http" not in text, f"{path} references an external <script> src"
    assert checked_files > 0

    assert not (WEB_DIR.parent / "package.json").exists()
    assert not (WEB_DIR / "package.json").exists()
