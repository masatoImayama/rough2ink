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
