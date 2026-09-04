"""パラメータプリセット（`core.presets` / `api.routes_presets`）のテスト。

`workspace/` はテストごとに `ROUGH2INK_WORKSPACE_DIR` で隔離する
（`tests/test_routes_ingest.py` と同じパターン）。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from rough2ink.app import app
from rough2ink.core import presets
from rough2ink.core.params import AnalysisParams


def test_sanitize_preset_name_replaces_illegal_windows_chars() -> None:
    assert presets.sanitize_preset_name('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_preset_name_replaces_control_chars() -> None:
    assert presets.sanitize_preset_name("a\x00b\x1fc") == "a_b_c"


def test_sanitize_preset_name_strips_trailing_dots_and_spaces() -> None:
    # Windows はファイル名末尾のドット・空白を暗黙に無視するため、事前に取り除く。
    assert presets.sanitize_preset_name("  my preset.  ") == "my preset"


def test_sanitize_preset_name_avoids_reserved_device_names() -> None:
    assert presets.sanitize_preset_name("CON") == "CON_"
    assert presets.sanitize_preset_name("nul") == "nul_"
    assert presets.sanitize_preset_name("COM1") == "COM1_"


def test_sanitize_preset_name_empty_result_falls_back() -> None:
    assert presets.sanitize_preset_name("   ") == "preset"
    assert presets.sanitize_preset_name("...") == "preset"


def test_core_save_load_list_delete_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    params = AnalysisParams()
    params.fill.black_threshold = 42

    saved_name = presets.save_preset("my/preset", params)
    assert saved_name == "my_preset"  # "/" はパストラバーサル対策も兼ねてサニタイズされる
    assert presets.list_presets() == ["my_preset"]

    loaded = presets.load_preset("my/preset")
    assert loaded.fill.black_threshold == 42

    assert presets.delete_preset("my/preset") is True
    assert presets.list_presets() == []
    assert presets.delete_preset("my/preset") is False


def test_core_load_missing_preset_raises_file_not_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(FileNotFoundError):
        presets.load_preset("does-not-exist")


def test_api_preset_put_get_list_apply_delete_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    name = "my preset"
    encoded = quote(name, safe="")

    params = AnalysisParams()
    params.panel.close_kernel = 21

    put_response = client.put(f"/api/presets/{encoded}", json=params.model_dump(mode="json"))
    assert put_response.status_code == 200
    assert put_response.json()["panel"]["close_kernel"] == 21

    list_response = client.get("/api/presets")
    assert list_response.status_code == 200
    assert list_response.json() == [name]

    # 適用（再取得して AnalysisParams として読み込めること）
    get_response = client.get(f"/api/presets/{encoded}")
    assert get_response.status_code == 200
    applied = AnalysisParams.model_validate(get_response.json())
    assert applied.panel.close_kernel == 21

    delete_response = client.delete(f"/api/presets/{encoded}")
    assert delete_response.status_code == 204

    get_after_delete = client.get(f"/api/presets/{encoded}")
    assert get_after_delete.status_code == 404


def test_api_get_unknown_preset_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.get("/api/presets/does-not-exist")

    assert response.status_code == 404


def test_api_delete_unknown_preset_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    response = client.delete("/api/presets/does-not-exist")

    assert response.status_code == 404


def test_api_put_sanitizes_unsafe_windows_filename_chars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))
    client = TestClient(app)

    # "/" は URL のパス区切りと衝突するため単一セグメントの経路では表現できない
    # （その組み合わせは `test_core_save_load_list_delete_roundtrip` が core 層で直接検証する）。
    # ここでは URL 経路上でも表現できる禁止文字（`:` `*` `?`）でサニタイズを確認する。
    unsafe_name = "a:b*c?d"
    put_response = client.put(
        f"/api/presets/{quote(unsafe_name, safe='')}",
        json=AnalysisParams().model_dump(mode="json"),
    )
    assert put_response.status_code == 200

    saved_files = list((tmp_path / "ws" / "presets").glob("*.json"))
    assert len(saved_files) == 1
    assert saved_files[0].name == "a_b_c_d.json"
