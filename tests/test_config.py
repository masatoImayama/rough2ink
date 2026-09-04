"""`workspace/` `out/` のパス解決のテスト。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rough2ink.core import config


def test_get_workspace_dir_defaults_under_project_root() -> None:
    workspace = config.get_workspace_dir(create=False)
    assert workspace == config.get_project_root() / "workspace"


def test_get_out_dir_defaults_under_project_root() -> None:
    out_dir = config.get_out_dir(create=False)
    assert out_dir == config.get_project_root() / "out"


def test_workspace_dir_env_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "custom-workspace"
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(override))

    resolved = config.get_workspace_dir(create=True)

    assert resolved == override.resolve()
    assert resolved.is_dir()


def test_presets_dir_is_created_under_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    presets_dir = config.get_presets_dir(create=True)

    assert presets_dir == (tmp_path / "ws" / "presets").resolve()
    assert presets_dir.is_dir()


# --- resolve_page_dir: パストラバーサル対策（Review #21） ------------------------


def test_resolve_page_dir_returns_pages_subdir_for_valid_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    page_dir = config.resolve_page_dir("abc123_def-01.page")

    assert page_dir == (tmp_path / "ws" / "pages" / "abc123_def-01.page").resolve()


def test_resolve_page_dir_rejects_backslash_traversal(tmp_path: Path, monkeypatch) -> None:
    """Review #21 の実測ケース: uvicorn は URL デコード後にルーティングするため、
    バックスラッシュ入りの page_id が単一パスセグメントとして届きうる。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(ValueError):
        config.resolve_page_dir("..\\..\\..\\Users\\victim\\preview")


def test_resolve_page_dir_rejects_forward_slash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(ValueError):
        config.resolve_page_dir("../../etc/passwd")


def test_resolve_page_dir_rejects_dot_dot_only(tmp_path: Path, monkeypatch) -> None:
    """`..` 単体はホワイトリスト（`[A-Za-z0-9_.-]+`）には一致してしまうため、
    `Path.is_relative_to` 側のチェックで弾けることも確認する。"""
    monkeypatch.setenv("ROUGH2INK_WORKSPACE_DIR", str(tmp_path / "ws"))

    with pytest.raises(ValueError):
        config.resolve_page_dir("..")
