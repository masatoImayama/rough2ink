"""`workspace/` `out/` のパス解決のテスト。"""

from __future__ import annotations

import os
from pathlib import Path

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
