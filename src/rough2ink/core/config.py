"""`workspace/` `out/` のパス解決。

Windows ローカル実行を前提とし、パスは常に `pathlib.Path` で扱う
（文字列連結でパスを組まない。日本語・空白入りファイル名を通す）。

既定ではプロジェクトルート直下の `workspace/` `out/` を使うが、
テストや別環境からの起動に対応するため環境変数での上書きを許可する。
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PROJECT_ROOT = "ROUGH2INK_PROJECT_ROOT"
_ENV_WORKSPACE_DIR = "ROUGH2INK_WORKSPACE_DIR"
_ENV_OUT_DIR = "ROUGH2INK_OUT_DIR"


def get_project_root() -> Path:
    """プロジェクトルート（`pyproject.toml` のあるディレクトリ）を返す。"""
    override = os.environ.get(_ENV_PROJECT_ROOT)
    if override:
        return Path(override).resolve()
    # src/rough2ink/core/config.py から見て 3 階層上がプロジェクトルート
    return Path(__file__).resolve().parents[3]


def get_workspace_dir(*, create: bool = True) -> Path:
    """アップロード原稿・GT マッピング・プリセットを置く `workspace/` を返す。"""
    override = os.environ.get(_ENV_WORKSPACE_DIR)
    path = Path(override).resolve() if override else get_project_root() / "workspace"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_out_dir(*, create: bool = True) -> Path:
    """バッチ処理の中間成果物・レポートを書き出す `out/` を返す。"""
    override = os.environ.get(_ENV_OUT_DIR)
    path = Path(override).resolve() if override else get_project_root() / "out"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_presets_dir(*, create: bool = True) -> Path:
    """パラメータプリセットの保存先 `workspace/presets/` を返す。"""
    path = get_workspace_dir(create=create) / "presets"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_gt_dir(*, create: bool = True) -> Path:
    """GT レイヤーマッピングの保存先 `workspace/gt/` を返す。"""
    path = get_workspace_dir(create=create) / "gt"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
