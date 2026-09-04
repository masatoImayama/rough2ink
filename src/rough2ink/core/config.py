"""`workspace/` `out/` のパス解決。

Windows ローカル実行を前提とし、パスは常に `pathlib.Path` で扱う
（文字列連結でパスを組まない。日本語・空白入りファイル名を通す）。

既定ではプロジェクトルート直下の `workspace/` `out/` を使うが、
テストや別環境からの起動に対応するため環境変数での上書きを許可する。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_PROJECT_ROOT = "ROUGH2INK_PROJECT_ROOT"
_ENV_WORKSPACE_DIR = "ROUGH2INK_WORKSPACE_DIR"
_ENV_OUT_DIR = "ROUGH2INK_OUT_DIR"

# `page_id` として許可する文字集合（英数字・`_`・`.`・`-` のみ）。
# uvicorn は URL をデコードしてからルーティングするため、パスパラメータであっても
# バックスラッシュ入りの値（`..%5C..%5C..` 等）が単一セグメントとして一致しうる
# （Review #21: Windows 上で `workspace\pages\..\..\..\Users\victim\...` に解決されることを実測）。
# `/` はスラッシュとして FastAPI のルーティングで弾かれるが、`\` はこの正規表現で弾く。
_PAGE_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


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


def resolve_page_dir(page_id: str) -> Path:
    """URL パスパラメータ `page_id` を検証したうえで `workspace/pages/<page_id>/` を返す。

    パストラバーサル対策として二重に検証する（Review #21）:
    - ホワイトリスト（英数字・`_`・`.`・`-` のみ）に一致しない値を拒否する。
      `page_id` はスラッシュを含まない前提の値だが、URL デコード後にバックスラッシュが
      入っていると Windows 上ではパス区切りとして解釈されてしまうため、この時点で弾く。
    - 念のため、解決後のパスが `workspace/pages/` の配下であることも `Path.is_relative_to`
      で確認する。

    不正な `page_id` には `ValueError` を送出する。呼び出し側（各 API ルート）はこれを
    捕捉して 404 に変換すること（存在しないページと区別しない ── 攻撃者に「弾かれた」
    という情報を与えないため）。
    """
    if not _PAGE_ID_PATTERN.fullmatch(page_id):
        raise ValueError(f"invalid page_id: {page_id!r}")

    pages_dir = (get_workspace_dir(create=True) / "pages").resolve()
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_dir = (pages_dir / page_id).resolve()
    if not page_dir.is_relative_to(pages_dir):
        raise ValueError(f"invalid page_id: {page_id!r}")
    return page_dir
