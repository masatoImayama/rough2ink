"""パラメータプリセット JSON の保存・一覧・取得・削除（Epic 仕様書 9 節）。

`workspace/presets/<name>.json` に `AnalysisParams` をそのまま JSON として永続化する。
UI のスライダー調整・プリセット保存・バッチ実行の 3 経路が同じ `AnalysisParams` を
共有できるよう、保存・読込ともにこのモデル 1 つを介する。

**プリセット名は Windows のファイル名として安全な文字にサニタイズする**
（Epic 仕様書 9 節 / タスク完了条件）。パストラバーサル（`../` 等）や予約デバイス名
（`CON` `NUL` 等）で `workspace/presets/` の外へ書き込まれないようにする。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rough2ink.core.config import get_presets_dir
from rough2ink.core.params import AnalysisParams

# Windows でファイル名に使えない文字（制御文字 + 予約記号）。
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Windows の予約デバイス名（拡張子の有無に関わらず使用不可）。
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_FALLBACK_NAME = "preset"


def sanitize_preset_name(name: str) -> str:
    """プリセット名を Windows のファイル名として安全な文字列に変換する。

    - 禁止文字（`<>:"/\\|?*` および制御文字）は `_` に置換する
    - 前後の空白・ドットは取り除く（Windows はこれらを暗黙に無視するため）
    - 予約デバイス名（`CON` `NUL` 等）は末尾に `_` を付けて回避する
    - 結果が空文字列になる場合は `preset` にフォールバックする
    """
    sanitized = _INVALID_CHARS_RE.sub("_", name)
    sanitized = sanitized.strip(" .")
    if not sanitized:
        return _FALLBACK_NAME
    if sanitized.upper() in _RESERVED_NAMES:
        sanitized = f"{sanitized}_"
    return sanitized


def _preset_path(name: str) -> Path:
    return get_presets_dir() / f"{sanitize_preset_name(name)}.json"


def save_preset(name: str, params: AnalysisParams) -> str:
    """プリセットを保存する。サニタイズ後の実際のファイル名（拡張子なし）を返す。"""
    sanitized = sanitize_preset_name(name)
    path = get_presets_dir() / f"{sanitized}.json"
    path.write_text(
        json.dumps(params.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sanitized


def load_preset(name: str) -> AnalysisParams:
    """プリセットを読み込む。存在しない場合は `FileNotFoundError`。"""
    path = _preset_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"preset not found: {name!r}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnalysisParams.model_validate(data)


def list_presets() -> list[str]:
    """保存済みプリセット名（拡張子なし）の一覧を、辞書順で返す。"""
    return sorted(path.stem for path in get_presets_dir().glob("*.json"))


def delete_preset(name: str) -> bool:
    """プリセットを削除する。存在して削除できたら `True`、元々無ければ `False`。"""
    path = _preset_path(name)
    if not path.is_file():
        return False
    path.unlink()
    return True
