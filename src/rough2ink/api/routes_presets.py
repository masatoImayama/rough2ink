"""パラメータプリセット CRUD API（Epic 仕様書 7・9 節）。

`workspace/presets/<name>.json` への保存・一覧・取得・削除を行う。
名前は `core.presets.sanitize_preset_name` で Windows のファイル名として安全な文字に
サニタイズしてから永続化する（パストラバーサル対策も兼ねる）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rough2ink.core import presets
from rough2ink.core.params import AnalysisParams

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("", response_model=list[str])
def list_presets() -> list[str]:
    """保存済みプリセット名の一覧を返す。"""
    return presets.list_presets()


@router.get("/{name}", response_model=AnalysisParams)
def get_preset(name: str) -> AnalysisParams:
    """プリセットを取得する（UI での再適用に使う）。"""
    try:
        return presets.load_preset(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{name}", response_model=AnalysisParams)
def put_preset(name: str, params: AnalysisParams) -> AnalysisParams:
    """現在のパラメータをプリセットとして保存する（既存なら上書き）。"""
    saved_name = presets.save_preset(name, params)
    return presets.load_preset(saved_name)


@router.delete("/{name}", status_code=204)
def delete_preset(name: str) -> None:
    """プリセットを削除する。"""
    if not presets.delete_preset(name):
        raise HTTPException(status_code=404, detail=f"preset not found: {name!r}")
