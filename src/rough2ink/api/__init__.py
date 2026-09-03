"""API ルータの受け口。

各タスクで作成するルータ（routes_ingest / routes_analyze / routes_gt /
routes_presets / routes_batch など）をここからエクスポートし、
`rough2ink.app` の `include_router` で登録する。
"""
