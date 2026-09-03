# rough2ink 前処理検証ツール（Phase 0 素材検証）PoC

漫画原稿の前処理パイプライン（PSD レイヤー抽出・分解器・コマ分割・品質ゲート・
フキダシ検出）を実測するためのローカル Web ツール。詳細は `manga-assist-model-design.md`
および GitHub の Epic issue を参照。

## 起動手順（uv）

[uv](https://docs.astral.sh/uv/) がインストール済みであることを前提とする。

```bash
uv sync
uv run uvicorn rough2ink.app:app --reload
```

起動後、ブラウザで `http://127.0.0.1:8000/` を開く。
`GET /api/health` が `{"status": "ok"}` を返せば起動確認は完了。

## 起動手順（uv 未導入時の pip fallback）

uv が使えない環境では、標準の venv + pip で代替できる。

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"  # 開発依存（pytest 等）が必要な場合
# もしくは pip install -e .

uvicorn rough2ink.app:app --reload
```

## テスト

```bash
uv run pytest
```

## ディレクトリ構成

```
src/rough2ink/
  app.py                  FastAPI アプリ・静的配信・ルータ登録
  core/
    config.py              workspace/ out/ のパス解決
    params.py               AnalysisParams スキーマと既定値
    types.py                 PageDocument / LayerInfo / PanelInfo / QualityReport
  api/                        各機能の API ルータ（後続タスクで追加）
web/                           静的フロントエンド（ビルド不要・Node 依存なし）
tests/                         pytest によるユニットテスト
workspace/                     アップロード原稿・GT マッピング・プリセット（gitignore 対象）
out/                            バッチ処理の中間成果物・レポート（gitignore 対象）
```

## 非機能要件（抜粋）

- 解析は常に **原寸** で行う。縮小はブラウザ表示用プレビュー生成の一点のみ
- パスは全て `pathlib.Path` で扱う。Windows ローカル実行・日本語ファイル名を前提とする
- フロントエンドは静的ファイル配信のみ。npm・ビルドステップは導入しない
