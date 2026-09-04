# rough2ink 前処理検証ツール（Phase 0 素材検証）PoC

漫画原稿の前処理パイプライン（PSD レイヤー抽出・分解器・コマ分割・品質ゲート・
フキダシ検出）を実測するためのローカル Web ツール。目的は `manga-assist-model-design.md`
7章「未決事項」のうち **分解器の精度が実用に足るか** / **コマ分割の例外発生率** を、
感覚ではなく実測値で判断できる状態にすること。詳細は `manga-assist-model-design.md`
および GitHub の Epic issue (#1) を参照。

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

## Windows での注意点

- **パスはすべて `pathlib.Path` で扱う**実装になっている（`core.config` / 各ローダ）。
  日本語ファイル名・空白入りファイル名のアップロードも通る設計だが、リポジトリ自体を
  置くディレクトリに極端に深い階層（Windows のデフォルト上限 260 文字）を使うと、
  `.venv` 配下の依存パッケージのパスが長くなり `pip`/`uv` のインストールに失敗することがある。
  リポジトリは `C:\work\rough2ink` のようになるべく浅い階層に置くことを推奨する
  （どうしても深い階層が必要な場合は Windows のグループポリシーで長いパスを有効化する）。
- PowerShell の実行ポリシーで `.venv\Scripts\Activate.ps1` の実行がブロックされる場合は、
  管理者権限不要の範囲で `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を検討する。
- 原寸解析は 5000px 級の画像を扱うため、`workspace/` `out/` の置き場所は容量に余裕のある
  ドライブを選ぶこと（既定はプロジェクトルート直下。`ROUGH2INK_WORKSPACE_DIR` /
  `ROUGH2INK_OUT_DIR` 環境変数で変更できる）。

## 使い方

Web UI（`http://127.0.0.1:8000/`）は以下の順で使う。番号は画面のセクション番号
（`web/index.html`）と対応する。

1. **アップロード**（画面1） — `.png` / `.jpg` / `.jpeg` / `.psd` / `.pdf` をアップロードする。
   PDF は複数ページに展開され、それぞれ独立したページとして扱われる
   (`POST /api/ingest`)。
2. **ページ一覧**（画面2）からページを選択すると、そのページのプレビューと解析結果が
   表示される。
3. **PSD レイヤー役割マッピング**（画面3、PSD ページのみ） — レイヤー一覧に対して
   `line` / `fill` / `tone` / `text` / `ignore` を手動で割り当てて保存する
   (`PUT /api/pages/{page_id}/gt`)。**レイヤー命名規則は一貫していない前提**のため
   自動判定はしない。保存すると `workspace/gt/<page_id>.json` に永続化され、次回以降の
   解析・バッチ処理で自動的に GT 指標（IoU/F1）算出に使われる。
4. **プリセット**（画面4） — 現在のパラメータに名前を付けて保存・一覧・適用できる
   (`workspace/presets/<name>.json`)。バッチ実行 API (`POST /api/batch`) に
   `preset` 名を渡すと、そのプリセットのパラメータで一括処理できる。
5. **パラメータ**（画面5） — 4節「パラメータとプリセット」のスライダーを動かすと
   400ms のデバウンス後に自動で再解析される。各パラメータの意味は次節の対応表を参照。
6. **オーバーレイ表示**（画面6） — 線（青）/ベタ（赤）/トーン（緑）/コマポリゴン（黄枠）/
   フキダシ（半透明紫）を個別に ON/OFF・不透明度調整しながら目視比較できる。
7. **指標**（画面7） — 品質ゲートの判定、コマ分割の検出数・成功率・例外フラグ件数、
   および（GT マッピング済みの PSD ページのみ）IoU/Precision/Recall/F1 を表示する。

単一ページの調整が済んだら、**バッチ実行**でフォルダ一括処理する。

```bash
curl -X POST http://127.0.0.1:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"input_dir": "C:/path/to/manuscripts", "preset": "my-preset"}'
# => {"job_id": "...", "status": "running", ...}

curl http://127.0.0.1:8000/api/batch/<job_id>   # ポーリングで進捗・完了レポートを取得
```

`out_dir` を省略すると既定の `out/` に書き出される。処理内容はページ選択画面で調整した
パラメータ（`preset` 省略時は既定値）と同じ解析一式（品質ゲート→分解→コマ分割→
フキダシ検出、GT マッピング済みなら指標も）である。

### レポートの読み方

バッチ完了後、`out/report.md`（人間可読）と `out/report.json`（機械可読）に集計結果が出る。

- `total_pages` / `included_count` / `excluded_count` — 品質ゲートで `fail` 判定され
  除外されたページ数。除外理由は `report.md` の「除外ページ」節、または各ページの
  `quality.json` の `reasons` を見る
- `decompose_macro_average` — 分解器（線/ベタ/トーン）の IoU/Precision/Recall/F1 を、
  GT マッピング済みの全ページでマクロ平均したもの。**分解器の精度が実用に足るか**
  （設計書 4章 B・7章）の実測値
- `panel_metrics.success_rate` / `panel_metrics.flags` — コマ分割の成功率と、
  フラグ別（`cut_off` / `unclosed` / `oblique` / `overflow` / `spread` / `effect_lines`）の
  発生件数・発生ページ比率。**コマ分割の例外発生率**（設計書 4章 C・7章）の実測値

GT 指標が欲しい場合は、事前に対象ページを一度 UI からアップロードして PSD レイヤー役割
マッピングを保存しておく（`workspace/gt/<page_id>.json` に保存されていれば、
バッチ処理・単一ページ解析 API のどちらからでも同じ規則で指標が算出される）。

## パラメータと設計書検証項目の対応表

`AnalysisParams`（`src/rough2ink/core/params.py`）の全パラメータが、`manga-assist-model-design.md`
4章「前処理の検証事項」のどの項目（A〜D）の実測に対応するかを示す。

| パラメータ | 既定値 | 対応する検証項目 | 何を測るためのつまみか |
|---|---|---|---|
| `quality.min_short_side` | 2000 | D（取り込み形式ごとの品質ゲート） | 「web用縮小（網点のモアレ化）」を入口で弾く短辺画素数の閾値 |
| `quality.jpeg_block_threshold` | 0.15 | D | 「配信用PDF（低解像度＋JPEG圧縮で線エッジ消失）」を検出する8px格子ブロックノイズの閾値 |
| `quality.binary_ratio_threshold` | 0.98 | D | 「二値かグレースケールか」の判定境界（情報記録用。失格条件そのものではない） |
| `tone.window` / `tone.stride` | 64 / 32 | B（分解器の成立性・最重要） | 「グラデーショントーンの検出可否」を左右するFFTブロックサイズ（ブロック単位判定が空間変化に対処する仕組み） |
| `tone.bandpass_low` / `tone.bandpass_high` | 0.08 / 0.45 | B | 「テクスチャとして使用しているトーンの扱い」を左右する周波数帯域（網点周期の判定範囲） |
| `tone.energy_threshold` | 0.20 | B | トーン候補として採用する帯域エネルギー比の閾値（「ベタとトーン濃度の高い領域の判別」に影響） |
| `fill.black_threshold` | 64 | B | 「ベタとトーン濃度の高い領域の判別」の二値化閾値 |
| `fill.min_area_ratio` | 0.0008 | B | 微小な黒領域をベタ扱いしないための面積下限（線との誤分類防止） |
| `fill.erosion_radius` | 2 | B | 「キャラの主線とトーンが重なる箇所の分離精度」に関わる、細線をベタから除外する収縮半径 |
| `line.black_threshold` | 128 | B | 「線の上にトーンを重ねる使い方」の頻度が高い原稿で線検出の感度を左右する閾値 |
| `panel.close_kernel` | 9 | C（コマ分割の例外処理） | 「枠線が独立レイヤーとして取れるか」に依らず、途切れた枠線を連結するクローズカーネルサイズ |
| `panel.min_panel_area_ratio` | 0.01 | C | ノイズ領域をコマとして誤検出しないための面積下限 |
| `panel.approx_epsilon_ratio` | 0.01 | C | 「斜めコマ・変形コマ」をポリゴン近似する際の頂点簡略化度合い |
| `panel.virtual_frame_margin` | 8 | C | 「断ち切りコマ — ページ端で枠線が途切れ閉領域検出が失敗」への仮想枠補完の延長量 |
| `panel.oblique_angle_deg` | 5.0 | C | 「斜めコマ・変形コマ」を検出するための、水平/垂直からの傾き許容角度 |
| `panel.spread_aspect_ratio` | 1.2 | C | 「見開き — PDFでは1ページ結合の場合と分割の場合がある」を疑うアスペクト比の閾値 |
| `panel.effect_line_density` | 0.15 | C | 「効果線・集中線のコマ — 枠線検出を誤らせる」を検知する直線密度の閾値 |
| `balloon.min_area_ratio` | 0.002 | A（フキダシは作画レイヤーに含まれるか、独立しているか）※注1 | フキダシ候補として拾う白領域の面積下限 |
| `balloon.white_threshold` / `balloon.white_fill_ratio` | 235 / 0.90 | A※注1 | フチなしフキダシも拾うための内部白画素判定 |
| `balloon.min_solidity` | 0.90 | A※注1 | 楕円・角丸の凸形状を検出する凸包面積比（solidity）の下限 |
| `balloon.dilate_radius` | 12 | A※注1 | 損失マスクを「過剰に広く取ってよい」（誤検出のコストが非対称）方針での膨張半径 |
| `balloon.text_rect_dilate` | 24 | A※注1 | PSD の `type` レイヤー bbox をフキダシ手がかりとして使う際の膨張半径 |
| `preview.max_long_side` | 1600 | 対応なし（UI表示専用） | 解析結果はブラウザ返却用に縮小するだけの表示パラメータで、A〜Dのいずれの実測項目にも対応しない |

※注1: フキダシ検出（設計書 3章「フキダシのノイズ対策」、Epic 仕様書スコープ E）は
4章 A〜D に独立した項目を持たない。もっとも近いのは 4章 A「フキダシは作画レイヤーに
含まれるか、独立しているか」であり、`balloon.*` パラメータ群はこの実態確認を踏まえて
損失マスクの過検出・過少検出のバランスを調整するためのものである。

## 出力ディレクトリ構成

`out/`（バッチ処理の書き出し先。`ROUGH2INK_OUT_DIR` で変更可能）は、後続の学習パイプライン
（設計書 5章 Phase 1「パッチ抽出」）がそのまま読み込める形式で書き出す。

```
out/
  <page_id>/
    page.png                   原寸グレースケール
    quality.json                品質ゲートの判定と各指標（D の測定結果そのもの）
    masks/line.png               0/255 二値・原寸（分解器の出力。B の測定対象）
    masks/fill.png
    masks/tone.png
    masks/balloon.png            損失マスク（255 = 無視。学習時にこの領域を除外する）
    panels.json                  ポリゴン頂点列・例外フラグ・面積（C の測定結果）
    panels/panel_000.png         ポリゴン外を白で埋めた bbox 切り出し
    panels/panel_000_mask.png    ポリゴンマスク
    metrics.json                 GT マッピング済みの場合のみ。IoU/Precision/Recall/F1
  report.json                    全ページの集計（機械可読）
  report.md                      同内容の人間可読サマリー
```

品質ゲートで `fail` と判定されたページは `quality.json` のみが書かれ、以降の処理
（`page.png` 以下）は行われない（`report.json` の `pages[].reasons` に理由が残る）。

`workspace/`（`ROUGH2INK_WORKSPACE_DIR` で変更可能）はアップロード原稿・GT マッピング・
プリセットを置く作業領域であり、`out/` と異なり後続パイプラインの入力形式ではない
（UI のセッション間で状態を保持するための内部データ）。

```
workspace/
  pages/<page_id>/meta.json     取り込み時のメタ情報（ファイル名・サイズ・レイヤー一覧）
  pages/<page_id>/page.png      原寸グレースケール
  pages/<page_id>/preview.png   ブラウザ表示用プレビュー
  pages/<page_id>/masks/*.png   直近の単一ページ解析で永続化されたマスク
  gt/<page_id>.json             PSD レイヤー役割の手動マッピング
  presets/<name>.json           パラメータプリセット
```

## スコープ外（今回作らない）

Epic 仕様書のとおり、本 PoC は「測るための配管」に限定する。以下は対象外。

- **学習ループ**（モデルの学習・推論そのもの）
- **GPU 処理**（すべて CPU 上の古典的画像処理で完結する）
- **ステガノグラフィ・マーキング**（提供データへの AI 生成物混入防止の技術的検出）
- **ビルドマニフェスト**（②のカタログ化に向けたデータセット管理の仕組み）
- **クリスタプラグイン連携**（`.clip` はクリスタ側で PSD 書き出しする前提。PSD 対応で足りる）
- **モデル本体**（分解器・コマ分割・品質ゲート・フキダシ検出はいずれも学習を伴わない
  古典的画像処理のプロトタイプ）

## ディレクトリ構成（コード）

```
src/rough2ink/
  app.py                  FastAPI アプリ・静的配信・ルータ登録
  core/
    config.py              workspace/ out/ のパス解決
    params.py               AnalysisParams スキーマと既定値
    types.py                 PageDocument / LayerInfo / PanelInfo / QualityReport
    loaders/                  画像 / PSD / PDF の3系統ローダ
    quality.py               D. 取り込み品質ゲート
    decompose.py             B. 分解器プロトタイプ
    panels.py                C. コマ分割ポリゴン抽出
    balloons.py              フキダシ検出 → 損失マスク
    gt.py                     PSD レイヤー役割マッピング → GT マスク生成
    metrics.py                IoU/F1・コマ例外率の集計
    presets.py                パラメータプリセット CRUD
    batch.py                  バッチ処理・中間成果物書き出し・レポート
  api/                        各機能の API ルータ
web/                           静的フロントエンド（ビルド不要・Node 依存なし）
tests/                         pytest によるユニットテスト・E2E テスト
workspace/                     アップロード原稿・GT マッピング・プリセット（gitignore 対象）
out/                            バッチ処理の中間成果物・レポート（gitignore 対象）
```

## 非機能要件（抜粋）

- 解析は常に **原寸** で行う。縮小はブラウザ表示用プレビュー生成の一点のみ
- パスは全て `pathlib.Path` で扱う。Windows ローカル実行・日本語ファイル名を前提とする
- フロントエンドは静的ファイル配信のみ。npm・ビルドステップは導入しない
