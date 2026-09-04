/**
 * バックエンド REST API の薄いラッパー（T10）。
 *
 * 実際のエンドポイント・リクエスト/レスポンス形式は以下の実装に合わせてある
 * （推測で書かない）:
 * - src/rough2ink/api/routes_ingest.py  … アップロード / ページ一覧 / プレビュー / レイヤー一覧
 * - src/rough2ink/api/routes_gt.py      … GT 役割マッピングの保存/取得
 * - src/rough2ink/api/routes_analyze.py … 単一ページ解析 / マスク取得
 * - src/rough2ink/api/routes_presets.py … プリセット CRUD
 * - src/rough2ink/app.py                … /api/health, /api/params/defaults
 */

const BASE = "/api";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // レスポンスが JSON でない場合は statusText をそのまま使う。
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res;
}

/** `POST /api/ingest`。アップロードしたファイルを取り込み、PageSummary のリストを返す。 */
export async function ingest(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await handleResponse(await fetch(`${BASE}/ingest`, { method: "POST", body: form }));
  return res.json();
}

/** `GET /api/pages`。永続化済み全ページの PageSummary リストを返す。 */
export async function listPages() {
  const res = await handleResponse(await fetch(`${BASE}/pages`));
  return res.json();
}

/** `GET /api/pages/{page_id}/preview` の URL（プレビュー PNG、長辺 1600px 以下）。 */
export function previewUrl(pageId) {
  return `${BASE}/pages/${encodeURIComponent(pageId)}/preview`;
}

/** `GET /api/pages/{page_id}/layers`。PSD レイヤー一覧（画像/PDF は空配列）。 */
export async function getLayers(pageId) {
  const res = await handleResponse(await fetch(`${BASE}/pages/${encodeURIComponent(pageId)}/layers`));
  return res.json();
}

/** `GET /api/pages/{page_id}/gt`。保存済み GT 役割マッピングを取得する（未保存なら空 mapping）。 */
export async function getGTMapping(pageId) {
  const res = await handleResponse(await fetch(`${BASE}/pages/${encodeURIComponent(pageId)}/gt`));
  return res.json();
}

/** `PUT /api/pages/{page_id}/gt`。GT 役割マッピング `{layer_path: role}` を保存する。 */
export async function putGTMapping(pageId, mapping) {
  const res = await handleResponse(
    await fetch(`${BASE}/pages/${encodeURIComponent(pageId)}/gt`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mapping }),
    })
  );
  return res.json();
}

/** `POST /api/pages/{page_id}/analyze`。`AnalysisParams` を渡して解析結果を得る。 */
export async function analyzePage(pageId, params) {
  const res = await handleResponse(
    await fetch(`${BASE}/pages/${encodeURIComponent(pageId)}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
  return res.json();
}

/**
 * `GET /api/pages/{page_id}/mask/{kind}` の URL（プレビュー解像度、`kind` は
 * "line"|"fill"|"tone"|"balloon"）。再解析のたびにマスク PNG が更新されるため、
 * ブラウザの HTTP キャッシュに古い画像を掴ませないようキャッシュバスティングを付ける。
 */
export function maskUrl(pageId, kind) {
  return `${BASE}/pages/${encodeURIComponent(pageId)}/mask/${kind}?preview=1&_t=${Date.now()}`;
}

/** `GET /api/params/defaults`（app.py 直下のエンドポイント）。 */
export async function getParamsDefaults() {
  const res = await handleResponse(await fetch(`${BASE}/params/defaults`));
  return res.json();
}

/** `GET /api/presets`。保存済みプリセット名の一覧。 */
export async function listPresets() {
  const res = await handleResponse(await fetch(`${BASE}/presets`));
  return res.json();
}

/** `GET /api/presets/{name}`。プリセットを取得する。 */
export async function getPreset(name) {
  const res = await handleResponse(await fetch(`${BASE}/presets/${encodeURIComponent(name)}`));
  return res.json();
}

/** `PUT /api/presets/{name}`。現在のパラメータをプリセットとして保存する。 */
export async function putPreset(name, params) {
  const res = await handleResponse(
    await fetch(`${BASE}/presets/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
  );
  return res.json();
}

/** `DELETE /api/presets/{name}`。プリセットを削除する。 */
export async function deletePreset(name) {
  await handleResponse(await fetch(`${BASE}/presets/${encodeURIComponent(name)}`, { method: "DELETE" }));
}
