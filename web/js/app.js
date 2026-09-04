/**
 * UI のエントリポイント（T10）。アップロード・ページ一覧・レイヤーマッピング・
 * プリセット・パラメータスライダー・オーバーレイ表示を束ねる。
 *
 * ビルド不要の素の ES モジュールとして動く（Node 依存なし、CDN なし）。
 */

import * as api from "./api.js";
import { debounce, renderParamControls } from "./params.js";
import { OverlayRenderer } from "./overlay.js";
import { renderLayerMapping } from "./layers.js";
import { renderPresetControls } from "./presets.js";

const RASTER_MASK_KINDS = ["line", "fill", "tone", "balloon"];
const TOGGLE_KINDS = ["line", "fill", "tone", "panel", "balloon"];
// 再解析のデバウンス間隔(ms)。スライダー連打でリクエストが詰まらないようにする。
const REANALYZE_DEBOUNCE_MS = 400;

const state = {
  pageId: null,
  params: null,
};

let overlay;
let debouncedReanalyze;

function byId(id) {
  return document.getElementById(id);
}

async function init() {
  state.params = await api.getParamsDefaults();
  overlay = new OverlayRenderer(byId("overlay-canvas"));
  debouncedReanalyze = debounce(reanalyze, REANALYZE_DEBOUNCE_MS);

  renderParamControls(byId("param-controls"), state.params, () => debouncedReanalyze());
  setupLayerToggles();
  await renderPresetControls(byId("preset-controls"), () => state.params, applyPresetParams);
  await refreshPageList();

  byId("upload-form").addEventListener("submit", onUpload);
}

function setupLayerToggles() {
  for (const kind of TOGGLE_KINDS) {
    const visibleCheckbox = byId(`layer-${kind}-visible`);
    const opacityInput = byId(`layer-${kind}-opacity`);
    visibleCheckbox.addEventListener("change", () => {
      overlay.setLayerState(kind, { visible: visibleCheckbox.checked });
    });
    opacityInput.addEventListener("input", () => {
      overlay.setLayerState(kind, { opacity: Number(opacityInput.value) });
    });
  }
}

async function onUpload(event) {
  event.preventDefault();
  const fileInput = byId("file-input");
  const file = fileInput.files[0];
  if (!file) return;

  byId("upload-status").textContent = "アップロード中...";
  try {
    await api.ingest(file);
    byId("upload-status").textContent = "アップロード完了。";
    fileInput.value = "";
    await refreshPageList();
  } catch (err) {
    byId("upload-status").textContent = `アップロードに失敗しました: ${err.message}`;
  }
}

async function refreshPageList() {
  const pages = await api.listPages();
  const list = byId("page-list");
  list.innerHTML = "";
  for (const page of pages) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${page.filename} (${page.width}x${page.height}, ${page.source_kind})`;
    button.addEventListener("click", () => selectPage(page.page_id));
    li.appendChild(button);
    list.appendChild(li);
  }
}

async function selectPage(pageId) {
  state.pageId = pageId;
  byId("page-status").textContent = `選択中: ${pageId}`;

  // Review #23: バッチ処理した meta.json のみのページ（page.png/preview.png を持たない）
  // が一覧に出ることがあり、選ぶと各 API が 404 を返す。ここで捕捉せず投げっぱなしにすると
  // 未処理の Promise 拒否になり画面に何も出ない（呼び出し元のクリックハンドラは await しない）。
  try {
    const layers = await api.getLayers(pageId);
    await renderLayerMapping(byId("layer-mapping"), pageId, layers, byId("gt-status"));

    await overlay.loadBase(api.previewUrl(pageId));
    overlay.redraw();

    await reanalyze();
  } catch (err) {
    byId("page-status").textContent = `ページの読み込みに失敗しました: ${err.message}`;
  }
}

async function applyPresetParams(params) {
  state.params = params;
  renderParamControls(byId("param-controls"), state.params, () => debouncedReanalyze());
  await reanalyze();
}

async function reanalyze() {
  if (!state.pageId) return;

  byId("analysis-status").textContent = "解析中...";
  try {
    const result = await api.analyzePage(state.pageId, state.params);

    await Promise.all(
      RASTER_MASK_KINDS.map((kind) => overlay.loadMask(kind, api.maskUrl(state.pageId, kind)))
    );
    overlay.setPanels(result.panels);
    overlay.redraw();

    renderQuality(result.quality);
    renderPanelMetrics(result.panels);
    renderGTMetrics(result.metrics);

    byId("analysis-status").textContent = "解析完了。";
  } catch (err) {
    byId("analysis-status").textContent = `解析に失敗しました: ${err.message}`;
  }
}

function renderQuality(quality) {
  const el = byId("quality-report");
  el.innerHTML = "";
  const dl = document.createElement("dl");
  const rows = [
    ["判定", quality.status],
    ["短辺(px)", quality.short_side],
    ["階調", quality.tone_depth],
    ["JPEGブロックスコア", quality.jpeg_block_score.toFixed(4)],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    dl.append(dt, dd);
  }
  if (quality.reasons.length > 0) {
    const dt = document.createElement("dt");
    dt.textContent = "理由";
    const dd = document.createElement("dd");
    dd.textContent = quality.reasons.join(", ");
    dl.append(dt, dd);
  }
  el.appendChild(dl);
}

/**
 * コマ例外率は `AnalysisResult.panels`（全ページで常に返る）からクライアント側で
 * 集計する。成功率 = 例外フラグが一つも立っていないコマ数 / 検出コマ総数。
 */
function renderPanelMetrics(panels) {
  const el = byId("panel-metrics");
  el.innerHTML = "";

  const total = panels.length;
  const successCount = panels.filter((panel) => panel.flags.length === 0).length;
  const successRate = total === 0 ? 0 : successCount / total;

  const flagCounts = {};
  for (const panel of panels) {
    for (const flag of panel.flags) {
      flagCounts[flag] = (flagCounts[flag] ?? 0) + 1;
    }
  }

  const heading = document.createElement("h3");
  heading.textContent = "コマ分割";
  el.appendChild(heading);

  const summary = document.createElement("p");
  summary.textContent = `検出コマ数: ${total} / 成功率: ${(successRate * 100).toFixed(1)}%`;
  el.appendChild(summary);

  if (Object.keys(flagCounts).length > 0) {
    const ul = document.createElement("ul");
    for (const [flag, count] of Object.entries(flagCounts)) {
      const li = document.createElement("li");
      li.textContent = `${flag}: ${count}件`;
      ul.appendChild(li);
    }
    el.appendChild(ul);
  }
}

/**
 * GT との IoU/F1 は `workspace/gt/<page_id>.json` に役割マッピングが保存されている
 * 場合のみサーバから返る (`AnalysisResult.metrics`, src/rough2ink/api/routes_analyze.py)。
 * マッピング未保存のページでは常に null なので、その場合は明示的にその旨を表示する。
 */
function renderGTMetrics(metrics) {
  const el = byId("gt-metrics");
  el.innerHTML = "";

  const heading = document.createElement("h3");
  heading.textContent = "GT 指標 (IoU / F1)";
  el.appendChild(heading);

  if (!metrics) {
    const p = document.createElement("p");
    p.textContent = "このページの GT 指標は利用できません（GT マッピング未保存）。";
    el.appendChild(p);
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>役割</th><th>IoU</th><th>Precision</th><th>Recall</th><th>F1</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const [role, values] of Object.entries(metrics)) {
    const tr = document.createElement("tr");
    const cells = [role, values.iou, values.precision, values.recall, values.f1];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = typeof cell === "number" ? cell.toFixed(3) : String(cell);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  el.appendChild(table);
}

init();
