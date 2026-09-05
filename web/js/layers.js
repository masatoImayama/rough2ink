/**
 * PSD レイヤー役割マッピング UI（T10 / Epic 仕様書 5節）。
 *
 * `GET /api/pages/{page_id}/layers`（routes_ingest.py）でレイヤー一覧を、
 * `GET /api/pages/{page_id}/gt`（routes_gt.py）で保存済みマッピングを取得し、
 * 役割を選択して `PUT /api/pages/{page_id}/gt` で保存する。
 *
 * マッピングのキーは `layer.id`（`LayerInfo.id`）を使う。`layer.path` は表示専用であり、
 * 同名レイヤーがあると一意性を保証しないため（#20）、送信キーには使わない。
 *
 * 設計書 4章 A の通り**レイヤー名だけでは役割を判断できない**。実測でも
 * `集中線（円形）` という名前でページの 74% が墨（黒地に白で放射線を抜いたベタフラッシュ、
 * 機能的にはベタ）という例があった。そのため各行に**墨の被覆率とサムネイル**を出し、
 * 墨が多い順に並べる。判断を誤りやすいレイヤーが自然に上へ来る。
 */

import * as api from "./api.js";

const ROLES = ["line", "fill", "tone", "text", "ignore"];

// 墨がページのこの割合を超えるレイヤーは、名前が何であれ確認が要る（ベタの可能性が高い）。
const HIGH_INK_PAGE_RATIO = 0.05;

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * レイヤーマッピング UI を `container` に描画する。
 *
 * @returns 保存対象として編集中のマッピングオブジェクト（`{layer_id: role}`）。
 */
export async function renderLayerMapping(container, pageId, layers, statusEl) {
  container.innerHTML = "";
  if (statusEl) statusEl.textContent = "";

  if (!layers || layers.length === 0) {
    container.textContent = "このページにはレイヤー情報がありません（PSD 以外の取り込み）。";
    return {};
  }

  const existing = await api.getGTMapping(pageId);
  const mapping = { ...existing.mapping };

  if (statusEl) statusEl.textContent = "レイヤーの墨の量を集計中...";
  let stats = [];
  let suggestions = [];
  try {
    // 初回はレイヤーをすべてラスタライズするため数十秒かかる（以降はキャッシュ）。
    [stats, suggestions] = await Promise.all([
      api.getLayerStats(pageId),
      api.getRoleSuggestions(pageId),
    ]);
    if (statusEl) statusEl.textContent = "";
  } catch (err) {
    // 統計が取れなくても割当自体はできるべきなので、致命的には扱わない。
    if (statusEl) statusEl.textContent = `レイヤー統計を取得できませんでした: ${err.message}`;
  }

  const statById = new Map(stats.map((stat) => [stat.layer_id, stat]));
  const suggestionById = new Map(suggestions.map((item) => [item.layer_id, item]));

  // グループは画素を持たず割当対象にならないので除外し、墨の多い順に並べる。
  const rows = layers
    .filter((layer) => layer.kind !== "group")
    .map((layer) => ({ layer, stat: statById.get(layer.id) }))
    .sort((a, b) => (b.stat?.ink_ratio_page ?? -1) - (a.stat?.ink_ratio_page ?? -1));

  const controls = document.createElement("div");
  controls.className = "layer-controls";

  const suggestButton = document.createElement("button");
  suggestButton.type = "button";
  suggestButton.textContent = "レイヤー名から推定";
  suggestButton.title = "未設定の行にだけ推定値を入れます（既に設定した行は変更しません）";
  controls.appendChild(suggestButton);

  const note = document.createElement("span");
  note.className = "hint";
  note.textContent = "推定は初期値です。墨の割合とサムネイルで必ず確認してください。";
  controls.appendChild(note);
  container.appendChild(controls);

  const table = document.createElement("table");
  table.className = "layer-table";
  const thead = document.createElement("thead");
  thead.innerHTML =
    "<tr><th></th><th>レイヤー</th><th>墨/ページ</th><th>墨/レイヤー</th><th>役割</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  const selects = new Map();

  for (const { layer, stat } of rows) {
    const tr = document.createElement("tr");
    if (stat && !stat.has_pixels) tr.className = "layer-row--empty";
    else if (stat && stat.ink_ratio_page >= HIGH_INK_PAGE_RATIO) tr.className = "layer-row--heavy";

    const thumbTd = document.createElement("td");
    if (stat?.has_thumbnail) {
      const img = document.createElement("img");
      img.className = "layer-thumb";
      img.loading = "lazy";
      img.alt = "";
      img.src = api.layerThumbnailUrl(pageId, layer.id);
      thumbTd.appendChild(img);
    }

    const nameTd = document.createElement("td");
    nameTd.textContent = layer.path;
    const suggestion = suggestionById.get(layer.id);
    if (suggestion?.matched_keyword) {
      const badge = document.createElement("span");
      badge.className = "layer-badge";
      badge.textContent = `名前: ${suggestion.matched_keyword}`;
      nameTd.appendChild(document.createTextNode(" "));
      nameTd.appendChild(badge);
    }
    if (stat && !stat.has_pixels) {
      const badge = document.createElement("span");
      badge.className = "layer-badge layer-badge--warn";
      badge.textContent = "画素なし";
      nameTd.appendChild(document.createTextNode(" "));
      nameTd.appendChild(badge);
    } else if (stat && stat.opaque_ratio_layer > 0 && stat.ink_ratio_layer === 0) {
      // 不透明だが墨が無い＝白抜き。GT に入れても線としては何も寄与しない。
      const badge = document.createElement("span");
      badge.className = "layer-badge layer-badge--warn";
      badge.textContent = "白のみ";
      nameTd.appendChild(document.createTextNode(" "));
      nameTd.appendChild(badge);
    }

    const inkPageTd = document.createElement("td");
    inkPageTd.className = "layer-num";
    inkPageTd.textContent = stat ? percent(stat.ink_ratio_page) : "-";

    const inkLayerTd = document.createElement("td");
    inkLayerTd.className = "layer-num";
    inkLayerTd.textContent = stat ? percent(stat.ink_ratio_layer) : "-";

    const roleTd = document.createElement("td");
    const select = document.createElement("select");
    const unsetOption = document.createElement("option");
    unsetOption.value = "";
    unsetOption.textContent = "(未設定)";
    select.appendChild(unsetOption);
    for (const role of ROLES) {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      select.appendChild(option);
    }
    select.value = mapping[layer.id] ?? "";
    select.addEventListener("change", () => {
      if (select.value === "") delete mapping[layer.id];
      else mapping[layer.id] = select.value;
    });
    selects.set(layer.id, select);
    roleTd.appendChild(select);

    tr.append(thumbTd, nameTd, inkPageTd, inkLayerTd, roleTd);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);

  suggestButton.addEventListener("click", () => {
    let applied = 0;
    for (const [layerId, select] of selects) {
      if (select.value !== "") continue; // 既に人が決めた行は上書きしない
      const role = suggestionById.get(layerId)?.role;
      if (!role) continue;
      select.value = role;
      mapping[layerId] = role;
      applied += 1;
    }
    if (statusEl) {
      statusEl.textContent = `${applied} 行に推定値を入れました（未保存）。内容を確認してから保存してください。`;
    }
  });

  const saveButton = document.createElement("button");
  saveButton.type = "button";
  saveButton.textContent = "マッピングを保存";
  saveButton.addEventListener("click", async () => {
    try {
      await api.putGTMapping(pageId, mapping);
      if (statusEl) statusEl.textContent = "GT マッピングを保存しました。";
    } catch (err) {
      if (statusEl) statusEl.textContent = `保存に失敗しました: ${err.message}`;
    }
  });
  container.appendChild(saveButton);

  return mapping;
}
