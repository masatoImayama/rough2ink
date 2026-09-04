/**
 * PSD レイヤー役割マッピング UI（T10 / Epic 仕様書 5節）。
 *
 * `GET /api/pages/{page_id}/layers`（routes_ingest.py）でレイヤー一覧を、
 * `GET /api/pages/{page_id}/gt`（routes_gt.py）で保存済みマッピングを取得し、
 * 役割を選択して `PUT /api/pages/{page_id}/gt` で保存する。
 */

import * as api from "./api.js";

const ROLES = ["line", "fill", "tone", "text", "ignore"];

/**
 * レイヤーマッピング UI を `container` に描画する。
 *
 * @returns 保存対象として編集中のマッピングオブジェクト（`{layer_path: role}`）。
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

  const table = document.createElement("table");
  table.className = "layer-table";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>レイヤー</th><th>種別</th><th>役割</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const layer of layers) {
    const tr = document.createElement("tr");

    const nameTd = document.createElement("td");
    nameTd.textContent = layer.path;

    const kindTd = document.createElement("td");
    kindTd.textContent = layer.kind;

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
    select.value = mapping[layer.path] ?? "";
    select.addEventListener("change", () => {
      if (select.value === "") delete mapping[layer.path];
      else mapping[layer.path] = select.value;
    });
    roleTd.appendChild(select);

    tr.append(nameTd, kindTd, roleTd);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);

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
