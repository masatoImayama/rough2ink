/**
 * パラメータプリセットの保存・一覧・適用 UI（T10 / routes_presets.py）。
 */

import * as api from "./api.js";

/**
 * プリセット操作 UI を `container` に描画する。
 *
 * @param getParams 現在のパラメータ（`AnalysisParams` 相当）を返す関数
 * @param applyParams プリセット適用時に呼ばれるコールバック（取得した params を受け取る）
 */
export async function renderPresetControls(container, getParams, applyParams) {
  container.innerHTML = "";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "プリセット名";

  const saveButton = document.createElement("button");
  saveButton.type = "button";
  saveButton.textContent = "現在の設定を保存";

  const select = document.createElement("select");

  const applyButton = document.createElement("button");
  applyButton.type = "button";
  applyButton.textContent = "適用";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "削除";

  const statusEl = document.createElement("p");
  statusEl.className = "status";

  async function refreshList() {
    const names = await api.listPresets();
    const previousValue = select.value;
    select.innerHTML = "";
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (names.includes(previousValue)) select.value = previousValue;
  }

  saveButton.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    if (!name) {
      statusEl.textContent = "プリセット名を入力してください。";
      return;
    }
    try {
      await api.putPreset(name, getParams());
      statusEl.textContent = `プリセット "${name}" を保存しました。`;
      nameInput.value = "";
      await refreshList();
    } catch (err) {
      statusEl.textContent = `保存に失敗しました: ${err.message}`;
    }
  });

  applyButton.addEventListener("click", async () => {
    if (!select.value) return;
    try {
      const params = await api.getPreset(select.value);
      await applyParams(params);
      statusEl.textContent = `プリセット "${select.value}" を適用しました。`;
    } catch (err) {
      statusEl.textContent = `適用に失敗しました: ${err.message}`;
    }
  });

  deleteButton.addEventListener("click", async () => {
    if (!select.value) return;
    try {
      await api.deletePreset(select.value);
      statusEl.textContent = "プリセットを削除しました。";
      await refreshList();
    } catch (err) {
      statusEl.textContent = `削除に失敗しました: ${err.message}`;
    }
  });

  container.append(nameInput, saveButton, document.createElement("br"), select, applyButton, deleteButton, statusEl);

  await refreshList();
}
