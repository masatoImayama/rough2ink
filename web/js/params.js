/**
 * `AnalysisParams`（src/rough2ink/core/params.py）に対応するスライダー UI（T10）。
 *
 * スキーマはサーバの `AnalysisParams` の構造・既定値と一致させてある。ここで持つ
 * min/max/step はスライダー表示専用の目安であり、サーバ側のバリデーションを兼ねない。
 */

export const PARAM_SCHEMA = [
  {
    group: "quality",
    label: "D. 取り込み品質ゲート",
    fields: [
      { key: "min_short_side", label: "最小短辺(px)", min: 500, max: 6000, step: 50 },
      { key: "jpeg_block_threshold", label: "JPEGブロック閾値", min: 0, max: 1, step: 0.01 },
      { key: "binary_ratio_threshold", label: "二値判定比率閾値", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    group: "line",
    label: "B. 線検出",
    fields: [{ key: "black_threshold", label: "黒判定閾値", min: 0, max: 255, step: 1 }],
  },
  {
    group: "fill",
    label: "B. ベタ検出",
    fields: [
      { key: "black_threshold", label: "黒判定閾値", min: 0, max: 255, step: 1 },
      { key: "min_area_ratio", label: "最小面積比", min: 0, max: 0.05, step: 0.0001 },
      { key: "erosion_radius", label: "収縮半径(px)", min: 0, max: 20, step: 1 },
    ],
  },
  {
    group: "tone",
    label: "B. トーン検出",
    fields: [
      { key: "window", label: "ウィンドウ(px)", min: 8, max: 256, step: 8 },
      { key: "stride", label: "ストライド(px)", min: 4, max: 256, step: 4 },
      { key: "bandpass_low", label: "帯域下限", min: 0, max: 1, step: 0.01 },
      { key: "bandpass_high", label: "帯域上限", min: 0, max: 1, step: 0.01 },
      { key: "energy_threshold", label: "エネルギー閾値", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    group: "panel",
    label: "C. コマ分割",
    fields: [
      { key: "close_kernel", label: "クローズカーネル", min: 1, max: 51, step: 2 },
      { key: "min_panel_area_ratio", label: "最小面積比", min: 0, max: 0.2, step: 0.001 },
      { key: "approx_epsilon_ratio", label: "多角形近似epsilon比", min: 0, max: 0.05, step: 0.001 },
      { key: "virtual_frame_margin", label: "仮想枠マージン(px)", min: 0, max: 100, step: 1 },
      { key: "oblique_angle_deg", label: "傾き判定(度)", min: 0, max: 45, step: 0.5 },
      { key: "spread_aspect_ratio", label: "見開きアスペクト比", min: 1, max: 3, step: 0.05 },
      { key: "effect_line_density", label: "効果線密度閾値", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    group: "balloon",
    label: "E. フキダシ検出",
    fields: [
      { key: "min_area_ratio", label: "最小面積比", min: 0, max: 0.1, step: 0.001 },
      { key: "white_threshold", label: "白判定閾値", min: 0, max: 255, step: 1 },
      { key: "white_fill_ratio", label: "白充填率", min: 0, max: 1, step: 0.01 },
      { key: "min_solidity", label: "最小凸包充填率", min: 0, max: 1, step: 0.01 },
      { key: "dilate_radius", label: "膨張半径(px)", min: 0, max: 60, step: 1 },
      { key: "text_rect_dilate", label: "テキスト矩形膨張(px)", min: 0, max: 100, step: 1 },
    ],
  },
];

/** 汎用デバウンス。スライダー連打で再解析リクエストが詰まらないようにする。 */
export function debounce(fn, delayMs) {
  let timer = null;
  return (...args) => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delayMs);
  };
}

function getValueAtPath(obj, path) {
  return path.split(".").reduce((acc, key) => (acc == null ? acc : acc[key]), obj);
}

function setValueAtPath(obj, path, value) {
  const keys = path.split(".");
  let target = obj;
  for (let i = 0; i < keys.length - 1; i += 1) {
    target = target[keys[i]];
  }
  target[keys[keys.length - 1]] = value;
}

/**
 * `container` にスライダー群を描画する。`params` は `AnalysisParams` 相当のオブジェクトで、
 * 操作するとその場で書き換える（呼び出し側の再解析トリガーは `onChange` に委ねる）。
 */
export function renderParamControls(container, params, onChange) {
  container.innerHTML = "";
  for (const group of PARAM_SCHEMA) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = group.label;
    fieldset.appendChild(legend);

    for (const field of group.fields) {
      const path = `${group.group}.${field.key}`;
      const value = getValueAtPath(params, path);

      const row = document.createElement("label");
      row.className = "param-row";

      const nameSpan = document.createElement("span");
      nameSpan.className = "param-name";
      nameSpan.textContent = field.label;

      const valueSpan = document.createElement("span");
      valueSpan.className = "param-value";
      valueSpan.textContent = String(value);

      const input = document.createElement("input");
      input.type = "range";
      input.min = String(field.min);
      input.max = String(field.max);
      input.step = String(field.step);
      input.value = String(value);
      input.addEventListener("input", () => {
        const numValue = Number(input.value);
        setValueAtPath(params, path, numValue);
        valueSpan.textContent = String(numValue);
        onChange(params);
      });

      row.append(nameSpan, input, valueSpan);
      fieldset.appendChild(row);
    }

    container.appendChild(fieldset);
  }
}
