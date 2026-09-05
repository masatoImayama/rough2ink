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
      {
        key: "min_short_side",
        label: "最小短辺(px)",
        min: 500,
        max: 6000,
        step: 50,
        hint: "これより小さい画像は「解像度不足」として解析から外します。web用に縮小された画像は網点がつぶれて使えないため、入口で弾くための設定です。600dpiのB4原稿なら短辺3000px以上が普通です。",
      },
      {
        key: "jpeg_block_threshold",
        label: "JPEGのアラの許容",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "JPEG保存でできる8pxごとの格子状のムラを、どのくらい厳しく検出するかです。小さくすると厳しくなり、圧縮された画像をより多く弾きます。線画は圧縮の影響を受けやすいので、できれば入稿用の元データを使ってください。",
      },
      {
        key: "binary_ratio_threshold",
        label: "白黒2色かの判定",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "白と黒だけでできた画像か、灰色を含む画像かを見分ける割合です。結果はレポートに記録されるだけで、解析の中身は変わりません。",
      },
    ],
  },
  {
    group: "line",
    label: "B. 線検出",
    fields: [
      {
        key: "black_threshold",
        label: "線とみなす濃さ",
        min: 0,
        max: 255,
        step: 1,
        hint: "どのくらい濃ければ「線」とみなすかです（0が真っ黒、255が真っ白）。大きくすると薄いカスレた線まで拾い、小さくすると濃い線だけになります。",
      },
    ],
  },
  {
    group: "fill",
    label: "B. ベタ検出",
    fields: [
      {
        key: "black_threshold",
        label: "ベタとみなす濃さ",
        min: 0,
        max: 255,
        step: 1,
        hint: "どのくらい濃ければ「ベタ」の候補とみなすかです。線の判定より小さめ（＝厳しめ）にしておくと、薄い線がベタに混ざりません。",
      },
      {
        key: "min_area_ratio",
        label: "ベタの最小の大きさ",
        min: 0,
        max: 0.05,
        step: 0.0001,
        hint: "これより小さい黒い塊はベタとみなしません（ページ全体に対する割合）。大きくすると、小さなベタは線として扱われます。",
      },
      {
        key: "erosion_radius",
        label: "ベタとみなす太さ",
        min: 0,
        max: 30,
        step: 1,
        hint: "太さがこの値の2倍より細いものは「線」として残します。髪の毛や輪郭の太いストロークがベタ扱いされてしまうときは、この値を大きくしてください。600dpiのB4原稿では8前後が目安です。",
      },
    ],
  },
  {
    group: "tone",
    label: "B. トーン検出",
    fields: [
      {
        key: "window",
        label: "判定する区画の大きさ(px)",
        min: 8,
        max: 256,
        step: 8,
        hint: "網点かどうかを、この大きさの正方形ごとにまとめて判定します。小さくすると判定は細かくなりますが、網点の並びを読み取れなくなり精度が落ちます。",
      },
      {
        key: "stride",
        label: "区画をずらす間隔(px)",
        min: 4,
        max: 256,
        step: 4,
        hint: "判定する正方形をずらす間隔です。小さくすると境目がなめらかになりますが、処理時間が急に増えます（半分にすると4倍）。",
      },
      {
        key: "bandpass_low",
        label: "探す網点の細かさ・下限",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "探す網点の細かさの範囲（下限）です。上げると、粗い網点を無視して細かい網点だけを探します。ふつうは変更しません。",
      },
      {
        key: "bandpass_high",
        label: "探す網点の細かさ・上限",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "探す網点の細かさの範囲（上限）です。下げると、細かい網点を無視して粗い網点だけを探します。ふつうは変更しません。",
      },
      {
        key: "energy_threshold",
        label: "模様の量の下限",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "その区画にどれだけ模様らしさがあれば候補にするかです。下げるとトーンを拾いやすくなりますが、誤検出も増えます。",
      },
      {
        key: "sharpness_threshold",
        label: "並びの規則正しさ",
        min: 100,
        max: 30000,
        step: 100,
        hint: "網点のように「規則正しく並んでいるか」を見る厳しさです。上げるほど、はっきりした網点だけをトーンとみなします。カケアミ（手描きの斜線）がトーン扱いされるときは上げてください。",
      },
      {
        key: "min_block_std",
        label: "濃淡のばらつきの下限",
        min: 0,
        max: 40,
        step: 1,
        hint: "真っ黒や真っ白など、模様の無い平らな部分をトーンから除外します。ベタ塗りがトーン扱いされるのを防ぎます。",
      },
      {
        key: "min_direction_ratio",
        label: "並ぶ向きの多さ",
        min: 0,
        max: 1,
        step: 0.05,
        hint: "網点は縦横やななめなど2方向以上に規則がありますが、まっすぐな線は1方向だけです。その違いでトーンと線を見分けます。上げるほど厳しくなります。",
      },
    ],
  },
  {
    group: "panel",
    label: "C. コマ分割",
    fields: [
      {
        key: "close_kernel",
        label: "枠線のつなぎ幅(px)",
        min: 1,
        max: 51,
        step: 2,
        hint: "途切れた枠線をどれだけつないで閉じた形にするかです。枠線がかすれてコマが取れないときは大きくします。大きくしすぎると隣のコマとくっつきます。",
      },
      {
        key: "min_panel_area_ratio",
        label: "コマの最小の大きさ",
        min: 0,
        max: 0.2,
        step: 0.001,
        hint: "これより小さい領域はコマとみなしません（ページ全体に対する割合）。小さなコマが無視されるときは下げてください。",
      },
      {
        key: "approx_epsilon_ratio",
        label: "輪郭の簡略化の強さ",
        min: 0,
        max: 0.05,
        step: 0.001,
        hint: "コマの輪郭をどれだけ単純な形に丸めるかです。大きくすると角が減って単純な形になり、小さくすると輪郭に忠実になります。",
      },
      {
        key: "virtual_frame_margin",
        label: "断ち切りの許容幅(px)",
        min: 0,
        max: 100,
        step: 1,
        hint: "ページの端からこの距離以内にある枠線は「断ち切りコマ」とみなし、ページの外へ仮に延長して形を閉じます。",
      },
      {
        key: "oblique_angle_deg",
        label: "斜めコマとみなす角度",
        min: 0,
        max: 45,
        step: 0.5,
        hint: "縦横からこの角度以上ずれた辺があるコマに「斜めコマ」の印を付けます。印を付けるだけで、切り出し方は変わりません。",
      },
      {
        key: "spread_aspect_ratio",
        label: "見開きとみなす横長さ",
        min: 1,
        max: 3,
        step: 0.05,
        hint: "ページの横幅が縦の何倍を超えたら「見開き」とみなすかです。縦長のページでは印は付きません。",
      },
      {
        key: "effect_line_density",
        label: "効果線とみなす密度",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "コマの中に直線が多すぎる場合「効果線・集中線のコマ」として印を付けます。枠線の検出を誤りやすい箇所を知らせるためのものです。",
      },
      {
        key: "min_solidity",
        label: "コマの形の許容",
        min: 0,
        max: 1,
        step: 0.05,
        hint: "コマの形がどれだけ凹んでいたら「おかしい」とみなすかです。絵の一部を枠線と間違えると内側に食い込んだ形になるので、それを見つけて印を付けます。1に近いほど厳しくなります。",
      },
      {
        key: "fallback_to_page",
        label: "枠線が無ければページ全体を1コマに",
        type: "boolean",
        hint: "信用できるコマが1つも取れなかったとき、ページ全体を1コマとして扱います。全面が絵で枠線が無いページでは、これが正しい答えになります。",
      },
    ],
  },
  {
    group: "balloon",
    label: "E. フキダシ検出",
    fields: [
      {
        key: "min_area_ratio",
        label: "フキダシの最小の大きさ",
        min: 0,
        max: 0.1,
        step: 0.001,
        hint: "これより小さい白い領域はフキダシとみなしません（ページ全体に対する割合）。",
      },
      {
        key: "max_area_ratio",
        label: "フキダシの最大の大きさ",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "これより大きい白い領域はフキダシとみなしません。ページの余白全体を誤ってフキダシ扱いしないための上限です。",
      },
      {
        key: "white_threshold",
        label: "白とみなす明るさ",
        min: 0,
        max: 255,
        step: 1,
        hint: "どのくらい明るければ「白」とみなすかです（255が真っ白）。下げると、少しくすんだ部分も白として扱います。",
      },
      {
        key: "white_fill_ratio",
        label: "中が白である割合",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "その領域の中がどれだけ白ければフキダシとみなすかです。下げると絵を巻き込みやすくなります。",
      },
      {
        key: "min_solidity",
        label: "フキダシの丸みの下限",
        min: 0,
        max: 1,
        step: 0.01,
        hint: "楕円や角丸のような「へこみの無い形」だけをフキダシとみなします。1に近いほど厳しくなります。",
      },
      {
        key: "dilate_radius",
        label: "覆う範囲の広げ幅(px)",
        min: 0,
        max: 60,
        step: 1,
        hint: "見つけたフキダシを少し大きめに覆います。フチや文字のはみ出しまで含めるためで、多少広くても問題ありません（そこを学習に使わないだけです）。",
      },
      {
        key: "text_rect_dilate",
        label: "文字まわりの広げ幅(px)",
        min: 0,
        max: 100,
        step: 1,
        hint: "PSDの文字レイヤーの周囲をどれだけ広く覆うかです。フキダシの白地ごと覆いたいので、文字の大きさより広めにします。",
      },
      {
        key: "require_text_overlap",
        label: "セリフのある白地だけをフキダシに",
        type: "boolean",
        hint: "PSDに文字レイヤーがあるとき、セリフと重なる白い領域だけをフキダシとみなします。モニタ画面や白い床が誤ってフキダシ扱いされるのを防ぎます。文字レイヤーが無い画像・PDFでは効きません。",
      },
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
// ヒントの吹き出しは1つだけ作って使い回す。サイドバーは `overflow-y: auto` なので、
// 中に絶対配置すると枠で切られてしまう。body 直下に `position: fixed` で置いて回避する。
let hintTooltip = null;

function ensureHintTooltip() {
  if (hintTooltip) return hintTooltip;
  hintTooltip = document.createElement("div");
  hintTooltip.className = "param-tooltip";
  hintTooltip.setAttribute("role", "tooltip");
  hintTooltip.hidden = true;
  document.body.appendChild(hintTooltip);
  return hintTooltip;
}

function showHint(anchor, text) {
  const tooltip = ensureHintTooltip();
  tooltip.textContent = text;
  tooltip.hidden = false;

  // いったん表示してから実寸を測り、画面外へはみ出さない位置に寄せる。
  const anchorRect = anchor.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const margin = 8;

  let left = anchorRect.left;
  if (left + tooltipRect.width + margin > window.innerWidth) {
    left = window.innerWidth - tooltipRect.width - margin;
  }
  let top = anchorRect.bottom + 6;
  if (top + tooltipRect.height + margin > window.innerHeight) {
    top = anchorRect.top - tooltipRect.height - 6; // 下に入らなければ上に出す
  }
  tooltip.style.left = `${Math.max(margin, left)}px`;
  tooltip.style.top = `${Math.max(margin, top)}px`;
}

function hideHint() {
  if (hintTooltip) hintTooltip.hidden = true;
}

/** ヒント用の「?」を作る。ホバーとキーボードフォーカスの両方で吹き出しを出す。 */
function createHintMark(text) {
  const mark = document.createElement("span");
  mark.className = "param-hint";
  mark.textContent = "?";
  mark.tabIndex = 0;
  // 支援技術向け。視覚的な吹き出しは自前で出すので `title` は付けない
  // （付けるとブラウザ標準の吹き出しと二重に出る）。
  mark.setAttribute("aria-label", text);
  mark.addEventListener("mouseenter", () => showHint(mark, text));
  mark.addEventListener("mouseleave", hideHint);
  mark.addEventListener("focus", () => showHint(mark, text));
  mark.addEventListener("blur", hideHint);
  return mark;
}

/**
 * `container` にスライダー群を描画する。`params` は `AnalysisParams` 相当のオブジェクトで、
 * 操作するとその場で書き換える（呼び出し側の再解析トリガーは `onChange` に委ねる）。
 */
export function renderParamControls(container, params, onChange) {
  hideHint();
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

      if (field.type === "boolean") {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = Boolean(value);
        checkbox.addEventListener("change", () => {
          setValueAtPath(params, path, checkbox.checked);
          onChange(params);
        });
        row.append(checkbox, nameSpan);
        if (field.hint) row.appendChild(createHintMark(field.hint));
        fieldset.appendChild(row);
        continue;
      }

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

      row.append(nameSpan);
      if (field.hint) row.appendChild(createHintMark(field.hint));
      row.append(input, valueSpan);
      fieldset.appendChild(row);
    }

    container.appendChild(fieldset);
  }
}
