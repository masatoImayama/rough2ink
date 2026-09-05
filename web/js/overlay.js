/**
 * オーバーレイ描画（T10）。プレビュー画像の上に、線(青)/ベタ(赤)/トーン(緑)/
 * コマポリゴン(黄枠)/フキダシ(半透明紫) を重ねて canvas に描く。
 *
 * マスク PNG（`GET /api/pages/{page_id}/mask/{kind}`）はグレースケール 0/255 の
 * 画像として返る（`src/rough2ink/core/imageio.py` の `write_mask_png`）。ブラウザは
 * これを R=G=B のグレー画素として復号するため、画素値で前景/背景を判定し
 * canvas 上で任意の色に着色し直す。
 */

export const MASK_COLORS = {
  line: [30, 100, 255], // 青
  fill: [230, 40, 40], // 赤
  tone: [30, 170, 60], // 緑
  balloon: [160, 40, 200], // 紫
};

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load image: ${url}`));
    img.src = url;
  });
}

/** グレースケールマスク画像を、指定色・アルファ 0/255 の canvas に変換する。 */
function tintMask(img, color) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const [r, g, b] = color;
  for (let i = 0; i < data.length; i += 4) {
    const isForeground = data[i] > 127; // グレースケールなので R チャンネルのみ見ればよい
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = isForeground ? 255 : 0;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

const RASTER_MASK_KINDS = ["fill", "tone", "line", "balloon"]; // 描画順（優先順位の低い順に下から重ねる）

export const ZOOM_MIN = 0.1;
export const ZOOM_MAX = 4;

/** 拡大時にマスク境界を補間でぼかさない閾値（画素単位で分解結果を確認するため）。 */
const PIXELATED_FROM = 2;

export class OverlayRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.baseImage = null;
    this.maskCanvases = {};
    this.panels = [];
    this.layerState = {
      line: { visible: true, opacity: 0.6 },
      fill: { visible: true, opacity: 0.6 },
      tone: { visible: true, opacity: 0.6 },
      balloon: { visible: true, opacity: 0.4 },
      panel: { visible: true, opacity: 1.0 },
    };
    // 表示倍率。canvas の内部解像度（プレビュー実サイズ）は変えず、CSS 幅だけを変える。
    // 内部解像度を変えると再描画のたびにマスクを描き直すことになり、拡大操作が重くなる。
    this.zoom = 1;
    // "fit" のときはコンテナ幅の変化に追従する（ページ切り替え・ウィンドウリサイズ時）。
    this.zoomMode = "fit";
  }

  /** 表示倍率を設定する。範囲外は丸める。実際に適用された倍率を返す。 */
  setZoom(zoom, { mode = "manual" } = {}) {
    const clamped = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom));
    this.zoom = clamped;
    this.zoomMode = mode;
    this.applyZoom();
    return clamped;
  }

  /** 現在の倍率を CSS 幅として canvas に反映する。 */
  applyZoom() {
    if (!this.baseImage) return;
    this.canvas.style.width = `${this.canvas.width * this.zoom}px`;
    this.canvas.style.height = "auto";
    this.canvas.style.imageRendering = this.zoom >= PIXELATED_FROM ? "pixelated" : "auto";
  }

  /** コンテナの内寸に合わせて倍率を決める（"幅に合わせる"）。適用された倍率を返す。 */
  fitToWidth(containerWidth) {
    if (!this.baseImage || !containerWidth) return this.zoom;
    // 枠線とスクロールバーの分を少し引いて、横スクロールが出ないようにする。
    const available = Math.max(1, containerWidth - 18);
    return this.setZoom(available / this.canvas.width, { mode: "fit" });
  }

  setLayerState(kind, partialState) {
    Object.assign(this.layerState[kind], partialState);
    this.redraw();
  }

  /** ページ選択時に一度だけ呼ぶ。canvas のサイズをプレビュー画像の実サイズに合わせる。 */
  async loadBase(previewUrl) {
    this.baseImage = await loadImage(previewUrl);
    this.canvas.width = this.baseImage.naturalWidth;
    this.canvas.height = this.baseImage.naturalHeight;
    // 内部解像度が変わったので、現在の倍率を新しいサイズに対して掛け直す。
    this.applyZoom();
  }

  /** 再解析のたびに呼ぶ。`kind` は "line"|"fill"|"tone"|"balloon"。 */
  async loadMask(kind, url) {
    const img = await loadImage(url);
    this.maskCanvases[kind] = tintMask(img, MASK_COLORS[kind]);
  }

  /** `AnalysisResult.panels`（プレビュー座標系のポリゴンを含む）を設定する。 */
  setPanels(panels) {
    this.panels = panels;
  }

  redraw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.baseImage) {
      ctx.drawImage(this.baseImage, 0, 0, this.canvas.width, this.canvas.height);
    }

    for (const kind of RASTER_MASK_KINDS) {
      const state = this.layerState[kind];
      const maskCanvas = this.maskCanvases[kind];
      if (!state.visible || !maskCanvas) continue;
      ctx.save();
      ctx.globalAlpha = state.opacity;
      ctx.drawImage(maskCanvas, 0, 0, this.canvas.width, this.canvas.height);
      ctx.restore();
    }

    const panelState = this.layerState.panel;
    if (panelState.visible && this.panels.length > 0) {
      ctx.save();
      ctx.globalAlpha = panelState.opacity;
      ctx.strokeStyle = "rgb(230, 195, 0)";
      ctx.lineWidth = 2;
      for (const panel of this.panels) {
        const polygon = panel.polygon;
        if (!polygon || polygon.length === 0) continue;
        ctx.beginPath();
        polygon.forEach(([x, y], index) => {
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
      }
      ctx.restore();
    }
  }
}
