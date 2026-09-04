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
