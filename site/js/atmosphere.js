// Wind speed color ramp and level naming helpers.

export const SPEED_MAX = 60; // m/s used for color normalization

// Perceptual-ish ramp: calm blue -> cyan -> green -> yellow -> orange -> red -> magenta
const STOPS = [
  [0.0, [63, 90, 190]],
  [0.17, [64, 175, 205]],
  [0.33, [95, 200, 120]],
  [0.5, [228, 210, 95]],
  [0.67, [240, 145, 65]],
  [0.83, [230, 75, 60]],
  [1.0, [220, 70, 180]],
];

export function rampColor(t) {
  t = Math.min(Math.max(t, 0), 1);
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const f = (t - t0) / (t1 - t0);
      return c0.map((c, k) => Math.round(c + (c1[k] - c) * f));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export function rampTextureData(n = 256) {
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = rampColor(i / (n - 1));
    data.set([r, g, b, 255], i * 4);
  }
  return data;
}

export function levelName(level) {
  if (level.kind === "height_agl") return `${level.value} m (surface)`;
  const km = (level.heightMeters / 1000).toFixed(1);
  return `${level.value} hPa (~${km} km)`;
}

export function drawLegend(canvas) {
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  for (let x = 0; x < w; x++) {
    const [r, g, b] = rampColor(x / (w - 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, 0, 1, 14);
  }
  ctx.fillStyle = "#a8b6ca";
  ctx.font = "9px sans-serif";
  for (const ms of [0, 15, 30, 45, 60]) {
    const x = Math.min((ms / SPEED_MAX) * w, w - 12);
    ctx.fillText(String(ms), x, 24);
  }
  ctx.fillText("m/s", w - 20, 33);
}
