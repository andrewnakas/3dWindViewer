// DOM control wiring.

import { levelName, drawLegend, volumetricIndices } from "./atmosphere.js";

export function initUI(map, layer, meta) {
  const $ = (id) => document.getElementById(id);

  // --- header ---
  const init = new Date(meta.init_time);
  $("init-time").textContent = `init ${init.toISOString().slice(0, 13)}Z`;

  // --- level selector ---
  const sel = $("level-select");
  const groups = { Surface: [], "Pressure levels": [] };
  for (const lv of meta.levels) {
    groups[lv.kind === "height_agl" ? "Surface" : "Pressure levels"].push(lv);
  }
  for (const [name, levels] of Object.entries(groups)) {
    const og = document.createElement("optgroup");
    og.label = name;
    for (const lv of levels) {
      const opt = document.createElement("option");
      opt.value = lv.index;
      opt.textContent = levelName(lv);
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  sel.value = String(layer.volumetric
    ? meta.levels.findIndex((l) => l.id === "10m")
    : layer.levelIndices[0]);
  sel.addEventListener("change", () => layer.setLevels([Number(sel.value)]));

  // --- volumetric toggle ---
  const vol = $("volumetric");
  vol.checked = layer.volumetric;
  sel.disabled = vol.checked;
  vol.addEventListener("change", () => {
    sel.disabled = vol.checked;
    layer.volumetric = vol.checked;
    layer.setLevels(vol.checked ? volumetricIndices(meta) : [Number(sel.value)]);
  });

  // --- wind altitude scale ---
  const alt = $("alt-scale");
  alt.value = String(layer.altScale);
  $("alt-val").textContent = `${layer.altScale}×`;
  alt.addEventListener("input", () => {
    layer.altScale = Number(alt.value);
    $("alt-val").textContent = `${alt.value}×`;
  });

  // --- panel collapse ---
  const panel = document.getElementById("panel");
  $("panel-toggle").addEventListener("click", () => panel.classList.toggle("collapsed"));
  if (window.innerWidth < 700) panel.classList.add("collapsed");

  // --- particle count ---
  const pc = $("particles-select");
  pc.value = String(layer.particleCount);
  pc.addEventListener("change", () => layer.setParticleCount(Number(pc.value)));

  // --- terrain exaggeration ---
  const ex = $("exaggeration");
  ex.addEventListener("input", () => {
    const v = Number(ex.value);
    $("exag-val").textContent = v.toFixed(1);
    layer.exaggeration = v;
    map.setTerrain(v > 0 ? { source: "terrain-dem", exaggeration: v } : null);
  });

  // --- particle speed ---
  const sf = $("speed-factor");
  sf.addEventListener("input", () => {
    layer.speedFactor = Number(sf.value);
    $("speed-val").textContent = Number(sf.value).toFixed(1);
  });

  // --- time slider + playback ---
  const slider = $("time-slider");
  const label = $("time-label");
  const playBtn = $("play");
  const maxLead = meta.frames[meta.frames.length - 1].lead_hours;
  slider.max = String(maxLead);
  let playing = false;
  let lastTs = 0;
  const HOURS_PER_SEC = 0.7;

  function setTime(t, fromSlider = false) {
    layer.time = Math.max(0, Math.min(maxLead, t));
    if (!fromSlider) slider.value = String(Math.round(layer.time));
    const valid = new Date(init.getTime() + layer.time * 3600e3);
    const opts = { month: "short", day: "numeric", hour: "numeric" };
    label.textContent = `+${Math.round(layer.time)} h · ${valid.toLocaleString(undefined, opts)}`;
    window.dispatchEvent(new CustomEvent("windtime", { detail: layer.time }));
  }

  slider.addEventListener("input", () => {
    pause();
    setTime(Number(slider.value), true);
  });

  function tick(ts) {
    if (!playing) return;
    const dt = lastTs ? (ts - lastTs) / 1000 : 0;
    lastTs = ts;
    let t = layer.time + dt * HOURS_PER_SEC;
    if (t > maxLead) t = 0; // loop
    setTime(t);
    requestAnimationFrame(tick);
  }
  function pause() {
    playing = false;
    playBtn.textContent = "▶";
  }
  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "❚❚" : "▶";
    if (playing) {
      lastTs = 0;
      requestAnimationFrame(tick);
    }
  });

  setTime(0);
  drawLegend($("legend"));
}
