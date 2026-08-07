// Bootstrap: map + terrain + wind layer + UI.

import { WindLayer } from "./windLayer.js";
import { initUI } from "./ui.js";

const TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

function fail(msg) {
  const el = document.getElementById("error");
  el.textContent = msg;
  el.hidden = false;
}

async function main() {
  let meta;
  try {
    const r = await fetch("data/meta.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    meta = await r.json();
  } catch (e) {
    fail("Wind data not available yet (data/meta.json missing). " +
      "The GitHub Action may still be running its first build.");
    throw e;
  }

  const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) || window.innerWidth < 700;
  let map;
  try {
    map = new maplibregl.Map({
      container: "map",
      style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: [-98.5, 39.5],
      zoom: isMobile ? 3 : 4,
      pitch: 55,
      bearing: 0,
      maxPitch: 80,
      antialias: !isMobile, // MSAA is heavy on mobile GPUs
    });
  } catch (e) {
    fail(`Couldn't start the map (WebGL2 unavailable?): ${e.message}`);
    throw e;
  }
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  window.__map = map; // debugging hook

  map.on("load", () => {
    map.addSource("terrain-dem", {
      type: "raster-dem",
      tiles: [TERRAIN_TILES],
      tileSize: 256,
      maxzoom: 12,
      encoding: "terrarium",
      attribution: "Terrain: AWS Terrain Tiles / Mapzen",
    });
    map.addSource("hillshade-dem", {
      type: "raster-dem",
      tiles: [TERRAIN_TILES],
      tileSize: 256,
      maxzoom: 12,
      encoding: "terrarium",
    });
    map.addLayer({
      id: "hillshade",
      type: "hillshade",
      source: "hillshade-dem",
      paint: { "hillshade-exaggeration": 0.35 },
    });
    map.setTerrain({ source: "terrain-dem", exaggeration: 1.5 });

    const layer = new WindLayer(map, meta, {
      exaggeration: 1.5,
      onReady: () => initUI(map, layer, meta),
    });
    if (isMobile) layer.particleCount = 32768; // ~3k per level in volumetric mode
    try {
      map.addLayer(layer);
    } catch (e) {
      fail(`This browser can't run the wind layer: ${e.message}`);
      throw e;
    }

    // ?debug: overlay frame 0 atlas tile georeference check
    if (new URLSearchParams(location.search).has("debug")) {
      const b = meta.bounds;
      map.addSource("debug-atlas", {
        type: "image",
        url: `data/${meta.frames[0].file}`,
        coordinates: [
          [b.west, b.north], [b.east, b.north],
          [b.east, b.south], [b.west, b.south],
        ],
      });
      map.addLayer({
        id: "debug-atlas",
        type: "raster",
        source: "debug-atlas",
        paint: { "raster-opacity": 0.55 },
      });
    }
  });

  map.on("error", (e) => console.warn("map error:", e?.error?.message ?? e));
}

main();
