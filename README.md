# 3D Wind Viewer

Browser-based 3D visualization of NOAA HRRR CONUS wind — GPU particle animation at **41 vertical levels** (10 m, 80 m, and all 39 pressure levels from 1000 to 50 hPa) over 3D terrain, with a 0–48 h forecast time slider.

**Live site:** https://andrewnakas.github.io/3dWindViewer/

## How it works

```
dynamical.org noaa-hrrr-forecast-48-hour-virtual (icechunk/zarr, all HRRR levels)
        │  GitHub Actions, every 6 h (Python)
        ▼
reproject LCC → lat/lon · downsample to ~12 km · rotate winds grid→earth
        ▼
49 texture-atlas PNGs (one per forecast hour; 41 level tiles each, u/v in R/G,
per-level scaling in meta.json)  ≈ 150 MB
        │  deployed with the static site as one GitHub Pages artifact
        ▼
MapLibre GL JS + custom WebGL2 layer: GPU particle advection sampling the
atlas, particles drawn at each level's altitude above 3D terrain
```

- **Data**: [NOAA HRRR via dynamical.org](https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour-virtual/) — the "virtual" map-optimized dataset with every vertical level. Read with `dynamical-catalog` + xarray in `pipeline/`.
- **No data in git**: `site/data/` exists only inside the Pages deployment artifact; a failed build leaves the previous forecast live.
- **Terrain**: AWS Terrain Tiles (terrarium encoding) with adjustable exaggeration; Carto Dark Matter basemap.
- **Particles**: classic [webgl-wind](https://github.com/mapbox/webgl-wind) technique upgraded to WebGL2 — ping-pong float state textures, advection in a fragment shader, streaks drawn as `GL_LINES` at pressure-level altitudes (standard-atmosphere heights). "Volumetric" mode stacks 8 levels from the surface to the jet stream.

## Local development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# build a few test frames (~2 min)
PYTHONPATH=pipeline .venv/bin/python pipeline/build_frames.py --out site/data --leads "0 6 12" --workers 3

python3 -m http.server 8931 -d site   # open http://localhost:8931
```

`?debug` overlays the raw atlas image for georeference checks.

## Deploy / operations

- `.github/workflows/build-and-deploy.yml` runs at 02:30/08:30/14:30/20:30 UTC (≈2.5 h after each HRRR init), on push to main, and manually via *Run workflow* (with an optional lead-hours subset for quick tests).
- First-time setup: repo Settings → Pages → Source: **GitHub Actions** (or `gh api -X POST repos/<user>/3dWindViewer/pages -f build_type=workflow`).

## Related projects

- [Leaflet_3d_terrain_maps](https://github.com/andrewnakas/Leaflet_3d_terrain_maps) — the 3D terrain approach this builds on
- [windplayground-](https://github.com/andrewnakas/windplayground-) — wind forecast modeling experiments

## License

MIT
