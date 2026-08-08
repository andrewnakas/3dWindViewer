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
   + terrain.png — one ~2.1 km elevation + curvature texture for terrain flow
        │  deployed with the static site as one GitHub Pages artifact
        ▼
MapLibre GL JS + custom WebGL2 layer: GPU particle advection sampling the
atlas, particles drawn at each level's altitude above 3D terrain
```

- **Data**: [NOAA HRRR via dynamical.org](https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour-virtual/) — the "virtual" map-optimized dataset with every vertical level. Read with `dynamical-catalog` + xarray in `pipeline/`.
- **No data in git**: `site/data/` exists only inside the Pages deployment artifact; a failed build leaves the previous forecast live.
- **Terrain**: AWS Terrain Tiles (terrarium encoding) with adjustable exaggeration; Carto Dark Matter basemap.
- **Particles**: [webgl-wind](https://github.com/mapbox/webgl-wind)-style GPU advection upgraded to WebGL2 with a **continuous vertical coordinate** — particles rise and sink with HRRR's vertical velocity (omega, atlas B channel), so orographic uplift and convection are real 3D motion between levels. Levels sit at each run's actual mean geopotential heights, and below-ground pressure levels (e.g. 925 hPa over the Rockies) are masked out using surface pressure. Falls back from float to packed-RGBA8 state on GPUs without float rendering (iOS).

### Terrain flow physics

HRRR's 12 km-regridded wind cannot show how air behaves against a specific ridge, so the update shader downscales it with four standard terrain-flow terms, each faded out with height above ground (toggle: *Terrain flow*).

| Term | Effect |
|---|---|
| `w = u·∇h` (kinematic boundary condition) | air crossing a slope must climb it — windward updrafts, lee downdrafts |
| Liston–Elder `Ww = 1 + γs·Ωs + γc·Ωc` | ridge crests and windward faces accelerate; valleys and lee slopes slow |
| Ryan (1977) diverting | oblique flow turns toward alignment with the contours |
| Winstral `Sx` | terrain standing upwind casts a wake, slowing the air in its lee |

These read the *slope* of the ground and have to place particles on the surface you can see, so terrain ships as its own ~1.4 km texture (`terrain.png`: 16-bit elevation plus precomputed curvature) built from the **same terrarium DEM MapLibre renders**, not from HRRR's orography — the model smooths summits by 700–1600 m, which floats particles over valleys and buries them inside peaks. Downsampling keeps the max, not the mean, so summits survive: regional high points land within ~50 m of true.

**Particles fly at true heights above ground.** HRRR carries only 10 m and 80 m as genuine above-ground winds — everything above is a pressure surface whose height depends on the terrain beneath it, so riding those surfaces means a particle's height above the ground drifts as it crosses a valley. Instead particles ride a fixed ladder of AGL heights (10, 40, 80, 150, 250, 400, 600, 900, 1400, 2000 m), with the wind at each rung interpolated from whichever model levels bracket that height over that terrain. The 10 m wind is 10 m off the deck over the valley floor and the summit alike. The ladder stops at 2 km — the layer terrain actually shapes; *Full column* swaps in a deep ladder to 11 km. Over SW Montana at 20 m/s the physics gives updrafts of ~2.8 m/s (99th percentile, peaks above 5 m/s) and ±4 m/s of speed variation, decaying to near nothing by 1500 m AGL. Total frame cost is about 1%.

This is a downscaling correction, not new data: the point-forecast panel deliberately keeps showing the raw model sounding.

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
