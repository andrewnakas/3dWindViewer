"""Build web-ready wind frames from the dynamical.org virtual HRRR dataset.

Usage:
  python pipeline/build_frames.py --out site/data [--leads "0 6 12"] [--workers 4]
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from config import LEAD_HOURS, LEVELS, PRESSURE_LEVELS, TILE_H, TILE_W
from encode import compute_scales, write_output
from hrrr_source import open_datasets, pick_init
from reproject import build_index_map, regrid, rotate_winds

log = logging.getLogger("build_frames")

SFC_VARS = [("wind_u_10m", "wind_v_10m"), ("wind_u_80m", "wind_v_80m")]


def build_frame(sfc, prs, init, lead, index_map, attempts=3):
    """Read + regrid + rotate all levels for one lead hour.

    Returns (nlev, TILE_H, TILE_W, 2) float16.
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return _build_frame(sfc, prs, init, lead, index_map)
        except Exception as e:  # noqa: BLE001 - network reads; retry then fail
            last_err = e
            log.warning("lead %d attempt %d failed: %s", lead, attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"lead {lead} failed after {attempts} attempts") from last_err


def _build_frame(sfc, prs, init, lead, index_map):
    frame = np.empty((len(LEVELS), TILE_H, TILE_W, 2), dtype=np.float16)

    sfc_t = sfc.sel(init_time=init).isel(lead_time=lead)
    for i, (uname, vname) in enumerate(SFC_VARS):
        u = regrid(sfc_t[uname].values, index_map)
        v = regrid(sfc_t[vname].values, index_map)
        frame[i, :, :, 0], frame[i, :, :, 1] = rotate_winds(u, v, index_map)

    prs_t = prs.sel(init_time=init).isel(lead_time=lead)
    u_slab = prs_t.wind_u.values  # (y, x, nplev)
    v_slab = prs_t.wind_v.values
    plev_axis = list(prs.pressure_level.values)
    for j, p in enumerate(PRESSURE_LEVELS):
        k = plev_axis.index(p)
        u = regrid(u_slab[:, :, k], index_map)
        v = regrid(v_slab[:, :, k], index_map)
        idx = 2 + j
        frame[idx, :, :, 0], frame[idx, :, :, 1] = rotate_winds(u, v, index_map)

    if np.isnan(frame[:, TILE_H // 2, TILE_W // 2, :]).any():
        raise ValueError(f"lead {lead}: NaN wind at domain center")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--leads", default="", help='e.g. "0 1 2"; default all 0-48')
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", default="auto", help="auto or ISO time")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    leads = [int(x) for x in args.leads.split()] if args.leads.strip() else LEAD_HOURS

    t0 = time.time()
    sfc, prs = open_datasets()
    init = pick_init(prs, args.init)
    init_iso = np.datetime_as_string(init, unit="s") + "Z"
    log.info("init=%s leads=%s", init_iso, leads)

    index_map = build_index_map(prs.x.values, prs.y.values)

    frames_by_lead = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_frame, sfc, prs, init, t, index_map): t for t in leads}
        for fut in as_completed(futs):
            lead = futs[fut]
            frames_by_lead[lead] = fut.result()
            log.info("lead %02d done (%d/%d)", lead, len(frames_by_lead), len(leads))

    ordered = [frames_by_lead[t] for t in sorted(frames_by_lead)]
    scales = compute_scales(ordered)
    meta = write_output(args.out, frames_by_lead, scales, init_iso)

    log.info(
        "wrote %d frames to %s in %.1f min (init %s)",
        len(meta["frames"]), args.out, (time.time() - t0) / 60, init_iso,
    )


if __name__ == "__main__":
    sys.exit(main())
