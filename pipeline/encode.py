"""Quantize regridded wind fields into texture-atlas PNGs + meta.json."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    ATLAS_COLS,
    ATLAS_H,
    ATLAS_ROWS,
    ATLAS_W,
    DATASET_ID,
    EAST,
    LEVELS,
    NORTH,
    SCALE_PAD,
    SOUTH,
    TILE_H,
    TILE_W,
    WEST,
    height_meters,
)


def compute_scales(frames):
    """Per-level u/v min/max over all frames (list of (nlev,H,W,2) arrays)."""
    stack = np.stack(frames)  # (nframes, nlev, H, W, 2)
    scales = []
    for i in range(stack.shape[1]):
        u = stack[:, i, :, :, 0]
        v = stack[:, i, :, :, 1]
        umin, umax = np.nanmin(u), np.nanmax(u)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        upad = max((umax - umin) * SCALE_PAD, 0.5)
        vpad = max((vmax - vmin) * SCALE_PAD, 0.5)
        scales.append(
            {
                "uMin": float(umin - upad),
                "uMax": float(umax + upad),
                "vMin": float(vmin - vpad),
                "vMax": float(vmax + vpad),
            }
        )
    return scales


def encode_frame(frame, scales):
    """frame: (nlev, TILE_H, TILE_W, 2) float -> RGBA atlas array."""
    atlas = np.zeros((ATLAS_H, ATLAS_W, 4), dtype=np.uint8)
    for i in range(frame.shape[0]):
        r0 = (i // ATLAS_COLS) * TILE_H
        c0 = (i % ATLAS_COLS) * TILE_W
        u = frame[i, :, :, 0].astype(np.float64)
        v = frame[i, :, :, 1].astype(np.float64)
        s = scales[i]
        valid = ~(np.isnan(u) | np.isnan(v))
        uq = np.clip((u - s["uMin"]) / (s["uMax"] - s["uMin"]), 0, 1)
        vq = np.clip((v - s["vMin"]) / (s["vMax"] - s["vMin"]), 0, 1)
        tile = atlas[r0 : r0 + TILE_H, c0 : c0 + TILE_W]
        tile[:, :, 0] = np.where(valid, np.round(uq * 255), 0).astype(np.uint8)
        tile[:, :, 1] = np.where(valid, np.round(vq * 255), 0).astype(np.uint8)
        tile[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return atlas


def write_output(out_dir, frames_by_lead, scales, init_time_iso):
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    frame_entries = []
    for lead, frame in sorted(frames_by_lead.items()):
        name = f"frames/f{lead:02d}.png"
        atlas = encode_frame(frame, scales)
        Image.fromarray(atlas, "RGBA").save(out / name, optimize=True)
        frame_entries.append({"lead_hours": lead, "file": name})

    meta = {
        "dataset": DATASET_ID,
        "init_time": init_time_iso,
        "bounds": {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH},
        "tile": {"width": TILE_W, "height": TILE_H},
        "atlas": {"cols": ATLAS_COLS, "rows": ATLAS_ROWS},
        "frames": frame_entries,
        "levels": [
            {
                "index": i,
                "id": lid,
                "kind": kind,
                "value": value,
                "heightMeters": height_meters(lid, kind, value),
                **scales[i],
            }
            for i, (lid, kind, value) in enumerate(LEVELS)
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return meta
