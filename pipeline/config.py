"""Single source of truth for the data pipeline.

The contract between pipeline and frontend is meta.json + the atlas PNG
layout defined here. Change tile size / level list here only.
"""

DATASET_ID = "noaa-hrrr-forecast-48-hour-virtual"

# HRRR pressure levels in the dataset, ordered surface -> top (hPa).
PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
    500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
    250, 225, 200, 175, 150, 125, 100, 75, 50,
]

# Atlas tile order: index 0 = 10m, 1 = 80m, then pressure levels surface->top.
# Each entry: (id, kind, value)
LEVELS = [("10m", "height_agl", 10), ("80m", "height_agl", 80)] + [
    (str(p), "pressure", p) for p in PRESSURE_LEVELS
]  # 41 levels total

# Target regular lat/lon grid (covers the HRRR LCC domain; cells outside the
# native grid get alpha=0 in the atlas). Bounds chosen from the dataset's
# 2D latitude/longitude coords, rounded outward slightly.
WEST, EAST = -134.1, -60.9
SOUTH, NORTH = 21.1, 52.7
TILE_W, TILE_H = 450, 265  # lon x lat samples per level tile (~12 km effective)

ATLAS_COLS, ATLAS_ROWS = 7, 6  # 42 slots for 41 levels
ATLAS_W = ATLAS_COLS * TILE_W   # 3150
ATLAS_H = ATLAS_ROWS * TILE_H   # 1590
assert len(LEVELS) <= ATLAS_COLS * ATLAS_ROWS
assert ATLAS_W <= 4096 and ATLAS_H <= 4096, "atlas exceeds safe WebGL texture size"

# HRRR native grid / projection (verified against the dataset's spatial_ref).
LCC_PROJ = (
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
)
REF_LON = -97.5
ROTCON = 0.6225146  # sin(38.5 deg): grid->earth wind rotation constant

LEAD_HOURS = list(range(49))  # 0..48

# Quantization: per-level min/max over all frames, padded by this fraction.
SCALE_PAD = 0.05


def height_meters(level_id: str, kind: str, value: float) -> float:
    """Approximate geometric altitude for a level (standard atmosphere for
    pressure levels; AGL height used directly for 10m/80m)."""
    if kind == "height_agl":
        return float(value)
    return round(44330.0 * (1.0 - (value / 1013.25) ** 0.1903), 0)
