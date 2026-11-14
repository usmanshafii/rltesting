"""Sky-sector geometry (task M1-04): cells, each camera's field at each zoom step, and what a frame covers.

The sector is a grid of `cell` x `cell` degree cells (rl/config/frozen.yaml: 60 x 15 deg in 5-deg cells, 12 x 3).
Row 0 is the lowest elevation, column 0 the leftmost azimuth. The camera points at a cell centre. At a zoom step the
uncooled LWIR camera has the field ZOOM_HFOV_DEG[z]; every other camera matches it where its lens allows, i.e. uses
min(field, its widest field). Frames are treated as flat rectangles in (azimuth, elevation): at <= 46 deg and these
elevations the error is far below a cell.

Two fractions per cell describe a frame:
  f = overlap / cell area   -> the cell is covered when f >= covered_min_fraction (0.5)
  g = overlap / frame area  -> the cell's share of the frame's false detections (pfa_cell = pfa_frame * g)
"""
from __future__ import annotations

import math

import numpy as np

from rl import contracts as C

ASPECT = C.FROZEN["cameras"]["image_px"][1] / C.FROZEN["cameras"]["image_px"][0]   # 512 / 640
WIDTH_PX = C.FROZEN["cameras"]["image_px"][0]


def az_edges(grid: C.Grid = C.GRID) -> np.ndarray:
    return grid.az_min + grid.cell * np.arange(grid.n_az + 1)


def el_edges(grid: C.Grid = C.GRID) -> np.ndarray:
    return grid.el_min + grid.cell * np.arange(grid.n_el + 1)


def cell_center(row: int, col: int, grid: C.Grid = C.GRID) -> tuple[float, float]:
    """(azimuth, elevation) of a cell centre, degrees."""
    return grid.az_min + (col + 0.5) * grid.cell, grid.el_min + (row + 0.5) * grid.cell


def cell_of(az_deg: float, el_deg: float, grid: C.Grid = C.GRID) -> tuple[int, int] | None:
    """The cell holding a direction, or None outside the sector."""
    if not (grid.az_min <= az_deg < grid.az_max and grid.el_min <= el_deg < grid.el_max):
        return None
    return int((el_deg - grid.el_min) // grid.cell), int((az_deg - grid.az_min) // grid.cell)


def band_hfov(band: str, zoom: int) -> float:
    """Horizontal field of `band` at zoom step `zoom` (0 = widest)."""
    return min(C.ZOOM_HFOV_DEG[zoom], float(C.FROZEN["cameras"]["max_hfov_deg"][band]))


def frame_box(pointing: tuple[int, int], hfov_deg: float, grid: C.Grid = C.GRID) -> tuple[float, float, float, float]:
    """(az0, az1, el0, el1) of a frame pointed at a cell centre."""
    az, el = cell_center(*pointing, grid)
    h, v = hfov_deg / 2.0, hfov_deg * ASPECT / 2.0
    return az - h, az + h, el - v, el + v


def overlap_fractions(pointing: tuple[int, int], hfov_deg: float, grid: C.Grid = C.GRID
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(f, g): per cell, overlap / cell area and overlap / frame area. Shapes = grid.shape."""
    a0, a1, e0, e1 = frame_box(pointing, hfov_deg, grid)
    ae, ee = az_edges(grid), el_edges(grid)
    da = np.clip(np.minimum(ae[1:], a1) - np.maximum(ae[:-1], a0), 0.0, None)      # [n_az]
    de = np.clip(np.minimum(ee[1:], e1) - np.maximum(ee[:-1], e0), 0.0, None)      # [n_el]
    overlap = de[:, None] * da[None, :]
    return overlap / grid.cell ** 2, overlap / ((a1 - a0) * (e1 - e0))


def covered(f: np.ndarray) -> np.ndarray:
    return f >= float(C.FROZEN["cameras"]["covered_min_fraction"])


def in_frame(az_deg: float, el_deg: float, pointing: tuple[int, int], hfov_deg: float,
             grid: C.Grid = C.GRID) -> bool:
    a0, a1, e0, e1 = frame_box(pointing, hfov_deg, grid)
    return a0 <= az_deg <= a1 and e0 <= el_deg <= e1


def size_px(span_m: float, range_m: float, hfov_deg: float, width_px: int = WIDTH_PX) -> float:
    """Apparent span in pixels on the 640-wide grid every band is emulated on (the census's size_px)."""
    ifov = math.radians(hfov_deg) / width_px
    return span_m / (max(range_m, 1.0) * ifov)


def coverage_tables(bands=C.MODELLED_BANDS, grid: C.Grid = C.GRID) -> dict:
    """Precomputed (covered, g) for every band, zoom step and pointing cell: {band: [zoom][row][col] -> (cov, g)}."""
    out = {}
    for b in bands:
        out[b] = [[[None] * grid.n_az for _ in range(grid.n_el)] for _ in range(C.N_ZOOM)]
        for z in range(C.N_ZOOM):
            for r in range(grid.n_el):
                for c in range(grid.n_az):
                    f, g = overlap_fractions((r, c), band_hfov(b, z), grid)
                    cov = covered(f)
                    out[b][z][r][c] = (cov, np.where(cov, g, 0.0))
    return out
