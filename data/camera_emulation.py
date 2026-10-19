"""Emulate each camera of the scene's rig (render/cameras.py) on the output grid. Pure NumPy.

A scene is rendered once per band on the output grid: the student's uncooled LWIR frame. A camera differs from that
grid in pixel size and coverage: one camera pixel spans r grid pixels (r < 1 finer, r > 1 coarser), and a zoom camera
that cannot widen to the grid sees only its centre. The loader therefore takes the covered part of the composited grid
image to the camera's own pixels, applies that camera's sensor model there (noise, fixed pattern, gain control), and
brings the result back to the grid:

    grid -> camera:  area average when the camera is coarser (r > 1: its pixel aperture); linear interpolation when
                     it is finer (r < 1: the render holds no finer detail)
    sensor model:    on the camera's own pixels, so noise is independent per CAMERA pixel
    camera -> grid:  area average when the camera is finer (r < 1: binning, so temporal noise falls as r); linear
                     interpolation when it is coarser (r > 1: correlated, blocky noise as on an upscaled real frame)

Outside the covered region the camera sees nothing: zero, and the returned mask is 0 there.
"""
from __future__ import annotations

import numpy as np

UNITY = 0.02            # |r - 1| below this and full coverage: the camera is the grid


def resample_matrix(n_out: int, n_in: int, mode: str) -> np.ndarray:
    """(n_out, n_in) linear operator. 'area': each output pixel averages the input it overlaps (box integration);
    'linear': each output pixel centre interpolates the two nearest input centres (edges clamped)."""
    if mode == "area":
        s = n_in / n_out
        edges_out = np.arange(n_out + 1) * s
        lo, hi = edges_out[:-1, None], edges_out[1:, None]
        j = np.arange(n_in)[None, :]
        overlap = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
        return overlap / s
    if mode == "linear":
        x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        x = np.clip(x, 0.0, n_in - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        t = x - i0
        M = np.zeros((n_out, n_in))
        M[np.arange(n_out), i0] += 1.0 - t
        M[np.arange(n_out), i1] += t
        return M
    raise ValueError(mode)


def _apply(M_y: np.ndarray, img: np.ndarray, M_x: np.ndarray) -> np.ndarray:
    """M_y img M_x^T per channel. The operators have two or three non-zeros per row, so they are applied as sparse
    matrices (a dense product of a 1920-pixel camera frame took over a second)."""
    from scipy import sparse
    sy, sx = sparse.csr_matrix(M_y), sparse.csr_matrix(M_x)

    def one(a):
        return np.asarray((sx @ (sy @ a).T).T, np.float32)
    if img.ndim == 2:
        return one(np.asarray(img, np.float32))
    return np.stack([one(np.asarray(img[..., c], np.float32)) for c in range(img.shape[2])], -1)


def covered_box(h: int, w: int, coverage) -> tuple:
    """(y0, y1, x0, x1) of the centred region of the grid a camera covers (fractions of the grid's tangent extent)."""
    cw, ch = float(min(1.0, coverage[0])), float(min(1.0, coverage[1]))
    ww, hh = max(1, int(round(w * cw))), max(1, int(round(h * ch)))
    x0, y0 = (w - ww) // 2, (h - hh) // 2
    return y0, y0 + hh, x0, x0 + ww


def camera_shape(h: int, w: int, r: float, coverage) -> tuple:
    """The camera's own pixel count over the covered region."""
    y0, y1, x0, x1 = covered_box(h, w, coverage)
    return max(1, int(round((y1 - y0) / r))), max(1, int(round((x1 - x0) / r)))


def emulate(img: np.ndarray, r: float, coverage, sensor) -> tuple[np.ndarray, np.ndarray]:
    """img: the grid image (H x W or H x W x C); sensor: f(camera-pixel image) -> image of the same shape.
    Returns (grid image, coverage mask)."""
    h, w = img.shape[:2]
    full = coverage[0] >= 1.0 and coverage[1] >= 1.0
    mask = np.zeros((h, w), np.float32)
    y0, y1, x0, x1 = covered_box(h, w, coverage)
    mask[y0:y1, x0:x1] = 1.0
    if full and abs(r - 1.0) < UNITY:
        return np.asarray(sensor(img), np.float32), mask
    sub = img[y0:y1, x0:x1]
    nh, nw = camera_shape(h, w, r, coverage)
    down = "area" if r > 1.0 else "linear"
    up = "area" if r < 1.0 else "linear"
    cam = _apply(resample_matrix(nh, y1 - y0, down), sub, resample_matrix(nw, x1 - x0, down))
    cam = np.asarray(sensor(cam), np.float32)
    back = _apply(resample_matrix(y1 - y0, nh, up), cam, resample_matrix(x1 - x0, nw, up))
    out = np.zeros(img.shape[:2] + back.shape[2:], np.float32)
    out[y0:y1, x0:x1] = back
    return out, mask


def grid_sigma(sigma_camera_px: float, r: float, sigma0: float) -> float:
    """A PSF of sigma camera pixels spans sigma x r grid pixels; the render already applied sigma0 (the floor)."""
    return max(float(sigma0), float(sigma_camera_px) * float(r))
