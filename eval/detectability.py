"""Matched-filter detectability d' of a target in one camera's frame (docs/privileged_teacher_case.md, section 3).

The box-against-ring contrast-to-noise (tools/m1_det2.cnr) is not noise-referenced for small targets: its 2-px ring
holds the target's own blur and the background, so it barely changes when the sensor noise is removed (2026-10-18
check). d' is the textbook measure of how well an ideal linear detector can find a known target in its background:

    T  = the target's exact signal in the frame (the frame with minus the frame without the target, same noise draw),
         zero-mean and scaled to unit norm over the box and a margin
    d' = T . (P - A) / std over positions of (T correlated with the target-free frame A)

The spread over positions holds the sensor noise and the scene's own structure (clutter). Correlating with
(A - A2) / sqrt(2), where A2 is the target-free frame with another noise draw, cancels the scene and leaves the sensor
noise alone (temporal and fixed pattern): noise_share = that variance / the total.

Thresholds, a definition: a Gaussian matched filter detects with probability 0.5 when d' equals its threshold, and
0.005-0.05 false alarms per 640 x 512 frame over about 2e4-3e5 independent positions need 4.9-5.6 noise widths, so
D (detectable alone) is d' >= 5. F (2-5) is below that, yet within reach of fusion or of averaging a few frames. N (< 2)
is too faint for either.
"""
from __future__ import annotations

import numpy as np

DETECTABLE, FAINT = 5.0, 2.0
NONE = dict(d=0.0, d_noise=0.0, noise_share=None)


def dprime(P, A, A2, box, pad=6, win=40, cover=None) -> dict:
    """P, A: the frame with and without the target (H x W x C), same noise draw; A2: target-free, another noise draw;
    box: (x0, y0, x1, y1) in pixels; cover: (y0, y1, x0, x1), the part of the frame the camera sees. Template and noise
    window stay inside it, so the edge of a zoom camera's view is not counted as clutter. Returns d, d_noise (sensor
    noise only) and noise_share."""
    from scipy.signal import correlate
    H, W = P.shape[:2]
    cy0, cy1, cx0, cx1 = cover if cover is not None else (0, H, 0, W)
    x0, y0, x1, y1 = box
    i0, i1 = max(cy0, int(np.floor(y0)) - pad), min(cy1, int(np.ceil(y1)) + pad)
    j0, j1 = max(cx0, int(np.floor(x0)) - pad), min(cx1, int(np.ceil(x1)) + pad)
    if i1 - i0 < 1 or j1 - j0 < 1:
        return dict(NONE)
    sig = (np.asarray(P, np.float64) - np.asarray(A, np.float64))[i0:i1, j0:j1]
    T = sig - sig.mean()
    n = float(np.sqrt((T ** 2).sum()))
    if n < 1e-9:
        return dict(NONE)
    T /= n
    S = float((T * sig).sum())
    a0, a1, b0, b1 = max(cy0, i0 - win), min(cy1, i1 + win), max(cx0, j0 - win), min(cx1, j1 + win)

    def spread(X):
        R = correlate(np.asarray(X, np.float64)[a0:a1, b0:b1], T, mode="valid")
        return float(R.std()) if R.size > 1 else float("inf")
    tot, noi = spread(A), spread((np.asarray(A, np.float64) - np.asarray(A2, np.float64)) / np.sqrt(2.0))
    return dict(d=S / max(tot, 1e-12), d_noise=S / max(noi, 1e-12), noise_share=min(1.0, (noi / max(tot, 1e-12)) ** 2))


def d_after(dp: dict, frames: int) -> float:
    """d' after averaging `frames` frames of a static scene: only the sensor-noise share of the spread averages
    (d'_N = 1 / sqrt(clutter^2 + noise^2 / N), with both in units of the signal)."""
    if dp["d"] <= 0:
        return 0.0
    noise = (1.0 / dp["d_noise"]) ** 2
    clutter = max(0.0, (1.0 / dp["d"]) ** 2 - noise)
    return float(1.0 / np.sqrt(clutter + noise / frames))


def signal_class(d: float) -> str:
    return "D" if d >= DETECTABLE else ("F" if d >= FAINT else "N")
