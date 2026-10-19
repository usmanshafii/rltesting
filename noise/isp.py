"""What a camera does to its sensor data before the video leaves it, applied by the data loader. NumPy + SciPy.

The sensor models (noise/rgb_noise.py, noise/lwir_fpn.py) produce raw sensor data; real cameras then correct and
display-map it, and a detector trained on synthetic data meets the camera's output, not its sensor. Physically
modelled camera effects on synthetic training data raised detection on real images by 4-7 AP (Carlson et al., ECCV
Workshops 2018, Tables 1-3; plain additive Gaussian noise did not do this). Each stage below follows a manufacturer's
description of its own camera (S1) or a peer-reviewed model (S2); docs/sensor_noise.md has the table, and the
parameter ranges are in configs/sensors.yaml `camera_pipelines`.

Thermal cameras. FLIR Boson product specification 102-2013-40 (secs. 5, 6.6, 6.8, 11.3, 11.4): defective pixels
(< 1-2 % of the array, single pixels, small clusters, a few bad rows or columns) are replaced from their neighbours;
filters reduce column, row and random spatial noise and low-frequency shading ("side-to-side gradients and
circularly-symmetric gradients"); image non-uniformities (INUs) are localised blobs, type A of radius < 10 px and type
B > 10 px; the AGC is a variant of histogram equalization (information-based weighting of detail, plateau, tail
rejection, max gain, linear percent, ACE) with digital detail enhancement (DDE), or linear. Cooled cameras: defect
replacement, AGC / dynamic-range compression and spatial / temporal noise reduction (SCD Mini Blackbird datasheet).
RGB zoom block. Sony IMX327 (Bayer RGB, 10 / 12-bit ADC) and FCB-EV7520 technical manual (AE gain to 50 dB, noise
reduction Off and levels 1-5 acting more strongly at high gain, edge enhancement in 16 steps, gamma standard / straight).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# ============================================================================ defects
def defect_mask(rng, h: int, w: int, frac: float, bad_lines: int = 0, cluster_prob: float = 0.15) -> np.ndarray:
    """Defective pixels: `frac` of the array, some as 2-pixel clusters (a camera spec allows small clusters but no 3x3
    cluster, Boson Tables 14-15), plus `bad_lines` whole rows or columns."""
    m = np.zeros((h, w), bool)
    n = int(round(frac * h * w))
    if n:
        idx = rng.choice(h * w, n, replace=False)
        y, x = np.divmod(idx, w)
        m[y, x] = True
        pair = rng.uniform(size=n) < cluster_prob                   # a neighbour of some defects is defective too
        dy, dx = rng.integers(-1, 2, n), rng.integers(-1, 2, n)
        m[np.clip(y[pair] + dy[pair], 0, h - 1), np.clip(x[pair] + dx[pair], 0, w - 1)] = True
    for _ in range(int(bad_lines)):
        if rng.uniform() < 0.5:
            m[:, int(rng.integers(w))] = True
        else:
            m[int(rng.integers(h)), :] = True
    return m


def replace_bad_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bad-pixel replacement: each defective pixel becomes the mean of its good 3x3 neighbours (a bad column takes the
    columns beside it); a pixel with no good neighbour takes the frame median."""
    if not mask.any():
        return img
    good = (~mask).astype(np.float32)
    k = np.ones((3, 3), np.float32)
    s = ndimage.convolve(np.where(mask, 0.0, img).astype(np.float32), k, mode="nearest")
    c = ndimage.convolve(good, k, mode="nearest")
    out = img.copy()
    fill = np.where(c > 0, s / np.maximum(c, 1e-6), float(np.median(img[~mask])) if (~mask).any() else 0.0)
    out[mask] = fill[mask]
    return out


# ============================================================================ smooth and localised non-uniformity
def shading_field(rng, h: int, w: int) -> np.ndarray:
    """Low-frequency shading, peak |value| 1: a side-to-side gradient plus a circularly symmetric term about a point
    near the centre (lens roll-off, housing heating, narcissus), in random proportion and sign."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    ang = rng.uniform(0, 2 * np.pi)
    grad = np.cos(ang) * u + np.sin(ang) * v
    cx, cy = rng.uniform(-0.2, 0.2, 2)
    radial = (u - cx) ** 2 + (v - cy) ** 2
    radial = radial - radial.mean()
    a = rng.uniform(0, 1)
    f = a * grad / max(float(np.abs(grad).max()), 1e-6) + (1 - a) * np.sign(rng.uniform(-1, 1)) * radial / max(float(np.abs(radial).max()), 1e-6)
    return (f / max(float(np.abs(f).max()), 1e-6)).astype(np.float32)


def inu_blobs(rng, h: int, w: int, n_small: int, n_large: int) -> np.ndarray:
    """Image non-uniformities: Gaussian blobs of random sign, type A of radius < 10 px and type B > 10 px (Boson sec.
    11.4); peak |value| about 1."""
    out = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for n, (r_lo, r_hi) in ((int(n_small), (2.0, 9.0)), (int(n_large), (10.0, 40.0))):
        for _ in range(n):
            r = rng.uniform(r_lo, r_hi)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            s = r / 2.0                                    # radius ~ 2 sigma
            out += np.sign(rng.uniform(-1, 1)) * rng.uniform(0.5, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return out


# ============================================================================ thermal AGC
def agc_linear(x: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip((x - lo) / max(float(hi - lo), 1.0), 0.0, 1.0).astype(np.float32)


def agc_histogram(x: np.ndarray, p: dict) -> np.ndarray:
    """Histogram-based AGC with detail enhancement, after the Boson AGC (product specification sec. 6.8):

      * the image is split into a low-pass part (Gaussian of `smoothing_px`) and the high-pass rest;
      * information-based equalization (`ibe`): pixels weigh 1 + |high-pass| / its median in the histogram, so detail
        gets more grey shades than a uniform background;
      * tail rejection (`tail`, %): the transfer function ignores that share of outliers at each end;
      * plateau (`plateau`, fraction of the histogram): no bin counts more than this;
      * max gain (`max_gain`): the transfer slope is at most this many output grey shades per input count, so a
        nearly uniform sky is not stretched until its noise fills the display;
      * linear percent (`linear`): this share of the transfer function is linear between the tails;
      * ACE (`ace`): exponent 1 / ace on the transfer function (ace < 1 darkens a white-hot image);
      * DDE (`dde`): the high-pass part is added back times dde and the local slope (dde < 1 attenuates, > 1 amplifies).
    Returns an image in [0, 1] (the 8-bit video is formed by quantisation later)."""
    x = np.asarray(x, np.float32)
    lp = ndimage.gaussian_filter(x, p["smoothing_px"])
    hp = x - lp
    lo, hi = np.percentile(x, [p["tail"], 100.0 - p["tail"]])
    span = max(float(hi - lo), 1.0)
    nb = int(min(4096, max(16, np.ceil(span))))                      # ~1 count per bin, at most 4096 bins
    edges = np.linspace(lo, hi, nb + 1)
    wts = None
    if p["ibe"]:
        a = np.abs(hp)
        wts = (1.0 + a / max(float(np.median(a)), 1e-6)).ravel()
    hist = np.histogram(np.clip(lp, lo, hi).ravel(), bins=edges, weights=wts)[0].astype(np.float64)
    hist = np.minimum(hist, p["plateau"] * hist.sum())                # plateau
    out_span = min(1.0, p["max_gain"] * span / 255.0)                 # max gain caps the output range ...
    cap = p["max_gain"] * (span / nb) / 255.0 / out_span              # ... and every bin's slope (grey shades / count)
    for _ in range(4):                                                # clipping lowers the total: iterate
        hist = np.minimum(hist, cap * hist.sum())
    cdf = np.concatenate([[0.0], np.cumsum(hist)])
    t = (1.0 - p["linear"]) * cdf / max(cdf[-1], 1e-12) + p["linear"] * np.linspace(0.0, 1.0, nb + 1)
    t = 0.5 + (t - 0.5) * out_span                                    # a low-contrast scene stays mid-grey
    t = np.clip(t, 0.0, 1.0) ** (1.0 / p["ace"])
    y = np.interp(lp, edges, t)
    slope = np.gradient(t, edges)
    y = y + p["dde"] * np.interp(lp, edges, slope) * hp
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def sample_agc(rng, cfg: dict) -> dict:
    """One AGC setting for a frame, from the ranges in configs/sensors.yaml camera_pipelines.<camera>.agc."""
    u = lambda k: float(rng.uniform(*cfg[k]))  # noqa: E731
    if rng.uniform() < cfg["histogram_prob"]:
        return dict(mode="histogram", plateau=u("plateau"), tail=u("tail_pct"), max_gain=u("max_gain"), linear=u("linear"),
                    ace=u("ace"), dde=u("dde"), ibe=bool(rng.uniform() < cfg["ibe_prob"]), smoothing_px=u("smoothing_px"))
    return dict(mode="linear", lo_pct=u("linear_lo_pct"), hi_pct=u("linear_hi_pct"))


def apply_agc(x: np.ndarray, p: dict) -> np.ndarray:
    return agc_histogram(x, p) if p["mode"] == "histogram" else agc_linear(x, p["lo_pct"], p["hi_pct"])


# ============================================================================ RGB
def mosaic_rggb(img: np.ndarray) -> np.ndarray:
    """Bayer RGGB colour-filter sampling: one colour per pixel."""
    h, w, _ = img.shape
    raw = np.empty((h, w), img.dtype)
    raw[0::2, 0::2] = img[0::2, 0::2, 0]
    raw[0::2, 1::2] = img[0::2, 1::2, 1]
    raw[1::2, 0::2] = img[1::2, 0::2, 1]
    raw[1::2, 1::2] = img[1::2, 1::2, 2]
    return raw


def demosaic_bilinear(raw: np.ndarray) -> np.ndarray:
    """Bilinear demosaicking of an RGGB mosaic (the reference interpolation of Brooks et al. CVPR 2019's pipeline)."""
    h, w = raw.shape
    m = np.zeros((h, w, 3), np.float32)
    m[0::2, 0::2, 0] = 1
    m[0::2, 1::2, 1] = 1
    m[1::2, 0::2, 1] = 1
    m[1::2, 1::2, 2] = 1
    k_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32)
    k_g = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], np.float32)
    out = np.empty((h, w, 3), np.float32)
    for c, k in ((0, k_rb), (1, k_g), (2, k_rb)):
        out[..., c] = ndimage.convolve(raw * m[..., c], k, mode="mirror")
    return out


def tukey_lambda(rng, lam: float, scale: float, shape) -> np.ndarray:
    """Tukey-lambda samples (inverse CDF), the read-noise distribution of Wei et al. (CVPR 2020, eq. 4): Gaussian-like at
    lam = 0.14, heavier-tailed below. Scaled so that the standard deviation is `scale`."""
    u = np.clip(rng.uniform(size=shape), 1e-7, 1 - 1e-7)
    if abs(lam) < 1e-6:
        q = np.log(u / (1 - u))
        sd = np.pi / np.sqrt(3.0)
    else:
        q = (u ** lam - (1 - u) ** lam) / lam
        from math import gamma
        if lam > -0.5:                                    # variance exists: 2/l^2 (1/(1+2l) - G(l+1)^2/G(2l+2))
            sd = np.sqrt(2.0 / lam ** 2 * (1.0 / (1 + 2 * lam) - gamma(lam + 1) ** 2 / gamma(2 * lam + 2)))
        else:
            sd = float(np.std(q))
    return (q * (scale / sd)).astype(np.float32)


def noise_reduction(img: np.ndarray, sigma: float) -> np.ndarray:
    """Spatial noise reduction as a Gaussian of `sigma` camera pixels per channel (0: off)."""
    if sigma <= 0.05:
        return img
    if img.ndim == 3:
        return np.stack([ndimage.gaussian_filter(img[..., c], sigma) for c in range(img.shape[2])], -1).astype(np.float32)
    return ndimage.gaussian_filter(img, sigma).astype(np.float32)


def sharpen(img: np.ndarray, amount: float, radius: float = 1.0) -> np.ndarray:
    """Edge enhancement (unsharp mask): img + amount (img - blur)."""
    if amount <= 0:
        return img
    blur = noise_reduction(img, radius)
    return (img + amount * (img - blur)).astype(np.float32)


def chromatic_aberration(img: np.ndarray, k_r: float, k_b: float) -> np.ndarray:
    """Lateral chromatic aberration as Carlson et al. (2018) model it: the red and blue planes scaled about the image
    centre by 1 + k relative to green."""
    if abs(k_r) < 1e-6 and abs(k_b) < 1e-6:
        return img
    h, w, _ = img.shape
    out = img.copy()
    for c, k in ((0, k_r), (2, k_b)):
        sy, sx = _scale_operator(h, k), _scale_operator(w, k)            # separable: a scale about the centre
        out[..., c] = np.asarray((sx @ (sy @ img[..., c]).T).T, np.float32)
    return out


def _scale_operator(n: int, k: float):
    """Sparse n x n linear-interpolation operator that scales a line of pixels by 1 + k about its centre."""
    from scipy import sparse
    c = (n - 1) / 2.0
    x = np.clip(c + (np.arange(n) - c) / (1.0 + k), 0.0, n - 1.0)
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    t = x - i0
    rows = np.repeat(np.arange(n), 2)
    cols = np.stack([i0, i0 + 1], 1).ravel()
    vals = np.stack([1.0 - t, t], 1).ravel()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def gamma_encode(x: np.ndarray, mode: str) -> np.ndarray:
    """Display gamma: 'standard' = the ITU-R BT.709 transfer curve (video cameras), 'straight' = power 1/2.2."""
    x = np.clip(x, 0.0, 1.0)
    if mode == "standard":
        return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099).astype(np.float32)
    return np.power(x, 1 / 2.2).astype(np.float32)
