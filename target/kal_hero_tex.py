"""Surface textures for the high-detail KAL presentation model (pure NumPy + Pillow, no Blender).

One 4096 x 4096 atlas per airframe variant, drawn in metric units from the same derived geometry as
kal_geometry.py, so seams and markings land where the mesh is:

    albedo (sRGB), roughness (linear), normal (tangent space, OpenGL +Y), all UV-mapped by kal_hero.py

Atlas regions (pixels, row 0 at the top):
    WU  wing upper surface, planform projection   x (aft, span units) -> rows,  y -> columns   [0, 2048) x [0, 2048)
    WL  wing lower surface, same projection                                                    [2048, 4096) x [0, 2048)
    FU  fuselage, unrolled: axial station -> columns, angle from the bottom -> rows            [0, 4096) x [2048, 3328)
    FS  starboard fin outer, FP port fin outer, FI fin inner faces: side projection            rows [3328, 4096)

Status of the content (evidence levels as in configs/target.yaml):
    markings (KAL wing text, fin wordmark, flag, white band)  B - P1/P3/P8, positions from target.yaml
    satin dark finish                                          B - P1/P3/P8 (look="photo": matte, set in
                                                                   kal_hero's shader; wider structural seams)
    seams, hatches, fasteners, elevon gaps, wear               C - presentation detail; the photos show seams
                                                                   and a fastened forebody hatch (P3) but not
                                                                   their exact layout
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

N = 4096
WING = 2048                      # pixels per span unit in the wing regions
FUS_ROWS = (2048, 3328)
FIN_ROWS = (3328, 4096)
FIN_PX = 1600.0                  # pixels per span unit in the fin regions
FIN_COLS = dict(FS=0, FP=700, FI=1400)


# ============================================================================ coordinate maps
def wing_px(g, x, y, lower=False):
    """Internal (x aft, y starboard) -> atlas pixel (col, row)."""
    return (np.asarray(y) + 0.5) * WING + (WING if lower else 0), np.asarray(x) / (g["te_x"] * 1.02) * WING


def fus_extent(g):
    return g["nose_x"], g["eng0"]


def fus_px(g, x, a):
    """Axial station x and ring angle a (radians, a = -pi/2 at the bottom, increasing towards
    starboard, top, port) -> atlas pixel."""
    x0, x1 = fus_extent(g)
    v = (np.asarray(a) + math.pi / 2) / (2 * math.pi)
    return (np.asarray(x) - x0) / (x1 - x0) * N, FUS_ROWS[0] + v * (FUS_ROWS[1] - FUS_ROWS[0])


def fin_origin(g):
    return g["te_x"] - 0.20, g["wing_z"] + 0.20          # (x0, z_top) of the projection window


def fin_px(g, x, z, region="FS"):
    x0, zt = fin_origin(g)
    return FIN_COLS[region] + (np.asarray(x) - x0) * FIN_PX, FIN_ROWS[0] + (zt - np.asarray(z)) * FIN_PX


# ============================================================================ drawing primitives
class Canvas:
    def __init__(self):
        self.h = np.zeros((N, N), np.float32)            # height, millimetres
        self.alb = np.zeros((N, N, 3), np.float32)       # linear albedo
        self.rough = np.zeros((N, N), np.float32)

    def _box(self, xs, ys, pad):
        x0, x1 = int(max(0, math.floor(min(xs) - pad))), int(min(N, math.ceil(max(xs) + pad)))
        y0, y1 = int(max(0, math.floor(min(ys) - pad))), int(min(N, math.ceil(max(ys) + pad)))
        return x0, x1, y0, y1

    def groove(self, p0, p1, width_px, depth_mm, dark=0.55, rough=0.62):
        """Panel seam along a segment: a soft V groove, darker and rougher (dirt in the gap)."""
        (ax, ay), (bx, by) = p0, p1
        x0, x1, y0, y1 = self._box((ax, bx), (ay, by), 3 * width_px + 2)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / max(L2, 1e-9), 0, 1)
        d = np.hypot(xx - (ax + t * dx), yy - (ay + t * dy))
        prof = np.exp(-(d / max(width_px, 0.5)) ** 2)
        self.h[y0:y1, x0:x1] = np.minimum(self.h[y0:y1, x0:x1], -depth_mm * prof)
        self.alb[y0:y1, x0:x1] *= (1 - (1 - dark) * prof)[..., None]
        self.rough[y0:y1, x0:x1] = np.maximum(self.rough[y0:y1, x0:x1], rough * prof)

    def polyline(self, pts, width_px, depth_mm, closed=False, **kw):
        pts = list(pts)
        for a, b in zip(pts, pts[1:] + ([pts[0]] if closed else [])):
            self.groove(a, b, width_px, depth_mm, **kw)

    def fastener(self, c, r_px, depth_mm=0.15):
        """Countersunk screw head: a shallow dish with a slightly lighter, smoother metal face."""
        cx, cy = c
        x0, x1, y0, y1 = self._box((cx,), (cy,), 2 * r_px + 2)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        d = np.hypot(xx - cx, yy - cy) / max(r_px, 0.5)
        ring = np.exp(-((d - 1.0) / 0.28) ** 2)
        face = np.clip(1.2 - d, 0, 1)
        self.h[y0:y1, x0:x1] += -depth_mm * ring + 0.03 * face
        self.alb[y0:y1, x0:x1] = self.alb[y0:y1, x0:x1] * (1 - 0.5 * face[..., None]) + 0.5 * face[..., None] * np.array([0.09, 0.09, 0.095])
        self.rough[y0:y1, x0:x1] = self.rough[y0:y1, x0:x1] * (1 - 0.5 * face) + 0.5 * face * 0.32

    def fasteners_along(self, pts, spacing_px, r_px, offset_px=0.0, closed=False):
        pts = list(pts) + ([pts[0]] if closed else [])
        carry = spacing_px / 2
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            L = math.hypot(bx - ax, by - ay)
            if L < 1e-6:
                continue
            nx, ny = -(by - ay) / L, (bx - ax) / L
            s = carry
            while s < L:
                t = s / L
                self.fastener((ax + t * (bx - ax) + offset_px * nx, ay + t * (by - ay) + offset_px * ny), r_px)
                s += spacing_px
            carry = s - L


def rounded_rect(cx, cy, w, h, r, n=6):
    """Outline points of a rounded rectangle (pixel space)."""
    r = min(r, w / 2, h / 2)
    pts = []
    for qx, qy, a0 in ((cx + w / 2 - r, cy - h / 2 + r, -90), (cx + w / 2 - r, cy + h / 2 - r, 0),
                       (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((qx + r * math.cos(a), qy + r * math.sin(a)))
    return pts


def _value_noise(shape, cell, rng, octaves=4):
    """Smooth fractal noise in [0, 1] (bilinear value noise), for mottling and wear."""
    H, W = shape
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cell / (2 ** o)))
        gh, gw = H // c + 2, W // c + 2
        grid = rng.random((gh, gw)).astype(np.float32)
        yi = np.arange(H, dtype=np.float32) / c
        xi = np.arange(W, dtype=np.float32) / c
        y0, x0 = yi.astype(int), xi.astype(int)
        fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = grid[y0][:, x0]
        b = grid[y0][:, x0 + 1]
        c_ = grid[y0 + 1][:, x0]
        d = grid[y0 + 1][:, x0 + 1]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c_ * (1 - fx) + d * fx) * fy)
        tot += amp
        amp *= 0.5
    return out / tot


def _paste_decal(cv, atlas, region, dst_px, uv_of_px, white=0.72):
    """Paint a decal into the albedo: for every destination pixel in the window, (u, v) from
    uv_of_px(cols, rows); alpha from the decal atlas region; paint is a matte off-white."""
    u0, v0, u1, v1 = region
    x0, x1, y0, y1 = dst_px
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(N, int(x1) + 1), min(N, int(y1) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    u, v = uv_of_px(xx, yy)
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    Ha, Wa = atlas.shape[:2]
    au = (u0 + np.clip(u, 0, 1) * (u1 - u0)) * (Wa - 1)
    av = (1 - (v0 + np.clip(v, 0, 1) * (v1 - v0))) * (Ha - 1)          # image rows run top-down
    a = _bilinear(atlas[..., 3], au, av) * inside
    rgb = np.stack([_bilinear(atlas[..., c], au, av) for c in range(3)], -1)
    col = np.clip(rgb, 0, 1) ** 2.2 * white
    cv.alb[y0:y1, x0:x1] = cv.alb[y0:y1, x0:x1] * (1 - a[..., None]) + col * a[..., None]
    cv.rough[y0:y1, x0:x1] = cv.rough[y0:y1, x0:x1] * (1 - a) + 0.55 * a
    cv.h[y0:y1, x0:x1] += 0.04 * a                                     # paint film thickness


def _bilinear(img, x, y):
    H, W = img.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


# ============================================================================ the atlas
BASE = np.array([0.0150, 0.0156, 0.0168], np.float32)       # satin charcoal, linear (P1/P3/P8: near black, cool)


def build_atlas(g: dict, decal_atlas: np.ndarray | None, decal_regions: dict, seed: int = 7,
                look: str = "survey") -> dict:
    """Returns dict(albedo=uint8 HxWx3 sRGB, rough=uint8, normal=uint8 HxWx3). look="photo" lays the
    fin wordmark along the fin and adds the spine emblem (P1); decal_regions then come from
    assets/decals/hero_decal_atlas.json."""
    rng = np.random.default_rng(seed)
    cv = Canvas()
    span_mm = g["span_m"] * 1000.0
    mm_wing = span_mm / WING * g["te_x"] * 1.02          # mm per pixel along rows; columns: span_mm / WING
    # ---- base finish: mottled satin charcoal, slightly dustier underneath and on the fuselage belly
    mott = _value_noise((N, N), 512, rng, 5)
    fine = _value_noise((N, N), 24, rng, 3)
    cv.alb[:] = BASE * (0.86 + 0.28 * mott[..., None]) * (0.96 + 0.08 * fine[..., None])
    cv.rough[:] = 0.36 + 0.10 * mott + 0.05 * (fine - 0.5)
    cv.h[:] = 0.012 * (fine - 0.5)                         # orange-peel paint texture
    dust = _value_noise((N, N), 256, rng, 4)
    lower = np.zeros((N, N), np.float32)
    lower[0:2048, 2048:4096] = 1.0
    rows = np.arange(FUS_ROWS[0], FUS_ROWS[1])
    v = (rows - FUS_ROWS[0]) / (FUS_ROWS[1] - FUS_ROWS[0])
    lower[FUS_ROWS[0]:FUS_ROWS[1]] = np.clip(1 - np.abs(v - 0.0) * 4, 0, 1)[:, None] + np.clip(1 - np.abs(v - 1.0) * 4, 0, 1)[:, None]
    dmask = np.clip((dust - 0.45) * 2.2, 0, 1) * lower
    del dust, lower, mott
    cv.alb *= (1 - 0.25 * dmask[..., None])
    cv.alb += 0.25 * dmask[..., None] * np.array([0.050, 0.046, 0.041], np.float32)
    cv.rough += 0.12 * dmask
    del dmask

    W_SEAM = 0.9 / (span_mm / WING)                      # ~0.9 mm half-width seams (pixels)
    D_SEAM = 0.35
    # P1 shows a few structural lines (root joint, main spar line, elevon cuts) at a few mm wide
    K_MAIN = 3.3 if look == "photo" else 1.0

    # ---- wing, both surfaces: root fairing seam, spar lines, elevon gaps, access panels, LE seam
    te, ts, ec = g["te_x"], g["tan_sweep"], g["elevon_c"]
    b0, b1, b2 = g["elevon_breaks"]
    for lower_side in (False, True):
        P = lambda x, y: tuple(float(c) for c in wing_px(g, x, y, lower_side))  # noqa: E731
        for s in (1, -1):
            # wing-to-body joint, just outside the fuselage tube
            yr = s * (g["R_tube"] + 0.006)
            cv.polyline([P(abs(yr) * ts + 0.01, yr), P(te - 0.004, yr)], W_SEAM * K_MAIN, D_SEAM * K_MAIN ** 0.5)
            # leading-edge strip and main spar line (chord fractions), root to tip
            for frac in (0.035, 0.28, 0.62):
                pts = [P(abs(y) * ts + frac * (te - abs(y) * ts), s * y) for y in np.linspace(g["R_tube"] + 0.006, 0.497, 24)]
                k = K_MAIN if frac == 0.28 else 0.8
                cv.polyline(pts, W_SEAM * k, D_SEAM * 0.8 * k ** 0.5, dark=0.4 if k > 1 else 0.55)
            # elevon: hinge line and the two side cuts, a real gap (deeper, darker)
            def hinge_x(y):
                return te - ec
            for ya, yb in ((b0, b1), (b1 + 0.004, b2 - 0.002)):
                cv.polyline([P(hinge_x(ya), s * ya), P(hinge_x(yb), s * yb)], 1.6 * W_SEAM * K_MAIN ** 0.75, 1.2, dark=0.25, rough=0.7)
                cv.polyline([P(hinge_x(ya), s * ya), P(te, s * ya)], 1.6 * W_SEAM * K_MAIN ** 0.75, 1.2, dark=0.25, rough=0.7)
                cv.polyline([P(hinge_x(yb), s * yb), P(te, s * yb)], 1.6 * W_SEAM * K_MAIN ** 0.75, 1.2, dark=0.25, rough=0.7)
                # hinge fasteners just ahead of the gap
                cv.fasteners_along([P(hinge_x(ya) - 0.012, s * ya), P(hinge_x(yb) - 0.012, s * yb)], 70, 2.2)
            # access panels near the root: two rounded rectangles with screws (upper) / one large (lower)
            # placed between the spar lines (0.28 and 0.62 chord) so no seam crosses a panel
            panels = [(0.595, 0.16, 0.09, 0.07), (0.68, 0.25, 0.08, 0.055)] if not lower_side else [(0.62, 0.20, 0.10, 0.07)]
            for xc, yc, lx, ly in panels:
                cx, cy = P(xc, s * yc)
                w, h = ly * WING, lx * WING / (te * 1.02)
                pts = rounded_rect(cx, cy, w, h, 0.012 * WING)
                cv.polyline(pts, W_SEAM, D_SEAM, closed=True)
                cv.fasteners_along(rounded_rect(cx, cy, w - 18, h - 18, 0.008 * WING), 42, 2.3, closed=True)
    # wing markings: "KAL" on both upper panels (P1, P3), as in kal_geometry.build_mesh
    if decal_atlas is not None and "wing_text" in decal_regions:
        cx_t = g["wing_text_center_over_root"] * te
        L, H = g["wing_text_size"]
        for s in (1, -1):
            cy_t = s * g["wing_text_y"]
            c0, r0 = wing_px(g, cx_t - H / 2, cy_t - L / 2)
            c1, r1 = wing_px(g, cx_t + H / 2, cy_t + L / 2)

            def uv(xx, yy, cy_t=cy_t, cx_t=cx_t, L=L, H=H):
                y = xx / WING - 0.5
                x = yy / WING * te * 1.02
                return (y - (cy_t - L / 2)) / L, ((cx_t + H / 2) - x) / H
            _paste_decal(cv, decal_atlas, decal_regions["wing_text"], (c0, c1, r0, r1), uv)

    # ---- fuselage
    nx, x1f = fus_extent(g)
    PF = lambda x, a: tuple(float(c) for c in fus_px(g, x, a))  # noqa: E731
    a_lo, a_hi = -math.pi / 2, 3 * math.pi / 2
    circ = lambda x: [PF(x, a) for a in np.linspace(a_lo, a_hi, 64)]  # noqa: E731
    mm_fus = (x1f - nx) * span_mm / N
    Wf = 0.9 / mm_fus
    stations = [nx + g["dome_len"] * 0.98, g["collar_x"] + g["collar_step_len"] + 0.004,
                g["collar_x"] + g["collar_step_len"] + (g["pod0"] - g["collar_x"]) * 0.36,
                g["pod0"] - 0.004, g["pod1"], g["neck_x"] - 0.03]
    for xs in stations:
        cv.polyline(circ(xs), Wf, D_SEAM)
    # radome / nose-cap fasteners
    cv.fasteners_along(circ(nx + g["dome_len"] * 0.98 + 0.012), 64, 2.4)
    # forebody top hatch with fasteners (P3: a fastened panel and a round cap on the upper forebody)
    xh0 = g["flag_x_over_root"] * g["te_x"] + g["flag_size"][0] / 2 + 0.02     # aft of the flag
    xh1 = g["band0"] - 0.02
    a_top = math.pi / 2
    if xh1 > xh0 + 0.04:
        c0x, c0y = PF((xh0 + xh1) / 2, a_top)
        wpx = (xh1 - xh0) / (x1f - nx) * N
        hpx = 0.9 * (FUS_ROWS[1] - FUS_ROWS[0]) / (2 * math.pi)     # ~0.9 rad around the top
        cv.polyline(rounded_rect(c0x, c0y, wpx, hpx, 22), Wf, D_SEAM, closed=True)
        cv.fasteners_along(rounded_rect(c0x, c0y, wpx - 26, hpx - 26, 14), 46, 2.6, closed=True)
        # round filler cap aft of the hatch
        cx_c, cy_c = PF(xh1 + 0.012, a_top - 0.25)
        cv.polyline([(cx_c + 16 * math.cos(t), cy_c + 16 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 25)], Wf, D_SEAM)
    # belly access panel under the straight tube and a longitudinal belly seam
    tb0 = g["collar_x"] + g["collar_step_len"]
    cv.polyline([PF(tb0 + 0.01, a_lo + 0.001), PF(g["pod0"] - 0.01, a_lo + 0.001)], Wf, D_SEAM)
    cv.polyline([PF(tb0 + 0.01, a_hi - 0.001), PF(g["pod0"] - 0.01, a_hi - 0.001)], Wf, D_SEAM)
    bx0, bx1 = tb0 + 0.05, tb0 + 0.05 + (g["pod0"] - tb0) * 0.45
    for ab in (a_lo + 0.42, a_hi - 0.42):
        cv.polyline([PF(bx0, ab), PF(bx1, ab)], Wf, D_SEAM)
    cv.polyline([PF(bx0, a_lo + 0.001), PF(bx0, a_lo + 0.42)], Wf, D_SEAM)
    cv.polyline([PF(bx1, a_lo + 0.001), PF(bx1, a_lo + 0.42)], Wf, D_SEAM)
    cv.polyline([PF(bx0, a_hi - 0.42), PF(bx0, a_hi - 0.001)], Wf, D_SEAM)
    cv.polyline([PF(bx1, a_hi - 0.42), PF(bx1, a_hi - 0.001)], Wf, D_SEAM)
    for ab in (a_lo + 0.36, a_hi - 0.36):
        cv.fasteners_along([PF(bx0 + 0.01, ab), PF(bx1 - 0.01, ab)], 52, 2.3)
    # white band ahead of the collar (P1)
    c_b0, _ = fus_px(g, g["band0"], 0)
    c_b1, _ = fus_px(g, g["band1"], 0)
    band_col = np.array([0.62, 0.62, 0.60], np.float32)
    cv.alb[FUS_ROWS[0]:FUS_ROWS[1], int(c_b0):int(c_b1)] = band_col * (0.93 + 0.07 * fine[FUS_ROWS[0]:FUS_ROWS[1], int(c_b0):int(c_b1), None])
    cv.rough[FUS_ROWS[0]:FUS_ROWS[1], int(c_b0):int(c_b1)] = 0.5
    # flag on the upper forebody (P1)
    if decal_atlas is not None and "flag" in decal_regions:
        fx = g["flag_x_over_root"] * te
        fl, fw = g["flag_size"]
        R = g["R_fore"]
        c0, _ = fus_px(g, fx - fl / 2, 0)
        c1, _ = fus_px(g, fx + fl / 2, 0)
        da = fw / R
        _, r0 = fus_px(g, 0, a_top - da)
        _, r1 = fus_px(g, 0, a_top + da)

        def uvf(xx, yy, fx=fx, fl=fl, fw=fw, R=R):
            x = nx + xx / N * (x1f - nx)
            a = (yy - FUS_ROWS[0]) / (FUS_ROWS[1] - FUS_ROWS[0]) * 2 * math.pi - math.pi / 2
            y = R * np.cos(a)
            return (x - (fx - fl / 2)) / fl, (y + fw / 2) / fw
        _paste_decal(cv, decal_atlas, decal_regions["flag"], (c0, c1, r0, r1), uvf, white=1.0)
    # spine emblem on the boom ahead of the pod, arrow pointing forward (P1)
    if look == "photo" and decal_atlas is not None and "emblem" in decal_regions:
        ex, el = g["pod0"] - 0.05, 0.050
        u0, v0, u1, v1 = decal_regions["emblem"]
        ew = el * (v1 - v0) / (u1 - u0)
        R = g["R_tube"]
        c0, _ = fus_px(g, ex - el / 2, 0)
        c1, _ = fus_px(g, ex + el / 2, 0)
        da = ew / 2 / R
        _, r0 = fus_px(g, 0, a_top - da)
        _, r1 = fus_px(g, 0, a_top + da)

        def uve(xx, yy, ex=ex, el=el, ew=ew, R=R):
            x = nx + xx / N * (x1f - nx)
            a = (yy - FUS_ROWS[0]) / (FUS_ROWS[1] - FUS_ROWS[0]) * 2 * math.pi - math.pi / 2
            y = R * np.cos(a)
            return ((ex + el / 2) - x) / el, (y + ew / 2) / ew
        _paste_decal(cv, decal_atlas, decal_regions["emblem"], (c0, c1, r0, r1), uve, white=0.85)

    # ---- fins: outer faces carry the wordmark (P1), inner faces plain; a seam along the wing junction
    zw = g["wing_z"]
    for reg in ("FS", "FP", "FI"):
        PFn = lambda x, z, reg=reg: tuple(float(c) for c in fin_px(g, x, z, reg))  # noqa: E731
        poly = [(x, z) for x, z in _fin_outline(g)]
        cv.polyline([PFn(x, z) for x, z in poly], 1.2, 0.25, closed=True, dark=0.7)
        cv.polyline([PFn(g["te_x"] - g["fin_root_chord"] - 0.02, zw + 0.004), PFn(g["te_x"] + 0.01, zw + 0.004)], 1.0, 0.3)
        cv.polyline([PFn(g["te_x"] - g["fin_root_chord"] - 0.02, zw - 0.004), PFn(g["te_x"] + 0.01, zw - 0.004)], 1.0, 0.3)
        cv.fasteners_along([PFn(g["te_x"] - g["fin_root_chord"] + 0.012, zw + 0.012), PFn(g["te_x"] - 0.012, zw + 0.012)], 34, 2.0)
        cv.fasteners_along([PFn(g["te_x"] - g["fin_root_chord"] + 0.012, zw - 0.012), PFn(g["te_x"] - 0.012, zw - 0.012)], 34, 2.0)
    if decal_atlas is not None and "fin_text" in decal_regions and g.get("fin_text", True):
        poly = _fin_outline(g)
        hd, hu = g["fin_down_height"], g["fin_up_height"]
        z0, z1 = zw - 0.45 * hd, zw + 0.2 * hu

        def le_at(z):
            (xa_, za_), (xb_, zb_) = (poly[0], poly[1]) if z >= zw else (poly[0], poly[5])
            return xa_ + (z - za_) / (zb_ - za_) * (xb_ - xa_)
        xa = max(le_at(z0), le_at(z1)) + 0.006
        xb = min(g["fin_te_top_x"], g["fin_te_bot_x"], g["te_x"]) - 0.008
        if look == "photo":
            # logo + wordmark read down the fin at mid-chord with their own proportions (P1)
            z0, z1 = zw - 0.62 * hd, zw + 0.62 * hu
            u0_, v0_, u1_, v1_ = decal_regions["fin_text"]
            hgt = (z1 - z0) * (v1_ - v0_) / (u1_ - u0_)
            xm = 0.5 * (le_at(zw) + g["te_x"])
            xa, xb = xm - hgt / 2, xm + hgt / 2
        x0w, ztw = fin_origin(g)
        # photo layout: the starboard outer face read mirror-reversed with the flip (checked on renders)
        for reg, mirror in (("FP", False), ("FS", look != "photo")):
            c0, r0 = fin_px(g, xa, z1, reg)
            c1, r1 = fin_px(g, xb, z0, reg)

            def uvt(xx, yy, reg=reg, mirror=mirror):
                x = x0w + (xx - FIN_COLS[reg]) / FIN_PX
                z = ztw - (yy - FIN_ROWS[0]) / FIN_PX
                u = (z1 - z) / (z1 - z0)
                v = (xb - x) / (xb - xa)
                return u, (1 - v) if mirror else v
            _paste_decal(cv, decal_atlas, decal_regions["fin_text"], (c0, c1, r0, r1), uvt)

    # ---- edge wear on leading edges (lighter, smoother paint rubbed to a sheen), wing and fins
    wear = _value_noise((N, N), 300, rng, 3)
    for lower_side in (False, True):
        for s in (1, -1):
            pts = [tuple(float(c) for c in wing_px(g, abs(y) * ts + 0.004, s * y, lower_side)) for y in np.linspace(0.06, 0.495, 40)]
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                x0, x1, y0, y1 = cv._box((ax, bx), (ay, by), 14)
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
                dx, dy = bx - ax, by - ay
                t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
                d = np.hypot(xx - (ax + t * dx), yy - (ay + t * dy))
                m = np.exp(-(d / 9.0) ** 2) * np.clip((wear[y0:y1, x0:x1] - 0.3) * 1.6, 0, 1)
                cv.alb[y0:y1, x0:x1] = np.maximum(cv.alb[y0:y1, x0:x1], cv.alb[y0:y1, x0:x1] * (1 + 0.35 * m[..., None]))
                cv.rough[y0:y1, x0:x1] = cv.rough[y0:y1, x0:x1] * (1 - 0.15 * m)

    # ---- encode
    gy, gx = np.gradient(cv.h)
    mm_px = span_mm / WING
    k = 1.0 / mm_px
    nrm = np.stack([-gx * k, gy * k, np.ones_like(gx)], -1)    # +V = up in Blender, rows run down
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    alb = np.clip(cv.alb, 0, 1) ** (1 / 2.2)
    return dict(albedo=(alb * 255 + 0.5).astype(np.uint8), rough=(np.clip(cv.rough, 0, 1) * 255 + 0.5).astype(np.uint8),
                normal=((nrm * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8))


def _fin_outline(g):
    te, zw = g["te_x"], g["wing_z"]
    hu, hd = g["fin_up_height"], g["fin_down_height"]
    frc, ftc = g["fin_root_chord"], g["fin_tip_chord"]
    tt, tb = g["fin_te_top_x"], g["fin_te_bot_x"]
    return [(te - frc, zw), (tt - ftc, zw + hu), (tt, zw + hu), (te, zw), (tb, zw - hd), (tb - ftc, zw - hd)]


def atlas_files(g: dict, cache_dir: str, decal_png: str | None, decal_regions: dict, look: str = "survey") -> dict:
    """Write (or reuse) the three PNGs for this geometry; keyed by a hash of the inputs (geometry,
    look, this file and the decal image). Inside Blender (no Pillow) the atlas is drawn by the
    system Python in a subprocess."""
    keys = sorted(k for k, v in g.items() if isinstance(v, (int, float, str, list, tuple)))
    gk = {k: g[k] for k in keys}
    decal_bytes = open(decal_png, "rb").read() if decal_png and os.path.exists(decal_png) else b""
    ident = hashlib.sha1(json.dumps(gk, sort_keys=True, default=str).encode() + look.encode()
                         + open(__file__, "rb").read() + decal_bytes).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    paths = {k: os.path.join(cache_dir, f"kal_hero_{ident}_{k}.png") for k in ("albedo", "rough", "normal")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    try:
        from PIL import Image
    except ImportError:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(dict(g=gk, cache_dir=cache_dir, decal_png=decal_png, regions=decal_regions, look=look), f,
                      default=str)
        subprocess.run(["python", os.path.abspath(__file__), f.name], check=True)
        os.unlink(f.name)
        return paths
    atlas = None
    if decal_png and os.path.exists(decal_png):
        atlas = np.asarray(Image.open(decal_png).convert("RGBA"), np.float32) / 255.0
    tex = build_atlas(g, atlas, decal_regions, look=look)
    Image.fromarray(tex["albedo"]).save(paths["albedo"])
    Image.fromarray(tex["rough"]).save(paths["rough"])
    Image.fromarray(tex["normal"]).save(paths["normal"])
    return paths


if __name__ == "__main__":
    import sys
    job = json.load(open(sys.argv[1]))
    print(atlas_files(job["g"], job["cache_dir"], job["decal_png"], job["regions"], job.get("look", "survey")))
