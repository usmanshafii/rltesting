"""Exhaust plume of the L550E in the thermal bands (LWIR uncooled, LWIR cooled, MWIR). Pure NumPy; docs/thermal_bands.md.

Gas. Products of petrol CH_y (y = 1.87-2.25, thermal/engine_heat.FUEL_H_TO_C) burnt at the equivalence ratio phi of the
scene (1.0-1.2, [AUSSERER] via engine_heat.PHI): carbon, hydrogen and oxygen balances, and for rich mixtures the water-
gas shift equilibrium CO2 + H2 = CO + H2O, K = x_CO x_H2O / (x_CO2 x_H2) = 3.5 (Heywood, Internal Combustion Engine
Fundamentals, 1988, section 4.9.1), which puts CO at about three times H2 as measured in rich exhausts. A share of the fresh charge short-circuits to the exhaust in a crankcase-scavenged two-stroke and
dilutes the products with air (fuel vapour is not modelled: its C-H band lies at 3.3-3.5 um, below the MWIR bands).
Exit temperature: the stub barely cools the gas (NTU 0.04-0.07, engine_heat.stub_state), so the gas leaves at the
setting's gas/wall ratio times the scene's stub wall draw, clipped to the setting's port-gas band.

Mixing. Each of the four outlets (target/engine_l550e.outlets, 28.5 mm bore) issues a round jet that the free stream
bends backwards (exit velocity / flight speed from the exhaust mass flow, both ~40-50 m/s). Mixture fraction on the
axis f_c = min(1, x_c / x), x_c = K D (round-jet similarity: K ~ 5 for a free jet in still air, Pope, Turbulent Flows,
2000, ch. 5; slower decay in a co-flowing stream, faster in the pusher propeller's slipstream: K drawn 5-20, C);
Gaussian radial profile with half-width S x (S = 0.08-0.11 around Pope's 0.094, C). Temperature and composition mix
linearly with f (equal diffusivities). Optical model: thermal/gas_bands.py along rays through the field
(SP-3080 / HITRAN narrow bands, Curtis-Godson), as for the atmosphere.

The pusher propeller ends the hot plume: its plane lies 0.13-0.27 m behind the outlets and its slipstream (~14 kg/s
through the disc at cruise, against ~0.05 kg/s of exhaust) dilutes the gas some 300-fold, to within ~2 K of the air,
so the field is cut at the propeller plane (C+: the jets may pass near the tips, where the tip vortices mix them as
fast). The four jets are treated as separate and optically independent (C: overlap matters only where the CO2 band
centre is opaque); the airframe does not occlude them (the stubs point down and the jets leave behind the engine).
"""
from __future__ import annotations

import math

import numpy as np

from thermal import gas_bands as gb

K_WGS = 3.5                         # Heywood 1988, 4.9.1: K = x_CO x_H2O / (x_CO2 x_H2) for rich products
AIR = {"N2": 0.7808, "O2": 0.2095, "CO2": 0.00042, "H2O": 0.0}


def exhaust_composition(phi: float, h_to_c: float, short_circuit: float = 0.0) -> dict:
    """Mole fractions of the exhaust (wet) per mole of mixture, after diluting the burnt products with a share
    `short_circuit` (by moles of fresh air relative to the air burnt) of unburnt air."""
    y = float(h_to_c)
    o2 = (1.0 + y / 4.0) / phi                                    # mol O2 supplied per mol C
    n2 = 3.773 * o2
    if phi <= 1.0:
        prod = {"CO2": 1.0, "H2O": y / 2.0, "CO": 0.0, "H2": 0.0, "O2": o2 - (1.0 + y / 4.0), "N2": n2}
    else:
        # unknowns a = CO2, b = CO, c = H2O, d = H2: a + b = 1, c + d = y/2, 2a + b + c = 2 o2, K = b c / (a d).
        # With a = 1 - b: c = 2 o2 - 2 + b, d = y/2 - c; b c - K a d grows monotonically with b on the feasible range.
        def resid(b):
            a = 1.0 - b
            c = 2.0 * o2 - 2.0 * a - b
            d = y / 2.0 - c
            return b * c - K_WGS * a * d, (a, b, c, d)
        lo, hi = max(1e-12, 2.0 - 2.0 * o2), min(1.0, 2.0 + y / 2.0 - 2.0 * o2) - 1e-12   # c >= 0 and d >= 0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            r, _abcd = resid(mid)
            if r > 0:
                hi = mid
            else:
                lo = mid
        a, b, c, d = resid(0.5 * (lo + hi))[1]
        prod = {"CO2": a, "CO": b, "H2O": c, "H2": d, "O2": 0.0, "N2": n2}
    air_moles = (o2 + n2) * float(short_circuit)                    # short-circuited fresh air
    prod["O2"] += 0.2095 / 0.9903 * air_moles
    prod["N2"] += 0.7808 / 0.9903 * air_moles
    tot = sum(prod.values())
    return {k: v / tot for k, v in prod.items()}


def gas_exit_excess(theta_wall: float, setting: str) -> float:
    """Gas excess at the outlet [K] for a stub wall excess drawn for the setting: wall excess x the setting's
    gas/wall ratio (thermal/engine_heat.stub_state at the envelope centre), clipped to the setting's port-gas band."""
    from thermal import engine_heat as eh
    s = eh.POWER_SETTINGS[setting]
    mid = lambda r: 0.5 * (r[0] + r[1])          # noqa: E731
    op = eh.operating_state(mid(s["f"]), mid(eh.INPUTS["bsfc_g_kwh"]), mid(eh.INPUTS["afr"]))
    st = [eh.stub_state(egt, op["exhaust_kg_s_per_cyl"], mid(eh.ENV["airspeed_mps"]), mid(eh.ENV["alt_m"]),
                        mid(eh.ENV["t_air_k"]), mid(eh.INPUTS["eps_stub"]), mid(eh.INPUTS["f_pulse"]),
                        mid(eh.INPUTS["k_stub"])) for egt in s["egt_excess_k"]]
    ratio = float(np.mean([x["theta_gas_out"] / x["theta_wall"] for x in st]))
    lo, hi = st[0]["theta_gas_out"], st[1]["theta_gas_out"]
    return float(np.clip(ratio * float(theta_wall), lo, hi))


def exit_velocity_ratio(setting: str, theta_out: float, t_air_k: float, airspeed: float, alt_m: float = 0.0) -> float:
    """Jet exit velocity over flight speed, from the setting's exhaust mass flow per cylinder and the gas density."""
    from thermal import engine_heat as eh
    s = eh.POWER_SETTINGS[setting]
    mid = lambda r: 0.5 * (r[0] + r[1])          # noqa: E731
    m = eh.operating_state(mid(s["f"]), mid(eh.INPUTS["bsfc_g_kwh"]), mid(eh.INPUTS["afr"]))["exhaust_kg_s_per_cyl"]
    rho = eh.pressure_isa(alt_m) / (eh.R_AIR * (t_air_k + theta_out))
    area = math.pi * (eh.stub_geometry()["d_i"] / 2.0) ** 2
    return float(m / (rho * area) / max(airspeed, 1.0))


class Jet:
    """One exhaust jet: mixture-fraction field around a straight centreline (world coordinates, metres)."""

    def __init__(self, origin, axis, d_m: float, theta_out: float, t_air_k: float, p_atm: float, comp: dict,
                 k_mix: float, spread: float, cut_point=None, cut_normal=None):
        self.o = np.asarray(origin, float)
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.d, self.theta, self.t_air, self.p = float(d_m), float(theta_out), float(t_air_k), float(p_atm)
        self.comp, self.k, self.s = comp, float(k_mix), float(spread)
        self.x_c = self.k * self.d
        self.length = 50.0 * self.x_c                         # f_c = 0.02 at the end
        self.cut = None
        if cut_point is not None and cut_normal is not None:   # the propeller plane: no hot gas behind it
            n = np.asarray(cut_normal, float) / np.linalg.norm(cut_normal)
            self.cut = (np.asarray(cut_point, float), n)
            along = float(self.a @ n)
            if along > 1e-6:
                self.length = min(self.length, float((self.cut[0] - self.o) @ n) / along)

    def mixture_fraction(self, P):
        rel = np.atleast_2d(P) - self.o
        x = rel @ self.a
        r = np.linalg.norm(rel - x[:, None] * self.a, axis=1)
        b = np.maximum(0.5 * self.d, self.s * np.maximum(x, 0.0))
        fc = np.where(x > 0, np.minimum(1.0, self.x_c / np.maximum(x, 1e-9)), 0.0)
        f = np.where((x > 0) & (x < self.length), fc * np.exp(-math.log(2.0) * (r / b) ** 2), 0.0)
        if self.cut is not None:
            f = np.where((np.atleast_2d(P) - self.cut[0]) @ self.cut[1] > 0.0, 0.0, f)
        return f

    def state(self, f: float) -> dict:
        p_atm = {k: self.p * (f * self.comp.get(k, 0.0) + (1.0 - f) * AIR.get(k, 0.0))
                 for k in ("CO2", "H2O", "CO", "N2", "O2")}
        return dict(T=self.t_air + f * self.theta, p_atm=p_atm)


def ray_spectrum(jet: Jet, start, direction, w, length_m: float, n_steps: int = 18):
    """Spectral radiance leaving a ray segment through the jet toward its start point, and its transmittance,
    from emission along it (SP-3080 eq. A-3 with Curtis-Godson, gas_bands.path_depths)."""
    s = (np.arange(n_steps) + 0.5) / n_steps * length_m
    P = np.asarray(start, float)[None, :] + s[:, None] * np.asarray(direction, float)[None, :]
    # mixture fraction quantised to 1/64 (repeated gas states reuse the narrow-band lookups, gas_bands._params_memo);
    # samples below 1/128 are ambient air, whose absorption over a sub-metre chord is negligible
    f = np.round(jet.mixture_fraction(P) * 64.0) / 64.0
    f = f[f > 0.0]
    if not len(f):
        return np.zeros_like(w), np.ones_like(w)
    ds = length_m / n_steps
    segs = [dict(length_m=ds, **jet.state(float(fi))) for fi in f]
    x = gb.path_depths(w, segs)
    tau = np.exp(-x)
    tau_prev = np.vstack([np.ones_like(w), tau[:-1]])
    T = np.array([g["T"] for g in segs])
    L = np.sum(gb.planck_wn(T[:, None], w[None, :]) * (tau_prev - tau), axis=0)
    return L, tau[-1]


def projected_rays(jet: Jet, view_dir, n_u: int = 12, n_v: int = 7):
    """Ray grid across the jet's projection on the plane normal to the view: (anchor points on the plane through the
    axis, area per ray [m2], entry points, chord length)."""
    v = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    a_perp = jet.a - (jet.a @ v) * v
    if np.linalg.norm(a_perp) < 1e-6:                       # looking straight along the jet
        a_perp = np.cross(v, [0.0, 0.0, 1.0] if abs(v[2]) < 0.9 else [1.0, 0.0, 0.0])
    a_perp /= np.linalg.norm(a_perp)
    e2 = np.cross(v, a_perp)
    L_vis = min(8.0 * jet.x_c, jet.length)                   # f_c down to 1/8 (the bright core), or the cut
    proj_len = L_vis * math.sqrt(max(1e-6, 1.0 - (jet.a @ v) ** 2)) + jet.d
    half_w = 3.0 * max(0.5 * jet.d, jet.s * L_vis)
    u = (np.arange(n_u) + 0.5) / n_u * proj_len - 0.5 * jet.d
    vv = ((np.arange(n_v) + 0.5) / n_v - 0.5) * 2.0 * half_w
    du, dv = proj_len / n_u, 2.0 * half_w / n_v
    anchors = jet.o[None, None, :] + u[:, None, None] * a_perp + vv[None, :, None] * e2
    depth = L_vis + 4.0 * half_w
    starts = anchors - 0.5 * depth * v                       # the ray runs from the camera side through the plane
    return anchors.reshape(-1, 3), du * dv, starts.reshape(-1, 3), depth, v


def jet_band_maps(jet: Jet, view_dir, band_ws: dict, bg_spectra: dict, n_u: int = 12, n_v: int = 7) -> dict:
    """Per band: for each ray of the projected grid, the band emission radiance [W m-2 sr-1], the band absorptance
    weighted by the background spectrum behind the jet, and the band emission spectrum summed over rays x area
    (for the path weighting). bg_spectra: band -> background spectral radiance on that band's w."""
    anchors, dA, starts, depth, v = projected_rays(jet, view_dir, n_u, n_v)
    out = {}
    for band, w in band_ws.items():
        L_emit, A_abs = np.zeros(len(starts)), np.zeros(len(starts))
        spec_sum = np.zeros_like(w)
        bg = np.asarray(bg_spectra[band], float)
        for i, st in enumerate(starts):
            Lw, tw = ray_spectrum(jet, st, v, w, depth)
            L_emit[i] = float(np.sum(Lw) * 5.0)
            A_abs[i] = float(np.sum((1.0 - tw) * bg) / max(np.sum(bg), 1e-30))
            spec_sum += Lw * dA
        out[band] = dict(L_emit=L_emit, A_abs=A_abs, spectrum_w_sr=spec_sum)
    return dict(anchors=anchors, dA=dA, bands=out)
