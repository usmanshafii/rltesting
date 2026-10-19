"""Per-scene draws of the band signatures in configs/signatures.yaml (pure NumPy).

Slots that point at the same group share one draw within a band. dT entries keyed by thermal
state are resolved with the scene's state. Bands share geometry only (spec 7.2 rule 4): every
band's values are separate draws, including the SWIR finish and the temperature used for SWIR
self-emission of hot parts (`emit_dT`, drawn from the LWIR range but not the LWIR value). The
fraction of the propeller disc swept by blades (`alpha`) is geometry and is shared.
"""
from __future__ import annotations

import colorsys

import numpy as np


def _is_range(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def draw(rng, v):
    return float(rng.uniform(v[0], v[1])) if _is_range(v) else v


def draw_group(rng, spec: dict, state: str | None = None) -> dict:
    """One draw per key. A value {mixture: {name: {weight, range}}} first picks a named family by weight
    (recorded as <key>_family), then draws uniformly inside that family's range."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, dict) and state in v:
            v = v[state]
        if isinstance(v, dict) and "mixture" in v:
            names = list(v["mixture"])
            w = np.array([v["mixture"][n]["weight"] for n in names], float)
            name = names[int(rng.choice(len(names), p=w / w.sum()))]
            out[f"{k}_family"] = name
            v = v["mixture"][name]["range"]
        out[k] = draw(rng, v)
    return out


def apply_surface_caps(slots: dict, air_temp_c: float, caps_c: dict | None, key: str = "dT") -> None:
    """Clip sampled surface offsets so air + offset stays below an absolute temperature cap per slot. key "dT"
    for LWIR, "emit_dT" for the SWIR emission draw. The uncapped draw and the cap are kept (in place):
    dT_cap_k / capped / dT_uncapped, or emit_dT_cap_k / emit_dT_capped / emit_dT_uncapped."""
    pre = "" if key == "dT" else f"{key}_"
    for slot, cap in (caps_c or {}).items():
        p = slots.get(slot)
        if p is None or key not in p:
            continue
        lim = float(cap) - float(air_temp_c)
        p[f"{key}_cap_k"] = lim
        p[f"{pre}capped"] = bool(p[key] > lim)
        if p[f"{pre}capped"]:
            p[f"{key}_uncapped"] = p[key]
            p[key] = lim


def swir_self_emission(p_swir: dict, T_kelvin: float, spec: dict, sig: dict) -> float:
    """SWIR self-emission in render units: (1 - reflectance) x Planck 0.9-1.7 um radiance x the render-unit
    calibration nishita_sun_units / inband_solar_irradiance (Kirchhoff: opaque surface, eps = 1 - rho).
    Target slots carry hot_emission_enabled: physics, on whenever the part is hotter than min_temp_c, day or
    night. Hard-negative slots follow the scene's optional swir_hot_emission switch."""
    from thermal.planck_lut import swir_lut
    he = sig["swir"]["hot_emission"]
    if T_kelvin < he["min_temp_c"] + 273.15:
        return 0.0
    if "hot_emission_enabled" in p_swir:
        if not p_swir["hot_emission_enabled"]:
            return 0.0
    elif not spec.get("swir_hot_emission"):
        return 0.0
    eps = 1.0 - p_swir["reflectance"]
    units = he["nishita_sun_units"] / he["inband_solar_irradiance"]
    return float(eps * swir_lut().radiance(min(T_kelvin, 1200.0)) * units)


def rgb_self_emission(p_rgb: dict, T_kelvin: float, sig: dict) -> tuple:
    """Night red glow in render units per colour channel (r, g, b): (1 - albedo_c) x Planck radiance in the
    channel's band x nishita_sun_units / inband_solar_irradiance_c (Kirchhoff: opaque surface; the same render-unit
    calibration as swir_self_emission). Zero below min_temp_c, where it is < 0.1 x the faintest night sky."""
    from thermal.planck_lut import band_radiance_direct
    he = sig["rgb"].get("hot_emission") or {}
    if not he.get("enabled") or T_kelvin < he["min_temp_c"] + 273.15:
        return (0.0, 0.0, 0.0)
    tint = p_rgb.get("color", [1.0, 1.0, 1.0])
    out = []
    for k, c in enumerate(("r", "g", "b")):
        eps = 1.0 - min(1.0, p_rgb.get("albedo", 0.0) * tint[k])
        L = float(band_radiance_direct(np.array([min(T_kelvin, 1500.0)]), tuple(he["bands_um"][c]), 400)[0])
        out.append(eps * L * he["nishita_sun_units"] / he["inband_solar_irradiance"][c])
    return tuple(out)


SUN_ANGULAR_RADIUS_RAD = 6.957e8 / 1.495978707e11      # IAU 2015 nominal solar radius / 1 au


def sun_glint_params(rng, sig: dict, sun_elevation_deg: float, eps_zenith: float | None) -> dict | None:
    """One draw per scene of the LWIR sun-glint lobe (signatures.yaml lwir.sun_glint), with the in-band solar
    irradiance at the ground E = E_toa x (1 - eps_z)^(1 / sin el). None at night or with the sun below the horizon.
    peak_per_reflectance: radiance at the mirror direction per unit reflectance (1 - eps), W m-2 sr-1."""
    g = sig["lwir"].get("sun_glint") or {}
    if not g.get("enabled") or sun_elevation_deg <= 0.0:
        return None
    eps_z = g["default_eps_zenith"] if eps_zenith is None else float(eps_zenith)
    tau = (1.0 - eps_z) ** (1.0 / max(np.sin(np.radians(sun_elevation_deg)), 1e-3))
    lobe = float(rng.uniform(*g["lobe_deg"]))
    fs = float(rng.uniform(*g["specular_fraction"]))
    sigma2 = np.radians(lobe) ** 2 + SUN_ANGULAR_RADIUS_RAD ** 2
    e_sun = float(g["sun_toa_irradiance"]) * tau
    return dict(lobe_deg=lobe, specular_fraction=fs, transmittance=float(tau), irradiance_w_m2=e_sun,
                sigma2_rad2=float(sigma2), peak_per_reflectance=float(fs * e_sun / (np.pi * sigma2)))


def glint_radiance(eps: float, glint: dict | None, cos_angle: float) -> float:
    """LWIR glint radiance (W m-2 sr-1) of a surface of emissivity eps whose mirror direction makes an angle with
    the sun of cosine cos_angle: (1 - eps) x peak x exp(-2 (1 - cos) / sigma^2) (2 (1 - cos) ~ angle^2)."""
    if not glint:
        return 0.0
    return float((1.0 - eps) * glint["peak_per_reflectance"] * np.exp(-2.0 * (1.0 - cos_angle) / glint["sigma2_rad2"]))


def add_rgb_glow_draw(rng, target: dict, sig: dict, state: str) -> None:
    """The RGB pass's own temperature draw for the glow of hot parts (glow_dT), independent of the LWIR and SWIR
    draws (spec 7.2 rule 4): an engine profile and a value in each LWIR range, in place on target["rgb"]."""
    lw = sig["lwir"]["target"]
    name, profile = _engine_profile(rng, lw)
    temps = {g: draw(rng, s["dT"]) for g, s in _thermal_groups(lw, "night" if state is None else state, profile).items()}
    for slot, p in target["rgb"].items():
        p["glow_dT"] = temps[lw["slots"][slot]]
        if "dT_profile" in lw["groups"][lw["slots"][slot]]:
            p["glow_thermal_profile"] = name


def _state_range(v, state):
    return v[state] if isinstance(v, dict) else v


def _engine_profile(rng, lw):
    """One engine power-setting profile per draw, picked by weight (signatures.yaml engine_profiles)."""
    profiles = lw["engine_profiles"]
    names = list(profiles)
    weights = np.array([profiles[n]["weight"] for n in names], float)
    name = str(rng.choice(names, p=weights / weights.sum()))
    return name, profiles[name]


def _thermal_groups(lw, state, profile):
    """Resolve operating-profile ranges without changing skin's day/crossover/night states."""
    return {name: {**{k: v for k, v in p.items() if k != "dT_profile"},
                   "dT": profile[p["dT_profile"]] if "dT_profile" in p else _state_range(p["dT"], state)}
            for name, p in lw["groups"].items()}


def sample_target(rng, sig: dict, state: str) -> dict:
    """{band: {slot: params}} for the KAL target."""
    out = {}
    for band in ("rgb", "lwir", "swir"):
        t = sig[band]["target"]
        specs = t["groups"]
        if band == "lwir":
            profile_name, profile = _engine_profile(rng, t)
            specs = _thermal_groups(t, state, profile)
        groups = {g: draw_group(rng, s, state) for g, s in specs.items()}
        if band == "lwir":
            for g, p in groups.items():
                if "dT_profile" in t["groups"][g]:
                    p["thermal_profile"] = profile_name
        out[band] = {slot: dict(groups[g], group=g) for slot, g in t["slots"].items()}
    alpha = draw(rng, sig["rgb"]["target"]["propeller_disc_alpha"])      # blade coverage: geometry
    out["rgb"]["propeller_disc"]["alpha"] = out["swir"]["propeller_disc"]["alpha"] = alpha
    lw = sig["lwir"]["target"]
    emit_name, emit_profile = _engine_profile(rng, lw)  # independent SWIR draw: no shared thermal latent
    emit = {g: draw(rng, s["dT"]) for g, s in _thermal_groups(lw, state, emit_profile).items()}
    for slot, p in out["swir"].items():
        p["emit_dT"] = emit[lw["slots"][slot]]
        p["hot_emission_enabled"] = sig["swir"]["target"]["hot_emission_enabled"]
        if "dT_profile" in lw["groups"][lw["slots"][slot]]:
            p["emit_thermal_profile"] = emit_name
    # sample_target returns uncapped draws; the scene applies any surface caps.
    return out


def sample_negative(rng, sig: dict, cls: str, state: str) -> dict:
    """{band: {slot: params}} for one hard-negative object."""
    out = {}
    for band in ("rgb", "lwir", "swir"):
        spec = sig[band]["negatives"][cls]
        d = {}
        for slot, s in spec.items():
            if "same_as" in s:
                continue
            p = draw_group(rng, s, state)
            if band == "rgb":
                if p.pop("saturated", False):
                    h, sat = rng.uniform(), rng.uniform(0.6, 1.0)
                    p["color"] = list(colorsys.hsv_to_rgb(h, sat, 1.0))
                else:
                    tint = rng.uniform(0.9, 1.1, 3)
                    p["color"] = list(np.clip(tint / tint.mean(), 0, 2))
            d[slot] = p
        for slot, s in spec.items():
            if "same_as" in s:
                d[slot] = dict(d[s["same_as"]])
        out[band] = d
    lw = sig["lwir"]["negatives"][cls]
    for slot, p in out["swir"].items():
        if "alpha" in out["rgb"].get(slot, {}):          # rotor / propeller disc coverage: geometry
            p["alpha"] = out["rgb"][slot]["alpha"]
        src = lw.get(slot) or lw.get(sig["swir"]["negatives"][cls].get(slot, {}).get("same_as", ""), {})
        if "dT" in src:
            p["emit_dT"] = draw(rng, _state_range(src["dT"], state))   # own draw, not the LWIR value
    return out


def sample_environment(rng, sig: dict, state: str) -> dict:
    env = {}
    r = sig["rgb"]["environment"]
    g = draw_group(rng, r["ground"])
    hue = rng.choice([[1.08, 1.0, 0.82], [0.9, 1.0, 0.78], [1.0, 1.0, 1.0], [1.1, 0.98, 0.9]])
    tint = 1.0 + g.pop("tint") * (np.array(hue) - 1.0) / 0.2
    g["color"] = list(np.clip(tint / tint.mean(), 0.5, 1.5))
    env["rgb"] = dict(ground=g, wall=draw_group(rng, r["wall"]), roof=draw_group(rng, r["roof"]))
    lw = sig["lwir"]["environment"]
    env["lwir"] = {k: draw_group(rng, lw[k], state) for k in ("sky", "clouds", "ground", "wall", "roof")}
    sw = sig["swir"]["environment"]
    env["swir"] = dict(ground=draw_group(rng, sw["ground"]), wall=draw_group(rng, sw["wall"]),
                       roof=draw_group(rng, sw["roof"]), cloud_brightness=draw(rng, sw["cloud_brightness"]),
                       cloud_brightness_scale=draw(rng, sw["cloud_brightness_scale"]))
    env["swir"]["sky_divisor"] = draw(rng, sig["swir"]["sky_divisor"])
    env["swir"]["night_airglow"] = float(np.exp(rng.uniform(*np.log(sig["swir"]["night_airglow"]))))
    return env


# ============================================================================ LWIR cooled and MWIR
THERMAL_EXTRA = ("lwir_cooled", "mwir")


def _requantile(v: float, src, dst) -> float:
    """The value at the same quantile of dst as v has in src (constant src: dst's centre)."""
    lo, hi = float(src[0]), float(src[1])
    u = 0.5 if hi <= lo else min(1.0, max(0.0, (float(v) - lo) / (hi - lo)))
    return float(dst[0] + u * (dst[1] - dst[0]))


def _lw_range(spec_entry):
    e = spec_entry["emissivity"]
    return e if _is_range(e) else (e, e)


def add_thermal_bands(rng, sigs: dict, sig: dict, neg_classes: list) -> None:
    """Signatures of the LWIR cooled and MWIR cameras, in place on sigs (target, negatives, environment): the LWIR
    draw's temperatures (and every other LWIR key) with the emissivity moved to the same quantile of the band's range
    (signatures.yaml lwir_cooled / mwir). The only new draws, on the caller's own stream: MWIR cloud reflectance."""
    lw = sig["lwir"]
    for band in THERMAL_EXTRA:
        bcfg = sig[band]
        tgt = {}
        for slot, p in sigs["target"]["lwir"].items():
            grp = p["group"]
            q = dict(p)
            q["emissivity"] = _requantile(p["emissivity"], _lw_range(lw["target"]["groups"][grp]),
                                          bcfg["target"]["emissivity"][grp])
            tgt[slot] = q
        sigs["target"][band] = tgt
        for neg, cls in zip(sigs["negatives"], neg_classes):
            d = {}
            for slot, p in neg["lwir"].items():
                q = dict(p)
                src = lw["negatives"][cls].get(slot)
                if src is not None and slot in bcfg["negatives"][cls]:
                    q["emissivity"] = _requantile(p["emissivity"], _lw_range(src), bcfg["negatives"][cls][slot])
                d[slot] = q
            neg[band] = d
        env_lw = sigs["environment"]["lwir"]
        env = {k: dict(env_lw[k]) for k in ("clouds", "ground", "wall", "roof")}
        for k in ("ground", "wall", "roof"):
            env[k]["emissivity"] = _requantile(env_lw[k]["emissivity"], _lw_range(lw["environment"][k]),
                                               bcfg["environment"][k])
        if "cloud_reflectance" in bcfg["environment"]:
            env["clouds"]["reflectance"] = draw(rng, bcfg["environment"]["cloud_reflectance"])
        sigs["environment"][band] = env


def thermal_glints(rng, sig: dict, sun_elevation_deg: float) -> dict | None:
    """Per-band glint lobe draws of the LWIR cooled and MWIR cameras (the irradiance comes from the scene's layered
    atmosphere at render time, render/thermal.py). None at night or with the sun below the horizon."""
    if sun_elevation_deg <= 0.0:
        return None
    out = {}
    for band in THERMAL_EXTRA:
        g = sig[band]["sun_glint"]
        lobe = float(rng.uniform(*g["lobe_deg"]))
        out[band] = dict(lobe_deg=lobe, specular_fraction=float(rng.uniform(*g["specular_fraction"])),
                         sigma2_rad2=float(np.radians(lobe) ** 2 + SUN_ANGULAR_RADIUS_RAD ** 2))
    return out


def plume_draws(rng, sig: dict) -> dict | None:
    """Per-scene exhaust plume parameters (signatures.yaml plume; thermal/plume.py)."""
    p = sig.get("plume") or {}
    if not p.get("enabled"):
        return None
    lo, hi = p["k_mix"]
    return dict(equivalence_ratio=float(rng.uniform(*p["equivalence_ratio"])),
                fuel_h_to_c=float(rng.uniform(*p["fuel_h_to_c"])),
                short_circuit=float(rng.uniform(*p["short_circuit"])),
                k_mix=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), spread=float(rng.uniform(*p["spread"])),
                rays=dict(p.get("rays", {"n_u": 12, "n_v": 7})))
