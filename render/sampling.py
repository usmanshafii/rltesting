"""Scene sampler (pure NumPy): everything about a scene except the pixels.

sample_scene(idx, cfgs, n_scenes) returns
    spec   JSON-serialisable description (camera, sun, weather, signatures, placements, ...)
    objs   in-memory geometry for the renderer (target mesh, negative meshes, terrain, structures)

The same function runs inside Blender (render/scene.py) and in plain Python (tests, tools), so the
placement that decides a scene's pixel bin can be checked without rendering.

Experiment overlays: if $KAL_CONFIG_OVERLAY names a YAML file (absolute, or relative to the repository
root), its keys are merged over configs/render.yaml, sensors.yaml and signatures.yaml (top-level keys
`render`, `sensors`, `signatures`; nested dicts merge, everything else replaces). Blender subprocesses
inherit the variable, so one export covers rendering, training and evaluation. M1 uses
configs/m1.yaml. Every scene records the overlay name and file hash.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (ROOT, os.path.join(ROOT, "target")):
    if p not in sys.path:
        sys.path.insert(0, p)

import kal_geometry as kg  # noqa: E402
import kal_measure as km  # noqa: E402
from render import backgrounds as bg  # noqa: E402
from render import cameras as cams  # noqa: E402
from render import negatives as neg  # noqa: E402
from render import placement as pl  # noqa: E402
from render import rotation_blur as rb  # noqa: E402
from render import signatures as sg  # noqa: E402
from thermal import atmosphere as atm  # noqa: E402
from thermal import atmosphere_bands as atb  # noqa: E402
from thermal import sky as thsky  # noqa: E402

BANDS = ("rgb", "lwir", "swir")                     # bands whose draws share the scene's main random stream
THERMAL_EXTRA = sg.THERMAL_EXTRA                     # LWIR cooled, MWIR: own streams (CLAUDE.md sensor set rule)
ALL_BANDS = ("rgb", "lwir", "lwir_cooled", "mwir", "swir")   # SWIR deferred: sampled, not rendered by default
DEFAULT_THERMAL_BANDS_UM = {"lwir": (8.0, 14.0), "lwir_cooled": (7.7, 9.3), "mwir": (3.7, 4.8)}
# Blender's Nishita sky (intern/sky/source/sky_nishita.cpp, Blender 4.2): aerosol scattering 2e-5 m-1 at sea level per
# unit dust density, extinction 1.11 x that, 1.2 km scale height (single scattering). Used to give the RGB sky the
# scene's own haze: dust = (3.912 / V) / 2.22e-5 m-1 (the visibility's surface extinction).
NISHITA_MIE_EXTINCTION_PER_M = 1.11 * 2e-5


def dust_for_visibility(visibility_km: float, lo: float = 0.5, hi: float = 400.0) -> float:
    return float(min(hi, max(lo, 3.912 / (visibility_km * 1000.0) / NISHITA_MIE_EXTINCTION_PER_M)))


# ============================================================================ climate record
_CLIMATE = {}


def climate_records(path: str) -> dict:
    """The station hours of the climate record (tools/source_measurements.lahore_climate), by lighting state."""
    if path not in _CLIMATE:
        import csv
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        by = {}
        with open(full, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by.setdefault(r["lighting"], []).append(r)
        _CLIMATE[path] = by
    return _CLIMATE[path]


def climate_conditions(rng, clim: dict, lighting: str) -> dict | None:
    """One hour of the record in the scene's lighting state, every hour of the window equally likely (so the draw
    follows the climatology). Returns the record and the scene values taken from it."""
    hours = climate_records(clim["file"]).get(lighting)
    if not hours:
        return None
    r = hours[int(rng.integers(len(hours)))]
    num = lambda k: float(r[k]) if r.get(k, "") != "" else None          # noqa: E731
    kind = {"fog": "fog", "haze": "haze", "dust": "haze", "rain": "haze", "clear": "clear"}[r["weather"]]
    vis = num("visibility_m")
    vis_km = None
    if vis is not None:
        vis_km = max(vis / 1000.0, clim.get("visibility_min_km", 0.05))
        ceiling = clim["visibility_report_ceiling_km"]
        if r["weather"] == "clear" and vis_km >= ceiling:
            # the station hardly ever reports beyond its ceiling: an unobscured hour at the ceiling is "at least"
            vis_km = float(math.exp(rng.uniform(math.log(vis_km), math.log(clim["visibility_beyond_ceiling_km"]))))
    okta = num("cloud_okta")
    return dict(record=dict(utc=r["utc"], local_hour=int(float(r["local_hour"])), month=int(r["month"]),
                            sun_elevation_deg=float(r["sun_elevation_deg"]), weather=r["weather"],
                            ww_codes=r["ww_codes"], visibility_m=vis, cloud_okta=okta),
                air_temp_c=float(r["t_air_c"]), rh=float(r["rh"]), weather_kind=kind, visibility_km=vis_km,
                cloud_okta=okta, station_pressure_hpa=num("station_pressure_hpa"))


# ============================================================================ configs
def load_yaml(path):
    """PyYAML if available, else target/_yamlite.py (Blender's bundled Python has no PyYAML)."""
    try:
        import yaml
    except ImportError:
        import _yamlite
        return _yamlite.load(path)
    with open(path) as f:
        return yaml.safe_load(f)


OVERLAY_ENV = "KAL_CONFIG_OVERLAY"
OVERLAY_KEYS = ("render", "sensors", "signatures")      # target.yaml is never overlaid (protected baseline)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _overlay_layers(path: str, root: str, seen: tuple = ()) -> list:
    """[(path, overlay)] for the overlay file at `path` and the overlays it extends (key `extends`), base first."""
    if path in seen:
        raise ValueError(f"{path}: overlays extend each other in a cycle")
    ov = load_yaml(path) or {}
    unknown = set(ov) - set(OVERLAY_KEYS) - {"name", "notes", "extends"}
    if unknown:
        raise ValueError(f"{path}: overlay keys must be {OVERLAY_KEYS}, got {sorted(unknown)}")
    base = ov.get("extends")
    layers = _overlay_layers(base if os.path.isabs(base) else os.path.join(root, base), root, seen + (path,)) if base else []
    return layers + [(path, ov)]


def load_configs(root: str = ROOT, overlay: str | None = None) -> dict:
    c = os.path.join(root, "configs")
    cfgs = dict(render=load_yaml(os.path.join(c, "render.yaml")),
                sensors=load_yaml(os.path.join(c, "sensors.yaml")),
                signatures=load_yaml(os.path.join(c, "signatures.yaml")),
                target=kg.load_config(os.path.join(c, "target.yaml")))
    if overlay is None:                               # None: use $KAL_CONFIG_OVERLAY; "": no overlay
        overlay = os.environ.get(OVERLAY_ENV)
    if overlay:
        path = overlay if os.path.isabs(overlay) else os.path.join(root, overlay)
        layers = _overlay_layers(path, root)          # an overlay may extend another (e.g. a thermal arm extends M1's)
        for _, ov in layers:
            for k in OVERLAY_KEYS:
                if k in ov:
                    cfgs[k] = deep_merge(cfgs[k], ov[k])
        import hashlib
        h = hashlib.sha1()
        for p, _ in layers:                           # one file: the file's own hash, as before
            with open(p, "rb") as f:
                h.update(f.read())
        ov = layers[-1][1]
        cfgs["overlay"] = dict(name=ov.get("name", os.path.basename(path)), path=os.path.relpath(path, root).replace(os.sep, "/"), sha1=h.hexdigest()[:12])
        if len(layers) > 1:
            cfgs["overlay"]["extends"] = [os.path.relpath(p, root).replace(os.sep, "/") for p, _ in layers[:-1]]
    return cfgs


def scene_id(idx: int) -> str:
    return f"s{idx:06d}"


def assign_splits(n: int, fractions: dict, seed: int = 0) -> list[str]:
    """Deterministic scene-level split: a fixed permutation cut 70/15/15 (spec 6)."""
    order = np.random.default_rng(seed + 991).permutation(n)
    out = [""] * n
    names = list(fractions)
    cuts = np.cumsum([fractions[k] for k in names])
    cuts = np.round(cuts / cuts[-1] * n).astype(int)
    start = 0
    for name, stop in zip(names, cuts):
        for i in order[start:stop]:
            out[int(i)] = name
        start = stop
    return out


def _choice(rng, probs: dict):
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _logu(rng, a, b):
    return float(math.exp(rng.uniform(math.log(a), math.log(b))))


# ============================================================================ poses
def _target_pose_fn(tcfg):
    def f(rng):
        h = float(rng.uniform(*tcfg["heading_deg"]))
        if rng.uniform() < tcfg["level_flight_prob"]:
            p = r = 0.0
        else:
            p, r = float(rng.uniform(*tcfg["pitch_deg"])), float(rng.uniform(*tcfg["roll_deg"]))
        return pl.object_rotation(h, p, r), dict(heading_deg=h, pitch_deg=p, roll_deg=r)
    return f


def look_elevation(rng, cc: dict, family: str, lo: float, hi: float) -> float:
    """Camera look elevation: uniform, or log-uniform (weighted towards the horizon) when
    render.yaml camera.look_elevation_sampling says so for the family. One draw either way."""
    if (cc.get("look_elevation_sampling") or {}).get(family) == "log_uniform" and lo > 0:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _uv_above(cam: pl.Camera, char_len: float, min_agl: float):
    """Ray sampler for the horizon family: only rows whose ray can clear `min_agl` at the distance
    implied by the requested size (distance ~ f * char_len / p). Rows below that would put the
    object underground and would only waste placement tries."""
    def f(rng, p_t):
        m = p_t / 2 + 2
        dist = cam.f * char_len / p_t
        s = (min_agl - cam.C[2]) / dist
        if s >= 0.999:
            return None
        el_min = math.degrees(math.asin(max(-0.999, s)))
        v_max = cam.H / 2 - cam.f * math.tan(math.radians(el_min - cam.elevation))
        v_hi = min(cam.H - m, v_max)
        if v_hi <= m:
            return None
        return rng.uniform(m, cam.W - m), rng.uniform(m, v_hi)
    return f


def _neg_pose_fn(cls, wind_dir_deg):
    lim = dict(bird=(20, 35), kite=(0, 0), light_aircraft=(8, 25), helicopter=(6, 8), small_uav=(12, 15),
               warm_clutter=(0, 0))[cls]

    def f(rng):
        h = float(rng.uniform(0, 360)) if cls != "kite" else float((wind_dir_deg + 180) % 360)
        p, r = float(rng.uniform(-lim[0], lim[0])), float(rng.uniform(-lim[1], lim[1]))
        return pl.object_rotation(h, p, r), dict(heading_deg=h, pitch_deg=p, roll_deg=r)
    return f


# ============================================================================ scene
def sample_scene(idx: int, cfgs: dict, n_scenes: int | None = None, attempts: int = 8):
    """Deterministic in idx. Geometry can make a bin impossible for a draw (a 64+ px KAL is only
    20-40 m from the wide camera, so at >= 20 m AGL it cannot appear in a horizon-looking frame).
    Such scenes are redrawn: fresh draws first, then the narrow camera, and as a last resort the
    open-sky family. `spec["resample"]` records what happened, so the balance can be audited."""
    last = None
    for attempt in range(attempts):
        force = {}
        if attempt >= attempts // 2:
            if cfgs["sensors"].get("cameras"):
                force["narrow"] = True                    # the student's longest lens (render/cameras.py)
            else:
                force["hfov"] = min(cfgs["sensors"]["hfov_deg"])
        if attempt >= attempts - 2:
            force["family"] = "open_sky"
        try:
            spec, objs = _sample_scene_once(idx, cfgs, n_scenes, attempt, force)
            spec["resample"] = dict(attempt=attempt, forced=force)
            return spec, objs
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"scene {idx}: {last}")


def _sample_scene_once(idx: int, cfgs: dict, n_scenes, attempt: int, force: dict):
    R, S, SIG, TGT = cfgs["render"], cfgs["sensors"], cfgs["signatures"], cfgs["target"]
    ds = R["dataset"]
    n_scenes = n_scenes or ds["n_scenes"]
    seed = int(ds.get("seed0", 0)) + idx
    rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
    edges = [tuple(e) for e in R["size_bins"]["edges_px"]]
    names = R["size_bins"]["names"]
    W, H = S["image"]["width"], S["image"]["height"]

    # ---- strata: bins balanced by index (spec 6), the rest drawn
    bin_idx = idx % len(edges)
    family = _choice(rng, ds["families"])
    lighting = _choice(rng, ds["lighting"])
    hfov = float(rng.choice(S["hfov_deg"], p=np.array(S["hfov_probs"]) / np.sum(S["hfov_probs"])))
    family = force.get("family", family)
    hfov = float(force.get("hfov", hfov))
    rig = None
    if S.get("cameras"):
        # one camera per sensor on a shared mount (render/cameras.py); the student's uncooled LWIR camera sets the
        # grid's field of view. Own stream: the camera draws do not shift any other draw of the scene.
        rig = cams.sample_rig(np.random.default_rng([seed, 43, attempt]), S["cameras"], W, H,
                              narrow=bool(force.get("narrow")))
        hfov = float(rig["grid"]["hfov_deg"])
    state = SIG["lwir"]["states"][lighting]

    # ---- camera
    cc = R["camera"]
    f_px = pl.focal_px(W, hfov)
    vfov = math.degrees(2 * math.atan(H / 2 / f_px))
    lo, hi = cc["look_elevation_deg"][family]
    if family == "open_sky":
        lo = max(lo, vfov / 2 + cc["open_sky_edge_margin_deg"])
    cam_el = look_elevation(rng, cc, family, lo, hi)
    cam_az = float(rng.uniform(0, 360))
    cam_h = float(rng.uniform(*cc["height_m"]))
    cam = pl.Camera(W, H, hfov, (0.0, 0.0, cam_h), cam_az, cam_el, cc.get("roll_deg", 0.0))

    # ---- sun, moon, atmosphere, clouds
    sk = R["sky"]
    near_los = lighting != "night" and rng.uniform() < sk["sun_near_los_prob"]
    s_lo, s_hi = sk["sun_elevation_deg"][lighting]
    if near_los:
        cone = sk["sun_near_los_cone_deg"]
        sun_az = (cam_az + rng.uniform(-cone, cone)) % 360
        sun_el = float(np.clip(cam_el + rng.uniform(-cone, cone), s_lo, s_hi))
    else:
        sun_az, sun_el = float(rng.uniform(0, 360)), float(rng.uniform(s_lo, s_hi))
    atmo = dict(air=float(rng.uniform(*sk["air"])), dust=float(rng.uniform(*sk["dust"])),
                ozone=float(rng.uniform(*sk["ozone"])))
    moon = None
    if lighting == "night" and rng.uniform() < sk["night"]["moon_prob"]:
        moon = dict(elevation_deg=float(rng.uniform(*sk["night"]["moon_elevation_deg"])),
                    azimuth_deg=float(rng.uniform(0, 360)),
                    ratio=_logu(rng, *sk["night"]["moon_to_sun_ratio"]))
    clouds = None
    if rng.uniform() < sk["clouds"]["prob"]:
        c = sk["clouds"]
        clouds = dict(coverage=float(rng.uniform(*c["coverage"])), scale=float(rng.uniform(*c["scale"])),
                      softness=float(rng.uniform(*c["softness"])), brightness=float(rng.uniform(*c["brightness"])),
                      offset=[float(x) for x in rng.uniform(-100, 100, 3)])
    airglow_rgb = _logu(rng, *sk["night"]["airglow_rgb"])
    gtex = {b: dict(scale_m=_logu(rng, *R["ground"]["texture_scale_m"]),      # own pattern per band (spec 7.2)
                    contrast=float(rng.uniform(*R["ground"]["texture_contrast"])), seed=float(rng.uniform(0, 100)))
            for b in BANDS}

    # ---- air temperature, exposure, weather, PSF
    at = R["air_temp_c"]
    if lighting == "day" and rng.uniform() < at["desert_day_prob"]:
        air_c = float(rng.uniform(*at["desert_day"]))
    else:
        air_c = float(rng.uniform(*at["default"]))
    exposure_ms = float(rng.uniform(*S["exposure_ms"][lighting]))
    wc = R["weather"]
    wkind = _choice(rng, wc["probs"])
    vis_km = _logu(rng, *wc["visibility_km"][wkind])
    ratios = {b: float(sg.draw(rng, wc["band_ratio"][wkind][b])) for b in BANDS}
    # ---- Lahore climate (CLAUDE.md sensor set rule 6): a real station hour in the scene's lighting state sets the
    # air temperature, humidity, weather kind, visibility, cloud cover and pressure. Own stream; the draws above are
    # still made (and replaced), so every other draw of the scene is unchanged.
    clim = None
    if (R.get("climate") or {}).get("enabled"):
        crng = np.random.default_rng([seed, 71, attempt])
        clim = climate_conditions(crng, R["climate"], lighting)
        if clim is not None:
            air_c = clim["air_temp_c"]
            if clim["visibility_km"] is not None:
                vis_km = clim["visibility_km"]
            if clim["weather_kind"] != wkind:
                wkind = clim["weather_kind"]
                ratios = {b: float(sg.draw(crng, wc["band_ratio"][wkind][b])) for b in BANDS}
            # the RGB (and SWIR) sky carries the same haze as the loader's extinction (the old dust range 0.5-6 is a
            # 29-350 km sky; Lahore's hours are mostly 2-5 km)
            atmo["dust"] = dust_for_visibility(vis_km)
            atmo["dust_from_visibility"] = True
            okta = clim["cloud_okta"]
            if okta is not None:
                if okta <= 0:
                    clouds = None
                else:
                    c = sk["clouds"]
                    clouds = dict(coverage=float(min(1.0, okta / 8.0)), scale=float(crng.uniform(*c["scale"])),
                                  softness=float(crng.uniform(*c["softness"])),
                                  brightness=float(crng.uniform(*c["brightness"])),
                                  offset=[float(x) for x in crng.uniform(-100, 100, 3)])
    psf = {b: dict(sigma0=float(S["psf_sigma_px"][b][0]), nominal=float(rng.uniform(*S["psf_sigma_px"][b])))
           for b in BANDS}
    # LWIR cooled and MWIR: the same aerosol as the uncooled LWIR camera (its ratio's quantile in their own ranges);
    # PSF from their own stream; the ground's temperature pattern is the LWIR one (same physical surface)
    lw_r = wc["band_ratio"][wkind]["lwir"]
    for b in THERMAL_EXTRA:
        if (wc.get("band_ratio_thermal") or {}).get(wkind, {}).get(b) is not None:
            ratios[b] = sg._requantile(ratios["lwir"], lw_r if sg._is_range(lw_r) else (lw_r, lw_r),
                                       wc["band_ratio_thermal"][wkind][b])
    prng = np.random.default_rng([seed, 67])
    for b in THERMAL_EXTRA:
        if b in S["psf_sigma_px"]:
            psf[b] = dict(sigma0=float(S["psf_sigma_px"][b][0]), nominal=float(prng.uniform(*S["psf_sigma_px"][b])))
    for b in THERMAL_EXTRA:
        gtex[b] = dict(gtex["lwir"])

    # ---- terrain and structures
    gcfg = R["ground"]
    if family == "horizon":
        terrain = bg.Terrain(np.random.default_rng(seed + 17), gcfg, (0.0, 0.0))
        blocks = bg.sample_structures(np.random.default_rng(seed + 23), R["structures"], terrain,
                                      (0.0, 0.0), cam_az, hfov)
    else:
        terrain = bg.flat_terrain(gcfg, (0.0, 0.0))
        blocks = []

    # ---- target
    variant = kg.sample_variant(TGT, seed)
    tmesh = kg.build_mesh({**kg.defaults(TGT), **variant})
    Vt = kg.to_output_frame(tmesh)
    tc = R["target"]
    air_rng = tc["airspeed_mps"]["range"]
    airspeed = float(rng.uniform(*air_rng))
    wind_speed = float(rng.uniform(*tc["wind_mps"]))
    wind_dir = float(rng.uniform(0, 360))
    wind = wind_speed * pl.direction(wind_dir, 0.0)

    def valid_air(min_agl, max_agl=None, max_alt=None, others=()):
        def f(pos, dist, box):
            gz = float(terrain.ground_z(pos[0], pos[1]))
            if pos[2] < gz + min_agl:
                return False
            if max_agl is not None and pos[2] > gz + max_agl:
                return False
            if max_alt is not None and pos[2] > max_alt:
                return False
            if any(pl.boxes_overlap(box, o, R["negatives"]["min_gap_px"]) for o in others):
                return False
            return family != "horizon" or bg.los_clear(cam.C, pos, terrain, blocks)
        return f

    t_agl = tc["min_agl_m"] if family == "horizon" else 0.0
    t_char = 0.8 * float(np.ptp(Vt, axis=0).max())
    tpl = pl.place(cam, Vt, rng, edges, bin_idx, _target_pose_fn(tc),
                   valid_air(t_agl, max_alt=tc.get("max_altitude_m")),
                   uv_fn=_uv_above(cam, t_char, t_agl + 5.0) if family == "horizon" else None,
                   sampling=R["size_bins"]["sampling"], tries=tc["place_tries"])
    tpl.velocity = airspeed * (tpl.R @ np.array([1.0, 0, 0])) + wind
    exp_s = exposure_ms / 1000.0
    tpl.bbox_blur = pl.motion_boxes(cam, Vt, tpl, exp_s)

    # ---- hard negatives
    nc = R["negatives"]
    negs = []
    boxes = [tpl.bbox_blur]
    n_neg = int(rng.integers(nc["count"][0], nc["count"][1] + 1))
    for k in range(n_neg):
        probs = dict(nc["class_probs"])
        for cname, ccfg in nc["classes"].items():
            if family not in ccfg.get("families", ["open_sky", "horizon"]):
                probs.pop(cname, None)
        cls = _choice(rng, probs)
        ccfg = nc["classes"][cls]
        nb = int(rng.choice(len(edges), p=np.array(nc["bin_probs"]) / np.sum(nc["bin_probs"])))
        span = float(rng.uniform(*ccfg["span_m"]))
        mrng = np.random.default_rng(seed * 101 + k)
        mesh = neg.build(cls, mrng, span)
        Vn = mesh["vertices"]
        try:
            if cls == "warm_clutter":
                npl = _place_clutter(rng, cam, Vn, edges, nb, terrain, blocks, boxes, nc, R)
            else:
                agl = ccfg.get("min_agl_m", 0.0)
                npl = pl.place(cam, Vn, rng, edges, nb, _neg_pose_fn(cls, wind_dir),
                               valid_air(agl, ccfg.get("max_agl_m"), others=boxes),
                               uv_fn=_uv_above(cam, 0.8 * float(np.ptp(Vn, axis=0).max()), agl + 2.0)
                               if family == "horizon" else None,
                               sampling=R["size_bins"]["sampling"], tries=150)
        except RuntimeError:
            continue
        speed = float(rng.uniform(*ccfg["speed_mps"]))
        npl.velocity = speed * (npl.R @ np.array([1.0, 0, 0]))
        npl.bbox_blur = pl.motion_boxes(cam, Vn, npl, exp_s)
        boxes.append(npl.bbox_blur)
        negs.append(dict(cls=cls, span_m=span, mesh=mesh, placement=npl))

    # ---- signatures
    sigs = dict(target=sg.sample_target(rng, SIG, state),
                negatives=[sg.sample_negative(rng, SIG, n["cls"], state) for n in negs],
                environment=sg.sample_environment(rng, SIG, state))
    sky = sigs["environment"]["lwir"]["sky"]
    if clim is not None:                             # the station hour's own humidity (no climatological cap)
        sky["rh"] = clim["rh"]
        sky["dew_point_max_c"] = None
    if sky.get("model") == "idso1981":               # humidity -> sky emittance; recorded for sensitivity analysis
        sky.update(thsky.humidity_state(air_c, sky["rh"], sky.get("dew_point_max_c")))
        from thermal.planck_lut import lwir_lut
        sky["dT_at_camera_k"] = float(thsky.sky_dT_k(lwir_lut(), air_c + 273.15, sky["emissivity_zenith"], cam_el))
    caps = SIG["lwir"]["target"].get("surface_temperature_cap_c")
    sg.apply_surface_caps(sigs["target"]["lwir"], air_c, caps)
    sg.apply_surface_caps(sigs["target"]["swir"], air_c, caps, key="emit_dT")
    # own streams: the RGB glow temperature draw and the LWIR sun-glint lobe do not shift any other draw
    sg.add_rgb_glow_draw(np.random.default_rng([seed, 37]), sigs["target"], SIG, state)
    sg.apply_surface_caps(sigs["target"]["rgb"], air_c, caps, key="glow_dT")
    glint = sg.sun_glint_params(np.random.default_rng([seed, 41]), SIG, sun_el, sky.get("emissivity_zenith"))
    wv = atm.scene_params(dict(air_temp_c=air_c, signatures=sigs), R["weather"].get("lwir_water_vapour"))
    swir_hot = rng.uniform() < SIG["swir"]["hot_emission"]["enabled_prob"][lighting]
    # ---- LWIR cooled and MWIR signatures, thermal atmosphere and exhaust plume: own streams, cheap draws only (the
    # radiative transfer runs once per scene at render time, thermal/atmosphere_bands.scene_tables)
    thermal = None
    if all(b in SIG for b in THERMAL_EXTRA):
        sg.add_thermal_bands(np.random.default_rng([seed, 53]), sigs, SIG, [n["cls"] for n in negs])
        TA = R.get("thermal_atmosphere") or {}
        bands_um = {b: tuple(v) for b, v in DEFAULT_THERMAL_BANDS_UM.items()}
        if rig:
            for b, cam_name in (("lwir", "lwir_uncooled"), ("lwir_cooled", "lwir_cooled"), ("mwir", "mwir")):
                if cam_name in rig["cameras"] and rig["cameras"][cam_name].get("band_um"):
                    bands_um[b] = tuple(rig["cameras"][cam_name]["band_um"])
        e_hpa = sky.get("vapour_pressure_hpa")
        if e_hpa is None:
            e_hpa = thsky.vapour_pressure_hpa(air_c, 0.5)
        thermal = dict(bands_um={b: list(v) for b, v in bands_um.items()},
                       layered_lwir=bool(TA.get("enabled")),
                       atmosphere=atb.sample_params(np.random.default_rng([seed, 61]), TA, air_c, e_hpa,
                                                    (clim or {}).get("station_pressure_hpa"))
                       if TA.get("enabled") else None,
                       plume=sg.plume_draws(np.random.default_rng([seed, 59]), SIG),
                       glint=sg.thermal_glints(np.random.default_rng([seed, 41, 1]), SIG, sun_el))

    # ---- crops (output px): union of the shutter boxes + 3 sigma of the widest PSF + margin
    sig_max = max(S["psf_sigma_px"][b][1] for b in ALL_BANDS if b in S["psf_sigma_px"])
    margin = int(math.ceil(3 * sig_max)) + int(R["render"]["crop_margin_px"])

    def crop(b):
        x0, y0 = max(0, int(math.floor(b[0])) - margin), max(0, int(math.floor(b[1])) - margin)
        x1, y1 = min(W, int(math.ceil(b[2])) + margin), min(H, int(math.ceil(b[3])) + margin)
        return [x0, y0, x1, y1]

    rc = R["render"]

    def ss_for(size):
        return int(rc["supersample_small_object"] if size < rc["small_object_px"] else rc["supersample_object"])

    def obj_entry(name, cls, p: pl.Placement, extra):
        az, el = pl.azimuth_elevation(p.position - cam.C)
        gz = float(terrain.ground_z(p.position[0], p.position[1]))
        return dict(name=name, cls=cls, class_id=neg.CLASSES.index(cls),
                    position=[float(x) for x in p.position], matrix_world=pl.matrix_world(p.R, p.position).tolist(),
                    velocity_mps=[float(x) for x in p.velocity], range_m=float(p.distance),
                    ray_azimuth_deg=az, elevation_deg=el, altitude_m=float(p.position[2]), agl_m=float(p.position[2] - gz),
                    bbox_px=_xywh(p.bbox), bbox_blur_px=_xywh(p.bbox_blur), size_px=float(p.size_px),
                    size_bin=names[p.bin], crop_px=crop(p.bbox_blur), supersample=ss_for(p.size_px),
                    pose=p.pose, **p.aspect(cam), **extra)

    objects = [obj_entry("target", "target", tpl, dict(
        lwir_wv_transmission=float(atm.transmittance(tpl.distance, wv, pl.azimuth_elevation(tpl.position - cam.C)[1])),
        airspeed_mps=airspeed, wind_mps=wind_speed, wind_dir_deg=wind_dir,
        projected_size_m=_projected_size(cam, Vt, tpl), hot_visible_cm2=_hot_visible(tmesh, cam, tpl)))]
    for k, n in enumerate(negs):
        objects.append(obj_entry(f"neg{k}", n["cls"], n["placement"], dict(span_m=n["span_m"],
                                                                         extra=n["mesh"]["extra"])))

    g = kg.derive({**kg.defaults(TGT), **variant})
    rot = None
    rcfg = R["render"].get("rotation_blur") or {}
    if rcfg.get("enabled"):
        # own stream: adding rotation does not shift any other draw of the scene
        rrng = np.random.default_rng([seed, 31])
        t_int = dict(rgb=exposure_ms, swir=exposure_ms, lwir=float(rrng.uniform(*S["lwir_integration_ms"])))
        for b, cam_name in (("lwir_cooled", "lwir_cooled"), ("mwir", "mwir")):   # cooled detectors: snapshot integration
            t_int[b] = float((rig or {}).get("cameras", {}).get(cam_name, {}).get("integration_ms") or 1.0)
        rot = rb.scene_rotation(rrng, g, g["prop_rpm"], sigs["target"]["rgb"]["propeller_disc"]["alpha"], t_int, rcfg)
    split = assign_splits(n_scenes, ds["splits"], int(ds.get("seed0", 0)))[idx] if idx < n_scenes else "extra"
    spec = dict(
        scene_id=scene_id(idx), idx=idx, seed=seed, split=split,
        signature_model=SIG["model_revision"], config_overlay=cfgs.get("overlay"),
        engine=kg.engine_metadata(g),
        family=family, lighting=lighting, thermal_state=state,
        camera=dict(hfov_deg=hfov, width=W, height=H, f_px=f_px, sensor_width_mm=S["sensor_width_mm"],
                    focal_mm=pl.focal_mm(S["sensor_width_mm"], hfov), position=[float(x) for x in cam.C],
                    azimuth_deg=cam_az, elevation_deg=cam_el, roll_deg=cam.roll, vfov_deg=vfov,
                    exposure_ms=exposure_ms, supersample_background=int(rc["supersample_background"]),
                    matrix_world=pl.matrix_world(cam.R, cam.C).tolist()),
        sun=dict(elevation_deg=sun_el, azimuth_deg=sun_az, near_line_of_sight=bool(near_los)),
        moon=moon, atmosphere=atmo, clouds=clouds, airglow_rgb=airglow_rgb, ground_texture=gtex,
        air_temp_c=air_c, weather=dict(kind=wkind, visibility_km=vis_km, band_ratio=ratios,
                                       aerosol_scale_height_m=wc.get("aerosol_scale_height_m"),
                                       lwir_water_vapour=wv),
        psf=psf, swir_hot_emission=bool(swir_hot), sun_glint=glint, cameras=rig, thermal=thermal,
        climate=(dict(source=R["climate"]["source"], **clim["record"]) if clim is not None else None),
        target_variant=dict(seed=seed, variant_index=variant["variant_index"], variant_seed=variant["variant_seed"],
                            params=variant, span_m=g["span_m"], root_chord=g["root_chord"],
                            tip_over_root=g["tip_over_root"], nose_over_root=g["nose_over_root"],
                            sweep_deg=g["sweep_deg"], forebody_dia=g["forebody_dia"],
                            fin_up_height=g["fin_up_height"], fin_down_height=g["fin_down_height"],
                            prop_dia=g["prop_dia"], length_m=g["length"] * g["span_m"]),
        objects=objects, signatures=_jsonable(sigs),
        terrain=dict(hill_amplitude_m=terrain.A, hill_wavelength_m=terrain.lam, edge_ridge_m=terrain.ridge,
                     grid=terrain.grid, size_m=terrain.size) if family == "horizon" else None,
        structures=[b.__dict__ for b in blocks], rotation=rot,
    )
    objs = dict(camera=cam, terrain=terrain, blocks=blocks, target_mesh=tmesh, target_verts=Vt,
                negatives=negs, target_placement=tpl)
    return spec, objs


def _place_clutter(rng, cam, Vn, edges, nb, terrain, blocks, boxes, nc, R):
    """Warm clutter on a roof (when a building is in view) or on the terrain."""
    def pose(r):
        h = float(r.uniform(0, 360))
        return pl.object_rotation(h), dict(heading_deg=h, pitch_deg=0.0, roll_deg=0.0)

    def valid(pos, dist, box):
        if any(pl.boxes_overlap(box, o, nc["min_gap_px"]) for o in boxes):
            return False
        top = pos + np.array([0, 0, np.ptp(Vn[:, 2]) * 0.5])
        others = [b for b in blocks if not (abs(b.cx - pos[0]) < b.w and abs(b.cy - pos[1]) < b.d)]
        return bg.los_clear(cam.C, top, terrain, others, clearance=0.2)

    buildings = [b for b in blocks if b.kind == "building"]
    if buildings and rng.uniform() < 0.5:
        for _ in range(40):
            b = buildings[int(rng.integers(len(buildings)))]
            P = b.roof_point(rng)
            R_, pose_d = pose(rng)
            Vw = Vn @ R_.T
            pos = P - np.array([0, 0, Vw[:, 2].min()])
            box = cam.bbox(Vw + pos)
            if box is None or not pl.box_inside(box, cam.W, cam.H, 1.0):
                continue
            p = pl.box_size(box)
            if pl.bin_index(p, edges) != nb or not valid(pos, 0, box):
                continue
            return pl.Placement(position=pos, R=R_, distance=float(np.linalg.norm(pos - cam.C)), bbox=box,
                                size_px=p, bin=nb, pose=pose_d)
    return pl.place_on_ground(cam, Vn, rng, edges, nb, terrain.ground_z, pose, valid,
                              sampling=R["size_bins"]["sampling"], tries=150)


def _hot_visible(mesh: dict, cam: pl.Camera, p: pl.Placement) -> dict:
    """Visible engine surface areas per slot, ray cast with airframe occlusion (geometry, not radiance).
    engine_total sums every engine slot; hot_total only the hottest (heads, barrels, exhaust stubs)."""
    areas = _surface_visible(mesh, cam, p, kg.ENGINE_SLOTS)
    return dict(areas, engine_total=float(sum(areas.values())),
                hot_total=float(sum(areas[s] for s in kg.HOT_SLOTS)))


def _surface_visible(mesh: dict, cam: pl.Camera, p: pl.Placement, slots) -> dict:
    """Geometric visible areas (cm^2) including occlusion."""
    to_cam = cam.C - p.position
    d = p.R.T @ (to_cam / np.linalg.norm(to_cam))          # body frame = kal_measure's output frame
    return {slot: float(km.hot_visibility(mesh, d[None], (slot,))[0] * 1e4) for slot in slots}


def _projected_size(cam, V, p: pl.Placement) -> dict:
    """Extent of the target perpendicular to the ray, in metres, along image x and y."""
    Vw = V @ p.R.T
    return dict(w=float(np.ptp(Vw @ cam.R[:, 0])), h=float(np.ptp(Vw @ cam.R[:, 1])))


def _xywh(b) -> dict:
    return dict(x=float(b[0]), y=float(b[1]), w=float(b[2] - b[0]), h=float(b[3] - b[1]))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
