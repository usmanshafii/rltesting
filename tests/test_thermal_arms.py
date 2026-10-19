"""The two thermal arms of pilot 2 (docs/privileged_teacher_case.md 6c): configs/m1_low_heat.yaml (engine at the air
temperature, no plume) and configs/m1_electric_like.yaml (the L550E's cooling surfaces at an electric drive's loss heat,
thermal/engine_heat.py S7). They extend M1's overlay and change only the target's propulsion heat, so a scene rendered
with either is M1's scene with another engine temperature."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

from render import sampling as S  # noqa: E402
from thermal import engine_heat as eh  # noqa: E402

ARMS = ("configs/m1_low_heat.yaml", "configs/m1_electric_like.yaml")


def _profiles(cfgs):
    return cfgs["signatures"]["lwir"]["target"]["engine_profiles"]


def test_m1_render_contract_is_unchanged_by_extends():
    import m1_det2 as D
    meta = json.loads((ROOT / "out/m1/dataset/meta/s000000.json").read_text()) \
        if (ROOT / "out/m1/dataset/meta/s000000.json").exists() else None
    exp = D.expected_contract()["config_sha256"]
    assert exp == "a2a7732ae30fe18c886a51666110daa0beabb62fc3a8943e041c46c4d0bb72f2"      # the 180 rendered scenes'
    if meta is not None:
        assert meta["extra"]["render_contract"]["config_sha256"] == exp


def test_arms_extend_m1_and_change_only_the_engine_heat():
    m1 = S.load_configs(overlay="configs/m1.yaml")
    for arm in ARMS:
        c = S.load_configs(overlay=arm)
        assert c["overlay"]["extends"] == ["configs/m1.yaml"]
        assert c["signatures"]["plume"]["enabled"] is False
        assert {k: v["weight"] for k, v in _profiles(c).items()} == {k: v["weight"] for k, v in _profiles(m1).items()}
        for name, p in _profiles(c).items():                 # same parts, every value still a 2-element range
            assert set(p) == set(_profiles(m1)[name]) and all(len(v) == 2 for k, v in p.items() if k != "weight")
        for key in ("render", "sensors", "target"):
            assert c[key] == m1[key]
        sig, sig1 = json.loads(json.dumps(c["signatures"])), json.loads(json.dumps(m1["signatures"]))
        for s in (sig, sig1):
            s["lwir"]["target"].pop("engine_profiles")
            s["plume"].pop("enabled")
        assert sig == sig1


def test_low_heat_is_zero_engine_heat():
    c = S.load_configs(overlay=ARMS[0])
    assert all(v == [0.0, 0.0] for p in _profiles(c).values() for k, v in p.items() if k != "weight")


def test_electric_like_lines_are_the_engine_model():
    c = S.load_configs(overlay=ARMS[1])
    model = eh.electric_like_yaml_ranges()
    for name, p in _profiles(c).items():
        assert {k: tuple(v) for k, v in p.items() if k != "weight"} == {k: tuple(v) for k, v in model[name].items()}
    combustion = eh.yaml_ranges()
    for name in model:                                         # an electric drive runs far colder than the engine
        for part in ("engine_head", "engine_barrel", "engine_heatsink"):
            assert model[name][part][1] < combustion[name][part][0] + 0.5 * (combustion[name][part][1] - combustion[name][part][0])
        assert model[name]["exhaust_stub"] == (0.0, 0.0)
    lo, hi = eh.MOTOR_EFFICIENCY                               # the sourced efficiencies sit inside the range used
    src = eh.MOTOR_EFFICIENCY_SOURCES
    assert lo <= src["ie3_15kw"][0] and src["ie4_37kw"][1] <= hi and lo <= src["hepperle_motor"] <= hi


def test_a_scene_under_either_arm_is_m1s_scene_with_another_engine_temperature():
    def flat(d, pre=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(flat(v, f"{pre}{k}."))
        elif isinstance(d, (list, tuple)) and d and not all(isinstance(x, (int, float)) for x in d):
            for i, v in enumerate(d):
                out.update(flat(v, f"{pre}{i}."))
        else:
            out[pre[:-1]] = d
        return out

    def spec(overlay):
        s = S.sample_scene(46, S.load_configs(overlay=overlay))[0]
        return flat(json.loads(json.dumps(s, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))

    m1 = spec("configs/m1.yaml")
    engine_slots = {"engine_barrel", "engine_case", "engine_flange", "engine_head", "engine_heatsink", "engine_intake",
                    "exhaust_stub", "tail_cone"}
    for arm in ARMS:
        a = spec(arm)
        for k in set(m1) | set(a):
            if m1.get(k) == a.get(k) or k.startswith(("config_overlay.", "thermal.plume.")):
                continue
            parts = k.split(".")
            assert parts[:2] == ["signatures", "target"] and parts[3] in engine_slots and \
                parts[-1] in ("dT", "emit_dT", "glow_dT", "dT_uncapped", "capped", "emit_dT_capped", "emit_dT_uncapped"), k
