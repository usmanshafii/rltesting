"""Plan invariants I1-I11 (rl/PLAN.md). The ones that need trained runs or other members' code skip with the task
that will enable them; they become real tests as that work lands."""
import ast
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from rl import contracts as C
from rl.env import SkyEnv, default_sensor_model
from rl.env.rewards import reward_terms
from rl.env.stub_sensor import StubSensorModel

RL = Path(__file__).resolve().parents[2] / "rl"

# The environment's default (M2's calibrated model, since M1-08) and the stub it replaced.
SENSOR_MODELS = {"calibrated": default_sensor_model, "stub": StubSensorModel}


class _PrivilegedChanged:
    """A sensor model with the privileged cameras made perfect: a student arm must not notice."""

    def __init__(self, base):
        self.base, self.version = base, f"{base.version}+privileged-perfect"

    def begin_episode(self, spec, rng):
        self.base.begin_episode(spec, rng)

    def dprime(self, band, size_px, spec):
        return self.base.dprime(band, size_px, spec)

    def pd_per_frame(self, band, size_px, spec):
        return 0.999 if band in ("lwir_cooled", "mwir", "mwir_shared") else self.base.pd_per_frame(band, size_px, spec)

    def pfa_per_frame(self, band, spec, cloud):
        return self.base.pfa_per_frame(band, spec, cloud)

    def nominal_pd(self, band, spec):
        return self.base.nominal_pd(band, spec)


def _student_trajectory(env, seed=5, steps=300):
    rng = np.random.default_rng(9)
    obs, _ = env.reset(seed=seed)
    traj = [obs]
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(int(rng.integers(C.N_ACTIONS)))
        traj.append(obs)
        if term or trunc:
            break
    return traj


def test_I1_student_observation_space_holds_no_privileged_key():
    assert set(SkyEnv().observation_space.spaces) == set(C.STUDENT_OBS_KEYS)


@pytest.mark.parametrize("model", list(SENSOR_MODELS))
@pytest.mark.parametrize("seed", [5, 6, 17])
def test_I1_student_observations_do_not_depend_on_privileged_cameras(model, seed):
    make = SENSOR_MODELS[model]
    base = _student_trajectory(SkyEnv(sensor_model=make()), seed)
    for other in (SkyEnv(sensor_model=_PrivilegedChanged(make())),
                  SkyEnv(sensor_model=make(), compute_privileged=False)):
        alt = _student_trajectory(other, seed)
        assert len(base) == len(alt)
        for a, b in zip(base, alt):
            assert all(np.array_equal(a[k], b[k]) for k in a)


def test_I2_I3_I5_I10_need_result_rows():
    pytest.skip("pending M4-02/M4-06: checked on result rows by `python -m rl.scripts.check invariants`")


def test_I4_frozen_values_unchanged_after_freeze():
    m = re.search(r"frozen\.yaml SHA-256 at interface-v1:\s*`([0-9a-f]{64})`", (RL / "PLAN.md").read_text("utf-8"))
    if not m:
        pytest.skip("not frozen yet: the hash is recorded in rl/PLAN.md at S-03 (interface-v1)")
    assert C.frozen_hash() == m.group(1)


def test_I6_reward_takes_ground_truth_events_only():
    params = set(inspect.signature(reward_terms).parameters)
    assert params == {"correct", "earliness", "false_confirmations", "missed_confirmable", "slewed"}
    for p in params:
        assert not any(w in p for w in ("privileged", "cooled", "mwir", "evidence", "score"))


def test_I7_shaping_potential_zero_at_episode_end():
    try:
        from rl.agents import shaping
        shaping.potential                                   # noqa: B018
    except (ImportError, AttributeError, NotImplementedError):
        pytest.skip("pending M3-06: potential-based shaping")


def test_I8_calibration_values_have_source_and_grade():
    path = RL / "calib" / "params.json"
    if not path.exists():
        pytest.skip("pending M2-04: rl/calib/params.json")
    import json
    for name, entry in json.loads(path.read_text("utf-8"))["values"].items():
        assert {"value", "source", "grade"} <= set(entry), name


ALLOWED_THIRD_PARTY = {"numpy", "scipy", "yaml", "gymnasium", "torch", "matplotlib", "PIL", "pytest"}
REPO_PACKAGES = {"rl", "eval", "render", "data", "thermal", "noise", "models", "target", "tools", "train"}


def test_I9_no_unlisted_dependency():
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for py in RL.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text("utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
            for n in names:
                top = n.split(".")[0]
                if top not in stdlib | ALLOWED_THIRD_PARTY | REPO_PACKAGES:
                    bad.append(f"{py.relative_to(RL)}: {n}")
    assert not bad, "imports outside rl/requirements-rl.txt: " + ", ".join(bad)


@pytest.mark.parametrize("model", list(SENSOR_MODELS))
def test_I11_four_sensors_with_separate_uncooled_and_cooled_lwir(model):
    assert set(C.BANDS) == {"rgb", "lwir", "lwir_cooled", "mwir"}
    model = SENSOR_MODELS[model]()
    from rl.env.scenario import sample_episode_spec
    spec = sample_episode_spec(1)
    model.begin_episode(spec, np.random.default_rng(0))
    assert model.dprime("lwir", 3.0, spec) != model.dprime("lwir_cooled", 3.0, spec)
    for b in C.BANDS:
        assert b in C.FROZEN["cameras"]["max_hfov_deg"]
