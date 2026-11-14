"""Tasks M1-03 to M1-06: stub environment, geometry and flight, evidence and confirmation, the environment itself.
The slower acceptance checks (oracle, random against oracle, speed, 1,000 stub episodes) are in
`python -m rl.scripts.check wp1`."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl import contracts as C
from rl.env import SkyEnv, StubSkyEnv
from rl.env import evidence as ev
from rl.env import flight as fl
from rl.env import geometry as geo
from rl.env.scenario import sample_episode_spec

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================ M1-03 stub environment
def test_stub_env_passes_gymnasium_checker():
    check_env(StubSkyEnv(), skip_render_check=True)


def test_stub_env_runs_random_episodes():
    env = StubSkyEnv(end_prob=0.05)
    for ep in range(50):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        assert set(C.INFO_KEYS) <= set(info)


# ============================================================================ M1-04 geometry
def test_known_direction_lands_in_expected_cell():
    assert geo.cell_of(-30.0, 5.0) == (0, 0)
    assert geo.cell_of(-0.1, 12.4) == (1, 5)
    assert geo.cell_of(0.0, 12.5) == (1, 6)
    assert geo.cell_of(29.9, 19.9) == (2, 11)
    assert geo.cell_of(30.0, 10.0) is None and geo.cell_of(0.0, 4.9) is None
    assert geo.cell_center(1, 6) == (2.5, 12.5)


@pytest.mark.parametrize("band,zoom,cols,rows", [
    ("lwir", 0, 9, 3), ("lwir", 1, 3, 3), ("lwir", 2, 1, 1),
    ("rgb", 0, 9, 3),                                   # the RGB zoom matches the uncooled field
    ("lwir_cooled", 0, 5, 3),                           # widest 24.9 deg: only the central cells
    ("mwir", 0, 7, 3),                                  # widest 36.4 deg
])
def test_zoom_coverage(band, zoom, cols, rows):
    f, _ = geo.overlap_fractions((1, 6), geo.band_hfov(band, zoom))
    cov = geo.covered(f)
    assert cov.any(axis=0).sum() == cols and cov.any(axis=1).sum() == rows


def test_cooled_lwir_at_widest_zoom_covers_only_central_cells():
    cov_u = geo.covered(geo.overlap_fractions((1, 6), geo.band_hfov("lwir", 0))[0])
    cov_c = geo.covered(geo.overlap_fractions((1, 6), geo.band_hfov("lwir_cooled", 0))[0])
    assert cov_c.sum() < cov_u.sum() and not (cov_c & ~cov_u).any()
    assert set(np.flatnonzero(cov_c.any(axis=0))) == {4, 5, 6, 7, 8}


def test_frame_share_sums_to_covered_part_of_frame():
    _, g = geo.overlap_fractions((1, 6), 15.0)
    assert 0.0 < g.sum() <= 1.0 + 1e-9


def test_size_px_formula():
    hfov = 46.0
    ifov = math.radians(hfov) / 640
    assert geo.size_px(2.0, 1000.0, hfov) == pytest.approx(2.0 / (1000.0 * ifov))


# ============================================================================ M1-04 flight: the port against the original
def _original_flight_path():
    """render/track_sequence.flight_path, compiled from its own source: that module imports Blender."""
    src = (ROOT / "render" / "track_sequence.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "flight_path")
    ns = {"np": np, "math": math, "G": 9.81, "pl": SimpleNamespace(direction=fl.direction)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "track_sequence.flight_path", "exec"), ns)
    return ns["flight_path"]


def test_flight_port_matches_original_on_fixed_input():
    a = SimpleNamespace(duration=26.0, fps=25.0, start_m=1450.0, offset_m=170.0, alt_m=150.0, weave_deg=10.0,
                        weave_s=22.0, speed=44.0)
    orig = _original_flight_path()(np.zeros(3), 12.0, a)
    port = fl.flight_path(np.zeros(3), 12.0, a)
    assert len(orig) == len(port)
    for o, p in zip(orig, port):
        assert np.array_equal(o["position"], p["position"]) and o["heading"] == p["heading"]
    vec = fl.flight_positions(np.zeros(3), 12.0, a)
    assert np.allclose(vec, np.array([p["position"] for p in port]), atol=1e-6)


def test_track_climbs_through_the_sector():
    spec = next(s for s in (sample_episode_spec(i) for i in range(50)) if s.drone_present)
    tr = fl.Track(spec, C.FROZEN["episode"]["max_steps"])
    assert geo.cell_of(*tr.at(0)[:2]) is not None
    assert 0 < tr.exit_step
    k = min(tr.exit_step - 1, 200)
    assert tr.el_deg[k] > tr.el_deg[0] and tr.range_m[k] < tr.range_m[0]


# ============================================================================ M1-05 evidence and confirmation
def test_llr_matches_hand_computation():
    want = 2 * math.log(0.2 / 0.01) + 3 * math.log(0.8 / 0.99)
    assert ev.llr_counts(2, 5, 0.2, 0.01) == pytest.approx(want)
    assert ev.expected_llr(0.01, 5, 0.2, 0.01) < 0 < ev.expected_llr(0.3, 5, 0.2, 0.01)


def test_cusum_clamps_at_zero_confirms_and_resets():
    cu = ev.Cusum((3, 12))
    cov = np.zeros((3, 12), bool)
    cov[1, 6] = True
    cu.update(np.full((3, 12), -5.0), cov)
    assert cu.L.min() == 0.0
    cu.update(np.full((3, 12), 6.0), cov)
    cu.update(np.full((3, 12), 6.0), cov)
    assert cu.L[1, 6] == 12.0 and cu.L.sum() == 12.0     # uncovered cells untouched
    assert cu.crossing(10.0, cov) == (1, 6) and cu.crossing(13.0, cov) is None
    cu.reset_cell((1, 6))
    assert cu.L.sum() == 0.0


def test_false_confirmations_stay_below_lordens_bound():
    """Drone-free detections, one confirmation test per cell: false confirmations per cell-step <= e^-A."""
    rng = np.random.default_rng(0)
    A, n, pd_nom, pfa = 4.0, 5, 0.2, 0.01
    cells, steps = 2000, 400
    L = np.zeros(cells)
    alarms = 0
    for _ in range(steps):
        k = rng.binomial(n, pfa, size=cells)
        L = np.maximum(0.0, L + ev.llr_counts(k, n, pd_nom, pfa))
        hit = L >= A
        alarms += int(hit.sum())
        L[hit] = 0.0
    assert alarms / (cells * steps) <= math.exp(-A)


def test_motion_spread_never_raises_the_maximum():
    rng = np.random.default_rng(1)
    for _ in range(20):
        cu = ev.Cusum((3, 12))
        cu.L = rng.exponential(3.0, size=(3, 12))
        top = cu.L.max()
        cu.spread(C.FROZEN["confirmation"]["spread_rho"])
        assert cu.L.max() == top


# ============================================================================ M1-06 the environment
def test_sky_env_passes_gymnasium_checker():
    check_env(SkyEnv(), skip_render_check=True)


def _rollout(env, seed, actions):
    obs, info = env.reset(seed=seed)
    out = [(obs, 0.0)]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((obs, r))
        if term or trunc:
            break
    return out


def test_one_seed_gives_an_identical_episode_twice():
    actions = list(np.random.default_rng(3).integers(0, C.N_ACTIONS, size=300))
    a, b = _rollout(SkyEnv(), 7, actions), _rollout(SkyEnv(), 7, actions)
    assert len(a) == len(b)
    for (oa, ra), (ob, rb) in zip(a, b):
        assert ra == rb and all(np.array_equal(oa[k], ob[k]) for k in oa)


def test_drone_free_share_is_thirty_percent():
    share = np.mean([not sample_episode_spec(s).drone_present for s in range(10_000)])
    assert 0.28 <= share <= 0.32


def test_reward_components_sum_to_the_reward():
    env = SkyEnv()
    rng = np.random.default_rng(4)
    for seed in range(3):
        env.reset(seed=seed)
        for _ in range(400):
            _, r, term, trunc, info = env.step(int(rng.integers(C.N_ACTIONS)))
            assert r == pytest.approx(sum(info["reward_terms"].values()))
            assert set(info["reward_terms"]) == set(C.REWARD_TERMS)
            if term or trunc:
                break


def test_replaying_a_spec_gives_the_seeded_episode():
    spec = sample_episode_spec(21)
    o1, _ = SkyEnv().reset(seed=21)
    o2, _ = SkyEnv().reset(options={"episode_spec": spec.to_dict()})
    assert all(np.array_equal(o1[k], o2[k]) for k in o1)


def test_drone_free_episode_runs_to_the_time_limit():
    seed = next(s for s in range(100) if not sample_episode_spec(s).drone_present)
    env = SkyEnv()
    env.reset(seed=seed)
    steps, done = 0, False
    while not done:
        _, _, term, trunc, info = env.step(C.DWELL)
        steps += 1
        done = term or trunc
    assert trunc and not term and steps == C.FROZEN["episode"]["max_steps"]
    assert not info["truth"]["ever_confirmable"]


def test_learned_declaration_adds_one_action_only_when_on():
    assert SkyEnv().action_space.n == C.N_ACTIONS
    assert SkyEnv(learned_declaration=True).action_space.n == C.N_ACTIONS_LEARNED_DECLARE
