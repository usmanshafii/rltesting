"""Acceptance checks per work package and the plan invariants (rl/PLAN.md, "Tests and verification").

    python -m rl.scripts.check wp1            M1's environment checks (M1-03, M1-06, M1-07)
    python -m rl.scripts.check wp2 ... wp5    the other work packages (PENDING until their tasks land)
    python -m rl.scripts.check invariants     I1-I11 (tests/rl/test_invariants.py)
    python -m rl.scripts.check --quick        a short end-to-end run
    python -m rl.scripts.check all

Each check prints PASS, FAIL or PENDING (the task that will enable it) and writes a JSON summary to
rl/results/checks/<name>.json. The exit code is 1 if anything FAILED; PENDING does not fail a run, but a gate is
passed only when its checks are PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from rl import contracts as C

RL = Path(__file__).resolve().parents[1]
OUT = RL / "results" / "checks"


class Report:
    def __init__(self, name: str):
        self.name, self.rows = name, []

    def add(self, check: str, status: str, detail: str = "", **values):
        self.rows.append(dict(check=check, status=status, detail=detail, **values))
        print(f"  [{status:7s}] {check}" + (f": {detail}" if detail else ""), flush=True)

    def save(self) -> bool:
        OUT.mkdir(parents=True, exist_ok=True)
        meta = dict(name=self.name, when=time.strftime("%Y-%m-%d %H:%M"), env_version=C.ENV_VERSION,
                    interface_version=C.INTERFACE_VERSION, frozen_sha256=C.frozen_hash(), python=sys.version.split()[0],
                    machine=platform.processor() or platform.machine(), cores=os.cpu_count())
        (OUT / f"{self.name}.json").write_text(json.dumps(dict(meta=meta, rows=self.rows), indent=1, default=str))
        return not any(r["status"] == "FAIL" for r in self.rows)


def _run(env, policy, seed: int) -> dict:
    """One episode; policy(env, obs) -> action. Returns what happened."""
    obs, info = env.reset(seed=seed)
    ret, steps, done, false_n = 0.0, 0, False, 0
    while not done:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        ret += r
        steps += 1
        false_n += int(info["events"]["false_confirmation"])
        done = term or trunc
    t = info["truth"]
    return dict(seed=seed, present=t["drone_present"], confirmable=t["ever_confirmable"],
                correct=bool(info["events"]["correct"]), false=false_n, steps=steps, ret=ret)


def wp1(episodes: int = 200, random_episodes: int = 60) -> bool:
    """M1's checks: stub (M1-03), environment basics (M1-06), oracle and speed (M1-07)."""
    from gymnasium.utils.env_checker import check_env

    from rl.env import OraclePolicy, SkyEnv, StubSkyEnv
    from rl.env.scenario import sample_episode_spec

    rep = Report("wp1")
    print("WP1: environment")
    stub = StubSkyEnv(end_prob=0.05)
    check_env(stub, skip_render_check=True)
    for ep in range(1000):
        stub.reset(seed=ep)
        done = False
        while not done:
            _, _, te, tr, _ = stub.step(stub.action_space.sample())
            done = te or tr
    rep.add("M1-03 stub: check_env and 1,000 random episodes", "PASS")

    env = SkyEnv()
    check_env(env, skip_render_check=True)
    rep.add("M1-06 SkyEnv: Gymnasium check_env", "PASS")
    share = float(np.mean([not sample_episode_spec(s).drone_present for s in range(10_000)]))
    rep.add("M1-06 drone-free share 30 +- 2 %", "PASS" if 0.28 <= share <= 0.32 else "FAIL", f"{share:.3f}",
            value=share)

    oracle = OraclePolicy()
    runs = [_run(env, lambda e, o: oracle(e), 10_000 + s) for s in range(episodes)]
    conf = [r for r in runs if r["present"] and r["confirmable"]]
    o_share = float(np.mean([r["correct"] for r in conf])) if conf else float("nan")
    need = float(C.FROZEN["gates"]["oracle_min_confirm_share"])
    rep.add(f"M1-07 oracle confirms >= {need:.0%} of confirmable drones",
            "PASS" if conf and o_share >= need else "FAIL", f"{o_share:.3f} of {len(conf)} confirmable drones",
            value=o_share, n=len(conf))

    rng = np.random.default_rng(0)
    rand = [_run(env, lambda e, o: int(rng.integers(C.N_ACTIONS)), 10_000 + s) for s in range(random_episodes)]
    o_sub = runs[:random_episodes]                      # the same episodes as the oracle's first ones

    def rate(rs):
        p = [r for r in rs if r["present"]]
        return float(np.mean([r["correct"] for r in p])) if p else float("nan")

    r_rate, o_rate = rate(rand), rate(o_sub)
    rep.add("M1-07 random < oracle (drones confirmed)", "PASS" if r_rate < o_rate else "FAIL",
            f"random {r_rate:.3f}, oracle {o_rate:.3f}", random=r_rate, oracle=o_rate)
    try:
        from rl.baselines.raster import RasterPolicy
        raster = RasterPolicy()
        ras = [_run(env, lambda e, o: raster(o), 10_000 + s) for s in range(random_episodes)]
        ok = r_rate < rate(ras) < o_rate
        rep.add("M1-07 random < raster < oracle", "PASS" if ok else "FAIL", f"raster {rate(ras):.3f}",
                raster=rate(ras))
    except NotImplementedError:
        rep.add("M1-07 random < raster < oracle", "PENDING", "needs M4-03 (rl/baselines/raster.py)")

    steps, t0 = 0, time.perf_counter()
    env.reset(seed=1)
    while steps < 5000:
        _, _, te, tr, _ = env.step(int(rng.integers(C.N_ACTIONS)))
        steps += 1
        if te or tr:
            env.reset(seed=steps)
    sps = steps / (time.perf_counter() - t0)
    need_sps = float(C.FROZEN["gates"]["min_steps_per_second_per_core"])
    rep.add(f"M1-07 >= {need_sps:.0f} steps per second per core", "PASS" if sps >= need_sps else "FAIL",
            f"{sps:.0f} steps/s on this laptop (a busy laptop measures low: rerun when idle)", value=sps)
    return rep.save()


def pending(name: str, tasks: str) -> bool:
    rep = Report(name)
    print(f"{name.upper()}:")
    rep.add(f"{name} acceptance checks", "PENDING", f"needs {tasks}")
    return rep.save()


def invariants() -> bool:
    import pytest
    print("Invariants I1-I11 (tests/rl/test_invariants.py)")
    code = pytest.main(["-q", "-p", "no:cacheprovider", str(RL.parent / "tests" / "rl" / "test_invariants.py")])
    rep = Report("invariants")
    rep.add("I1-I11 (skips = pending tasks)", "PASS" if code == 0 else "FAIL", f"pytest exit code {code}")
    return rep.save()


def quick() -> bool:
    from rl.env import OraclePolicy, SkyEnv
    rep = Report("quick")
    print("Quick end-to-end")
    env, oracle, t0 = SkyEnv(), OraclePolicy(), time.perf_counter()
    runs = [_run(env, lambda e, o: oracle(e), 30_000 + s) for s in range(5)]
    rep.add("environment and oracle, 5 episodes", "PASS", f"{time.perf_counter() - t0:.1f} s",
            correct=sum(r["correct"] for r in runs))
    try:
        from rl.agents.ppo import train
        train({"quick": True})
        rep.add("PPO quick run", "PASS")
    except NotImplementedError:
        rep.add("PPO quick run", "PENDING", "needs M3-03")
    return rep.save()


WP_PENDING = {"wp2": "M2-08 (calibrated sensor model)", "wp3": "M3-05 (toy gate)",
              "wp4": "M4-03/M4-04 (baselines, evaluation)", "wp5": "S-04 (integration)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["wp1", "wp2", "wp3", "wp4", "wp5", "invariants", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=200, help="oracle episodes for wp1")
    a = ap.parse_args(argv)
    if a.quick:
        return 0 if quick() else 1
    ok = True
    if a.what in ("wp1", "all"):
        ok &= wp1(a.episodes)
    for wp, tasks in WP_PENDING.items():
        if a.what in (wp, "all"):
            ok &= pending(wp, tasks)
    if a.what in ("invariants", "all"):
        ok &= invariants()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
