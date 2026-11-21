"""M2 calibration: real data, numerical checks, and the documented missing-render fallback."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rl import contracts as C
from rl.calib.join_specs import join_census_with_specs
from rl.calib.fit_dprime import fit, predict_dprime
from rl.calib.score_model import GaussianScoreModel, pd_from_dprime
from rl.calib.clutter import split_spread, dwell_dprime, NoiseClutter
from rl.calib.sensor_model import CalibratedSensorModel
from rl.env.scenario import sample_episode_spec
from eval.detectability import d_after

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "docs/reference/privileged/data/census.json"

@pytest.fixture(scope="module")
def rows():
    return join_census_with_specs(str(CENSUS))

@pytest.fixture(scope="module")
def fitted(rows):
    return fit(rows)

def test_census_join_matches_census(rows):
    assert len(rows) == 126
    assert len({r["scene_id"] for r in rows}) == 126
    for r in rows:
        assert r["split"] != "test"
        assert r["join_error"]["range_relative"] < .001
        assert r["join_error"]["size_relative"] < .001
        assert 0 <= r["aspect_deg"] <= 180
        assert r["engine_setting"]

def test_join_rejects_drift(tmp_path):
    data = json.loads(CENSUS.read_text())
    sid = next(iter(data))
    data = {sid: data[sid]}
    data[sid]["range_m"] *= 1.1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="join"):
        join_census_with_specs(str(p))

def test_fit_reproduces_census_ratios(rows, fitted):
    for b, expected in [("lwir_cooled", 1.23), ("mwir", 1.50)]:
        seen = [r for r in rows if r["dprime"][b]["d"] > 0 and r["dprime"]["lwir"]["d"] > 0]
        ratio = np.median([predict_dprime(fitted,b,r) / predict_dprime(fitted,"lwir",r) for r in seen])
        assert abs(ratio / expected - 1) <= .10
    for b in C.BANDS:
        m = fitted["bands"][b]
        assert len(m["cv_folds"]) == 5
        assert m["cv_log_rmse"] > 0
        assert m["residual_std"] > 0
        assert all(s.startswith("s") for s in m["fit_scene_ids"])
        assert not set(m["fit_scene_ids"]) & {r["scene_id"] for r in rows if r["split"] == "val"}

def test_validation_does_not_change_fit(rows, fitted):
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["split"] == "val":
            for b in C.BANDS:
                r["dprime"][b]["d"] *= 100
    other = fit(changed)
    for b in C.BANDS:
        assert other["bands"][b]["coefficients"] == fitted["bands"][b]["coefficients"]
        assert other["bands"][b]["residual_std"] == fitted["bands"][b]["residual_std"]

def test_score_model():
    model = GaussianScoreModel()
    rng = np.random.default_rng(631)
    assert abs(np.mean(model.sample(rng, 0, 200_000) > model.threshold) - .01) < .002
    for d in (5, 10, 43, 114, 500):
        p = pd_from_dprime("lwir", d)
        assert model.detection_probability(d) == pytest.approx(p)
        empirical = np.mean(model.sample(rng, d, 60_000) > model.threshold)
        assert abs(empirical - p) < .012
    assert pd_from_dprime("rgb", 0) == 0
    with pytest.raises(ValueError):
        pd_from_dprime("swir", 10)
    with pytest.raises(ValueError):
        model.mean(float("nan"))

def test_frame_false_alarm_allocation():
    model = GaussianScoreModel()
    for n in (1, 4, 36):
        p = model.cell_false_alarm_probabilities(np.full(n, 1/n))
        assert 1 - np.prod(1-p) == pytest.approx(.01)
    with pytest.raises(ValueError):
        model.cell_false_alarm_probabilities(np.array([.8,.8]))

def test_clutter_and_dwell():
    rng = np.random.default_rng(46)
    model = NoiseClutter({b:.5 for b in C.BANDS})
    clutter, noise = model.draw(rng, 40_000)
    assert np.corrcoef(clutter["lwir"],clutter["lwir_cooled"])[0,1] > .99
    assert abs(np.corrcoef(noise["lwir"],noise["lwir_cooled"])[0,1]) < .025
    for dp in ({"d":8.,"d_noise":10.}, {"d":9.,"d_noise":8.}, {"d":0.,"d_noise":0.}):
        for n in (1,5,50):
            assert dwell_dprime(dp,n) == d_after(dp,n)
    with pytest.raises(ValueError):
        dwell_dprime({"d":1.,"d_noise":1.},0)
    c,n = split_spread({"d":8.,"d_noise":10.})
    assert c*c+n*n == pytest.approx(1/64)

def test_sensor_model_reproducible_and_contract():
    s=sample_episode_spec(11)
    a,b=CalibratedSensorModel(),CalibratedSensorModel()
    assert isinstance(a,C.SensorModel)
    for m in (a,b):
        m.begin_episode(s,np.random.default_rng(8))
    for band in C.MODELLED_BANDS:
        assert a.dprime(band,3,s) == b.dprime(band,3,s)
        assert a.pd_per_frame(band,3,s) == b.pd_per_frame(band,3,s)
        assert a.dprime(band,0,s) == 0
        assert a.dprime(band,.5,s) < a.dprime(band,1,s)
    assert a.dprime("mwir_shared",3,s) <= a.dprime("mwir",3,s)
    assert a.dprime("lwir",3,s) != a.dprime("lwir_cooled",3,s)
    assert a.nominal_pd("rgb",replace(s,is_day=False)) is None
    assert a.nominal_pd("lwir",s) == C.FROZEN["confirmation"]["pd_nominal"]
    with pytest.raises(ValueError,match="low.heat"):
        a.begin_episode(replace(s,engine="low_heat"),np.random.default_rng(8))

def test_calibrated_environment_seeding_and_no_privileged_leak():
    from rl.env import SkyEnv
    envs=[SkyEnv(sensor_model=CalibratedSensorModel(),compute_privileged=p) for p in (True,False)]
    obs=[e.reset(seed=19)[0] for e in envs]
    for i in range(40):
        assert all(np.array_equal(obs[0][k],obs[1][k]) for k in C.STUDENT_OBS_KEYS)
        results=[e.step(i%C.N_ACTIONS) for e in envs]
        assert results[0][1:4] == results[1][1:4]
        obs=[r[0] for r in results]
        if any(r[2] or r[3] for r in results):break

def test_fallback_is_explicit_and_sources_verify():
    m=CalibratedSensorModel()
    assert m.params["availability"]["low_heat"] == "unavailable"
    assert m.params["availability"]["plume"] == "analytic-grade-C"
    m.verify_sources()
