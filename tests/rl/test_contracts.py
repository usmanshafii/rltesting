"""Task M1-02: the interface (rl/contracts.py) and the frozen values (rl/config/frozen.yaml)."""
import numpy as np
import pytest

from rl import contracts as C
from rl.env.scenario import sample_episode_spec
from rl.env.stub_sensor import StubSensorModel

STAR_VALUES = [
    ("grid", "az_deg"), ("grid", "el_deg"), ("grid", "cell_deg"),
    ("cameras", "frames_per_step"), ("cameras", "zoom_hfov_deg"), ("cameras", "max_hfov_deg"),
    ("episode", "max_steps"), ("episode", "drone_free_share"), ("episode", "speed_mps_train"),
    ("episode", "speed_mps_heldout"), ("episode", "entry_range_m"),
    ("confirmation", "threshold_A"), ("confirmation", "confirmable_K_steps"),
    ("confirmation", "raster_false_per_hour_target"),
    ("reward", "correct"), ("reward", "earliness_weight"), ("reward", "false_confirmation"),
    ("reward", "missed_confirmable"), ("reward", "step"), ("reward", "slew"),
    ("gates", "toy_within_points"), ("evaluation", "seeds_core"), ("evaluation", "seeds_ablation"),
]


@pytest.mark.parametrize("section,key", STAR_VALUES)
def test_every_star_value_is_frozen(section, key):
    assert key in C.FROZEN[section], f"{section}.{key} missing from rl/config/frozen.yaml"


def test_grid_is_12_by_3():
    assert C.GRID_SHAPE == (3, 12)
    assert C.ZOOM_HFOV_DEG == (46.0, 15.0, 5.0)


def test_eleven_actions_plus_declare():
    core = {C.DWELL, *C.SLEW, C.ZOOM_IN, C.ZOOM_OUT}
    assert core == set(range(C.N_ACTIONS)) and C.N_ACTIONS == 11
    assert C.DECLARE == C.N_ACTIONS and C.N_ACTIONS_LEARNED_DECLARE == 12
    assert len(C.ACTION_NAMES) == C.N_ACTIONS_LEARNED_DECLARE
    moves = list(C.SLEW.values())
    assert len(set(moves)) == 8 and all(max(abs(a), abs(b)) == 1 for a, b in moves)


def test_actor_keys_never_name_a_privileged_camera():
    assert not set(C.STUDENT_OBS_KEYS) & set(C.PRIVILEGED_KEYS)
    for k in C.STUDENT_OBS_KEYS:
        assert "cooled" not in k and "mwir" not in k


def test_band_sets():
    assert C.BAND_SETS["student"] == ("rgb", "lwir")
    assert not set(C.STUDENT_BANDS) & set(C.PRIVILEGED_BANDS)
    assert set(C.BANDS) == {"rgb", "lwir", "lwir_cooled", "mwir"}
    assert "mwir" not in C.BAND_SETS["four_shared"] and "mwir_shared" in C.BAND_SETS["four_shared"]


def test_frozen_hash_ignores_line_endings(tmp_path):
    text = C.FROZEN_PATH.read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert C.frozen_hash(lf) == C.frozen_hash(crlf) == C.frozen_hash()


def test_episode_spec_round_trip():
    spec = sample_episode_spec(3)
    assert C.EpisodeSpec.from_dict(spec.to_dict()) == spec


def _sensor_models():
    models = [StubSensorModel()]
    try:                                                # M2's model joins this test once it exists (M2-08)
        from rl.calib.sensor_model import CalibratedSensorModel
        models.append(CalibratedSensorModel())
    except (ImportError, NotImplementedError):
        pass
    return models


@pytest.mark.parametrize("model", _sensor_models(), ids=lambda m: m.version)
def test_sensor_model_contract(model):
    assert isinstance(model, C.SensorModel)
    for day in (True, False):
        spec = sample_episode_spec(11)
        spec = C.EpisodeSpec.from_dict({**spec.to_dict(), "is_day": day})
        model.begin_episode(spec, np.random.default_rng(0))
        for b in C.MODELLED_BANDS:
            for size in (0.3, 1.0, 3.0, 30.0):
                assert model.dprime(b, size, spec) >= 0.0
                assert 0.0 <= model.pd_per_frame(b, size, spec) <= 1.0
            clear, cloud = model.pfa_per_frame(b, spec, cloud=False), model.pfa_per_frame(b, spec, cloud=True)
            assert 0.0 < clear <= cloud < 1.0
            pd_nom = model.nominal_pd(b, spec)
            assert pd_nom is None or 0.0 < pd_nom < 1.0
        assert (model.nominal_pd("rgb", spec) is None) == (not day)       # RGB carries no evidence at night
