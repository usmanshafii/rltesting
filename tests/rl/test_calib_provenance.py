"""Calibration inputs and cached joins must fail closed on drift."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from rl.calib import calibrate
from rl.calib.sensor_model import CalibratedSensorModel
from rl.calib.validate import score_validation

HERE=Path(calibrate.__file__).parent

def test_reuse_join_refuses_modified_payload(tmp_path,monkeypatch):
    for name in ("joined_census.json","join_sources.json"):
        shutil.copyfile(HERE/name,tmp_path/name)
    with (tmp_path/"joined_census.json").open("a") as f:f.write(" ")
    monkeypatch.setattr(calibrate,"HERE",tmp_path)
    with pytest.raises(ValueError,match="join"):
        calibrate.build(reuse_join=True)

def test_parameter_sources_and_frozen_config(tmp_path):
    data=json.loads((HERE/"params.json").read_text())
    for entry in data["values"].values():
        assert entry["source"] in data["sources"]
        assert entry["grade"] in ("B","C")
    data["frozen_sha256"]="0"*64
    p=tmp_path/"bad.json";p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="frozen"):
        CalibratedSensorModel(p)

def test_bootstrap_handles_zero_signal_rows():
    result=score_validation(bootstrap=200)
    assert result["n"] == 55
    assert 0 < result["n_positive"] <= result["n"]
    for k in ("bootstrap_lower","bootstrap_upper","simulated_curve"):
        assert np.all(np.isfinite(result[k]))
    assert result["fa_pass"]
