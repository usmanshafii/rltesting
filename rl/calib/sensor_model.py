"""M2-08: the calibrated sensor model, behind rl.contracts.SensorModel, replacing rl/env/stub_sensor.py.

The environment needs no change when it arrives (M1-08 only swaps the object). It must pass
tests/rl/test_contracts.py::test_sensor_model_contract, which picks this class up automatically once it can be built.
"""
from __future__ import annotations


class CalibratedSensorModel:
    version = "calibrated-0.0"

    def __init__(self):
        raise NotImplementedError("M2-08: calibrated sensor model (built from rl/calib/params.json)")
