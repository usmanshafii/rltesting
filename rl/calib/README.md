# rl/calib: calibration and the sensor model (owner: M2)

Tasks M2-01 to M2-09 in rl/PLAN.md. Until M2-08, the environment uses M1's stub, `rl/env/stub_sensor.py`, which
implements the same `rl.contracts.SensorModel` protocol with placeholder numbers.

## `params.json` (written by M2-04, checked by invariant I8)

```json
{
  "version": "calibrated-1",
  "sources": {"census": {"path": "docs/reference/privileged/data/census.json", "sha256": "..."}},
  "values": {
    "lwir.day.intercept": {"value": 1.79, "source": "census", "grade": "B", "note": "five-fold CV error ..."}
  }
}
```

Every entry of `values` has `value`, `source` (a key of `sources`) and `grade` (the evidence rule in CLAUDE.md).
