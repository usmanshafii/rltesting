"""M2-04: fit log d' per band on log size, aspect group, day or night, log visibility and cloud; five-fold
cross-validation; residual spread kept as a random effect; write rl/calib/params.json (rl/calib/README.md schema).

Done when: tests/rl/test_calib.py passes (the fit reproduces the census medians, cooled 1.23x and MWIR 1.50x the
uncooled d' where seen, within 10 %) and the cross-validated error is reported.
"""
from __future__ import annotations


def fit(rows: list[dict]) -> dict:
    """Fitted coefficients per band, with cross-validated error and residual spread."""
    raise NotImplementedError("M2-04: fit log d' per band")
