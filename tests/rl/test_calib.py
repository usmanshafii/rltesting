"""M2-02, M2-04 to M2-07: calibration. Each test skips until its task lands; replace the skip with the real test."""
import pytest


@pytest.mark.skip(reason="pending M2-02: census join within 0.1 %")
def test_census_join_matches_census():
    ...


@pytest.mark.skip(reason="pending M2-04: fit reproduces cooled 1.23x and MWIR 1.50x the uncooled d' within 10 %")
def test_fit_reproduces_census_ratios():
    ...


@pytest.mark.skip(reason="pending M2-05: removing the plume barely changes LWIR d' and lowers MWIR d'")
def test_plume_free_census():
    ...


@pytest.mark.skip(reason="pending M2-06: detection curve inside the bootstrap interval; 0.01 +- 0.002 false alarms per frame")
def test_score_model():
    ...


@pytest.mark.skip(reason="pending M2-07: LWIR clutter correlated, sensor noise not; dwell gain equals d_after")
def test_clutter_and_dwell():
    ...
