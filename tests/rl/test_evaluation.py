"""M4-02 to M4-05: baselines and evaluation. Each test skips until its task lands; replace the skip with the real test."""
import pytest


@pytest.mark.skip(reason="pending M4-02: same episode-set hash on two laptops")
def test_episode_set_hash_is_stable():
    ...


@pytest.mark.skip(reason="pending M4-03: raster visits every cell in a fixed cycle; Bayes looks at the highest posterior")
def test_baselines():
    ...


@pytest.mark.skip(reason="pending M4-04: evaluation reproducible to the last digit; bootstrap recovers a planted difference")
def test_evaluation_and_bootstrap():
    ...
