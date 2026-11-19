"""Persistent spatial clutter must survive consecutive observations of the same sky."""
import numpy as np
from rl import contracts as C
from rl.calib.clutter import NoiseClutter

def test_persistent_clutter_and_fresh_noise():
    model=NoiseClutter({b:.5 for b in C.BANDS})
    rng=np.random.default_rng(71)
    model.begin_episode(rng,(3,12))
    c1,n1=model.sample_frames(rng,5)
    c2,n2=model.sample_frames(rng,5)
    for b in C.BANDS:
        assert c1[b].shape == (5,3,12)
        assert np.array_equal(c1[b],c2[b])
        assert np.array_equal(c1[b][0],c1[b][-1])
        assert not np.array_equal(n1[b],n2[b])
    assert np.array_equal(c1["lwir"],c1["lwir_cooled"])
