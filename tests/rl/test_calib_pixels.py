"""M2-only loader controls: keep the target, remove just plume, and preserve source layers."""
import numpy as np
import pytest
from data.composite import save_layers
from rl.calib.pixels import CalibrationDataset

@pytest.mark.parametrize("band", ["rgb","lwir","lwir_cooled","mwir"])
def test_plume_filter_is_local_and_nondestructive(tmp_path,band):
    path=tmp_path/(band+".npz")
    sky=np.ones((4,4,1),np.float32)
    def obj(name):
        return dict(name=name,P=np.ones((1,1,1)),A=np.ones((1,1)),crop=[0,0,1,1],range_m=10.)
    save_layers(str(path),dict(sky=sky,psf_sigma0=0),[obj("target"),obj("plume")],half=False)
    # No full SceneDataset needed to exercise the shared layer-loading path.
    def load(remove=False,absent=False):
        ds=object.__new__(CalibrationDataset)
        ds.cache_max=0
        ds._cache={}
        ds.remove_plume=remove
        ds.absent=absent
        return ds._layers(str(path))[1]
    assert [x["name"] for x in load()] == ["target","plume"]
    assert [x["name"] for x in load(True)] == ["target"]
    assert load(False,True) == []
    assert [x["name"] for x in load()] == ["target","plume"]
