"""M2-01/M2-05 recovery tool; requires the original complete four-band layers.

python -m rl.calib.pixels --dataset out/m1/dataset --output rl/calib/pixel_census.json
Add --no-plume to remove ONLY the separate plume layer, with no edit to data/dataset.py.
Use --overlay configs/m1_low_heat.yaml on an independently rendered low-heat folder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from data.dataset import SceneDataset, BAND_SALT, gt_objects, scene_camera
from data.camera_emulation import covered_box
from eval.detectability import dprime, NONE
from render.sampling import load_configs
from rl import contracts as C
from rl.calib.sensor_model import source_hash


class CalibrationDataset(SceneDataset):
    def __init__(self,*args,remove_plume=False,absent=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.remove_plume=bool(remove_plume)
        self.absent=bool(absent)

    def _layers(self,path):
        layer,objects=super()._layers(path)
        if self.absent:
            objects=[]
        elif self.remove_plume:
            objects=[o for o in objects if o["name"] != "plume"]
        return layer,objects


def measure(dataset: Path, remove_plume=False, overlay="configs/m1.yaml", limit=None):
    """The exact M1 noise draws, coverage, preprocessing and matched filter.

    Imports CNN preprocessing only after checking availability; no model is run.
    Never visits a test ID or rewrites the original census or source pixels.
    """
    dataset=Path(dataset)
    root=C.RL_ROOT.parent
    census=json.loads((root/"docs/reference/privileged/data/census.json").read_text())
    if limit is not None:
        census=dict(list(sorted(census.items()))[:limit])
    missing=[]
    for sid in sorted(census):
        p=dataset/"meta"/f"{sid}.json"
        if not p.is_file():
            missing.append(sid)
            continue
        meta=json.loads(p.read_text())
        if meta["split"] not in ("train","val"):
            raise ValueError(f"sealed split: {sid}")
        if not set(C.BANDS) <= set(meta.get("modalities",{})) or any(not (dataset/meta["modalities"][b]["layers"]).is_file() for b in C.BANDS):
            missing.append(sid)
    if missing:
        raise FileNotFoundError(f"missing complete four-band census scenes: {len(missing)}/{len(census)}; first: {missing[:5]}")
    from tools.m1_det2 import LWIR_REALISM, centre_inside
    from models.cnn_detector import preprocess, PREPROCESS_V2
    cfg=copy.deepcopy(load_configs(root=str(root),overlay=overlay))
    cfg["sensors"]["lwir_noise"].update(LWIR_REALISM)
    H=cfg["sensors"]["image"]["height"]
    W=cfg["sensors"]["image"]["width"]
    result={}
    checks={}
    file_hashes={}
    for sid in sorted(census):
        meta=json.loads((dataset/"meta"/f"{sid}.json").read_text())
        obj=gt_objects(meta)[0]
        size=max(obj["box"][2]-obj["box"][0],obj["box"][3]-obj["box"][1])
        if abs(obj["range_m"]/census[sid]["range_m"]-1)>=.001 or abs(size/census[sid]["size_px"]-1)>=.001:
            raise ValueError(f"pixel dataset geometry differs from census: {sid}")
        file_hashes[sid]={"meta":hashlib.sha256((dataset/"meta"/f"{sid}.json").read_bytes()).hexdigest(),
                          **{b:hashlib.sha256((dataset/meta["modalities"][b]["layers"]).read_bytes()).hexdigest() for b in C.BANDS}}
        # All calls use the same source scene and noise salts as the original census.
        def frames(seed,absent):
            ds=CalibrationDataset(str(dataset),None,C.BANDS,cfg,train=False,seed=seed,metas=[meta],
                aug=dict(hflip=0.,scale_prob=0.),remove_plume=remove_plume,absent=absent)
            m=ds.metas[0]
            cov={b:covered_box(H,W,(scene_camera(m,b) or {}).get("coverage",(1.,1.))) for b in C.BANDS}
            vis,ratio=ds._weather(m,ds._rng(0))
            imgs={b:ds.band_image(m,b,ds._rng(0,BAND_SALT[b]),vis,ratio,reg=False) for b in C.BANDS}
            return preprocess(imgs,PREPROCESS_V2,coverage=cov),cov
        present,cov=frames(17,False)
        absent,_=frames(17,True)
        absent2,_=frames(1017,True)
        box=gt_objects(meta)[0]["box"]
        stats={b:dprime(present[b],absent[b],absent2[b],box,cover=cov[b]) if centre_inside(box,cov[b]) else dict(NONE) for b in C.BANDS}
        result[sid]=dict(census[sid],dprime=stats)
        checks[sid]={b:(abs(stats[b]["d"]/census[sid]["dprime"][b]["d"]-1) if census[sid]["dprime"][b]["d"]>0 else abs(stats[b]["d"])) for b in C.BANDS}
        print(sid,flush=True)
    return dict(rows=result,remove_plume=remove_plume,overlay=overlay,relative_dprime_errors=checks,
                source_file_hashes=file_hashes,
                source_census_sha256=source_hash(root/"docs/reference/privileged/data/census.json"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--no-plume",action="store_true")
    ap.add_argument("--verify-five",action="store_true",help="Reproduce the first five census scenes within 2 percent")
    ap.add_argument("--overlay",default="configs/m1.yaml")
    a=ap.parse_args()
    protected=C.RL_ROOT.parent/"docs/reference/privileged/data/census.json"
    if a.output.resolve() == protected.resolve():
        raise ValueError("never overwrite the reference census")
    if a.verify_five and (a.no_plume or a.overlay != "configs/m1.yaml"):
        raise ValueError("five-scene reproduction requires the original M1 setting")
    result=measure(a.dataset,a.no_plume,a.overlay,limit=5 if a.verify_five else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    if a.verify_five and any(e>.02 for row in result["relative_dprime_errors"].values() for e in row.values()):
        raise SystemExit("five-scene pixel reproduction exceeds 2 percent; see output errors")

if __name__ == "__main__":
    main()
