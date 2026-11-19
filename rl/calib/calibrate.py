"""Reproducible M2 build: python -m rl.calib.calibrate [--reuse-join]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rl import contracts as C
from rl.calib.join_specs import join_census_with_specs
from rl.calib.fit_dprime import fit
from rl.calib.sensor_model import CalibratedSensorModel, source_hash

HERE=Path(__file__).resolve().parent
ROOT=C.RL_ROOT.parent
SOURCE_PATHS={
    "census":"docs/reference/privileged/data/census.json",
    "detector":"docs/reference/privileged/data/detector_vs_ideal.json",
    "detector_method":"tools/detector_vs_ideal.py",
    "census_method":"tools/privileged_census.py",
    "sampling":"render/sampling.py", "placement":"render/placement.py",
    "geometry":"target/kal_geometry.py", "measurement":"target/kal_measure.py",
    "engine_geometry":"target/engine_l550e.py",
    "signatures_method":"render/signatures.py", "cameras_method":"render/cameras.py",
    "overlay":"configs/m1.yaml", "low_heat_overlay":"configs/m1_low_heat.yaml",
    "render":"configs/render.yaml", "sensors":"configs/sensors.yaml",
    "signatures":"configs/signatures.yaml", "target":"configs/target.yaml",
    "frozen":"rl/config/frozen.yaml", "dwell":"eval/detectability.py",
    "plume_bounds":"docs/thermal_bands.md", "detector_correction":"docs/privileged_teacher_case.md",
    "clutter_estimates":"rl/env/stub_sensor.py",
}


def sources():
    result={name:dict(path=p,sha256=source_hash(ROOT/p),hash_convention="SHA-256, CRLF normalized to LF")
            for name,p in SOURCE_PATHS.items()}
    from render.sampling import load_configs
    climate=load_configs(root=str(ROOT),overlay="configs/m1.yaml")["render"]["climate"]["file"]
    result["climate"]=dict(path=climate,sha256=source_hash(ROOT/climate),hash_convention="SHA-256, CRLF normalized to LF")
    return result


def build(reuse_join=False):
    src=sources()
    joined=HERE/"joined_census.json"
    meta_path=HERE/"join_sources.json"
    if reuse_join:
        prior=json.loads(meta_path.read_text())
        if prior["sources"] != src or prior["joined_sha256"] != source_hash(joined):
            raise ValueError("join sources changed; regenerate without --reuse-join")
        rows=json.loads(joined.read_text())
    else:
        rows=join_census_with_specs(str(ROOT/SOURCE_PATHS["census"]))
        if sources() != src:
            raise ValueError("sources changed while regenerating census")
        joined.write_text(json.dumps(rows,indent=1)+"\n",encoding="utf-8")
        meta_path.write_text(json.dumps(dict(sources=src,joined_sha256=source_hash(joined)),indent=1)+"\n",encoding="utf-8")
    model=fit(rows)
    values={}
    def add(name,value,source,grade,note):
        values[name]=dict(value=value,source=source,grade=grade,note=note)
    for b,m in model["bands"].items():
        for name,value in zip(model["feature_names"],m["coefficients"]):
            add(f"{b}.{name}",value,"census","B","Derived from training scenes only; features from regenerated specs")
        for k in ("residual_std","cv_log_rmse","heldout_log_rmse"):
            add(f"{b}.{k}",m[k],"census","B","Log units; scene-level five-fold CV; validation is never fitted")
    add("plume_to_stub_ratio_lower",.1,"plume_bounds","C","Section 5: source plume/stub contrast lower bound, not plume/whole-target fraction")
    add("plume_to_stub_ratio_upper",.7,"plume_bounds","C","Section 5: source plume/stub upper bound; analytic sensitivity only")
    add("plume_to_stub_ratio_central",.4,"plume_bounds","C","Midpoint of .1-.7 range; conservative removal d/(1+r), not a measured no-plume census")
    for b in C.MODELLED_BANDS:
        base="mwir" if b == "mwir_shared" else b
        sh=[r["dprime"][base]["noise_share"] for r in rows if r["split"] == "train" and r["dprime"][base]["noise_share"] is not None]
        add(f"{b}.noise_share_median",float(np.median(sh)),"census","B","Training-only measured variance share; camera noise remains separate")
    from rl.calib.clutter import cloud_false_detection_factor
    for b in C.MODELLED_BANDS:
        for day in (True,False):
            add(f"{b}.cloud_factor.{day}",cloud_false_detection_factor(b,day),"clutter_estimates","C",
                "Sensitivity estimate inherited from stub, NOT measured cloud false alarms")
    for key,value in C.FROZEN["detector"].items():
        add(f"detector.{key}",value,"frozen","B","Frozen historical joint-detector surrogate; per-band use is grade C")
    params=dict(version=CalibratedSensorModel.version,sources=src,values=values,model=model,
        implementation_sha256={p.name:source_hash(p) for p in HERE.glob("*.py")},
        frozen_sha256=C.frozen_hash(),availability={"census_renders":"unavailable", "plume":"analytic-grade-C", "low_heat":"unavailable"},
        limitations=["Historical joint RGB+LWIR curve is a surrogate per band and was not a measured 0.01-FA test ROC.",
                    "No original four-band pixels: no five-scene pixel reproduction or empirical plume-free/low-heat census.",
                    "EpisodeSpec lacks aspect, power and visibility; draw matched training nuisance metadata once per episode.",
                    "SkyEnv v0 draws bands/frames independently; shared-clutter and dwell samplers require M1 integration."])
    (HERE/"params.json").write_text(json.dumps(params,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    return params


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse-join",action="store_true")
    args=ap.parse_args()
    p=build(args.reuse_join)
    print(f"Built {p['version']}; {len(p['model']['residual_bank'])} training scenes; held-out validation reported separately")

if __name__ == "__main__":
    main()
