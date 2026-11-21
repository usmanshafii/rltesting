"""WP2 acceptance with explicit fallback and integration status; no hidden skips."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from tools.detector_vs_ideal import _fit
from rl import contracts as C
from rl.calib.fit_dprime import predict_dprime
from rl.calib.clutter import NoiseClutter, dwell_dprime
from rl.calib.score_model import GaussianScoreModel
from rl.calib.sensor_model import CalibratedSensorModel, source_hash

HERE=Path(__file__).resolve().parent
ROOT=C.RL_ROOT.parent


def score_validation(seed=206, bootstrap=1000):
    rows=json.loads((ROOT/"docs/reference/privileged/data/detector_vs_ideal.json").read_text())["models"]["rgb_lwir"]["scenes"]
    positive=[r for r in rows if r["dprime"] > 0]
    x=np.log([r["dprime"] for r in positive])
    y=np.array([r["detected"] for r in positive],float)
    grid=np.geomspace(5,500,32)
    rng=np.random.default_rng(seed)
    curves=[]
    for _ in range(bootstrap):
        idx=rng.integers(len(positive),size=len(positive))
        a,b=_fit(x[idx],y[idx])
        curves.append(expit(a+b*np.log(grid)))
    lo,hi=np.quantile(curves,[.025,.975],axis=0)
    score=GaussianScoreModel()
    empirical=np.array([np.mean(score.sample(rng,float(d),100_000) > score.threshold) for d in grid])
    predicted=np.array([score.detection_probability(float(d)) for d in grid])
    null=float(np.mean(score.sample(rng,0,200_000)>score.threshold))
    return dict(seed=seed,n=len(rows),n_positive=len(positive),bootstrap_resamples=bootstrap,grid=grid.tolist(),
                bootstrap_lower=lo.tolist(),bootstrap_upper=hi.tolist(),historical_curve=predicted.tolist(),
                simulated_curve=empirical.tolist(),inside_bootstrap=bool(np.all((empirical>=lo)&(empirical<=hi))),
                fa_per_frame=null,fa_pass=abs(null-float(C.FROZEN["detector"]["fa_per_frame"])) <= .002,
                note="Bootstrap is of the historical 55-scene joint-detector curve. Actual achieved test FA was corrected in docs/privileged_teacher_case.md; Gaussian surrogate FA is synthetic.")


def acceptance(seed=206):
    sensor=CalibratedSensorModel()
    rows=json.loads((HERE/"joined_census.json").read_text())
    entries=[]
    def add(check,status,detail="",**extra):
        entries.append(dict(check=check,status=status,detail=detail,**extra))
    sensor.verify_sources()
    add("M2-02 census identity", "PASS" if len(rows)==126 and max(max(r["join_error"].values()) for r in rows)<.001 else "FAIL",n=len(rows))
    ratios={}
    for b,target in (("lwir_cooled",1.23),("mwir",1.50)):
        seen=[r for r in rows if r["dprime"][b]["d"]>0 and r["dprime"]["lwir"]["d"]>0]
        measured=float(np.median([r["dprime"][b]["d"]/r["dprime"]["lwir"]["d"] for r in seen]))
        predicted=float(np.median([predict_dprime(sensor.model,b,r)/predict_dprime(sensor.model,"lwir",r) for r in seen]))
        ratios[b]=dict(measured=measured,predicted=predicted,target=target)
        add(f"M2-04 {b} median ratio", "PASS" if abs(predicted/target-1)<=.1 else "FAIL",**ratios[b])
    scores=score_validation(seed)
    add("M2-06 historical detection curve bootstrap", "PASS" if scores["inside_bootstrap"] else "FAIL")
    add("M2-06 synthetic frame false alarms", "PASS" if scores["fa_pass"] else "FAIL",value=scores["fa_per_frame"])
    rng=np.random.default_rng(seed)
    shares={b:sensor.params["values"][f"{b}.noise_share_median"]["value"] for b in C.BANDS}
    c,n=NoiseClutter(shares).draw(rng,40_000)
    corr_c=float(np.corrcoef(c["lwir"],c["lwir_cooled"])[0,1])
    corr_n=float(np.corrcoef(n["lwir"],n["lwir_cooled"])[0,1])
    add("M2-07 shared clutter / independent sensor noise sampler", "PASS" if corr_c>.9 and abs(corr_n)<.025 else "FAIL",clutter_correlation=corr_c,noise_correlation=corr_n)
    from eval.detectability import d_after
    same=all(dwell_dprime(r["dprime"][b],50)==d_after(r["dprime"][b],50) for r in rows for b in C.BANDS)
    add("M2-07 exact repository dwell formula","PASS" if same else "FAIL")
    add("M2-08 probability contract", "PASS" if isinstance(sensor,C.SensorModel) else "FAIL")
    for b,m in sensor.model["bands"].items():
        good=m["heldout_log_rmse"] is not None and m["heldout_log_rmse"] <= m["cv_log_rmse"]
        add(f"M2-09 {b} held-out error <= CV", "PASS" if good else "FAIL",heldout_log_rmse=m["heldout_log_rmse"],cv_log_rmse=m["cv_log_rmse"])
    add("M2-01 original pixel reproduction", "UNAVAILABLE", "No original four-band census layers or Blender installation; source census retained unchanged.")
    add("M2-03 low-heat rendering", "UNAVAILABLE", "Plan WP2 fallback omits low-heat evaluation; sensor refuses low_heat instead of fabricating values.")
    add("M2-05 plume-free census", "FALLBACK", "Conservative analytic plume/stub bound, grade C; no claim of pixel validation.")
    add("M2-07 environment integration", "PENDING", "SkyEnv probability-only contract draws independent frames/bands; correlated score and dwell integration needs M1 (CR-M2-01).")
    add("M2-10/M2-11 trained-run analysis", "PENDING", "No M3 policies or M4 evaluation/seed bootstrap implementation or Tier 1 result rows exist.")
    return dict(meta=dict(seed=seed,env_version=C.ENV_VERSION,interface_version=C.INTERFACE_VERSION,
                frozen_sha256=C.frozen_hash(),calibration_version=sensor.version,
                configuration_hash=source_hash(HERE/"params.json"),
                score_dataset_sha256=sensor.params["sources"]["detector"]["sha256"],
                python=sys.version.split()[0],platform=platform.platform()),
                rows=entries,score_validation=scores,ratios=ratios,
                numerical_checks_pass=not any(r["status"]=="FAIL" for r in entries),
                full_wp2_complete=False)


def main():
    result=acceptance()
    dest=C.RL_ROOT/"results/checks/wp2.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    for r in result["rows"]:
        print(f"[{r['status']}] {r['check']}: {r.get('detail','')}")
    # A partial/fallback implementation must not silently pass the full gate.
    return 0 if result["full_wp2_complete"] and result["numerical_checks_pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
