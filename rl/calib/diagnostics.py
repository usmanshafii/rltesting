"""Generate held-out residual diagnostics and evaluate the prescribed simpler fallback."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from rl import contracts as C
from rl.calib.fit_dprime import features, predict_dprime


def main():
    here=Path(__file__).resolve().parent
    rows=json.loads((here/"joined_census.json").read_text())
    model=json.loads((here/"params.json").read_text())["model"]
    train=[r for r in rows if r["split"] == "train"]
    val=[r for r in rows if r["split"] == "val"]
    folds=model["fold_assignment"]
    comparison={}
    residuals={}
    for band in C.BANDS:
        def regress(rr):
            positive=[r for r in rr if r["dprime"][band]["d"]>0]
            return np.linalg.lstsq(np.array([features(r)[:4] for r in positive]),
                np.log([r["dprime"][band]["d"] for r in positive]),rcond=None)[0]
        cv=[]
        for fold in range(5):
            coef=regress([r for r in train if folds[r["scene_id"]] != fold])
            cv.extend(math.log(r["dprime"][band]["d"])-features(r)[:4]@coef
                      for r in train if folds[r["scene_id"]]==fold and r["dprime"][band]["d"]>0)
        coef=regress(train)
        errors=[math.log(r["dprime"][band]["d"])-features(r)[:4]@coef for r in val if r["dprime"][band]["d"]>0]
        comparison[band]=dict(full_cv=model["bands"][band]["cv_log_rmse"],
            simple_cv=float(np.sqrt(np.mean(np.square(cv)))),full_heldout=model["bands"][band]["heldout_log_rmse"],
            simple_heldout=float(np.sqrt(np.mean(np.square(errors)))))
        residuals[band]=[dict(scene_id=r["scene_id"],measured=r["dprime"][band]["d"],
            predicted=predict_dprime(model,band,r),log_residual=math.log(r["dprime"][band]["d"])-math.log(predict_dprime(model,band,r)))
            for r in val if r["dprime"][band]["d"]>0]
    result=dict(fallback_comparison=comparison,heldout_residuals=residuals,
        note="Diagnostic only. No coefficients, splits, uncertainty or acceptance limits are changed using held-out outcomes. The size/day fallback does not repair the LWIR failure.")
    (here/"diagnostics.json").write_text(json.dumps(result,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(comparison,indent=1))

if __name__ == "__main__":
    main()
