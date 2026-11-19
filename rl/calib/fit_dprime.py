"""M2-04/M2-09: train-only log-linear calibration and scene-level 5-fold CV.

All validation rows are excluded from coefficients, residuals and calibration
adjustments. Zero d' denotes no usable/covered signal and is not log-transformed.
No new physical constants: features use the regenerated scene metadata.
"""
from __future__ import annotations

import math
import numpy as np
from rl import contracts as C

FEATURES = ("intercept", "log_size", "night", "night_log_size", "log_visibility",
            "cloud", "head_on", "tail_on", "engine_low", "engine_high")


def features(row: dict) -> np.ndarray:
    s = math.log(max(float(row["size_px"]),1e-12))
    night = float(row["lighting"] != "day")
    aspect = float(row["aspect_deg"])
    return np.array([1.,s,night,night*s,math.log(max(float(row["visibility_km"]),1e-12)),
                     float(bool(row["clouds"])),float(aspect < 60),float(aspect > 120),
                     float(row["engine_setting"] == "low_power"),float(row["engine_setting"] == "high_power")])


def _regress(rows, band):
    valid = [r for r in rows if r["dprime"][band]["d"] > 0]
    if len(valid) <= len(FEATURES):
        raise ValueError(f"not enough positive calibration scenes for {band}")
    X=np.array([features(r) for r in valid])
    y=np.log([r["dprime"][band]["d"] for r in valid])
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    return coef, valid


def predict_dprime(model: dict, band: str, row: dict) -> float:
    if band not in C.BANDS:
        raise ValueError(band)
    logd=float(features(row) @ np.array(model["bands"][band]["coefficients"]))
    return math.exp(float(np.clip(logd,-700,700)))


def _fit_bands(rows):
    result={}
    for b in C.BANDS:
        coef,valid=_regress(rows,b)
        result[b]={"coefficients":coef.tolist(),"fit_scene_ids":[r["scene_id"] for r in valid]}
    # A median calibration on paired, actually-seen TRAINING scenes preserves
    # the primary census ratios without treating uncovered privileged views as 0.
    model={"bands":result}
    for b in C.PRIVILEGED_BANDS:
        paired=[r for r in rows if r["dprime"][b]["d"] > 0 and r["dprime"]["lwir"]["d"] > 0]
        target=float(np.median([r["dprime"][b]["d"]/r["dprime"]["lwir"]["d"] for r in paired]))
        predicted=float(np.median([predict_dprime(model,b,r)/predict_dprime(model,"lwir",r) for r in paired]))
        shift=math.log(target/predicted)
        result[b]["coefficients"][0]+=shift
        result[b]["paired_ratio_adjustment_log"]=shift
        result[b]["training_paired_ratio"]=target
    return result


def fit(rows: list[dict]) -> dict:
    if any(r["split"] == "test" for r in rows):
        raise ValueError("sealed test split must not be supplied to calibration")
    training=sorted([r for r in rows if r["split"] == "train"], key=lambda r:r["scene_id"])
    validation=[r for r in rows if r["split"] == "val"]
    if len({r["scene_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate scene IDs")
    # Scene IDs, not per-band rows, define folds. One fixed permutation; no search
    # for a seed or fold partition that gives more attractive results.
    order=np.random.default_rng(0).permutation(len(training))
    fold_ids={training[int(i)]["scene_id"]: k%5 for k,i in enumerate(order)}
    model={"feature_names":list(FEATURES),"bands":_fit_bands(training),"cv_seed":0,
           "fit_split":"train","heldout_split":"val","fold_assignment":fold_ids}
    residual_bank=[]
    for r in training:
        residuals={b:math.log(r["dprime"][b]["d"])-math.log(predict_dprime(model,b,r))
                   for b in C.BANDS if r["dprime"][b]["d"] > 0}
        residual_bank.append(dict(scene_id=r["scene_id"],lighting=r["lighting"],clouds=r["clouds"],
            aspect_deg=r["aspect_deg"],engine_setting=r["engine_setting"],visibility_km=r["visibility_km"],
            residuals=residuals,noise_share={b:r["dprime"][b]["noise_share"] for b in C.BANDS}))
    model["residual_bank"]=residual_bank
    cv={b:[] for b in C.BANDS}
    folds={b:[] for b in C.BANDS}
    for k in range(5):
        train=[r for r in training if fold_ids[r["scene_id"]] != k]
        hold=[r for r in training if fold_ids[r["scene_id"]] == k]
        fm={"bands":_fit_bands(train)}
        for b in C.BANDS:
            errors=[math.log(r["dprime"][b]["d"])-math.log(predict_dprime(fm,b,r)) for r in hold if r["dprime"][b]["d"] > 0]
            cv[b].extend(errors)
            folds[b].append(dict(fold=k,n=len(errors),log_rmse=float(np.sqrt(np.mean(np.square(errors))))))
    for b in C.BANDS:
        res=[x["residuals"][b] for x in residual_bank if b in x["residuals"]]
        errors=[math.log(r["dprime"][b]["d"])-math.log(predict_dprime(model,b,r)) for r in validation if r["dprime"][b]["d"] > 0]
        model["bands"][b].update(residual_std=float(np.std(res,ddof=1)),
            cv_log_rmse=float(np.sqrt(np.mean(np.square(cv[b])))),cv_folds=folds[b],
            heldout_log_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            heldout_n=len(errors))
    return model
