"""Train, evaluate and demonstrate a small CPU M1 pilot using saved weights.

python tools/m1_pilot.py prepare [--swir]
python tools/m1_pilot.py train [--models rgb_only lwir_only rgb_lwir]
python tools/m1_pilot.py evaluate
python tools/m1_pilot.py demo --model student
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target")]

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from data.dataset import SceneDataset, gt_objects
from data.provenance import atomic_json, digest
from eval import metrics
from eval.nwd import iou_matrix
from models.pilot import (PilotDetector, proposals, features, select_features, encode,
                          predict_features, predict_images, load_weights)
from render.sampling import load_configs, load_yaml

SPECS = {
    "rgb_only": (["rgb"], ["rgb"], None),
    "lwir_only": (["lwir"], ["lwir"], None),
    "rgb_lwir": (["rgb","lwir"], ["rgb","lwir"], None),
    "teacher": (["rgb","lwir","swir"], ["rgb","lwir"], None),
    "student": (["rgb","lwir"], ["rgb","lwir"], "teacher"),
    "student_no_swir_teacher": (["rgb","lwir"], ["rgb","lwir"], "rgb_lwir"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_sample(manifest, sid, bands):
    """Load requested inputs only; student demo never opens SWIR layers."""
    cfg = load_configs(overlay=manifest["overlay"])
    root = ROOT / manifest["source"]
    m = read(root / "meta" / (sid + ".json"))
    contract=m["extra"]["render_contract"]
    if contract["config_sha256"]!=manifest["config_sha256"] or contract["signature_model"]!=manifest["signature_model"]:
        raise ValueError(f"{sid}: RGB/LWIR source differs from the frozen configuration")
    images = {}
    base = [b for b in bands if b != "swir"]
    if base:
        images.update(SceneDataset(str(root),None,base,cfg,seed=manifest["seed"],metas=[m]).load(0)["images"])
    if "swir" in bands:
        sr = ROOT / manifest["swir"]
        sm = read(sr / "meta" / (sid + ".json"))
        # Numeric scene alignment is checked independently of different band/sample contracts.
        for key in ("scene_id","split","seed","target","camera","conditions"):
            if sm[key] != m[key]:
                raise ValueError(f"{sid}: unaligned SWIR {key}")
        if sm["extra"]["render_contract"]["signature_model"] != manifest["signature_model"]:
            raise ValueError("wrong SWIR signature model")
        images.update(SceneDataset(str(sr),None,["swir"],cfg,seed=manifest["seed"],metas=[sm]).load(0)["images"])
    return images,m


def prepare(out, with_swir, wait_for_render=False):
    m = read(out/"manifest.json")
    bands = ["rgb","lwir"] + (["swir"] if with_swir else [])
    cache = out / ("cache3" if with_swir else "cache2")
    cache.mkdir(exist_ok=True)
    identity_path=cache/"identity.json"
    if identity_path.exists():
        old=read(identity_path)
        if old["manifest_sha256"]!=sha(out/"manifest.json") or old["code_sha256"]!=sha(ROOT/"models/pilot.py"):
            raise ValueError("cache provenance changed; use a fresh pilot output directory")
    index = {}
    for sid in m["scene_ids"]:
        path = cache/(sid+".npz")
        if not path.exists():
            if with_swir and wait_for_render:
                deadline=time.monotonic()+7200
                while not (ROOT/m["swir"]/"meta"/(sid+".json")).exists():
                    if time.monotonic()>deadline: raise TimeoutError(f"SWIR render missing: {sid}")
                    time.sleep(2)
            old = out/"cache2"/(sid+".npz")
            images,meta = source_sample(m,sid,["swir"] if with_swir and old.exists() else bands)
            values = {}
            for key,pb in (("rgb",["rgb"]),("lwir",["lwir"]),("fused",["rgb","lwir"])):
                old = out/"cache2"/(sid+".npz")
                if with_swir and old.exists():
                    with np.load(old) as d:
                        boxes,score = d[key+"_boxes"],d[key+"_score"]
                        xf = np.c_[d[key+"_features"],features(images,boxes,["swir"])[:,4:]]
                else:
                    boxes,score = proposals(images,pb)
                    xf = features(images,boxes,bands)
                values.update({key+"_boxes":boxes,key+"_score":score,key+"_features":xf})
            np.savez_compressed(path,**values)
            print(f"prepared {sid}: " + ", ".join(f"{k} {len(values[k+'_boxes'])}" for k in ("rgb","lwir","fused")),flush=True)
        index[sid] = sha(path)
    atomic_json(cache/"identity.json",dict(bands=bands,files=index,manifest_sha256=sha(out/"manifest.json"),
                                            code_sha256=sha(ROOT/"models/pilot.py")))


def records_for(out, name, split, with_swir=False):
    m = read(out/"manifest.json")
    cache = out/("cache3" if with_swir else "cache2")
    loaded = ["rgb","lwir"]+(["swir"] if with_swir else [])
    identity=read(cache/"identity.json")
    bands,pb,_ = SPECS[name]
    key = pb[0] if len(pb)==1 else "fused"
    rows = []
    for sid in m["splits"][split]:
        meta = read(ROOT/m["source"]/"meta"/(sid+".json"))
        if sha(cache/(sid+".npz"))!=identity["files"][sid]:
            raise ValueError(f"cached input changed: {sid}")
        with np.load(cache/(sid+".npz")) as d:
            rows.append(dict(scene_id=sid,boxes=d[key+"_boxes"],raw_scores=d[key+"_score"],
                             x=select_features(d[key+"_features"],loaded,bands),
                             x_all=d[key+"_features"],gts=gt_objects(meta),
                             camera=meta["camera"],conditions=meta["conditions"]))
    return rows


def predicted(model, rows):
    return [dict(scene_id=r["scene_id"],gts=r["gts"],camera=r["camera"],conditions=r["conditions"],
                 preds=predict_features(model,r["x"],r["boxes"])) for r in rows]


def targets(rows):
    xs,ys,ds = [],[],[]
    for r in rows:
        gt = np.asarray([o["box"] for o in r["gts"]],np.float32)
        ious = iou_matrix(r["boxes"],gt)
        best = ious.argmax(1)
        ov = ious.max(1)
        # Midquality proposals are ignored for classification; no GT proposals are injected.
        y = np.where(ov>=0.3,1,np.where(ov<0.1,0,-1)).astype(np.float32)
        xs.append(r["x"]); ys.append(y); ds.append(encode(r["boxes"],gt[best]))
    return tuple(torch.from_numpy(np.concatenate(v)) for v in (xs,ys,ds))


def config_eval():
    cfg = load_yaml(str(ROOT/"configs/train.yaml"))["eval"]
    cfg["bootstrap"] = 1000
    return cfg


def train(out, names):
    m = read(out/"manifest.json")
    cfg = config_eval()
    torch.set_num_threads(1)
    for name in names:
        bands,pb,teacher_name = SPECS[name]
        run = out/"runs"/name
        if (run/"best.pt").exists():
            print(f"{name}: saved run exists; use a new pilot directory to change the experiment",flush=True)
            continue
        swir = "swir" in bands or teacher_name=="teacher"
        identity=read(out/("cache3" if swir else "cache2")/"identity.json")
        if identity["manifest_sha256"]!=sha(out/"manifest.json") or identity["code_sha256"]!=sha(ROOT/"models/pilot.py"):
            raise ValueError("cache is not from the current frozen pilot")
        tr,va = records_for(out,name,"train",swir),records_for(out,name,"val",swir)
        x,y,delta = targets(tr)
        active,pos = y>=0,y==1
        if pos.sum()<5:
            raise ValueError(f"only {int(pos.sum())} positive training proposals for {name}")
        torch.manual_seed(m["seed"])
        model = PilotDetector(bands)
        model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0).clamp(min=0.05))
        initial = copy.deepcopy(model.state_dict())
        teacher = None
        if teacher_name:
            teacher,tck = load_weights(out/"runs"/teacher_name/"best.pt")
            teacher.requires_grad_(False)
            loaded = ["rgb","lwir"]+(["swir"] if swir else [])
            xt = torch.from_numpy(np.concatenate([select_features(r["x_all"],loaded,teacher.bands) for r in tr]))
            with torch.no_grad():
                tl,td = teacher(xt)
        opt = torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.001)
        history = []
        best,best_state,best_ep = -1,None,0
        start = time.time()
        for ep in range(1,m["epochs"]+1):
            model.train()
            logits,bd = model(x)
            cls = F.binary_cross_entropy_with_logits(logits[pos],y[pos])*.5
            cls = cls + F.binary_cross_entropy_with_logits(logits[y==0],y[y==0])*.5
            box = F.smooth_l1_loss(bd[pos],delta[pos])
            kd = logits.sum()*0
            if teacher is not None:
                # Response KD only: a documented small pilot, not full spec 7.2 feature KD.
                kd = F.binary_cross_entropy_with_logits(logits[active]/2,torch.sigmoid(tl[active]/2))*4
                kd = kd + F.smooth_l1_loss(bd[pos],td[pos])
            loss = cls + 2*box + (0.25*kd if teacher is not None else 0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            row = dict(epoch=ep,loss=float(loss),classification=float(cls),box=float(box),kd=float(kd))
            if ep%10==0 or ep==1:
                model.eval()
                ap = metrics.ap(predicted(model,va),cfg=cfg)
                row["val_ap50"]=ap
                if ap>best:
                    best,best_state,best_ep = ap,copy.deepcopy(model.state_dict()),ep
                print(f"{name} epoch {ep}: loss {float(loss):.4f}, val AP50 {ap:.4f}",flush=True)
            history.append(row)
        model.load_state_dict(best_state); model.eval()
        vr = predicted(model,va)
        choices = []
        for threshold in np.arange(.05,.96,.05):
            cc = dict(cfg,score_thresh=float(threshold))
            op = metrics.operating_point(vr,cc)
            f1 = 2*op["precision"]*op["recall"]/max(op["precision"]+op["recall"],1e-9)
            choices.append((f1,float(threshold)))
        threshold = max(choices)[1]  # highest threshold breaks F1 ties, on validation only
        run.mkdir(parents=True,exist_ok=True)
        ck = dict(architecture="m1-proposal-mlp-v1",bands=bands,proposal_bands=pb,hidden=64,
                  state_dict=best_state,epoch=best_ep,val_ap50=best,threshold=threshold,
                  teacher=teacher_name,teacher_sha256=sha(out/"runs"/teacher_name/"best.pt") if teacher_name else None,
                  manifest_sha256=sha(out/"manifest.json"),code_sha256=sha(ROOT/"models/pilot.py"),
                  training_code_sha256=sha(Path(__file__)),cache_identity_sha256=digest(identity),
                  train_scenes=m["splits"]["train"],val_scenes=m["splits"]["val"],
                  seed=m["seed"],epochs=m["epochs"],training_proposals=int(len(x)),positive_proposals=int(pos.sum()),
                  seconds=time.time()-start,distillation="response only, T=2, lambda=0.25" if teacher else None)
        torch.save(ck,run/"best.pt")
        torch.save(dict(ck,state_dict=initial,epoch=0),run/"untrained.pt")
        atomic_json(run/"history.json",history)
        atomic_json(run/"training.json",{k:v for k,v in ck.items() if k!="state_dict"})
        atomic_json(run/"validation.json",dict(ap50=best,threshold=threshold,
                    threshold_selection="maximum validation F1 at IoU >= .5; .05:.05:.95, highest tie",records=vr))
        print(f"SAVED {run/'best.pt'}",flush=True)


def summary(records,cfg):
    return dict(n=len(records),ap50=metrics.ap(records,cfg=cfg),ap50_ci95=metrics.ap_interval(records,cfg),
                ap50_95=metrics.ap_50_95(records,cfg=cfg),op=metrics.operating_point(records,cfg),
                centre=metrics.operating_point(records,cfg,kind="centre",thr=0))


def evaluate(out):
    m = read(out/"manifest.json")
    cfg = config_eval()
    paths = {n:out/"runs"/n/"best.pt" for n in SPECS}
    if any(not p.exists() for p in paths.values()):
        raise ValueError("freeze all six planned runs before opening the held-out test")
    lock = dict(manifest_sha256=sha(out/"manifest.json"),weights={n:sha(p) for n,p in paths.items()},evaluation_settings=cfg)
    if (out/"test_lock.json").exists() and read(out/"test_lock.json")!=lock:
        raise ValueError("test was already opened with different weights; create a new untouched test cohort")
    atomic_json(out/"test_lock.json",lock)
    report,all_records = dict(models={},scope="96-scene synthetic clear open-sky compact pilot",seed=m["seed"],evaluation_settings=cfg),{}
    for name,path in paths.items():
        model,ck = load_weights(path)
        if ck["manifest_sha256"]!=sha(out/"manifest.json") or ck["code_sha256"]!=sha(ROOT/"models/pilot.py"):
            raise ValueError("checkpoint does not match the frozen pilot/code")
        rows = records_for(out,name,"test",with_swir="swir" in model.bands)
        records = predicted(model,rows)
        cfg1 = dict(cfg,score_thresh=ck["threshold"])
        untrained,_ = load_weights(path.with_name("untrained.pt"))
        result = summary(records,cfg1)
        result.update(bands=ck["bands"],checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
                      threshold=ck["threshold"],best_epoch=ck["epoch"],
                      untrained_ap50=metrics.ap(predicted(untrained,rows),cfg=cfg))
        proposal_records=[dict(scene_id=r["scene_id"],gts=r["gts"],
                               preds=dict(boxes=r["boxes"].tolist(),scores=r["raw_scores"].tolist(),
                                          labels=[0]*len(r["boxes"]))) for r in rows]
        result["proposal_only_ap50"]=metrics.ap(proposal_records,cfg=cfg)
        result["proposal_coverage_iou30"]=sum(bool(np.any(iou_matrix(r["boxes"],np.asarray([o["box"] for o in r["gts"]]))>=.3)) for r in rows)/len(rows)
        result["lighting"]={c:summary([r for r in records if r["conditions"]["lighting"]==c],cfg1) for c in ("day","night")}
        result["bins"]={b:dict(n=sum(o["size_bin"]==b for r in records for o in r["gts"]),
                                  ap50=metrics.ap(records,bin_name=b,cfg=cfg1))
                        for b in sorted({o["size_bin"] for r in records for o in r["gts"]})}
        report["models"][name]=result
        all_records[name]=records
    report["comparisons"] = {
        "student_minus_fusion":metrics.bootstrap_diff(all_records["student"],all_records["rgb_lwir"],lambda rr:metrics.ap(rr,cfg=cfg)),
        "student_minus_no_swir_teacher":metrics.bootstrap_diff(all_records["student"],all_records["student_no_swir_teacher"],lambda rr:metrics.ap(rr,cfg=cfg)),
    }
    atomic_json(out/"report.json",report)
    atomic_json(out/"predictions.json",all_records)
    lines=["# M1 compact pilot — held-out results","",f"Fixed split: {len(m['splits']['train'])} train / {len(m['splits']['val'])} validation / {len(m['splits']['test'])} test scenes. One training seed; synthetic data only.","",
           "Compact contrast-proposal + learned MLP detectors; these results do not validate the full P2 detector. All inputs include the repository sensor-noise model. SWIR is teacher-only privileged input.","",
           "| Model | Inputs | AP50 [95% scene bootstrap] | AP50:95 | Precision | Recall (TP/27) | FP/frame | Untrained AP50 | Proposal-only AP50 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in report["models"].items():
        op,ci=r["op"],r["ap50_ci95"]
        lines.append(f"| {name} | {' + '.join(r['bands'])} | {r['ap50']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] | {r['ap50_95']:.3f} | {op['precision']:.3f} | {op['recall']:.3f} ({op['tp']}/{op['n_gt']}) | {op['fppi']:.3f} | {r['untrained_ap50']:.3f} | {r['proposal_only_ap50']:.3f} |")
    lines += ["","Operating thresholds were selected separately on validation F1, then frozen. AP uses all predictions at score >= .001, IoU >= .5, one-to-one matching and 101-point interpolation. AP50:95 averages IoU .50:.05:.95. Intervals resample whole test scenes 1,000 times; they exclude training-seed and real-world uncertainty.","",
              "Untrained AP50 uses the same initial MLP with training-input normalization. Proposal-only AP50 ranks raw image-contrast components without learned classification or box correction; it retains up to 80 component boxes, whereas the learned output applies final NMS and keeps up to 30. These are diagnostic references, not additional trained systems.","","## Paired comparisons"]
    for name,r in report["comparisons"].items():
        lines.append(f"- {name}: AP50 difference {r['diff']:+.3f}, 95% paired interval [{r['lo']:+.3f}, {r['hi']:+.3f}].")
    lines += ["","A completed distillation experiment is not by itself evidence of an SWIR benefit. The no-SWIR-teacher control has the same student architecture, initialization, labels and training budget. A single seed and 27 test scenes only support a limited pilot conclusion.","",
              "Every frame contains one target. Background proposals provide negatives, but no bird/clutter or target-absent frame set was evaluated. FP/frame here is not a deployment false-alarm rate. Thermal/spectral priors are assumptions, not measured infrared signatures. SWIR used 16 render samples, RGB/LWIR 128. No protected aircraft/flight settings changed.","",
              "See report.json for day/night and size-bin results, predictions.json for every prediction, gallery/index.md for declared success/failure selection, and docs/m1_defense_checklist.md for the oral explanation."]
    (out/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    learning_curves(out)
    gallery(out)


def learning_curves(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    for name in SPECS:
        hist=read(out/"runs"/name/"history.json")
        val=[r for r in hist if "val_ap50" in r]
        ax[0].plot([r["epoch"] for r in val],[r["val_ap50"] for r in val],label=name)
        ax[1].plot([r["epoch"] for r in hist],[r["box"] for r in hist],label=name)
    ax[0].set(xlabel="Training step (full batch)",ylabel="Validation AP50",ylim=(0,1),title="Checkpoint selection used validation only")
    ax[1].set(xlabel="Training step (full batch)",ylabel="Training box loss",title="Fitting labels is not held-out performance")
    ax[0].legend(fontsize=7)
    for a in ax: a.grid(alpha=.2)
    fig.suptitle("M1 compact pilot | 58 train, 11 validation | seed 17")
    fig.tight_layout(); fig.savefig(out/"learning_curves.png",dpi=160); plt.close(fig)


def draw_example(images, record, threshold, title, path):
    bands=[b for b in ("rgb","lwir") if b in images]
    canvas=Image.new("RGB",(640*len(bands),844),(18,24,35))
    d=ImageDraw.Draw(canvas)
    font_path=Path("C:/Windows/Fonts/arial.ttf")
    font=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
    for col,b in enumerate(bands):
        arr=np.clip(images[b]*255,0,255).astype(np.uint8)
        if arr.shape[-1]==1: arr=np.repeat(arr,3,axis=-1)
        canvas.paste(Image.fromarray(arr),(640*col,48))
        d.text((640*col+10,28),b.upper()+" | green: label; orange: saved-weight prediction",fill="white",font=font)
        for o in record["gts"]:
            x0,y0,x1,y1=o["box"]
            d.rectangle((x0+640*col,y0+48,x1+640*col,y1+48),outline=(50,230,110),width=2)
        for box,score in zip(record["preds"]["boxes"],record["preds"]["scores"]):
            if score < threshold: continue
            x0,y0,x1,y1=box
            d.rectangle((x0+640*col,y0+48,x1+640*col,y1+48),outline=(255,155,40),width=2)
            d.text((max(0,x0+640*col),max(48,y0+31)),f"{score:.2f}",fill=(255,155,40),font=font)
        # Annotation location is used only to choose this explanatory zoom AFTER inference.
        gt=record["gts"][0]["box"]
        cx,cy=(gt[0]+gt[2])/2,(gt[1]+gt[3])/2
        size=max(96,int(max(gt[2]-gt[0],gt[3]-gt[1])*1.4))
        left=int(np.clip(cx-size/2,0,640-size)); top=int(np.clip(cy-size/2,0,512-size))
        crop=Image.fromarray(arr).crop((left,top,left+size,top+size)).resize((240,240),Image.Resampling.NEAREST)
        canvas.paste(crop,(640*col+10,594))
        factor=240/size
        def detail_box(box,color):
            x0,y0,x1,y1=box
            if x1<left or y1<top or x0>left+size or y0>top+size: return
            c=np.clip(np.array([x0-left,y0-top,x1-left,y1-top])*factor,0,239)
            d.rectangle((c[0]+640*col+10,c[1]+594,c[2]+640*col+10,c[3]+594),outline=color,width=2)
        detail_box(gt,(50,230,110))
        for box,score in zip(record["preds"]["boxes"],record["preds"]["scores"]):
            if score>=threshold: detail_box(box,(255,155,40))
        d.text((640*col+10,568),"Enlarged detail at label location (display only)",fill="white",font=font)
        d.text((640*col+270,612),f"Scene: {record['scene_id']}\nThreshold: {threshold:.2f}\n\nDetection ran on the full frame.\nThe zoom is for inspection only.",fill="white",font=font)
    d.text((10,6),title,fill="white",font=font)
    canvas.save(path)


def gallery(out):
    manifest=read(out/"manifest.json")
    recs=read(out/"predictions.json")
    report=read(out/"report.json")
    dest=out/"gallery"; dest.mkdir(exist_ok=True)
    lines=["# Held-out success and failure examples","","For each model: first two successes and first two failures in ascending scene ID at its frozen validation threshold. A success has a matched IoU >= .5 prediction; a failure has no such match. Extra false positives can coexist with a success. This gallery is illustrative; aggregate scores use all 27 test scenes.","",
           "Each figure overlays the named model's same predictions on aligned RGB and LWIR for inspection. This does not mean an LWIR model also ran an RGB detector. The enlarged detail is chosen from the label location only after full-frame inference.",""]
    selected=[]
    for name,rr in recs.items():
        threshold=report["models"][name]["threshold"]
        counts={True:0,False:0}
        for r in rr:
            preds=r["preds"]
            box=np.array(preds["boxes"]).reshape(-1,4)
            scores=np.array(preds["scores"])
            gt=np.array([o["box"] for o in r["gts"]])
            ok=bool(np.any(iou_matrix(box[scores>=threshold],gt)>=.5))
            if counts[ok]>=2: continue
            counts[ok]+=1
            status="success" if ok else "failure"
            if ok:
                reason="At least one prediction meets the frozen score threshold and IoU >= 0.5."
            elif np.any(iou_matrix(box,gt)>=.5):
                reason="A matching box exists below the frozen display threshold; the target is missed at the operating point."
            elif len(box[scores>=threshold]) and np.any(metrics._ok_quality(box[scores>=threshold],gt,"centre",0,config_eval())[0]):
                reason="A retained prediction is near the target centre but does not meet IoU >= 0.5 (localization failure)."
            else:
                reason="No retained prediction matches the target; inspect contrast, scale and proposal coverage without assuming a proven cause."
            images,_=source_sample(manifest,r["scene_id"],["rgb","lwir"])
            filename=f"{name}_{status}_{r['scene_id']}.png"
            draw_example(images,r,threshold,f"{name} | {r['scene_id']} | {status} | threshold {threshold:.2f}",dest/filename)
            selected.append(dict(model=name,scene_id=r["scene_id"],status=status,path=filename,reason=reason))
            lines.append(f"- {name}, {status}, {r['scene_id']}: [{filename}]({filename}). {reason}")
        if not counts[False]: lines.append(f"- {name}: no missed targets at the operating point in this test set; no failure invented.")
    atomic_json(dest/"selection.json",selected)
    (dest/"index.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def demo(out,name,scene=None):
    manifest=read(out/"manifest.json")
    model,ck=load_weights(out/"runs"/name/"best.pt")
    if ck["manifest_sha256"]!=sha(out/"manifest.json") or ck["code_sha256"]!=sha(ROOT/"models/pilot.py"):
        raise ValueError("demo checkpoint does not match the frozen pilot/code")
    dest=out/"demo"/name; dest.mkdir(parents=True,exist_ok=True)
    if scene:
        if scene not in manifest["splits"]["test"]: raise ValueError("demo scene must be held out")
        ids=[scene]
    else:
        ids=[r["scene_id"] for r in read(out/"gallery/selection.json") if r["model"]==name]
    records=[]; frames=[]
    start=time.time()
    for sid in ids:
        images,meta=source_sample(manifest,sid,model.bands)
        pred=predict_images(model,images,ck["proposal_bands"])
        record=dict(scene_id=sid,preds=pred,gts=gt_objects(meta))
        path=dest/(sid+".png")
        draw_example(images,record,ck["threshold"],f"SAVED WEIGHTS | {name} | {sid} | RGB/LWIR student has no SWIR input",path)
        records.append(record); frames.append(Image.open(path).copy())
    if frames: frames[0].save(dest/"demo.gif",save_all=True,append_images=frames[1:],duration=2200,loop=0)
    atomic_json(dest/"predictions.json",records)
    reference={r["scene_id"]:r for r in read(out/"predictions.json")[name]}
    identical=all(r["preds"]==reference[r["scene_id"]]["preds"] for r in records)
    atomic_json(dest/"reproduction.json",dict(checkpoint_sha256=sha(out/"runs"/name/"best.pt"),
                scenes=ids,bands_loaded=model.bands,identical_to_evaluation=identical,seconds=time.time()-start))
    if not identical: raise AssertionError("demo differs from held-out evaluation")
    print(f"Saved-weight demo reproduced exactly: {dest}")


def package(out):
    """Bundle only the completed experiment and its reproducible local inputs."""
    import zipfile
    m=read(out/"manifest.json")
    read(out/"report.json")
    for name in ("student","lwir_only"):
        if not read(out/"demo"/name/"reproduction.json")["identical_to_evaluation"]:
            raise ValueError("both saved-weight demonstrations must pass before packaging")
    files=set()
    for folder in ("data","models","eval","noise","render","target","thermal","train","tools"):
        files.update((ROOT/folder).glob("*.py"))
    files.update((ROOT/"configs").glob("*.yaml"))
    files.update((ROOT/"docs").glob("m1*.md"))
    files.update(ROOT/f for f in ("requirements.txt","requirements-ml.txt","AGENTS.md","CLAUDE.md","README.md"))
    for root in (ROOT/m["source"],ROOT/m["swir"]):
        for sid in m["scene_ids"]:
            path=root/"meta"/(sid+".json")
            meta=read(path); files.add(path)
            files.update(root/v["layers"] for v in meta["modalities"].values())
    for folder in ("cache2","cache3","runs","gallery","demo"):
        files.update(p for p in (out/folder).rglob("*") if p.is_file())
    files.update(p for p in out.iterdir() if p.is_file() and p.suffix in (".json",".md",".png"))
    files.discard(out/"validation_example.png")
    files.discard(out/"package_manifest.json")
    files.update(out/f for f in ("evidence_checks.log",))
    manifest={str(p.resolve().relative_to(ROOT)).replace("\\","/"):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    atomic_json(out/"package_manifest.json",manifest)
    files.add(out/"package_manifest.json")
    dest=out.parent/"m1_pilot_evidence.zip"
    with zipfile.ZipFile(dest,"w",compression=zipfile.ZIP_STORED) as z:
        for path in sorted(files): z.write(path,path.resolve().relative_to(ROOT))
        z.write(ROOT/"docs/m1_pilot_bundle_readme.md","START_HERE.md")
    atomic_json(dest.with_suffix(".json"),dict(path=dest.name,sha256=sha(dest),bytes=dest.stat().st_size,files=len(files)+1))
    print(f"Saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","train","evaluate","demo","package"])
    p.add_argument("--out",type=Path,default=ROOT/"out/m1/pilot")
    p.add_argument("--swir",action="store_true")
    p.add_argument("--wait-for-render",action="store_true",help="prepare SWIR as the active local render finishes scenes")
    p.add_argument("--models",nargs="+",choices=list(SPECS),default=list(SPECS))
    p.add_argument("--model",choices=list(SPECS),default="student")
    p.add_argument("--scene")
    a=p.parse_args(); a.out=a.out.resolve(); torch.set_num_threads(1)
    if a.stage=="prepare": prepare(a.out,a.swir,a.wait_for_render)
    elif a.stage=="train": train(a.out,a.models)
    elif a.stage=="evaluate": evaluate(a.out)
    elif a.stage=="package": package(a.out)
    else: demo(a.out,a.model,a.scene)


if __name__=="__main__": main()
