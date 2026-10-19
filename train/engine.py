"""Shared training loop for the baselines, the teacher and the distilled student (spec 7, Phase 6).

    AdamW, linear warm-up + cosine decay, EMA weights, gradient clipping, validation AP50 on the
    target class every `eval_every` epochs; best.pt / last.pt under <out>/<model name>/.

Student (configs/train.yaml models.student, distill_from: teacher; CLAUDE.md sensor set rule 5): the frozen teacher
(RGB + LWIR uncooled + LWIR cooled + MWIR, trained first with the detection loss alone) sees every band of the scene;
the student (RGB + LWIR uncooled) gets one modality subset per batch (train.yaml distill.modality_dropout), absent
bands entering as zeros with their presence flag. Loss: L_det + lambda_resp L_resp + lambda_feat L_feat
(models/distill.py: response KD at teacher-positive locations, masked feature KD through 1x1 adapters with
CNR-gap weights). Validation and checkpoints use the student alone; the adapters are discarded.
Device-agnostic: CUDA when available (training a real run on CPU is slow), CPU otherwise.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (ROOT, os.path.join(ROOT, "target")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.dataset import SceneDataset, collate, torch_dataset  # noqa: E402
from data.provenance import atomic_json, config_identity, dataset_identity, digest  # noqa: E402
from models.detector import build_detector  # noqa: E402
from models.loss import DetLoss, assign_batch  # noqa: E402
from render.sampling import load_yaml  # noqa: E402


def load_all_configs(train_cfg: str | None = None) -> dict:
    from render.sampling import load_configs
    cfgs = load_configs()
    cfgs["train"] = load_yaml(train_cfg or os.path.join(ROOT, "configs", "train.yaml"))
    return cfgs


def device_of(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class EMA:
    def __init__(self, model, decay=0.9998, tau=2000):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.n = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay * (1 - math.exp(-self.n / self.tau))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def save_ckpt(path, model, cfg_model: dict, bands, epoch, metrics, train_cfg, **extra):
    tmp = str(path) + ".tmp"
    torch.save(dict(state_dict=model.state_dict(), bands=list(bands), model_cfg=cfg_model, epoch=epoch,
                    metrics=metrics, detector=train_cfg["detector"], classes=train_cfg["classes"], **extra), tmp)
    os.replace(tmp, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(detector=ck["detector"], classes=ck["classes"])
    m = build_detector(cfg, ck["bands"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def train_model(name: str, data_root: str, out_root: str, cfgs: dict, teacher_ckpt: str | None = None,
                max_iters: int | None = None, epochs: int | None = None, log=print,
                resume: bool = False, gate_epoch: int | None = None, gate_ap50: float = 0.5,
                continue_after_gate: bool = False) -> str:
    T = cfgs["train"]
    tc = T["train"]
    mcfg = T["models"][name]
    bands = list(mcfg["bands"])
    dev = device_of(tc["device"])
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    teacher = None
    load_bands = list(bands)
    if mcfg.get("distill_from"):
        if not teacher_ckpt:
            raise ValueError(f"model {name} distils from {mcfg['distill_from']}: pass --teacher <best.pt>")
        teacher, tck = load_ckpt(teacher_ckpt, dev)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        want = list(T["models"][mcfg["distill_from"]]["bands"])
        if list(tck["bands"]) != want:
            raise ValueError(f"teacher checkpoint bands {tck['bands']} != train.yaml {mcfg['distill_from']} {want}")
        load_bands += [b for b in tck["bands"] if b not in load_bands]
    train_ds = SceneDataset(data_root, "train", load_bands, cfgs, train=True, seed=tc["seed"], aug=tc["augment"],
                            use_blur_box=tc.get("use_blur_box", False), cache_layers=tc.get("cache_layers", 0))
    val_ds = SceneDataset(data_root, "val", bands, cfgs, train=False, seed=tc["seed"], use_blur_box=tc.get("use_blur_box", False))
    if len(train_ds) == 0:
        raise RuntimeError(f"no training scenes under {data_root}")
    if gate_epoch and not len(val_ds):
        raise ValueError("the detector decision gate requires validation scenes")
    loader = torch.utils.data.DataLoader(torch_dataset(train_ds), batch_size=tc["batch"], shuffle=True,
                                         num_workers=tc["workers"], collate_fn=collate, drop_last=len(train_ds) > tc["batch"],
                                         persistent_workers=False)
    model = build_detector(T, bands).to(dev)
    crit = DetLoss(len(T["classes"]), **T["loss"])
    kd = _Distiller(model, teacher, bands, T["distill"], crit, tc["seed"]).to(dev) if teacher is not None else None
    params = list(model.parameters()) + (list(kd.parameters()) if kd is not None else [])
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    ema = EMA(model, tc["ema_decay"])
    n_ep = epochs if epochs is not None else tc["epochs"]
    if n_ep <= 0 or (max_iters is not None and max_iters <= 0):
        raise ValueError("epochs and max_iters must be positive")
    if gate_epoch and max_iters:
        raise ValueError("the convergence gate cannot be used with a smoke iteration limit")
    steps = max(1, len(loader)) * n_ep
    warm = max(1, len(loader)) * tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)) * 0.95 + 0.05))
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    last_path = os.path.join(out, "last.pt")
    if not resume and any(os.path.exists(os.path.join(out, p)) for p in ("last.pt", "best.pt", "config.json")):
        raise FileExistsError(f"{last_path} exists: use --resume or a new output directory")
    if resume and not os.path.exists(last_path) and os.path.exists(os.path.join(out, "best.pt")):
        raise ValueError("best.pt exists without resumable last.pt; use a fresh run directory")
    provenance = dict(config_sha256=config_identity(cfgs), overlay=cfgs.get("overlay"),
                      dataset_sha256=dataset_identity(train_ds.metas + val_ds.metas),
                      torch_version=str(torch.__version__), device=str(dev))
    # Workers and device can change on restart; experiment and optimisation settings cannot.
    train_identity = copy.deepcopy(T)
    for key in ("device", "workers", "cache_layers"):
        train_identity["train"].pop(key, None)
    run_hash = digest(dict(train=train_identity, model=name, epochs=n_ep,
                           data=provenance["dataset_sha256"], config=provenance["config_sha256"],
                           gate_epoch=gate_epoch, gate_ap50=gate_ap50, max_iters=max_iters))
    best, it, hist = -1.0, 0, []
    start_ep = 0
    gate = {}
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        state = ck.get("training_state")
        if state is None or state["run_hash"] != run_hash:
            raise ValueError("resume requires the same dataset, effective configs and epoch budget")
        if ck.get("decision_gate", {}).get("status") == "needs_review" and not continue_after_gate:
            raise RuntimeError("epoch-10 gate needs review: inspect preview/history, use the fallback, "
                               "or explicitly pass --continue-after-gate to continue the repository detector")
        model.load_state_dict(state["model"])
        ema.m.load_state_dict(ck["state_dict"])
        ema.n = state["ema_n"]
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        if dev.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        best, it, hist = state["best"], state["iters"], state["history"]
        gate = ck.get("decision_gate", {})
        start_ep = ck["epoch"] + 1
        atomic_json(os.path.join(out, "history.json"), hist)
        if start_ep >= n_ep or (max_iters and it >= max_iters):
            return os.path.join(out, "best.pt")
        log(f"[{name}] resumed at epoch {start_ep + 1}, iteration {it}")
    atomic_json(os.path.join(out, "config.json"), dict(model=name, bands=bands, train=T,
                teacher=teacher_ckpt, data=os.path.abspath(data_root), effective_configs=cfgs,
                provenance=provenance, epochs=n_ep, max_iters=max_iters, run_hash=run_hash))
    for ep in range(start_ep, n_ep):
        train_ds.set_epoch(ep)
        model.train()
        t0 = time.time()
        losses = []
        for imgs, tgts in loader:
            imgs = {b: v.to(dev, non_blocking=True) for b, v in imgs.items()}
            if kd is None:
                raw = model({b: imgs[b] for b in bands})
                loss, parts = crit(model, raw, tgts)
            else:
                loss, parts = kd.step(imgs, tgts, it)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at epoch {ep + 1}, iteration {it + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            opt.step()
            sched.step()
            ema.update(model)
            losses.append(float(loss.detach()))
            it += 1
            if it % 10 == 0 or it == 1:
                log(f"[{name}] ep {ep} it {it} loss {float(loss):.4f} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items())
                    + f" lr {sched.get_last_lr()[0]:.2e}")
            if max_iters and it >= max_iters:
                break
        log(f"[{name}] epoch {ep} done in {time.time() - t0:.0f}s")
        last = (ep + 1 == n_ep) or (max_iters and it >= max_iters)
        row = dict(epoch=ep, epoch_number=ep + 1, iters=it, train_loss=float(np.mean(losses)),
                   lr=sched.get_last_lr()[0], seconds=time.time() - t0)
        if len(val_ds) and ((ep + 1) % tc["eval_every"] == 0 or last or ep + 1 == gate_epoch):
            from eval.metrics import quick_ap50
            ap = quick_ap50(ema.m, val_ds, bands, T["eval"], dev)
            if not np.isfinite(ap):
                raise FloatingPointError("validation AP50 is undefined; check validation target labels")
            row["val_ap50_target"] = ap
            log(f"[{name}] val AP50(target) {ap:.4f}")
            if ap > best:
                best = ap
                save_ckpt(os.path.join(out, "best.pt"), ema.m, mcfg, bands, ep,
                          dict(val_ap50_target=ap), T, provenance=provenance)
        hist.append(row)
        if ep + 1 == gate_epoch:
            gate = dict(epoch_number=ep + 1, threshold=gate_ap50,
                        val_ap50_target=row["val_ap50_target"],
                        status="passed" if row["val_ap50_target"] >= gate_ap50 else "needs_review")
            atomic_json(os.path.join(out, "decision_gate.json"), gate)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     ema_n=ema.n, best=best, iters=it, history=hist, run_hash=run_hash,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)
        save_ckpt(last_path, ema.m, mcfg, bands, ep, dict(history=hist), T,
                  provenance=provenance, training_state=state, decision_gate=gate)
        atomic_json(os.path.join(out, "history.json"), hist)
        if gate.get("status") == "needs_review" and not continue_after_gate:
            raise RuntimeError(f"{name}: epoch-{gate_epoch} AP50 below {gate_ap50}; state saved. "
                               "Inspect preview/history and use the documented Ultralytics fallback.")
        if max_iters and it >= max_iters:
            break
    if not os.path.exists(os.path.join(out, "best.pt")):
        save_ckpt(os.path.join(out, "best.pt"), ema.m, mcfg, bands, ep, dict(note="no validation scenes"), T)
    atomic_json(os.path.join(out, "history.json"), hist)
    return os.path.join(out, "best.pt")


class _Distiller(torch.nn.Module):
    """One distillation step (spec 7.2): modality subset for the student, frozen teacher on all its bands, the
    detection, response and masked feature losses. Holds the feature adapters (trained, then discarded)."""

    def __init__(self, student, teacher, student_bands, cfg: dict, crit, seed: int):
        super().__init__()
        from models.distill import FeatureDistiller
        self.crit, self.cfg = crit, cfg
        self._models = (student, teacher)              # plain attributes: only the adapters are this module's params
        self.bands = list(student_bands)
        subsets, probs = [], []
        for key, p in cfg["modality_dropout"].items():
            sub = [b.strip() for b in key.split("+")]
            if not set(sub) <= set(self.bands):
                raise ValueError(f"modality subset {key} is not inside the student's bands {self.bands}")
            subsets.append(sub)
            probs.append(float(p))
        self.subsets, self.probs = subsets, np.array(probs) / np.sum(probs)
        self.rng = np.random.default_rng([seed, 7213])
        self.fd = FeatureDistiller(student.feat_channels, teacher.feat_channels, tuple(cfg["feat_levels"]))

    def step(self, imgs: dict, tgts: list, it: int):
        from models.distill import cnr_gap_weights, feature_masks, response_kd
        student, teacher = self._models
        sub = self.subsets[int(self.rng.choice(len(self.subsets), p=self.probs))]
        raw, sfeat = student({b: (imgs[b] if b in sub else None) for b in self.bands}, return_feats=True)
        flat = student.flatten(raw)
        l_det, parts = self.crit(student, raw, tgts, flat=flat)
        with torch.no_grad():
            traw, tfeat = teacher({b: imgs[b] for b in teacher.bands}, return_feats=True)
            tf = teacher.flatten(traw)
            t_pos = assign_batch(self.crit.assign, tf, tgts)[2]
        l_resp, rp = response_kd(flat, tf, t_pos, float(self.cfg["temperature"]))
        w = [cnr_gap_weights(t.get("cnr", torch.full((len(t["boxes"]), 5), float("nan"))), teacher.bands, sub,
                             float(self.cfg["cnr_beta"]), float(self.cfg["cnr_scale"])) if len(t["boxes"])
             else torch.zeros(0) for t in tgts]
        strides = dict(zip(teacher.feat_channels, teacher.strides))
        masks = feature_masks(tgts, {lv: tfeat[lv] for lv in self.fd.levels}, strides, w, float(self.cfg["mask_floor"]))
        l_feat, fp = self.fd(sfeat, tfeat, masks)
        loss = l_det + float(self.cfg["lambda_resp"]) * l_resp + float(self.cfg["lambda_feat"]) * l_feat
        parts = dict(parts, **rp, **fp, subset=len(sub))
        return loss, parts


def cli(default_model: str | None = None, needs_teacher: bool = False):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "out", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "runs"))
    ap.add_argument("--model", default=default_model, required=default_model is None)
    ap.add_argument("--teacher", default=None, required=needs_teacher, help="teacher best.pt (student only)")
    ap.add_argument("--config", default=None, help="train.yaml override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="resume saved optimiser, EMA and RNG state")
    ap.add_argument("--gate-epoch", type=int, default=None, help="M1 RGB decision gate (10)")
    ap.add_argument("--continue-after-gate", action="store_true")
    ap.add_argument("--no-p2", action="store_true", help="drop the stride-4 head (spec 7 baseline gap run)")
    a = ap.parse_args()
    cfgs = load_all_configs(a.config)
    tc = cfgs["train"]["train"]
    for k in ("device", "batch", "workers"):
        if getattr(a, k) is not None:
            tc[k] = getattr(a, k)
    if a.no_p2:
        cfgs["train"]["detector"]["p2_head"] = False
    name = a.model
    out = a.out if not a.no_p2 else os.path.join(a.out, "no_p2")
    path = train_model(name, a.data, out, cfgs, a.teacher, a.max_iters, a.epochs,
                       resume=a.resume, gate_epoch=a.gate_epoch, continue_after_gate=a.continue_after_gate)
    print("best checkpoint:", path)
