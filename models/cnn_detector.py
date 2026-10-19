"""Compact fully convolutional detector for the M1 CNN pilot (CPU-trainable).

A learned replacement for the M1 compact proposal/MLP pilot (models/pilot.py): every decision is
made by convolutions on pixels, with no hand-made proposal stage. It is still much smaller than the
P2 detector of spec 7.1 (models/detector.py), so results from it are not results for that model.

    per-band stems (stride 4)  ->  concat fusion (+ per-band coverage planes)  ->  encoder at
    strides 4 / 8 / 16 (dilated)  ->  top-down neck back to stride 4  ->  CenterNet-style heads

Coverage planes (models built with `presence=True`): one plane per band at stride 4, the share of each cell the band's
camera actually sees (its per-pixel coverage mask: a zoom camera that cannot widen to the grid sees only the centre,
data/camera_emulation.py) times the band's 0/1 flag (modality dropout). With no mask given the plane is the flag
alone, constant over the frame, which is what v1/v2 checkpoints were trained with.

Heads at stride 4: a target-centre heatmap (focal loss on Gaussian targets), the sub-cell centre
offset and log width / height in input pixels. A 2-4 px target is one heatmap cell; the offset
recovers its sub-cell position. Boxes are decoded from local heatmap maxima, so the detector can
return no box at all, or several.

Distillation (spec 7.2, compact form, models only - the training loop lives in tools/m1_det2.py):
  * response KD: the student's softened heatmap against the frozen teacher's (temperature T), and
    the box parameters at teacher-positive cells;
  * feature KD: the stride-4 and stride-8 neck outputs pass a 1x1 adapter, both sides are
    standardised per channel, and the squared error is weighted by M = sum_i w_i G_i + floor with
    a Gaussian G_i per object and a CNR-gap weight w_i. Adapters are discarded after training.
Inputs are only the bands a model is built for; a student never receives SWIR.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_CHANNELS = {"rgb": 3, "lwir": 1, "swir": 1, "lwir_cooled": 1, "mwir": 1}
STRIDE = 4
ARCH = "m1-cnn-centernet-v1"
ARCH_V2 = "m1-cnn-centernet-v2"          # same network; v2 training targets/losses and input preprocessing


# ============================================================================ input normalisation
def standardise(img: np.ndarray) -> np.ndarray:
    """Per-image, per-channel robust standardisation of a sensor frame (HxWxC in [0, 1]).

    (x - median) / (1.4826 MAD), clipped to +-30 and divided by 10. A pure function of the frame's
    own pixels: it removes auto-exposure / AGC level and gain so that a value is a contrast in
    noise units, the quantity that decides faint-target detection."""
    x = np.asarray(img, np.float32)
    x = x if x.ndim == 3 else x[..., None]
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        v = x[..., c]
        sub = v[::4, ::4]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826
        if mad < 1e-6:                       # quantised dark frames: fall back to the standard deviation
            mad = float(sub.std())
        out[..., c] = np.clip((v - med) / max(mad, 1e-4), -30.0, 30.0) / 10.0
    return out


# ============================================================================ blocks
class ConvBN(nn.Sequential):
    def __init__(self, ci, co, k=3, s=1, d=1, act=True):
        super().__init__(nn.Conv2d(ci, co, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True) if act else nn.Identity())


class Res(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.a = ConvBN(c, c, 3, 1, d)
        self.b = ConvBN(c, c, 3, 1, d, act=False)

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class Stem(nn.Module):
    """One band to stride 4. A lossless space-to-depth step (each 2x2 pixel block becomes four
    channels) keeps every pixel of a 2-4 px target, then convolutions at stride 2 and 4. This avoids
    full-resolution activations, which dominate CPU training time, without discarding pixels."""

    def __init__(self, ci, c):
        super().__init__()
        self.net = nn.Sequential(ConvBN(4 * ci, 24, 3, 1), ConvBN(24, c, 3, 2), ConvBN(c, c, 3, 1))

    def forward(self, x):
        return self.net(F.pixel_unshuffle(x, 2))


class CNNDetector(nn.Module):
    def __init__(self, bands, stem=24, widths=(32, 48, 64), presence=False):
        super().__init__()
        self.bands = list(bands)
        self.presence = bool(presence)
        self.widths = tuple(widths)
        self.stem_c = int(stem)
        c4, c8, c16 = widths
        self.stems = nn.ModuleDict({b: Stem(BAND_CHANNELS[b], stem) for b in self.bands})
        fuse_in = stem * len(self.bands) + (len(self.bands) if presence else 0)
        self.fuse = ConvBN(fuse_in, c4, 1)
        self.e4 = Res(c4)
        self.d8 = ConvBN(c4, c8, 3, 2)
        self.e8 = nn.Sequential(Res(c8), Res(c8, 2))
        self.d16 = ConvBN(c8, c16, 3, 2)
        self.e16 = nn.Sequential(Res(c16, 2), Res(c16, 4))
        self.l16 = ConvBN(c16, c8, 1)
        self.n8 = ConvBN(c8, c8, 3)
        self.l8 = ConvBN(c8, c4, 1)
        self.n4 = ConvBN(c4, c4, 3)
        self.hm = nn.Sequential(ConvBN(c4, 24, 3), nn.Conv2d(24, 1, 1))
        self.reg = nn.Sequential(ConvBN(c4, 24, 3), nn.Conv2d(24, 4, 1))   # dx, dy (cells), log w, log h (px)
        nn.init.constant_(self.hm[-1].bias, -4.6)                          # prior p = 0.01 (CenterNet)
        nn.init.zeros_(self.reg[-1].bias)
        nn.init.normal_(self.reg[-1].weight, std=0.01)

    def config(self) -> dict:
        return dict(bands=self.bands, stem=self.stem_c, widths=list(self.widths), presence=self.presence)

    def forward(self, images: dict, present: dict | None = None, masks: dict | None = None):
        """images: band -> [B, C, H, W] standardised; H, W multiples of 16. present: band -> [B]
        0/1 flags (modality dropout; a dropped band must also be zeroed by the caller). masks: band ->
        [B, 1, H, W] coverage, 1 where the band's camera sees the scene (default: everywhere)."""
        x0 = next(iter(images.values()))
        B, _, H, W = x0.shape
        feats = [self.stems[b](images[b]) for b in self.bands]
        if self.presence:
            h4, w4 = feats[0].shape[-2:]
            for b in self.bands:
                p = (torch.ones(B, device=x0.device) if present is None else present[b].float()).view(B, 1, 1, 1)
                m = None if masks is None else masks.get(b)
                feats.append(p.expand(B, 1, h4, w4) if m is None else F.avg_pool2d(m.float(), STRIDE) * p)
        f4 = self.e4(self.fuse(torch.cat(feats, 1)))
        f8 = self.e8(self.d8(f4))
        f16 = self.e16(self.d16(f8))
        n8 = self.n8(f8 + F.interpolate(self.l16(f16), size=f8.shape[-2:], mode="nearest"))
        n4 = self.n4(f4 + F.interpolate(self.l8(n8), size=f4.shape[-2:], mode="nearest"))
        return dict(hm=self.hm(n4), reg=self.reg(n4), n4=n4, n8=n8)


# ============================================================================ targets and losses
def gaussian_sigma_cells(w, h):
    """Heatmap Gaussian width in stride-4 cells; CenterNet-like, never below 0.5 cell."""
    return max(0.5, 0.15 * math.sqrt(max(w, 0.5) * max(h, 0.5)) / STRIDE)


def build_targets(boxes_list, H, W):
    """boxes_list: per image an [N, 4] xyxy array (input pixels). Returns heatmap [B,1,h,w],
    regression targets [B,4,h,w], a positive-cell mask [B,1,h,w] and per-object Gaussian masks
    (list of [N, h, w]) for feature distillation."""
    h, w = H // STRIDE, W // STRIDE
    B = len(boxes_list)
    hm = np.zeros((B, 1, h, w), np.float32)
    reg = np.zeros((B, 4, h, w), np.float32)
    pos = np.zeros((B, 1, h, w), np.float32)
    gauss = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i, boxes in enumerate(boxes_list):
        g_i = []
        for x0, y0, x1, y1 in np.asarray(boxes, np.float32).reshape(-1, 4):
            bw, bh = max(x1 - x0, 0.3), max(y1 - y0, 0.3)
            cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
            ix, iy = int(min(max(cx, 0), w - 1e-3)), int(min(max(cy, 0), h - 1e-3))
            s = gaussian_sigma_cells(bw, bh)
            g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * s * s))
            hm[i, 0] = np.maximum(hm[i, 0], g)
            hm[i, 0, iy, ix] = 1.0
            reg[i, :, iy, ix] = (cx - ix, cy - iy, math.log(bw), math.log(bh))
            pos[i, 0, iy, ix] = 1.0
            # feature-KD mask: sigma = max(1 cell, half the box size in cells) (spec 7.2)
            sf = max(1.0, 0.5 * max(bw, bh) / STRIDE)
            g_i.append(np.exp(-((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2) / (2 * sf * sf)))
        gauss.append(np.asarray(g_i, np.float32).reshape(-1, h, w))
    return torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(pos), gauss


def focal_loss(logits, target, pos):
    """CenterNet penalty-reduced focal loss, normalised by the number of objects."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos_l = -torch.log(p) * (1 - p) ** 2 * pos
    neg_l = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return (pos_l.sum() + neg_l.sum()) / n


def reg_loss(pred, target, pos):
    """L1 on centre offset and log size at the positive cells."""
    n = pos.sum().clamp(min=1.0)
    off = (F.l1_loss(pred[:, :2], target[:, :2], reduction="none") * pos).sum() / n
    size = (F.l1_loss(pred[:, 2:], target[:, 2:], reduction="none") * pos).sum() / n
    return off, size


def detection_loss(out, hm_t, reg_t, pos, w_off=1.0, w_size=1.0):
    hm = focal_loss(out["hm"], hm_t, pos)
    off, size = reg_loss(out["reg"], reg_t, pos)
    return hm + w_off * off + w_size * size, dict(hm=float(hm), off=float(off), size=float(size))


def _chan_std(f):
    m = f.mean(dim=(0, 2, 3), keepdim=True)
    s = f.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-3)
    return (f - m) / s


def feature_kd(student_f, teacher_f, adapter, mask):
    """Masked, per-channel-standardised MSE between adapted student and frozen teacher features."""
    s = _chan_std(adapter(student_f))
    t = _chan_std(teacher_f.detach())
    m = F.interpolate(mask, size=s.shape[-2:], mode="bilinear", align_corners=False) if mask.shape[-2:] != s.shape[-2:] else mask
    return (((s - t) ** 2).mean(1, keepdim=True) * m).sum() / m.sum().clamp(min=1e-6)


def response_kd(student_out, teacher_out, temperature=2.0, tau=0.1, keep=None):
    """Softened heatmap distillation plus box parameters where the teacher is confident. keep ([B, 1, h, w] in 0..1,
    kd_keep): cells to leave out, around targets the student's own frames do not show."""
    T = float(temperature)
    s = student_out["hm"] / T
    t = torch.sigmoid(teacher_out["hm"].detach() / T)
    # weight cells by how much either network thinks a target is there, so the empty sky
    # (almost every cell) does not drown the signal
    w = (t + torch.sigmoid(s.detach())).clamp(max=1.0) + 0.01
    if keep is not None:
        w = w * keep
    hm = (F.binary_cross_entropy_with_logits(s, t, reduction="none") * w).sum() / w.sum().clamp(min=1e-6)
    tp = (torch.sigmoid(teacher_out["hm"].detach()) > tau).float()
    if keep is not None:
        tp = tp * (keep > 0.5).float()
    n = tp.sum().clamp(min=1.0)
    box = (F.l1_loss(student_out["reg"], teacher_out["reg"].detach(), reduction="none") * tp).sum() / (4 * n)
    return hm * T * T + box


def kd_mask(gauss, weights, floor=0.05, gate=None):
    """M = sum_i w_i G_i + floor, per image; gauss: list of [N, h, w], weights: list of [N]. With `gate` (kd_gate), M is 0
    outside the gate, so no background feature is imitated."""
    out = []
    for n, (g, w) in enumerate(zip(gauss, weights)):
        m = np.full(g.shape[1:], floor, np.float32)
        for gi, wi in zip(g, w):
            m += float(wi) * gi
        if gate is not None:
            m = m * gate[n, 0].numpy()
        out.append(m)
    return torch.from_numpy(np.stack(out)[:, None])


def kd_gate(gauss, radius=2.0, floor=1e-3):
    """Per image, 1 on the cells near a labelled target (its Gaussian at or above `floor`, or within `radius` cells of its
    centre) and 0 elsewhere; a crop without a target is all 0. Distillation inside this gate only cannot teach the
    student the teacher's false alarms, which lie away from the labels (2026-10-19, docs/privileged_teacher_case.md 6c).
    gauss: list of [N, h, w]."""
    out = []
    for g in gauss:
        h, w = g.shape[1:]
        m = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for gi in g:
            iy, ix = np.unravel_index(int(np.argmax(gi)), gi.shape)
            m[(gi >= floor) | ((yy - iy) ** 2 + (xx - ix) ** 2 <= radius ** 2)] = 1.0
        out.append(m)
    return torch.from_numpy(np.stack(out)[:, None])


def cnr_gap_weight(c_teacher, c_student, beta=2.0, student_floor=None, unseen=1.0):
    """w = 1 + beta clip((C_T - C_S) / 5, 0, 1) (spec 7.2). With student_floor, a target the student's own frames do not
    show (C_S < student_floor) gets `unseen` instead. Its features cannot be learned from those frames, and the CNR-gap
    weight would otherwise be largest there (docs/privileged_teacher_case.md, sections 1 and 6)."""
    if student_floor is not None and c_student < student_floor:
        return float(unseen)
    return 1.0 + beta * float(np.clip((c_teacher - c_student) / 5.0, 0.0, 1.0))


def kd_keep(gauss, unseen):
    """Per image, 1 minus the union of the Gaussian cores of the targets the student does not see: the response KD
    leaves those cells out. gauss: list of [N, h, w]; unseen: list of N flags (empty: keep everything)."""
    out = []
    for g, u in zip(gauss, unseen):
        m = np.ones(g.shape[1:], np.float32)
        for gi, ui in zip(g, u):
            if ui:
                m = np.minimum(m, 1.0 - gi)
        out.append(m)
    return torch.from_numpy(np.stack(out)[:, None])


# ============================================================================ decoding
@torch.no_grad()
def decode(out, conf=0.01, topk=20, shape=None):
    """Local maxima of the heatmap -> boxes (xyxy, input pixels), scores. One dict per image."""
    out = dict(hm=out["hm"].detach().float().cpu(), reg=out["reg"].detach().float().cpu())   # any device -> CPU
    hm = torch.sigmoid(out["hm"])
    peak = F.max_pool2d(hm, 3, 1, 1)
    hm = hm * (peak == hm).float()
    B, _, h, w = hm.shape
    res = []
    for i in range(B):
        scores, idx = hm[i, 0].flatten().topk(min(topk, h * w))
        keep = scores >= conf
        scores, idx = scores[keep], idx[keep]
        iy, ix = (idx // w).float(), (idx % w).float()
        r = out["reg"][i][:, idx // w, idx % w]
        cx = (ix + r[0]) * STRIDE
        cy = (iy + r[1]) * STRIDE
        bw, bh = torch.exp(r[2].clamp(-2, 7)), torch.exp(r[3].clamp(-2, 7))
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        if shape is not None:
            Hs, Ws = shape
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, Ws)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, Hs)
        res.append(dict(boxes=boxes.numpy().tolist(), scores=scores.numpy().tolist(), labels=[0] * len(scores)))
    return res


def pad16(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2:]
    ph, pw = (-H) % 16, (-W) % 16
    return F.pad(x, (0, pw, 0, ph), mode="replicate") if ph or pw else x


# Image pyramid: training crops are 128 px, so a target larger than LARGE_PX is shown to the network
# box-filter downsampled by 2 or 4 (tools/m1_det2.py); inference mirrors this. Each level keeps only
# boxes whose size (original pixels) lies in its band; the bands overlap and NMS merges them.
LARGE_PX = 96.0
LEVELS = ((1, 0.0, 112.0), (2, 80.0, 224.0), (4, 176.0, 1e9))


def level_for(size_px: float) -> int:
    """Downsampling factor used in training for a target with this max side (pixels)."""
    return 1 if size_px <= LARGE_PX else (2 if size_px <= 2 * LARGE_PX else 4)


def nms(boxes, scores, iou=0.5, limit=20):
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    while len(order) and len(keep) < limit:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.clip(rb - lt, 0, None).prod(-1)
        ov = inter / (area[i] + area[rest] - inter + 1e-9)
        # also suppress a box whose centre lies inside the kept one (same target at two levels)
        c = (boxes[rest, :2] + boxes[rest, 2:]) / 2
        inside = (c[:, 0] >= boxes[i, 0]) & (c[:, 0] <= boxes[i, 2]) & (c[:, 1] >= boxes[i, 1]) & (c[:, 1] <= boxes[i, 3])
        order = rest[(ov <= iou) & ~inside]
    return np.asarray(keep, int)


def to_tensors(frames: dict, bands, present: dict | None = None):
    imgs = {b: torch.from_numpy(np.ascontiguousarray(frames[b].transpose(2, 0, 1)))[None].float() for b in bands}
    pres = None
    if present is not None:
        pres = {b: torch.tensor([float(present.get(b, 1))]) for b in bands}
        imgs = {b: v * pres[b].view(1, 1, 1, 1) for b, v in imgs.items()}
    return imgs, pres


def coverage_masks(coverage: dict | None, shape, bands) -> dict | None:
    """band -> HxW float32 mask from per-band covered boxes (y0, y1, x0, x1); bands without a box are fully covered."""
    if coverage is None:
        return None
    H, W = shape
    out = {}
    for b in bands:
        m = np.zeros((H, W), np.float32)
        y0, y1, x0, x1 = coverage.get(b) or (0, H, 0, W)
        m[y0:y1, x0:x1] = 1.0
        out[b] = m
    return out


@torch.no_grad()
def predict(model, frames: dict, conf=0.01, present: dict | None = None, levels=LEVELS, masks: dict | None = None):
    """frames: band -> HxWxC standardised numpy frame (full sensor frame). masks: band -> HxW coverage
    (coverage_masks); used by models with coverage planes. Runs the pyramid and returns dict(boxes,
    scores, labels) in original pixels."""
    model.eval()
    H, W = next(iter(frames.values())).shape[:2]
    dev = next(model.parameters()).device                   # the model's device (KAL_DEVICE in tools/m1_det2.py)
    imgs, pres = to_tensors(frames, model.bands, present)
    imgs = {b: v.to(dev) for b, v in imgs.items()}
    pres = {b: v.to(dev) for b, v in pres.items()} if pres is not None else None
    mk = None
    if masks is not None and model.presence:
        mk = {b: torch.from_numpy(np.asarray(masks[b], np.float32))[None, None].to(dev) for b in model.bands if b in masks}
    boxes, scores = [], []
    for f, lo, hi in levels:
        x = {b: pad16(F.avg_pool2d(v, f) if f > 1 else v) for b, v in imgs.items()}
        m = None if mk is None else {b: pad16(F.avg_pool2d(v, f) if f > 1 else v) for b, v in mk.items()}
        d = decode(model(x, pres, m), conf=conf)[0]
        if not d["scores"]:
            continue
        bx = np.asarray(d["boxes"]) * f
        side = np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1])
        ok = (side >= lo) & (side < hi)
        boxes.append(bx[ok])
        scores.append(np.asarray(d["scores"])[ok])
    if not boxes:
        return dict(boxes=[], scores=[], labels=[])
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, W)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, H)
    k = nms(boxes, scores)
    return dict(boxes=boxes[k].tolist(), scores=scores[k].tolist(), labels=[0] * len(k))


def save(path, model, arch=ARCH, **meta):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}     # CPU tensors from any device
    torch.save(dict(architecture=arch, config=model.config(), state_dict=state, **meta), path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("architecture") not in (ARCH, ARCH_V2):
        raise ValueError(f"{path}: not an {ARCH} or {ARCH_V2} checkpoint")
    c = ck["config"]
    model = CNNDetector(c["bands"], c["stem"], tuple(c["widths"]), c["presence"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


# ============================================================================ v2 (tools/m1_det2.py)
# Changes found by the validation diagnosis of the v1 CNN (2026-10-04): column stripes caused most
# LWIR false alarms, and boxes were in the right place but the wrong size. The network is unchanged.
PREPROCESS_V2 = "8bit+destripe+standardise"


def quantise8(img: np.ndarray) -> np.ndarray:
    """The 8-bit video a commodity camera delivers: the AGC image in [0, 1] rounded to 256 levels."""
    return np.round(np.clip(np.asarray(img, np.float32), 0.0, 1.0) * 255.0) / np.float32(255.0)


def _high_pass(profile: np.ndarray, width: int) -> np.ndarray:
    """Profile minus its running median (mirrored at the ends): the stripe part, without smooth trends."""
    from scipy.ndimage import median_filter
    return profile - median_filter(profile, size=width, mode="reflect")


def destripe(img: np.ndarray, width: int = 61, band: int = 256) -> np.ndarray:
    """Remove column and row stripes from a thermal frame (HxW or HxWx1); a pure function of the frame.

    Columns: real stripes are not constant down a column (column gain x a sky that brightens towards
    the horizon, 1/f drift of the column amplifiers), so column offsets are estimated as medians over
    overlapping bands of `band` rows (stride band/2) and interpolated linearly between band centres.
    Only the high-pass part of each profile (minus its running median over `width` columns) is
    removed, which keeps smooth left-right sky gradients. A band is tall enough that an 80 px tall
    target covers under a third of it, so the median ignores targets. Rows: the same high-pass on the
    row medians removes row noise and keeps the smooth vertical sky gradient."""
    x = np.asarray(img, np.float32)
    x2 = x if x.ndim == 2 else x[..., 0]
    H, W = x2.shape
    band = min(band, H)
    starts = list(range(0, H - band + 1, max(1, band // 2))) or [0]
    centres = np.array([s + band / 2 for s in starts])
    prof = np.stack([_high_pass(np.median(x2[s:s + band], axis=0), width) for s in starts])     # [n, W]
    if len(starts) == 1:
        stripes = np.broadcast_to(prof[0], (H, W))
    else:                       # linear between band centres, extrapolated linearly above/below them
        y = np.arange(H) + 0.5
        k = np.clip(np.searchsorted(centres, y) - 1, 0, len(centres) - 2)
        t = ((y - centres[k]) / (centres[k + 1] - centres[k]))[:, None]
        stripes = prof[k] * (1 - t) + prof[k + 1] * t
    out = x2 - stripes
    out = out - _high_pass(np.median(out, axis=1), width)[:, None]
    return (out if x.ndim == 2 else out[..., None]).astype(np.float32)


def _prep_one(band: str, x: np.ndarray, version: str | None) -> np.ndarray:
    if version == PREPROCESS_V2:
        x = quantise8(x)
        if band != "rgb":
            x = destripe(x)
    return standardise(x)


def preprocess(frames: dict, version: str | None = None, coverage: dict | None = None) -> dict:
    """Camera frames (band -> HxWxC in [0, 1]) -> network input. `version` is the checkpoint's
    "preprocess" entry: None for v1 checkpoints (standardise only), PREPROCESS_V2 for v2 (8-bit
    quantisation, destriping of every thermal band, then standardise).

    coverage: band -> (y0, y1, x0, x1), the part of the grid the band's camera sees (a zoom camera that
    cannot widen to the grid sees its centre only; data/camera_emulation.covered_box). That band is
    processed inside its box alone, so the empty border does not enter the stripe medians or the
    frame's median and spread, and is 0 outside it (no contrast); the model gets the coverage as a mask."""
    if version not in (None, PREPROCESS_V2):
        raise ValueError(f"unknown preprocessing {version!r}")
    out = {}
    for b, img in frames.items():
        x = np.asarray(img, np.float32)
        box = (coverage or {}).get(b)
        H, W = x.shape[:2]
        if box is not None and tuple(box) != (0, H, 0, W):
            y0, y1, x0, x1 = (int(v) for v in box)
            full = np.zeros((H, W, x.shape[2] if x.ndim == 3 else 1), np.float32)
            if y1 > y0 and x1 > x0:
                full[y0:y1, x0:x1] = _prep_one(b, x[y0:y1, x0:x1], version)
            out[b] = full
        else:
            out[b] = _prep_one(b, x, version)
    return out


def build_targets_dense(boxes_list, H, W, region=0.5):
    """v2 targets: heatmap, positive cells and KD Gaussians exactly as `build_targets`, plus box
    regression supervised at every cell whose heatmap target is >= `region` (TTFNet-style), weighted
    by that value. A peak one cell away from the centre then still decodes the right box; v1 trained
    the box only at the single centre cell. Tiny targets (sigma 0.5 cell) keep exactly one cell.

    Returns hm, reg, pos, gauss, reg_w [B,1,h,w] and gt [B,4,h,w] (the cell's box, xyxy pixels)."""
    hm, reg, pos, gauss = build_targets(boxes_list, H, W)
    h, w = H // STRIDE, W // STRIDE
    B = len(boxes_list)
    reg = np.zeros((B, 4, h, w), np.float32)
    reg_w = np.zeros((B, 1, h, w), np.float32)
    gt = np.zeros((B, 4, h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i, boxes in enumerate(boxes_list):
        for x0, y0, x1, y1 in np.asarray(boxes, np.float32).reshape(-1, 4):
            bw, bh = max(x1 - x0, 0.3), max(y1 - y0, 0.3)
            cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
            ix, iy = int(min(max(cx, 0), w - 1e-3)), int(min(max(cy, 0), h - 1e-3))
            s = gaussian_sigma_cells(bw, bh)
            g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * s * s))
            m = (g >= region) & (g > reg_w[i, 0])         # overlapping objects: the nearer centre wins
            for c, v in enumerate((cx - xx[m], cy - yy[m], np.full(m.sum(), math.log(bw)), np.full(m.sum(), math.log(bh)))):
                reg[i, c][m] = v
            for c, v in enumerate((x0, y0, x1, y1)):
                gt[i, c][m] = v
            reg_w[i, 0][m] = g[m]
    return hm, torch.from_numpy(reg), pos, gauss, torch.from_numpy(reg_w), torch.from_numpy(gt)


def giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU of matching xyxy boxes [N, 4]: IoU minus the empty share of the enclosing box."""
    ax0, ay0, ax1, ay1 = a.unbind(-1)
    bx0, by0, bx1, by1 = b.unbind(-1)
    inter = (torch.min(ax1, bx1) - torch.max(ax0, bx0)).clamp(min=0) * (torch.min(ay1, by1) - torch.max(ay0, by0)).clamp(min=0)
    union = (ax1 - ax0).clamp(min=0) * (ay1 - ay0).clamp(min=0) + (bx1 - bx0).clamp(min=0) * (by1 - by0).clamp(min=0) - inter
    hull = ((torch.max(ax1, bx1) - torch.min(ax0, bx0)) * (torch.max(ay1, by1) - torch.min(ay0, by0))).clamp(min=1e-6)
    return inter / union.clamp(min=1e-6) - (hull - union) / hull


def decoded_boxes(reg: torch.Tensor) -> torch.Tensor:
    """Every cell's box (xyxy, input pixels) from the regression map [B,4,h,w], as `decode` does."""
    B, _, h, w = reg.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=reg.dtype, device=reg.device),
                            torch.arange(w, dtype=reg.dtype, device=reg.device), indexing="ij")
    cx, cy = (xx + reg[:, 0]) * STRIDE, (yy + reg[:, 1]) * STRIDE
    bw, bh = torch.exp(reg[:, 2].clamp(-2, 7)), torch.exp(reg[:, 3].clamp(-2, 7))
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def detection_loss_v2(out, hm_t, pos, reg_t, reg_w, gt_box, w_off=1.0, w_size=1.0, w_giou=1.0):
    """Focal heatmap loss (as v1) + weighted L1 on centre offset and log size + GIoU of the decoded
    boxes, all over the dense regression region."""
    hm = focal_loss(out["hm"], hm_t, pos)
    n = reg_w.sum().clamp(min=1e-6)
    off = (F.l1_loss(out["reg"][:, :2], reg_t[:, :2], reduction="none") * reg_w).sum() / n
    size = (F.l1_loss(out["reg"][:, 2:], reg_t[:, 2:], reduction="none") * reg_w).sum() / n
    m = reg_w[:, 0] > 0
    if bool(m.any()):
        p = decoded_boxes(out["reg"]).permute(0, 2, 3, 1)[m]
        g = gt_box.permute(0, 2, 3, 1)[m]
        wm = reg_w[:, 0][m]
        gl = ((1.0 - giou(p, g)) * wm).sum() / wm.sum()
    else:
        gl = out["reg"].sum() * 0.0
    loss = hm + w_off * off + w_size * size + w_giou * gl
    return loss, dict(hm=float(hm), off=float(off), size=float(size), giou=float(gl))
