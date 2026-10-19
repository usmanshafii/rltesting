"""Small CPU proposal detector for M1, separate from the full P2 architecture.

Proposals and features use pixels only. No boxes, masks, scene IDs or temperatures
are inputs to inference. SWIR can enrich teacher features but never proposals.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label, find_objects, maximum, zoom
from torch import nn

from eval.nwd import iou_matrix

BANDS = ("rgb", "lwir", "swir")
PER_BAND = 72


def luminance(im):
    return im @ np.array([0.2126, 0.7152, 0.0722], np.float32) if im.shape[-1] == 3 else im[..., 0]


def residual(im):
    y = luminance(im)
    r = y - gaussian_filter(y, 12)
    noise = max(float(np.median(np.abs(r - np.median(r))) * 1.4826), 0.002)
    return y, r / noise


def suppress(boxes, scores, threshold=0.5, limit=100):
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while len(order) and len(keep) < limit:
        i = order[0]
        keep.append(i)
        order = order[1:]
        if len(order):
            order = order[iou_matrix(boxes[i:i+1], boxes[order])[0] <= threshold]
    return np.asarray(keep, dtype=int)


def proposals(images, bands):
    """Multiscale contrast components; deliberately independent of labels/SWIR."""
    if not bands or set(bands) - {"rgb", "lwir"}:
        raise ValueError("pilot proposals must use RGB and/or LWIR only")
    boxes, scores = [], []
    for band in bands:
        y = luminance(images[band])
        H, W = y.shape
        for sigma in (0.7, 1.5, 3.0, 6.0):
            r = gaussian_filter(y, sigma) - gaussian_filter(y, max(8, 4*sigma))
            sd = max(float(np.median(np.abs(r-np.median(r))) * 1.4826), 1e-4)
            z = np.abs(r) / sd
            for threshold in (3.5, 6.0):
                labs, n = label(z > threshold)
                slices = find_objects(labs)
                strength = maximum(z, labs, np.arange(1,n+1))
                # Screen by image contrast before extracting components; no label information.
                candidates = np.argsort(-strength,kind="stable")[:160]
                for j in candidates:
                    k,sl = j+1,slices[j]
                    if sl is None:
                        continue
                    sy, sx = sl
                    h, w = sy.stop-sy.start, sx.stop-sx.start
                    if w*h < 3 or max(w, h) > 180 or min(w, h) > 100:
                        continue
                    component = labs[sl] == k
                    if component.sum() < 3:
                        continue
                    score = float(np.percentile(z[sl][component], 90))
                    boxes.append([max(0,sx.start-1), max(0,sy.start-1), min(W,sx.stop+1), min(H,sy.stop+1)])
                    scores.append(score)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32)
    keep = suppress(boxes, scores, 0.75, 80)
    return boxes[keep], scores[keep]


def features(images, boxes, bands):
    """4 geometry features + 72 per band; all derived from pixels/proposals."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 1)
    out = [np.c_[np.log(wh)/5, np.log(wh[:, 0]/wh[:, 1]), np.log(wh.prod(1))/10]]
    for band in bands:
        y, r = residual(images[band])
        H, W = y.shape
        rows = []
        for box in boxes:
            x0, y0, x1, y1 = box.astype(int)
            pad = max(3, int(max(x1-x0, y1-y0)*0.3))
            X0,Y0,X1,Y1 = max(0,x0-pad),max(0,y0-pad),min(W,x1+pad),min(H,y1+pad)
            patch = r[Y0:Y1, X0:X1]
            small = zoom(patch, (8/patch.shape[0], 8/patch.shape[1]), order=1)
            # Signed contrast is useful; clipping avoids extreme hot pixels dominating.
            small = np.clip(small, -30, 30).reshape(-1)/10
            core = r[y0:y1, x0:x1]
            raw = y[y0:y1, x0:x1]
            stats = [core.mean()/10, core.std()/10, np.percentile(core,10)/10,
                     np.percentile(core,90)/10, np.abs(core).mean()/10,
                     raw.mean(), raw.std(), np.mean(np.abs(core)>3)]
            rows.append(np.r_[small, np.clip(stats,-10,10)])
        out.append(np.asarray(rows,np.float32).reshape(-1, PER_BAND))
    return np.concatenate(out, axis=1).astype(np.float32)


def select_features(x, loaded_bands, bands):
    indices = list(range(4))
    for band in bands:
        start = 4 + loaded_bands.index(band)*PER_BAND
        indices.extend(range(start,start+PER_BAND))
    return x[:, indices]


def encode(boxes, gt):
    wh = np.maximum(boxes[:,2:]-boxes[:,:2],1)
    center = (boxes[:,:2]+boxes[:,2:])/2
    gc = (gt[:,:2]+gt[:,2:])/2
    gw = np.maximum(gt[:,2:]-gt[:,:2],0.1)
    return np.c_[(gc-center)/wh,np.log(gw/wh)].astype(np.float32)


def decode(boxes, delta, shape=(512,640)):
    wh = np.maximum(boxes[:,2:]-boxes[:,:2],1)
    center = (boxes[:,:2]+boxes[:,2:])/2 + np.clip(delta[:,:2],-2,2)*wh
    wh = wh*np.exp(np.clip(delta[:,2:],-2,2))
    out = np.c_[center-wh/2,center+wh/2]
    return np.clip(out,0,np.array([shape[1],shape[0],shape[1],shape[0]]))


class PilotDetector(nn.Module):
    def __init__(self, bands, hidden=64):
        super().__init__()
        self.bands = list(bands)
        self.hidden = hidden
        n = 4 + PER_BAND*len(bands)
        self.register_buffer("mean",torch.zeros(n))
        self.register_buffer("scale",torch.ones(n))
        self.net = nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.cls = nn.Linear(hidden,1)
        self.box = nn.Linear(hidden,4)
        nn.init.zeros_(self.box.weight)
        nn.init.zeros_(self.box.bias)

    def forward(self,x):
        h = self.net(torch.clamp((x-self.mean)/self.scale, -10,10))
        return self.cls(h).squeeze(-1),self.box(h)


@torch.no_grad()
def predict_features(model, x, boxes, conf=0.001):
    logits,delta = model(torch.as_tensor(x,dtype=torch.float32))
    scores = logits.sigmoid().numpy()
    bx = decode(boxes,delta.numpy())
    keep = np.where((scores >= conf) & np.all(bx[:,2:]>bx[:,:2],axis=1))[0]
    k = keep[suppress(bx[keep],scores[keep],0.4,30)]
    return dict(boxes=bx[k].tolist(),scores=scores[k].tolist(),labels=[0]*len(k))


def load_weights(path):
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("architecture") != "m1-proposal-mlp-v1":
        raise ValueError("not an M1 compact pilot checkpoint")
    model = PilotDetector(ck["bands"],ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(),ck


def predict_images(model, images, proposal_bands):
    boxes,_ = proposals(images,proposal_bands)
    return predict_features(model,features(images,boxes,model.bands),boxes)
