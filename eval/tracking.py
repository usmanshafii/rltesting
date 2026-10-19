"""Multi-target tracker for video detections: the false-alarm unit for M1 is a *confirmed track*.

A detection starts a tentative track. A track is confirmed once it has detections in at least M of its
last N frames (M-of-N), and it is deleted after `max_coast` frames without a detection (tentative
tracks: once N frames pass without confirmation). Association is greedy by score inside a gate around
the predicted position (constant velocity, alpha-beta update). Noise false alarms rarely repeat in
the same place, so they seldom confirm; a false alarm that persists (a stripe, a fixed hot pixel)
does, which is why false alarms are counted on video with fixed camera patterns, not single frames.
"""
from __future__ import annotations

import math


class Track:
    __slots__ = ("id", "cx", "cy", "w", "h", "vx", "vy", "hits", "age", "misses", "confirmed", "score")

    def __init__(self, tid, box, score):
        self.id = tid
        self.cx, self.cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.w, self.h = box[2] - box[0], box[3] - box[1]
        self.vx = self.vy = 0.0
        self.hits = [1]
        self.age = 1
        self.misses = 0
        self.confirmed = False
        self.score = score

    def predicted(self):
        return self.cx + self.vx, self.cy + self.vy

    def box(self):
        return [self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2]


class MultiTracker:
    def __init__(self, m=3, n=5, gate_px=8.0, gate_scale=2.5, max_coast=10, alpha=0.6, beta=0.15):
        self.m, self.n = m, n
        self.gate_px, self.gate_scale = gate_px, gate_scale
        self.max_coast, self.alpha, self.beta = max_coast, alpha, beta
        self.tracks: list[Track] = []
        self.next_id = 1
        self.confirmed_ids: list[int] = []        # every track that ever confirmed, in order

    def step(self, dets):
        """dets: list of (xyxy box, score). Returns the live tracks after this frame."""
        free = list(self.tracks)
        matched = {}
        for box, s in sorted(dets, key=lambda d: -d[1]):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, best_d = None, None
            for t in free:
                px, py = t.predicted()
                gate = max(self.gate_px, self.gate_scale * max(t.w, t.h))
                d = math.hypot(cx - px, cy - py)
                if d <= gate and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                t = Track(self.next_id, box, s)
                self.next_id += 1
                self.tracks.append(t)
                matched[t.id] = True
                continue
            free.remove(best)
            px, py = best.predicted()
            rx, ry = cx - px, cy - py
            best.vx, best.vy = best.vx + self.beta * rx, best.vy + self.beta * ry
            best.cx, best.cy = px + self.alpha * rx, py + self.alpha * ry
            best.w = self.alpha * (box[2] - box[0]) + (1 - self.alpha) * best.w
            best.h = self.alpha * (box[3] - box[1]) + (1 - self.alpha) * best.h
            best.score = s
            best.hits.append(1)
            best.age += 1
            best.misses = 0
            matched[best.id] = True
        for t in free:                             # coast unmatched tracks
            t.cx, t.cy = t.predicted()
            t.hits.append(0)
            t.age += 1
            t.misses += 1
        alive = []
        for t in self.tracks:
            if not t.confirmed and sum(t.hits[-self.n:]) >= self.m:
                t.confirmed = True
                self.confirmed_ids.append(t.id)
            if t.confirmed and t.misses > self.max_coast:
                continue
            if not t.confirmed and t.age >= self.n:
                continue
            alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks]
