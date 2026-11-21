"""Evidence map and confirmation test (task M1-05).

Each step a covered cell receives k detections out of n frames from each band. Its log-likelihood ratio, drone in
the cell (detection probability pd_nom per frame) against no drone (false-detection probability pfa per frame), is

    LLR = k ln(pd_nom / pfa) + (n - k) ln((1 - pd_nom) / (1 - pfa))

summed over the bands the arm uses (detections of different cameras are independent given the drone's position).

Confirmation is the sequential probability ratio test in its repeated form, Page's CUSUM (Page 1954, Biometrika 41):
per cell, L <- max(0, L + LLR) on every step the cell is covered, and the cell is confirmed when L >= A. Under the
null, exp(LLR) has mean 1, so the mean time to a false confirmation of a monitored cell is at least e^A steps
(Lorden 1971, Ann. Math. Statist. 42); the false-confirmation rate per monitored cell-step is at most e^(-A).
A false confirmation resets that cell to 0.

Motion model: a drone crossing into a neighbouring cell should not lose all its evidence, so each step every cell
takes max(L, rho * max of its 8 neighbours). This never raises the map's maximum, so it cannot create a confirmation
on its own.
"""
from __future__ import annotations

import numpy as np


def llr_counts(k, n: int, pd_nom: float, pfa):
    """Log-likelihood ratio of k detections in n frames; k and pfa may be arrays of the same shape."""
    k = np.asarray(k, dtype=np.float64)
    pfa = np.clip(np.asarray(pfa, dtype=np.float64), 1e-12, 1 - 1e-12)
    return k * np.log(pd_nom / pfa) + (n - k) * np.log((1.0 - pd_nom) / (1.0 - pfa))


def expected_llr(pd: float, n: int, pd_nom: float, pfa: float) -> float:
    """Mean LLR per step when the true per-frame detection probability is pd."""
    return float(llr_counts(n * pd, n, pd_nom, pfa))


_NBR: dict[tuple[int, int], np.ndarray] = {}


def _neighbour_index(shape: tuple[int, int]) -> np.ndarray:
    """[cells, 8] flat indices of each cell's neighbours; cells outside the grid point at a padding slot."""
    if shape not in _NBR:
        h, w = shape
        pad = h * w
        idx = [[(r + dr) * w + (c + dc) if 0 <= r + dr < h and 0 <= c + dc < w else pad
                for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
               for r in range(h) for c in range(w)]
        _NBR[shape] = np.array(idx)
    return _NBR[shape]


def neighbour_max(L: np.ndarray) -> np.ndarray:
    """Max over each cell's 8 neighbours (cells outside the grid count as 0)."""
    flat = np.append(L.ravel(), 0.0)
    return flat[_neighbour_index(L.shape)].max(axis=1).reshape(L.shape)


class Cusum:
    """One evidence map: per-cell CUSUM of log-likelihood ratios."""

    def __init__(self, shape: tuple[int, int]):
        self.L = np.zeros(shape, dtype=np.float64)
        self._pad = np.zeros((shape[0] + 2, shape[1] + 2))   # zero border: cells outside the grid count as 0

    def update(self, increment: np.ndarray, covered: np.ndarray) -> None:
        self.L = np.where(covered, np.maximum(0.0, self.L + increment), self.L)

    def spread(self, rho: float) -> None:
        if 0 < rho <= 1:
            # Same result as max(L, rho * neighbour_max(L)), faster: with L >= 0 and rho <= 1, rho * L <= L, so the
            # 3 x 3 maximum may include the cell itself; that maximum is taken along rows, then along columns.
            p = self._pad
            p[1:-1, 1:-1] = self.L
            r = np.maximum(np.maximum(p[:, :-2], p[:, 1:-1]), p[:, 2:])
            self.L = np.maximum(self.L, rho * np.maximum(np.maximum(r[:-2], r[1:-1]), r[2:]))
        elif rho > 1:
            self.L = np.maximum(self.L, rho * neighbour_max(self.L))

    def crossing(self, A: float, covered: np.ndarray) -> tuple[int, int] | None:
        """The covered cell with the most evidence, if it has reached A."""
        masked = np.where(covered, self.L, -np.inf)
        idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
        return (int(idx[0]), int(idx[1])) if masked[idx] >= A else None

    def reset_cell(self, cell: tuple[int, int]) -> None:
        self.L[cell] = 0.0
