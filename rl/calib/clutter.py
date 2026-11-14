"""M2-07: noise and clutter. Split each band's spread with the census's d and d_noise; sensor noise independent per
camera, clutter shared by the two LWIR cameras; clouds as persistent static sources; dwell through
eval/detectability.d_after, imported, never copied.

Done when: in simulation the two LWIR cameras' clutter is correlated and their sensor noise is not, and the dwell
gain equals d_after's output exactly.
"""
from __future__ import annotations


def cloud_false_detection_factor(band: str, is_day: bool) -> float:
    raise NotImplementedError("M2-07: clutter model")
