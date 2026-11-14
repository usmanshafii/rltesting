"""Oracle policy (task M1-07): points at the drone it knows from the simulator's truth, at the zoom step with the most
expected evidence, and dwells. It is a reference for the WP1 check (it should confirm over 95 % of confirmable drones)
and an upper bound on pointing, never a deployable policy: it reads `env.unwrapped.oracle_plan()`, which no learned
policy or baseline may call.
"""
from __future__ import annotations

from rl import contracts as C
from rl.env.sky_env import slew_toward


class OraclePolicy:
    def __call__(self, env) -> int:
        e = env.unwrapped
        plan = e.oracle_plan()
        if plan is None:                                # no drone in the sector: wait at the widest field
            return C.ZOOM_OUT if e.zoom > 0 else C.DWELL
        cell, zoom = plan
        if e.pointing != cell:
            return slew_toward(e.pointing, cell)
        if e.zoom < zoom:
            return C.ZOOM_IN
        if e.zoom > zoom:
            return C.ZOOM_OUT
        return C.DWELL
