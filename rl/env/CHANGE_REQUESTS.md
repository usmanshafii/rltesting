# M1 change requests and answers to other members' requests (not approvals)

`AGENTS.md` rule 3: shared files (`rl/contracts.py`, `rl/config/frozen.yaml`, `rl/PLAN.md`) change only after the
team approves. M1's environment changes inside `rl/env/` are listed here too when they change the simulation.

## Done inside rl/env with M1-08 (2026-11-21)

- **Sensor model swapped:** `SkyEnv()` now uses M2's `CalibratedSensorModel`; the stub stays selectable.
  `info["sensor_version"]` is a new key (beyond `INFO_KEYS`, so no contract change) recording which model ran.
- **Exact false-detection partition (CR-M2-01, last item):** a cell holding a share g of a frame now gets
  1 - (1 - pfa)^g instead of pfa * g, so the cells' no-detection probabilities multiply back to the frame's 1 - pfa.
  At the clear-sky 0.01 the two differ by under 0.5 %; under cloud (pfa up to 0.3) by up to about 19 %.
- **Speed, same results:** per-step cache of the drone's detection probability (asked once per band and field per
  step), per-episode cache of each view's null probabilities, LLR coefficients shared across episodes, and a 3 x 3
  maximum by rows then columns for the evidence spread (tested equal to the neighbour formula).

## CR-M1-02 to M2: a faster scalar path in CalibratedSensorModel (speed gate)

With M2's model the WP1 speed check (all cameras, 2,000 steps/s per core) measured 1,370-1,570 steps/s on M1's
laptop, against 2,000-2,150 for the stub under the same load; student-only arms (`compute_privileged=False`) measured
2,170. About a quarter of a step is spent in `pd_per_frame`. Two changes inside `rl/calib` (M2's files; their hashes
are in `params.json`, so M1 did not edit them) were tested by patching them in at run time:

1. In `begin_episode`, cache per band a and b of log d' = a + b log(max(size, 1)): `features()` is linear in
   log(size) once the episode's row is fixed (a = features at size 1 times the coefficients; b = features at size e
   minus a). `dprime` then needs one `math.exp`. Largest relative difference from the current `dprime` over 300
   episodes x 42 sizes x 5 bands: 5e-15.
2. In `pd_per_frame`, the logistic with `math.exp` instead of `scipy.special.expit` on a scalar: largest difference
   1e-16.

Together: 1,830-1,890 steps/s under the same load. M1 then reruns `python -m rl.scripts.check wp1` on an idle laptop.

## CR-M1-01: bump ENV_VERSION (needs approval: rl/contracts.py)

`ENV_VERSION` is still `sky-env-0.1-stub-sensor`, but the environment changed (calibrated sensor model, exact
partition). Proposal: `sky-env-0.2-calibrated` now, and the env-v1 tag at S-04 after M1-09. Until then every result
also carries `info["sensor_version"]`, so stub and calibrated runs can be told apart.

## Answer to CR-M2-01: shared clutter and dwell gain

Agreed in principle, proposed for after env-v1 unless the team decides it is needed for the core comparison:

- The environment's evidence counts detections that are independent from frame to frame, and the gate flag
  multiplies one step's expected evidence by K. That is consistent with what the environment simulates now; using
  `eval/detectability.d_after` for the gate alone would make the gate disagree with the simulation.
- Simulating M2's persistent LWIR clutter needs a sensing interface that returns detection counts (or scores) from an
  episode-level noise state, instead of probabilities. That changes the `SensorModel` protocol, so it needs all four
  members. Requirements if adopted: separate random streams per camera stay (invariant I1), the gate flag and the
  simulation use the same model, and the false-confirmation and oracle checks are rerun.
- Nominal Pd and A are not moved to make any check pass (M1-09 sets A from drone-free raster runs only).

## Answer to CR-M2-02: visibility, aspect and power

Agreed that `EpisodeSpec` lacks them. M1 proposes no new frozen interval now. If the team wants a held-out visibility
condition, M1 adds the field to `EpisodeSpec` and `sample_episode_spec` after approval, with M2 supplying the sourced
range.

## Answer to CR-M2-03 and CR-M2-04

For M2 and the team: M1's threshold A (M1-09) is set from drone-free episodes in the environment itself, so it does
not depend on the historical 0.01 operating point being exact; the uncooled-LWIR validation shortfall stays reported
as M2 wrote it.
