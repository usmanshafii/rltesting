# rl/env: the camera-post environment (owner: M1)

A Gymnasium environment, NumPy only: one pan-tilt-zoom head watches a 60 x 15 deg sky sector for an incoming small
fixed-wing drone. The agent decides where to look, how far to zoom and how long to dwell; a fixed sequential test on
the deployed cameras' evidence decides when a drone is confirmed. Values and their sources: `rl/config/frozen.yaml`.

## Formulation (the brief's checklist)

| Element | In this environment | Code |
| --- | --- | --- |
| State, as the actor sees it | per cell (3 x 12): CUSUM evidence from RGB + uncooled LWIR, time since last look, current coverage; zoom step; time left; day or night | `sky_env.SkyEnv._obs` |
| Privileged, training only | the same evidence from cooled LWIR, MWIR and MWIR without the plume; their coverage; the truth, including the gate flag `confirmable` | `info["privileged"]`, `info["truth"]` |
| Actions (11) | dwell; slew one cell in 8 directions; zoom in; zoom out (a 12th, declare, only in the learned-declaration test) | `rl/contracts.py` |
| Reward | correct confirmation +1 + share of the approach left; false confirmation -1 (episode continues); missed confirmable drone -0.5; -0.001 per step; -0.0005 per slew | `rewards.reward_terms` |
| Dynamics | the drone flies the repo's S-turn approach (`render/track_sequence.flight_path`, ported) and climbs through the sector as it closes; each step every covered cell reports detections in 5 frames per camera; evidence = CUSUM of log-likelihood ratios, spread to neighbours | `flight.py`, `sky_env._sense`, `evidence.py` |
| Termination | correct confirmation; the drone leaves the sector; 900 steps (3 min). 30 % of episodes have no drone | `sky_env.SkyEnv.step` |
| Metrics | confirmation rate, range at confirmation, false confirmations per hour, return; by observability class | `info`, M4's evaluation |

## Design choices worth defending

- **Confirmation is not learned in the core.** It is Page's CUSUM (the repeated sequential probability ratio test):
  per cell L <- max(0, L + LLR), confirm at L >= A, the same A for every arm. Its false-confirmation rate per
  monitored cell-step is at most e^-A (Lorden 1971; `tests/rl/test_env.py`). Every arm is compared at the same
  false-alarm rate, and no arm can be taught to confirm on evidence its cameras lack.
- **Privileged data never enter a student's observation.** They are in `info` only, and each camera draws its
  detections from its own random stream, so a student's observations are identical whatever the privileged cameras see
  (`tests/rl/test_invariants.py`, invariant I1).
- **Search against resolution.** At 46 deg the frame covers 9 x 3 cells but a 2 m drone at 5 km is under half a pixel;
  at 5 deg it covers one cell with about 3 px on the drone. Each camera matches the uncooled field where its lens allows
  (cooled LWIR at most 24.9 deg, MWIR 36.4 deg).
- **Gate flag (Filter 2).** A drone is "confirmable" at a step if 10 steps of dwell at the best zoom would bring the
  expected student evidence to A, using the deployed detector's detection probability. Only such drones count as
  missed, and only on such steps may a teacher be imitated (M3).

## Calibration status

Until M2-08 the detection numbers come from `stub_sensor.StubSensorModel`: placeholders (grade C) from a crude fit to
`docs/reference/privileged/data/census.json` and the detector's logistic curve. M2's calibrated model replaces it
through the same `SensorModel` protocol (M1-08); nothing else changes. Threshold A, the entry range and the nominal
detection probability are placeholders until M1-09.

## Checks

`python -m pytest tests/rl/test_env.py tests/rl/test_invariants.py -q` and `python -m rl.scripts.check wp1`.
