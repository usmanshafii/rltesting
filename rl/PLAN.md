# RL course project (CSE 482): the plan, for people and AI assistants

Owner: M4 (plan keeper). Changes go through the change log at the end; an assistant never edits this file.
Sources of truth, highest first: the course brief (`RL-F26-BS-Project.pdf`), the "Course project plan" tab of the team's
doc, this file, `rl/contracts.py` with `rl/config/frozen.yaml`, then the tests in `tests/rl/` and
`python -m rl.scripts.check`.

## 1. Decisions in one paragraph

Theme D: a simulator of a ground camera post with one pan-tilt-zoom head over a 60 x 15 deg sky sector (12 x 3 cells of
5 deg), and an RL agent (PPO) that decides where to look, how far to zoom and how long to dwell, to find and confirm an
incoming drone as early as possible without false confirmations on clouds. The deployed agent sees only RGB + uncooled
LWIR evidence. The cooled LWIR and MWIR cameras exist only in training, where they reach the critic or a teacher after
two filters (photons the cheap cameras share; drones the student can confirm). Confirmation is a fixed sequential test
on the student's own evidence with one threshold A for every arm, so no arm can be taught to confirm on evidence it
lacks. Core arms: symmetric PPO, gated privileged-critic PPO, gated teacher with a DAgger student, four-band PPO; with
random, raster-scan and Bayesian-search baselines; 5 seeds; the primary comparison and its rule are fixed before the
first core run. The simulator is NumPy only, calibrated from the repo's census (`docs/reference/privileged/data/`).

## 2. Frozen values

All live in `rl/config/frozen.yaml`, each with its source or estimate grade. Status: **draft (interface-v0)**, to be
approved at S-03 and tagged `interface-v1`; placeholders (grade P) are fixed on days 8-9 and frozen at `env-v1`.

frozen.yaml SHA-256 at interface-v1: (recorded at S-03; invariant I4 compares against it)

| Value | Draft | Fixed when |
| --- | --- | --- |
| Sector, cells | az -30..30, el 5..20 deg, 5 deg cells (12 x 3) | interface-v1 |
| Zoom steps (uncooled LWIR field) | 46, 15, 5 deg; other cameras use min(field, their widest) | interface-v1 |
| Widest fields | RGB 63.7, LWIR 46, cooled LWIR 24.9, MWIR 36.4 deg | interface-v1 |
| Step | 0.2 s = 5 frames at 25 Hz; one-cell slew per step | interface-v1 |
| Episode | 900 steps (3 min); 30 % drone-free | interface-v1 |
| Speeds | 39-50 m/s training, 50-56 m/s held out | interface-v1 |
| Reward | correct +1, earliness +1 x share left, false -1, missed confirmable -0.5, step -0.001, slew -0.0005 | interface-v1 |
| Gate flag K | 10 steps of dwell | interface-v1 |
| Threshold A | 10 (placeholder) | env-v1, at ~1 false confirmation per hour for raster scan (M1-09) |
| Entry range | 4-7 km (placeholder) | env-v1, raster scan confirms 40-70 % (M1-09) |
| pd_nominal | 0.2 (placeholder) | env-v1 |
| PPO settings | none yet | env-v1 (M3-07) |
| Seeds | none yet | pre-registration, before Tier 1 (WP5) |

## 3. Interface summary (`rl/contracts.py`)

- Bands: `rgb`, `lwir`, `lwir_cooled`, `mwir`, plus `mwir_shared` (MWIR without the plume). Band sets: `student`
  (RGB + uncooled LWIR), `four`, `four_shared`.
- Actions (11): 0 dwell; 1-8 slew N, NE, E, SE, S, SW, W, NW; 9 zoom in; 10 zoom out. Action 11 (declare) exists only
  with learned declaration (a Tier-2 test).
- Observation of the actor: `evidence`, `since_look`, `coverage` (3 x 12 each), `zoom` (3), `time_left`, `is_day`.
- Privileged data only in `info["privileged"]`: `evidence_lwir_cooled`, `evidence_mwir`, `evidence_mwir_shared`,
  `coverage_lwir_cooled`, `coverage_mwir`. Ground truth only in `info["truth"]`, including the gate flag `confirmable`.
- `EpisodeSpec`: everything that defines an episode; evaluation sets are lists of them.
- `SensorModel` protocol: `begin_episode`, `dprime`, `pd_per_frame`, `pfa_per_frame`, `nominal_pd`. M1's stub
  (`rl/env/stub_sensor.py`) implements it now; M2's calibrated model replaces it (M2-08, M1-08) with no other change.
- Environment: `rl.env.SkyEnv(actor_bands="student"|"four"|"four_shared", condition=..., sensor_model=...,
  learned_declaration=False, compute_privileged=True)`.

## 4. Invariants (each is a test; `python -m rl.scripts.check invariants`)

| ID | Invariant | Checked by |
| --- | --- | --- |
| I1 | The actor never receives a privileged channel | observation-space keys; student observations identical whatever the privileged cameras see (`tests/rl/test_invariants.py`); M3's rollout assertion |
| I2 | Threshold A is the same for every arm | every result row carries A |
| I3 | Every arm uses the same evaluation episodes | episode-set hash in every result row |
| I4 | Frozen values do not change after their freeze | `frozen_hash()` against the hash recorded in section 2 |
| I5 | One environment version per tier | `ENV_VERSION` in every result row |
| I6 | Rewards use ground truth only | `rl/env/rewards.reward_terms` signature test |
| I7 | Shaping potential is 0 at every episode end | shaping sum = -Phi(s0) (M3-06) |
| I8 | Every calibration value has a source and a grade | `rl/calib/params.json` schema |
| I9 | No dependency outside the pinned list | import scan of `rl/` |
| I10 | Seeds are the pre-registered ones | seed lists against the pre-registration |
| I11 | All four sensors modelled; uncooled and cooled LWIR separate | band and sensor-model tests |

## 5. Tasks (ID, owner, days, done-when, status on 2026-11-14)

Shared: S-01 day 1 environment and repo tests; S-02 day 1 assistant check; S-03 day 2 interface approval
(`interface-v1`); S-04 days 7-9 integration, `check all` passes, `env-v1`; S-05 day 9 gate meeting; S-06 days 10-14
assigned runs; S-07 day 14 runs frozen (`results-v1`); S-08 days 18-20 report, README, AI-use logs.

| ID | Task | Done when | Status |
| --- | --- | --- | --- |
| M1-01 | Proposal | team approves; sent by day 2 | notes in `docs/rl_course/proposal_notes.md`; the team writes it |
| M1-02 | `contracts.py`, `frozen.yaml` | `tests/rl/test_contracts.py` passes; S-03 approves | draft done, tests pass; awaits S-03 |
| M1-03 | Stub environment | `check_env`; 1,000 random episodes | done (`rl/env/stub_env.py`) |
| M1-04 | Geometry and flight | cell, coverage and flight-port tests | done (`geometry.py`, `flight.py`) |
| M1-05 | Evidence and confirmation | hand-computed LLR; false confirmations below e^-A | done (`evidence.py`) |
| M1-06 | Rewards, episodes, gate flag, seeding, vector env | determinism; 30 +- 2 % drone-free; rewards sum; I6 | done, against the stub sensor model |
| M1-07 | Oracle and speed | `check wp1`: oracle >= 95 %; random < raster < oracle; >= 2,000 steps/s/core | oracle and random done; raster needs M4-03; speed to measure on an idle laptop |
| M1-08 | Swap in M2's sensor model | WP1 still passes | blocked by M2-08 |
| M1-09 | Difficulty, threshold A, hard slice, pre-registration part | values in `frozen.yaml`; `env-v1` | blocked by M2-08, M4-03 |
| M1-10 | Assigned runs | as S-06 | blocked by M3, M4 |
| M1-11 | Demo with M3 | clips from logged evaluation episodes | blocked by trained policies and M4-05 |
| M1-12 | Report sections, environment README | S-08 | environment README can start now |
| M2-01 | Census scenes found or re-rendered | 5 scenes within 2 % of `census.json` | open |
| M2-02 | Census join with regenerated specs | range and size within 0.1 % for 126 scenes | open |
| M2-03 | Low-heat render | render log complete | open |
| M2-04 | Fit log d' per band, `params.json` | `test_calib.py`; ratios 1.23x / 1.50x within 10 % | open |
| M2-05 | Plume-free and low-heat census | LWIR d' unchanged, MWIR d' lower without plume | open |
| M2-06 | Score model | curve inside bootstrap interval; 0.01 +- 0.002 FA/frame | open |
| M2-07 | Noise and clutter | LWIR clutter correlated, noise not; dwell = `d_after` | open |
| M2-08 | Sensor model v1 behind the contract | `check wp2`; passes `test_sensor_model_contract` | open |
| M2-09 to M2-12 | Validation, runs, analysis, report | see the guide | open |
| M3-01 | PPO on CartPole | 475 within 500k steps, 3 seeds | open |
| M3-02 | Actor/critic inputs, I1 audit | audit fires on injection | open |
| M3-03 | PPO on the stub | 50k-step run, log schema | open (stub ready) |
| M3-04 | Critic variants | one test per switch | open |
| M3-05 | Toy gate | within 10 points of the oracle | open (environment and oracle ready) |
| M3-06 | DAgger, shaping, learned declaration | mask, shaping sum, flag tests | open |
| M3-07 to M3-11 | PPO settings, gate evidence, runs, demo, report | see the guide | open |
| M4-01 | Scaffolding | assistants pass S-02 | done early by M1 (2026-11-14); M4 to review |
| M4-02 | Episode sets, log schema | same hash on two laptops | open |
| M4-03 | Baselines | raster cycle; Bayes highest posterior | open |
| M4-04 | Evaluation, bootstrap | reproducible; recovers planted difference | open |
| M4-05 | Viewer, run matrix, merge | replays; merges cleanly | open |
| M4-06 to M4-09 | Integration, report, runs, submission | see the guide | open |

## 6. Gates

| Day | Gate | Check |
| --- | --- | --- |
| 2 | Interface frozen | S-03 approves; tag `interface-v1`; record the hash in section 2 |
| 5 | Toy solved | M3-05 within 10 points of the oracle |
| 9 | Go or fallback | PPO's curve rising on the full task; four-band headroom above 0; Tier 1 fits in 4 laptop-days. Otherwise the confirm-only fallback or the hard slice |
| 14 | Runs frozen | `results-v1` |
| 20 | Submit | report, code, demo, AI-use disclosure |

## 7. Branches and workflow

- `rotary-eng` is the FYP base. `rl-course` is the integration branch, made from it when the first pull request is
  merged. Each member works on a personal branch named `<name>-m<k>` (M1: `aqib-m1`).
- One pull request per task ID (`.github/pull_request_template.md`), reviewed by the member in the guide's roles table,
  merged only when `python -m pytest tests/rl -q`, the repo's tests and `python -m rl.scripts.check --quick` pass.
- Python 3.11 virtual environment from `rl/requirements-rl.txt`, kept **outside** OneDrive-synced folders.

## 8. Change log

| Date | Who | Change | Approved |
| --- | --- | --- | --- |
| 2026-11-14 | M1 | Scaffolding of M4-01 done early (team works one member at a time): `rl/` folders, this file, `AGENTS.md`, `GEMINI.md`, `.github/` files, `CLAUDE.md` addition, `tests/rl/` | M4 to review |
| 2026-11-14 | M1 | `.gitignore`: its `env/` rule also hid `rl/env/`; added `!rl/env/` and `rl/runs/` | M4 to review |
| 2026-11-14 | M1 | Proposed for S-03: confirmation is the repeated SPRT (Page's CUSUM), whose false-confirmation rate per monitored cell-step is at most e^-A (Lorden 1971) | S-03 |
| 2026-11-14 | M1 | Proposed for S-03: evidence is built from per-frame detections at the detector's 0.01-per-frame threshold (the operating point the repo measured), with a nominal detection probability in the LLR; RGB carries no evidence at night | S-03 |
| 2026-11-14 | M1 | Proposed for S-03: an arm confirms on the bands its actor sees (student arms on RGB + uncooled LWIR, four-band arms on all four), always with the same A | S-03 |
| 2026-11-14 | M1 | Proposed for S-03: privileged data reach critics and teachers only through `info`, never the observation; each camera has its own random stream, so student observations cannot depend on the privileged cameras (I1 test) | S-03 |
| 2026-11-14 | M1 | Proposed for S-03: sector elevation 5-20 deg (inside M1's open-sky looks, `configs/m1.yaml`); MWIR widest 36.4 deg (`configs/sensors.yaml`) | S-03 |
