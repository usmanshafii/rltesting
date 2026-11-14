# rl/: RL course project (CSE 482)

Learning where a camera post should look for a drone with RGB + uncooled LWIR, when the cooled LWIR and MWIR cameras
help only in training. The plan is `rl/PLAN.md`; the interface is `rl/contracts.py`; assistants follow `AGENTS.md`.

## Setup (S-01)

Keep the virtual environment outside OneDrive-synced folders: syncing the ~1 GB of PyTorch files slows the laptop.

```
py -3.11 -m venv %USERPROFILE%\venvs\kal-rl
%USERPROFILE%\venvs\kal-rl\Scripts\python -m pip install -r rl/requirements-rl.txt
%USERPROFILE%\venvs\kal-rl\Scripts\python -m pytest tests/rl -q
```

## Commands

```
python -m pytest tests/rl -q                  unit tests (skips = tasks not yet done)
python -m rl.scripts.check wp1                M1's environment checks (oracle, speed, stub)
python -m rl.scripts.check invariants         plan invariants I1-I11
python -m rl.scripts.check --quick            short end-to-end run
python -m rl.scripts.check all                everything; JSON summaries in rl/results/checks/
```

## Using the environment

```python
from rl.env import SkyEnv, OraclePolicy

env = SkyEnv()                                  # student arm: actor sees RGB + uncooled LWIR evidence only
obs, info = env.reset(seed=1)                   # info["privileged"], info["truth"]: training only
obs, reward, terminated, truncated, info = env.step(0)   # 0 dwell, 1-8 slew, 9 zoom in, 10 zoom out
```

`SkyEnv(actor_bands="four")` gives a four-band actor (the headroom arm); `"four_shared"` is the gated teacher's view
(MWIR without the plume). `SkyEnv(compute_privileged=False)` skips the privileged cameras for arms that never read them.
Until M2-08 the sensor numbers are placeholders from `rl/env/stub_sensor.py` (grade C).

## Folders

| Folder | Owner | Contents |
| --- | --- | --- |
| `env/` | M1 | environment, geometry, flight, evidence and confirmation, rewards, episodes, stub sensor model, oracle |
| `calib/` | M2 | census join, d' fit, score model, clutter, calibrated sensor model |
| `agents/` | M3 | PPO, actor and critic inputs, DAgger, shaping |
| `baselines/` | M4 | random, raster scan, Bayesian search |
| `evaluation/` | M4 | frozen episodes, evaluation, bootstrap, viewer |
| `scripts/` | M4, M2 | checks, run matrix, merging, analysis |
| `results/` | all | committed result summaries |
