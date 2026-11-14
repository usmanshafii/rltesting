# Instructions for AI assistants: RL course project (CSE 482)

The RL course project lives in `rl/` and `tests/rl/`. Before any work, read `rl/PLAN.md`
(decisions, frozen values, invariants, task IDs, change log) and the repo's `CLAUDE.md`
(evidence rule, sensor set rule, scope rule). Both take precedence over your own suggestions.

## Rules
1. Work only on the task ID the user gives you, inside that task owner's folder.
2. Import interfaces, constants and formulas from `rl/contracts.py`, `rl/config/frozen.yaml`
   and the repo (for example `eval/detectability.d_after`); never re-implement or redefine them.
3. Do not edit `rl/contracts.py`, `rl/config/frozen.yaml`, `rl/PLAN.md` or another member's
   folder. If a change seems needed, write it as a change request for the team instead.
4. Write or update the task's tests first. The task is done only when
   `python -m pytest tests/rl -q` and the task's `python -m rl.scripts.check` pass.
5. Add no dependency outside `rl/requirements-rl.txt`. Python 3.11, numpy<2, Gymnasium API
   (reset returns obs, info; step returns obs, reward, terminated, truncated, info).
6. Never break these invariants: the actor receives only RGB and uncooled-LWIR channels;
   confirmation uses only the student's evidence, with one threshold A for every arm;
   rewards use ground truth only; the shaping potential is 0 at every episode end; every run
   records environment version, episode-set hash, threshold A, configuration hash and seed.
7. Every number traces to a repo file, or is marked an estimate with grade C (`CLAUDE.md`).
8. End every answer with: files changed, tests run and their result, assumptions made.

## Folders and owners
`rl/env/` M1 · `rl/calib/` M2 · `rl/agents/` M3 · `rl/baselines/`, `rl/evaluation/`, `rl/scripts/`, `rl/PLAN.md` M4 ·
`rl/contracts.py`, `rl/config/frozen.yaml` all four (change request only).
