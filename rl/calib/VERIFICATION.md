# M2 verification record

Date: 2026-11-19. Branch: ibad-m2. Repository: D:/RL project/rltesting.

## Runtime

Python 3.11.9. Exact package versions are recorded in runtime_verified.json:
NumPy 1.26.4, SciPy 1.17.1, PyYAML 6.0.3, matplotlib 3.11.2, Pillow 13.0.0,
pytest 9.2.0, Gymnasium 1.4.0, and PyTorch 2.15.0+cpu. The CPU build satisfies
the pinned torch==2.15.0 requirement. No application dependency was added.

A portable verification interpreter was prepared at D:/workspace/rl-python311/python.exe
because the shell's Python command resolved to the Microsoft Store alias. The earlier
background helper captured Python 3.12; this was caught, and the complete repository
run was repeated explicitly with the 3.11 executable printed at the start of its log.

## Tests and checks

| Check | Observed result |
| --- | --- |
| Full Python 3.11 repository run, single PyTorch CPU thread | 412 passed, 14 skipped, 2 failed initially, 1 existing tensor-conversion warning; 285.11 seconds |
| RL subset within that full run | All 83 tests passed; 9 existing skips; no M2 skip |
| Targeted rerun of both initial failures after input/temp correction | 2 passed in 0.43 seconds |
| Final fast RL check excluding the three expensive full-join/fit tests | 80 passed, 9 skipped, 3 deselected |
| Calibrated SkyEnv Gymnasium check_env | PASS |
| python -m rl.scripts.check invariants | 9 passed, 2 existing skips; exit 0 |
| python -m rl.scripts.check --quick | Environment/oracle pass; PPO remains PENDING M3-03; exit 0 |
| python -m rl.scripts.check wp2 | Exit 1: uncooled-LWIR held-out error fails, missing empirical evidence and pending integration are explicit |
| Calibration source/implementation hashes | PASS |
| git diff --check | PASS |
| Protected shared files, environment, agent, baseline, evaluation and reference census diffs | Empty |

The full test command was invoked through Python with torch.set_num_threads(1)
then pytest.main(["tests", "-q", "--tb=short"]). All RL tests ran before the legacy
physics tests. The 14 skips cover previously deferred work and unavailable optional
test assets; they are not counted as completed work packages.

The two initial failures were environmental:

1. tests/test_atmosphere_bands.py::test_g173_mwir_window_direct_beam needed the
   git-ignored ASTM_G173-03_from_NREL_workbook.csv. The registered workbook was
   recovered with its exact registered SHA-256, and its table exported and compared
   cell-for-cell. The export serialization differs from the historical CSV bytes;
   the numerical data are identical. See REFERENCE_RECOVERY.md and its JSON receipt.
2. tests/test_sky.py::test_overlay_merges_and_records_provenance used C: for pytest's
   temporary file while the repository is on D:. The existing sampler's relpath
   cannot cross Windows drives. It passes with a verified new --basetemp directory
   on D:. No source or test expectation was changed.

Both tests passed together after those corrections. The full suite was not repeated
unnecessarily after the targeted rerun. There are no unresolved regression-test
failures from that run; this does not turn the separate WP2 scientific gate green.

## Reproduce on this host

~~~powershell
$rlPython = 'D:\workspace\rl-python311\python.exe'
& $rlPython -m pytest tests/rl -q
& $rlPython -m rl.scripts.check wp2
& $rlPython -m rl.scripts.check invariants
~~~

For the whole suite on Windows, choose a fresh --basetemp directory on the repository's
D: drive. Use only an explicitly scoped test directory: pytest may clear that directory.
The recovered third-party reference files stay locally git-ignored per repository policy.
A fresh clone needs those source assets restored for the corresponding legacy test.

## Changed-file scope

- rl/calib/: census join, fits, score/clutter models, sensor implementation, build,
  diagnostics, pixel recovery, provenance data, reports and reproducibility notes.
- tests/rl/test_calib.py and the new test_calib_clutter.py, test_calib_pixels.py,
  test_calib_provenance.py: numerical, leakage, reproducibility and failure tests.
- rl/scripts/check.py: M2-08 WP2 dispatch only, explicitly assigned in the team guide.
- rl/results/checks/: WP2, invariant and quick-check summaries.
- docs/ai_use/m2/: session log and summary.
- Locally restored, git-ignored source workbook/CSV: exact verified document and
  value-for-value table export used only to restore the existing baseline test.

No edits to rl/contracts.py, rl/config/frozen.yaml, rl/PLAN.md, rl/env/,
rl/agents/, rl/baselines/, rl/evaluation/, data/dataset.py or the reference census.
No merge, push, trained policy, approved freeze or project submission is claimed.

## Assumptions

"Milestone 2" means the M2/WP2 calibration member in the supplied plan. The missing
original renders trigger the PDF's analytic-plume/omit-low-heat fallback. The frozen
historical detector curve is retained as required and labelled as a surrogate, with
its later false-alarm correction documented. Existing contracts remain authoritative;
required cross-owner changes are recorded instead of silently implemented.
