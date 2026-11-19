# M2 calibration handoff

Date: 2026-11-19 (Asia/Karachi). Branch: ibad-m2.

**Status: implementation delivered with documented fallback; full WP2 acceptance remains blocked.**
The implementation and its unit tests are distinct from the scientific acceptance
criteria. The WP2 command deliberately exits 1, rather than declaring an incomplete
gate passed.

## Scope and sources

"Milestone 2" was interpreted as member M2 / work package WP2 in the supplied team
guide and course plan, consistent with rl/PLAN.md and the calibration stubs. All five
PDFs were read, including the timeline and module diagrams. The course brief asks
for an implemented RL system, reproducible experiments and AI disclosure; the
literature review motivates the training/deployment information boundary. This
handoff implements the calibration member's work, not M3's policy training or M4's
complete experiment infrastructure.

Primary local requirements: Team implementation guide.pdf pp. 9-11; Course project
plan (2).pdf pp. 8-9, 12, 19-21. Source precedence follows rl/PLAN.md. The protected
contracts, frozen configuration, plan and other members' implementations remain
unchanged. The only shared script edit is the M2-08 WP2 dispatch explicitly assigned
to M2 by the guide.

## Completed implementation

- Regenerated every census scene with the original M1 overlay and 600-scene population;
  proved range and unblurred-box size identity for all 126 non-test rows at the required
  0.1% tolerance. Lighting, visibility, cloud and split identity are checked as well.
- Fitted four separate log-dprime models, with common scene-level five-fold CV,
  train-only paired ratio calibration and a persistent joint residual bank. All
  coefficients and random effects use the 107 training rows. The 19 validation rows
  never influence coefficients or residual estimates.
- Implemented historical-curve Gaussian scores, a synthetic 0.01 frame-null threshold,
  exact whole-frame false-alarm partitioning, shared persistent LWIR clutter with
  independent camera noise, and exact delegation to eval.detectability.d_after.
- Delivered the existing SensorModel contract, deterministic episode draws, source
  and implementation hash verification, negative-input checks and an explicit error
  for unsupported low-heat episodes.
- Added an isolated, tested plume-layer filter and pixel-census recovery command,
  preserving data/dataset.py and the reference census.
- Added reproducible build, diagnostics, WP2 acceptance and provenance outputs, and
  this calibration report. Runtime verification uses Python 3.11.9 and the pinned
  numerical packages.

## Numerical results

| Sensor | Training positives | CV log RMSE | Held-out positives | Held-out log RMSE |
| --- | ---: | ---: | ---: | ---: |
| rgb | 100 | 1.1398 | 16 | 0.9668 |
| lwir | 107 | 1.3613 | 18 | 1.9210 |
| lwir_cooled | 90 | 1.7380 | 15 | 1.6011 |
| mwir | 101 | 1.9714 | 18 | 1.8187 |

Errors are in natural-log dprime units; zero/unseen signals are excluded, with
positive-scene counts shown explicitly. They are synthetic-census fit errors, not
physical sensor measurements or RL performance estimates.

| Ratio | Census | Fitted | Plan target |
| --- | ---: | ---: | ---: |
| lwir_cooled / lwir | 1.2276 | 1.1490 | 1.23 +/- 10% |
| mwir / lwir | 1.4977 | 1.3503 | 1.50 +/- 10% |

The ratio tolerance passes, with MWIR near its lower limit. No threshold or reference
ratio was adjusted to obtain that result.

The simulated historical curve is inside the 95% pointwise bootstrap interval
from the stored 55-scene detector sample (positive dprime rows are used for log
fitting, matching the original method). The bootstrap has 1,000 resamples, seed 206,
and 32 logarithmically spaced evaluation points from dprime 5 to 500. Synthetic
frame false-alarm probability measured in 200,000 draws: 0.010030.
Detailed curves and hashes are in rl/results/checks/wp2.json.

## Unresolved scientific acceptance result

M2-09 fails for uncooled LWIR: held-out log RMSE
1.9210 exceeds training CV
1.3613. The prescribed size/day fallback is
worse: CV 1.7157 and held-out
2.0156. It was rejected, and no
split, coefficient, uncertainty or acceptance limit was changed to hide the failure.

Large validation residuals include cloudy scenes s000013 (measured dprime about 627,
fitted about 7.28) and s000090 (measured about 0.153, fitted about 6.64). A single
low-dimensional scene model does not explain that variation. diagnostics.json
contains every held-out residual and the reproducible simpler-model comparison.
A repair needs recovered pixels and new held-out validation, not fitting the
existing validation labels.

## Missing evidence and contract limits

The original four-band census dataset is absent: the recovery tool reports 126/126
missing complete scenes. The repository also has no available Blender 4.2 executable.
The PDF's fallback therefore applies: conservative analytic plume removal (grade C)
and omission of low-heat tests. Neither pixel reproduction nor measured no-plume
or low-heat results are claimed.

The .1-.7 plume bound is plume-to-stub contrast. It is not plume-to-whole-target
contrast. The fallback uses d/(1+r) as a conservative sensitivity estimate, only for
MWIR shared signal, and leaves the LWIR signal models unchanged. At deployment
there is no privileged input to the student actor.

The current environment draws independent Bernoulli detections through a
probability-only protocol and does not consume the new shared-clutter/score sampler
or sensor-only dwell gain. EpisodeSpec also lacks explicit visibility, aspect and
power fields. CHANGE_REQUESTS.md records these M1/shared-interface integrations;
no protected interface was edited without review.

The frozen historical detector curve is a joint RGB+LWIR surrogate. The later local
source correction says its nominal validation threshold did not achieve .01 FA/frame
on the reported test images. The new model reproduces the frozen curve and generates
synthetic .01 null events separately; it does not validate a real .01-FA ROC or
establish per-camera detection efficiency.

## Task disposition

| Task | Disposition |
| --- | --- |
| M2-01 | Pixel recovery/reproduction command implemented; empirical five-scene check blocked by missing layers |
| M2-02 | Complete: 126 regenerated scene identities checked |
| M2-03 | Low-heat rendering unavailable; documented plan fallback |
| M2-04 | Implemented; CV reported; required median ratios pass |
| M2-05 | Local filter tested on all four sensors; analytic grade-C fallback; empirical censuses unavailable |
| M2-06 | Historical score curve and synthetic null checks pass, with operating-point caveat |
| M2-07 | Standalone noise/clutter and dwell checks pass; environment integration requires M1 |
| M2-08 | Existing protocol passes; WP2 sign-off remains blocked |
| M2-09 | Held-out validation implemented; uncooled-LWIR acceptance fails honestly |
| M2-10 / M2-11 | Blocked by absent trained policies, evaluation rows and M4 bootstrap implementation |
| M2-12 | Calibration section/report delivered; RL results section awaits genuine experiments |

## Verification

The final test output and regression status are recorded in VERIFICATION.md. The
WP2 acceptance output is reproducible with python -m rl.scripts.check wp2. It
returns nonzero for the listed unresolved requirements. No training run or team
approval is invented. No merge, push, interface freeze or submission was performed.

The broader regression run and recovered legacy reference input are documented in
VERIFICATION.md and REFERENCE_RECOVERY.md. Those environment repairs do not change
the WP2 validation result or supply the missing original rendered scenes.
