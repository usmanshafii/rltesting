# M2 handoff requests (not approvals)

The PDFs and AGENTS.md reserve shared interfaces and other members' folders for review.
The following requests are recorded without changing those files.

## CR-M2-01: consume the calibrated stochastic model in M1

Owners: M1 and M4; all four approve any contract/frozen-value change.

The current SensorModel protocol returns probabilities. SkyEnv._sense generates
independent per-band, per-frame Bernoulli events, and _best_zoom multiplies
single-frame expected evidence by K. Neither function can consume a persistent
shared clutter realization or the measured sensor-only dwell gain through the
existing protocol. Passing CalibratedSensorModel works now, but does not implement
those additional stochastic features.

Proposed reviewed integration: extend the sensing interface to accept correlated
scores or detection counts from an episode-local noise/clutter state, while retaining
separate sensor-noise streams. Use eval.detectability.d_after for the dwell model,
check the student-only evidence gate against the same generated observations, and
rerun the false-confirmation and oracle checks. Preserve the rule that changing a
privileged camera never consumes a student random stream. Also decide whether to
replace fa*g with the exact partition 1-(1-fa)**g (implemented in score_model.py).
Do not silently change the evidence nominal Pd or A to make an acceptance result pass.

## CR-M2-02: explicit episode nuisance state

EpisodeSpec has no visibility, viewing aspect, or engine-power setting. Calibration
fits these fields from regenerated metadata, but the current implementation can only
sample matched TRAINING scene nuisance state once per episode. An actual held-out
visibility condition or changing aspect along a flight needs an interface change.
Do not invent a frozen visibility interval or claim that condition was evaluated.

## CR-M2-03: corrected detector operating point

The course plan/frozen file uses the historical logistic coefficients from
 detector_vs_ideal.json. docs/privileged_teacher_case.md section 3 records the later
correction: the nominal validation threshold did not produce 0.01 FA/frame on those
test frames. The Gaussian surrogate deliberately reproduces the frozen historical
curve, and separately generates a synthetic 0.01 null probability. These are not a
measured ROC at a common operating point. Review whether to freeze a remeasured
curve after source pixels become available. A change must version the environment
and rerun affected experiments.

## CR-M2-04: validation shortfall and remaining evidence

Uncooled LWIR held-out log RMSE exceeds the five-fold cross-validated estimate.
The prescribed size/day fallback is worse on training CV and held-out scenes, so it
was rejected. Do not calibrate on validation rows or move their split to hide this.
Recover the original four-band pixels, investigate the residuals, and validate any
repair on new held-out scenes before approving env-v1.

Original scenes/renderer are absent here; pixel reproduction, measured no-plume and
low-heat evidence remain unavailable. The plan authorizes analytic plume grade C
and omission of low-heat tests, but that fallback is explicitly distinct from empirical
validation. No Tier 1 policies/evaluation rows exist, so M2-10/M2-11 results analysis
and the results section of M2-12 remain blocked by M3/M4.
