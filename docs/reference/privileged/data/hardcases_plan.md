# Hard cases of the standard detector: analysis plan

Written 2026-10-19, before any hard-case number was computed. Code: `tools/privileged_hardcases.py`.

**Question.** On the test scenes where the standard RGB + uncooled LWIR detector (B, `rgb_lwir`) fails, does the
privileged student (P, `student`) find more targets than:
- another training run of B would, and
- the ordinary-distillation student (K, `student_ctrl`)?

**Why the design matters.** Selecting B's own misses and comparing P with that same B favours P by construction. Some of
B's misses are chance, and any other model, even B with another seed, recovers some of them (regression to the mean).
The primary hard set is therefore cross-fitted: B is never judged on a set it chose.

**Hard sets** (test split of `out/m1/privileged_ablation`, B's miss = below its own threshold for 0.01 false alarms per
target-free frame):
- **H1 (primary, cross-fitted):** for evaluation seed s, the scenes B misses in both other seeds.
- **H1+:** for seed s, the scenes B misses in at least one other seed.
- **H2 (descriptive):** the scenes B misses in all three seeds. B detects none of them by construction, so H2 is
  compared only between the other models and independent supervised runs.

**Measures:**
1. Detection at each model's own threshold for 0.01 false alarms per target-free frame (test-set curve). Secondary: at
   0.05, and at the scene-level point (at most one scene with any false alarm).
2. The false-alarm cost of each target: target-free detections per frame that score at or above the target's best
   centre-matched score, on a log10 scale (lower is better). This is threshold-free.
3. The JAFROC figure of merit: the probability that a target outscores the highest score of a target-free frame
   (Chakraborty & Berbaum 2004).

**Comparisons on H1** (paired, per seed and as the seed mean):
- P−B (primary);
- P−K, P−M, T−B, K−B.

Scene bootstrap with 10,000 resamples; thresholds and costs are recomputed in every resample. Seed-level t-interval.

**Baselines for the misses:**
- the other seeds' B (rescue by retraining alone);
- the noise ablation's independently trained supervised RGB + LWIR models on identical pixels.

**Mechanism checks:**
- Among B's misses, P versus K on the scenes the four-sensor teacher detects, in-sample and cross-fitted.
- Description of the hard scenes by d′, size, range, light and cloud.

**Rule.** A hard-case benefit is reported as shown only if, on H1:
- the seed mean of P−B is positive with a scene-bootstrap 95 % interval that excludes 0, and
- P−K is positive.

Otherwise it is reported as not shown. Every result is reported, whatever it is.

All of this is exploratory. The 55 test scenes have already been inspected (audit of 2026-10-18), so confirmation needs
fresh scenes.
