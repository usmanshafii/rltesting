# Better detection that survives a critic: what to do (2026-10-18)

**Status.** This is a plan built on evidence. No new model has been trained for it.
- The literature numbers are quoted from the saved papers and checked:
  - `docs/sources/detection_methods/readings.json`, by `tools/source_measurements.detection_method_readings()`;
  - `docs/sources/privileged/readings.json`, by `privileged_readings()`.
- The project numbers come from the signal census of the rendered M1 scenes outside the sealed test split
  (`tools/privileged_census.py`, `docs/reference/privileged/data/`).
- The companion document, on whether the privileged teacher adds anything, is `docs/privileged_teacher_case.md`.

## 1. A correction first: how detectability is measured

Until 2026-10-18 the regime cases of M1 and the census of 2026-10-17 used a target "contrast-to-noise": the box mean
against a 2-px ring, divided by the ring's spread (`tools/m1_det2.cnr`).
- **The measure is not referenced to noise for small targets.** With the sensor noise removed it barely changed:
  2.98 against 3.03, and 2.13 against 2.13. The ring holds the target's own blur and the background.
- **It also ranked the cameras wrongly.** In scene s000006 it called the target faint in every camera (uncooled LWIR
  1.6, cooled LWIR 2.1). The textbook measure finds it clear in all three thermal cameras: 6.6, 13.6 and 8.8.

**The replacement: the matched-filter detectability d′** (`eval/detectability.py`). It is the target's exact signal
measured against the spread of the same filter over the target-free frame.
- **Exact signal.** The frame with and the frame without the target share one noise draw, so their difference is
  exactly the target's signal.
- **Noise and clutter split.** A second, independent noise draw separates sensor noise from the scene's own structure.
- **Tested.** On white noise it returns the matched-filter SNR within 2 % (`tests/test_detectability.py`).
- **Threshold.** D ≥ 5 is where a Gaussian matched filter detects half the time at about 0.01 false alarms per
  640 × 512 frame.
- **Where it is used now:** the census and M1's regime report (`tools/m1_det2_eval.regimes`, computed from the cached
  test frames). The old ring measure is kept beside it for comparison.
- **What it replaces.** The figures of 2026-10-17 (38 %, 10 %, 9–16 %, cooled LWIR 0.93×) are withdrawn. Section 2
  replaces them.

## 2. Where the detection gap actually is

The census measured 126 rendered M1 scenes outside the sealed test split, 63 of them with targets of up to 16 px, one
noise draw each, with the current loader (`docs/reference/privileged/data/census.json`, `census_summary.json`).

**What the cameras carry** (matched-filter d′; Wilson 95 % intervals in brackets):

| | All targets | ≤ 16 px | ≤ 16 px at night |
|---|---|---|---|
| The student's own cameras give an ideal detector enough signal (d′ ≥ 5) | 95 % [90–98] | 90 % [81–96] | 83 % [66–93] |
| A privileged camera sees the target better than the student's best | 71 % [62–78] | 76 % [64–85] | 73 % [56–86] |
| …at least twice as well | 37 % [29–46] | 35 % [24–47] | 30 % [17–48] |
| Student faint (2–5) or blind (< 2) while a privileged camera is clear | 2 of 126 | 2 of 63 | 2 of 30 |
| All four cameras fused, ideal detector | 97 % | 94 % | 90 % |

- **The privileged cameras are better, but they cover less.** Where they see the target, the cooled LWIR has 1.23×
  the uncooled camera's d′ (1.35× for small targets) and the MWIR 1.50×. Ideal fusion of all four is 2.1× the student's
  two. The cooled LWIR misses 17 % of targets (27 % of small ones), because its zoom widens only to 25°.
- **For small targets the limit is sensor noise.** The median share of sensor noise in the spread is 0.78–0.84 in every
  camera. For larger targets clutter dominates (uncooled LWIR 0.40).

**The trained detector against the ideal one** (`tools/detector_vs_ideal.py`, `detector_vs_ideal.json`):
- *Data:* the noise verification's RGB + uncooled-LWIR models on the same 55 test scenes and the same noise draw.
- *Detection* is counted at each model's own threshold for 0.01 false alarms per frame.

| Student d′ | < 5 | 5–10 | 10–20 | 20–50 | ≥ 50 | All |
|---|---|---|---|---|---|---|
| Targets | 3 | 5 | 4 | 8 | 35 | 55 |
| Ideal detector | 0 | 5 | 4 | 8 | 35 | 52 |
| Trained, seed 29 | 0 | 0 | 0 | 5 | 31 | 36 |
| Trained, seed 31 | 0 | 0 | 1 | 3 | 23 | 27 |

- The trained detector finds half the targets at d′ ≈ 43 (seed 29) and ≈ 114 (seed 31), against about 5 for the ideal
  detector: 8–20× more signal.
- **The gap is in the detector, not in the sensors.** These models had 62 training scenes and 2,000 steps; M1's (360
  scenes, 3,000 steps) should be closer to the ideal. The gap is the room for "better detection".

**At the trained detector's working point** (d′ ≈ 43; `working_point.json`):

| Share of targets above the working point | All targets | ≤ 16 px |
|---|---|---|
| The student's own cameras, one frame | 63 % | 46 % |
| The student, 5 frames of a static scene (only sensor noise averages) | 67 % | 54 % |
| The student, 5 frames, bound for a moving target (clutter also suppressed) | 77 % | 67 % |
| The teacher, all four cameras fused | 83 % | 78 % |
| Above it for the teacher only | 25 of 126 (20 %) | 20 of 63 (32 %) |

This is where the privileged teacher has something to teach. For 20–32 % of targets the teacher's views are above the
level a trained network needs while the student's are not, although the student's frames still hold the signal for an
ideal detector. That is the "learns faster from a better teacher" case of the theory. How much of it reaches the
student has to be measured.

**Zooming the teacher's cameras in, measured** (loader-only, centred; `census_z2*`, `census_z1to3*`):

| Teacher's zoom cameras | Privileged camera better than the student | They miss the target | Above the working point for the teacher only |
|---|---|---|---|
| Matched to the student's field (M1 as it is) | 71 % | 17 % / 6 % | 25 of 126 |
| 2× narrower | 19 % | 75 % / 68 % | 6 of 126 |
| 1–3× narrower, drawn per scene | 37 % | 56 % / 51 % | 8 of 126 |

(The "miss" column is cooled LWIR / MWIR.)

- **Where a 2× camera sees the target, it is better** (cooled LWIR 1.46× the uncooled camera, up from 1.23×).
- **But it sees only the centre**, and M1's targets fall anywhere in the frame (median offset 0.64 of the half-frame).
- **So in M1 the matched field is the better choice**, and `teacher_zoom` stays off.
- The zoom cameras' real advantage, more pixels on a far target, pays only when the camera is pointed at the target.
  That is the slew-to-cue of a real system, which in training would leak the target's position to the teacher.
- Native resolution (a re-render) would raise the gain where the camera sees, but not its coverage.

## 3. What the latest work reports, and how it fits here

| Lever | Best evidence (checked) | Reported gain | Fit to this project |
|---|---|---|---|
| **Multi-frame (temporal) detection** of moving small targets | MOCID, AAAI-25 [ZHANG2025-MOCID]: on IRDST, single-frame DNANet 63.6 AP50 against multi-frame DTUM 92.1 and MOCID 94.7. In MOCID itself, 3 → 5 frames gives 87.6 → 95.9 AP50 (DAUB) | the largest of any lever, on dim moving infrared targets | uses only the deployed RGB + uncooled LWIR. The sensor-noise share of the spread averages over frames, and a moving target separates from static clutter |
| **Upsampled or sliced inference** for tiny objects | SAHI, ICIP 2022 [AKYON2022-SAHI]: +6.8 / +5.1 / +5.3 AP; with sliced fine-tuning, +12.7 / +13.4 / +14.5 | large where objects are a few pixels in a big frame | 2–4 px targets on a stride-4 heatmap. A 2× upsampled pyramid level (M1 only downsamples today) |
| **Privileged-sensor distillation** (a better sensor in training only) | LiDAR → camera: UniDistill +2.0–3.2 [ZHOU2023-UNIDISTILL], SimDistill +4.8 mAP [ZHAO2024-SIMDISTILL], MemDistill +9.2 mAP [KWON2025-MEMDISTILL]. IR → RGB aerial: +1.4 to +3.7 mAP [LOU2026] | +2 to +9 against plain baselines, not against a same-sensor distillation control | the four-sensor teacher. Claims only against `student_ctrl` (`docs/privileged_teacher_case.md`) |
| **High-resolution teacher → low-resolution student** | MSAD, CVPR 2021 [QI2021-MSAD]: +2.1 to +3.6 mAP | moderate | the evidence for zoomed teacher cameras (section 4.3) |
| **Small-target losses** | MSHNet, CVPR 2024 [LIU2024-MSHNET]: Pd 91.2 → 93.9 %, but false alarms 5.3 → 15.0 per million pixels | small, with a trade-off | low priority |
| **How to compare** | Bouthillier et al., MLSys 2021 [BOUTHILLIER2021]: "As many sources of variation as possible should be randomized"; decide on the probability of improvement, not on averages | — | seeds, paired tests, resampling |
| **Tiny UAVs remain hard** | CST Anti-UAV, ICCVW 2025 [XIE2025-CST]: the best tracker reaches 35.92 % state accuracy on tiny thermal UAVs, against 67.69 % on Anti-UAV410 | — | why the detection-range end, not the average, is the contest |

## 4. The plan: four contributions, each of which can be defended

### 4.1 Close the gap between the trained detector and the ideal observer (deployed sensors only)

- **False alarms on clouds.**
  - *Evidence:* in the noise verification (`docs/sensor_noise.md`), the camera-trained model's FROC more than tripled
    without two cloudy scenes (0.151 → 0.538), and clouds set its thresholds.
  - *Work:* render target-free, cloud-rich training scenes and mine their false alarms (the hard-negative machinery
    exists). Report per-scene false alarms with the leave-one-scene-out range.
- **Tiny targets.** Add a 2× upsampled pyramid level for targets of 2–8 px, trained on 2× crops, in the manner of
  sliced inference (SAHI's gains above). No re-render is needed.
- **Data at the same budget.** Rendering only RGB and uncooled LWIR gives 1.73× the scenes for the same time
  (`docs/privileged_teacher_case.md`). Data is the classic lever for a learned detector far from the ideal observer.
- **Seeds and longer training.** At least 3 seeds per model; two seeds of one training differed by up to 0.38 FROC
  in the noise verification.

### 4.2 A multi-frame student

- **Physics.** Averaging N frames divides the sensor-noise part of the spread by √N. A moving target against a static
  sky separates from the clutter by its motion: the literature's large gains.
- **No re-render.** Short clips can be made in the loader: over 0.2 s at 25 Hz the target's aspect barely changes and
  its image moves about a pixel per frame (range, airspeed and focal length are in each scene's metadata). The
  rendered target layer is composited at those positions, with fresh temporal noise and the camera's fixed pattern
  kept. This holds for clips of a few frames, not for long sequences.
- **Model.** The student's per-band stems take a stack of N = 3–5 frames, as MOCID's best length.
- **Test.** The same scenes, single frame against N frames, same seeds.

### 4.3 The privileged teacher, made fair

**Zoomed teacher cameras** (implemented 2026-10-18; loader-only, so rendered scenes stay valid):
- *Code:* `configs/sensors.yaml teacher_zoom`, `render/cameras.zoom_in`, `data/dataset.apply_teacher_zoom`.
- *What it does:* each privileged zoom camera can be set 1–3× narrower than the student's field, drawn per scene.
- *Measured (section 2):* where it sees the target it is better, but it sees only the centre. Above the detector's
  working point it leaves the teacher 6–8 targets ahead instead of 25. **It stays off for M1.**
- *Where it would pay:* a camera pointed at the target, at native resolution (a re-render). That is worth testing only
  on a set of far, central targets.

**Distillation where the student can see** (implemented, on in M1):
- A target whose student-camera contrast is below 1 gets no feature-distillation weight; its cells are left out of the
  response distillation (`models/cnn_detector.cnr_gap_weight`, `kd_keep`).
- Before, the weight peaked there.
- The floor is still on the old ring measure. Moving it to d′ needs an absent frame with the same noise draw for each
  training realisation, about 1.5–2 h more cache time.

**Claims:** only against the control student and an equal-budget B, with seeds (`docs/privileged_teacher_case.md`,
section 6).

### 4.4 An evaluation built to survive a critic

- **The ideal-observer reference.** Report each model's detection probability per d′ class against what an ideal
  matched filter would get. The gap is the room for improvement, and it cannot be argued away.
- **Operational metrics.** Detection range, the range at which detection probability falls to 0.5, by day and night.
  The FROC is reported with its leave-one-scene-out range.
- **Variance.**
  - at least 3 seeds per arm;
  - paired scene bootstrap;
  - the probability-of-improvement criterion [BOUTHILLIER2021].
- **A faint-target test set.** About 200 long-range scenes where the student's d′ is 2–5, if the privileged or temporal
  gain is to be shown where it lives.

## 5. What can be claimed, and what not

- **Not:** "+15 mAP from the privileged teacher".
  - The teacher is ahead above the network's working point for 20–32 % of targets (section 2).
  - Published privileged training passes on 8–21 % of what the privileged sensor is worth [LOU2026]. That
    suggests a few points, not 15.
  - mAP is not this project's metric.
- **Possible, from the detector:** a gain of 15 points or more in detection probability.
  - The trained detectors found 49–65 % of the test targets at 0.01 false alarms per frame.
  - An ideal detector with the same two cameras would find 95 % (section 2).
  - Closing that gap is what more data, false-alarm control, tiny-target resolution and frames do.
- **Yes, if measured:** "the multi-frame student detects X % more targets at 0.01 false alarms per frame than the
  single-frame one, and Y % of the ideal observer's".
  - That is the claim with large published support.
  - It applies to the deployed sensors only.
- **Yes, if measured:** "with zoomed privileged cameras, distillation adds Z over a same-sensor control, across N
  seeds", or "it does not". A null result is still a finding.

## 5a. The baseline and the gains to expect (estimates, 2026-10-18)

**The baseline, measured** (`tools/detector_vs_ideal.py ap` → `baseline_ap.json`): the noise verification's
RGB + uncooled-LWIR models (62 training scenes, 2,000 steps), two seeds, on 55 test scenes. M1's models (360 scenes,
3,000 steps) should start higher.

| Metric | Seed 29 | Seed 31 |
|---|---|---|
| Detection at 0.01 false alarms per frame (centre match) | 0.65 | 0.49 |
| AP, centre match (the infrared small-target convention M1 uses) | 0.646 | 0.655 |
| AP, centre match, targets up to 16 px | 0.453 | 0.539 |
| AP50 (IoU ≥ 0.5, the standard "mAP") | 0.218 | 0.217 |
| AP50, targets up to 16 px | 0.016 | 0.004 |
| An ideal detector on the same frames (detection at 0.01 false alarms per frame) | 0.95 | 0.95 |

IoU-based mAP is the wrong headline for 2–16 px targets. A box 1 px off on a 3-px target already fails IoU 0.5, so a
detector that finds the target scores nearly nothing. Report detection at a fixed false-alarm rate and the
centre-matched AP, with IoU-AP beside them.

**Projected gains** in detection at 0.01 false alarms per frame, in points (100 × probability):
- Each lever is measured against the M1 baseline with 3 seeds (section 4).
- They are not additive, and an ideal detector caps them all at 95.

| Lever | Projected gain | Basis | Grade |
|---|---|---|---|
| More training data (M1's 360 scenes; 620 for equal-budget B) | not projected: the largest unknown; the room is up to +30 to +46 | the detector needs 8–20× the ideal signal | to measure first (learning curve) |
| Multi-frame student, 5 frames | +4 to +11 (all targets), +5 to +14 (≤ 16 px), with the network's efficiency fixed; more if it learns motion | each model's measured detection curve applied to the census d′ (`projections.json`); MOCID 3 → 5 frames +8.4 AP50 | C |
| Cloud false-alarm control | +2 to +6 on this test set (shows at fixed false alarms, hardly in AP) | removing the two cloud scenes' false alarms gave the M1-noise model +0.056 FROC | C |
| Upsampled level for tiny targets | detection 0 to +5; IoU-AP for small targets well above today's ~0.01 | SAHI +5 to +7 AP; the detector misses below d′ 10–20 for lack of signal, not of pixels | C/D |
| Privileged teacher, against the control | +1 to +4, possibly 0 | the teacher's full signal would give the student +10 to +19 (`projections.json`); privileged training passes on 8–21 % of such a gap [LOU2026]; [XUE2023] and our pilot found 0 | C |
| Full detector (5.6 M parameters, Kaggle) | not projected | 15× the compact CNN | to measure |

**Combined**, if the multi-frame student and the data work as projected: +10 to +25 points in detection at 0.01 false
alarms per frame. That is from about 0.5–0.65 to about 0.65–0.85. Centre-matched AP would move by a similar amount;
IoU-AP much less unless localisation of tiny boxes improves.

## 5b. What has to be done, in order

| Step | Work | Laptop time | Code |
|---|---|---|---|
| A1 | Learning curve on the 126 scenes already rendered (31 / 62 / 107 training scenes, 2 seeds): how much more data gives | ~1.5 h | none |
| A2 | Render the remaining 420 M1 scenes | ~6.5 h | none |
| A3 | M1 with 3 seeds (`rgb_lwir`, `student_ctrl`, `student`, `teacher`; single-camera models once): the baseline, the teacher test, the gap to the ideal detector | ~12 h | none |
| B1 | Cloud-rich target-free scenes (~100) and their use as background and mined crops; retrain, 3 seeds | ~1.5 h render + ~1.5 h | ~0.5 day |
| B2 | An upsampled pyramid level for targets up to 8 px; retrain, 3 seeds | ~2 h | ~0.5 day |
| B3 | Multi-frame student: clips made in the loader, 5-frame stems, cache and train, 3 seeds | ~8 h | ~1.5 days |
| B4 | Equal-budget B: ~260 more RGB + uncooled-LWIR scenes; retrain, 3 seeds | ~2.3 h render + ~1.5 h | small |
| C1 | Working-point test set: ~200 longer-range scenes with student d′ 10–50 | ~3 h render | an overlay |
| D1 (optional) | Full detector on Kaggle: GPU and cache support, upload, train | GPU hours | ~0.5 day |
| E1 | Report: detection at fixed false alarms, centre and IoU AP, d′ classes against the ideal detector, detection range, seeds | — | small |

**Totals:** about 3–4 days of code and 35–45 h of laptop time (renders ~16 h, training and evaluation ~20–25 h). A1
first: it decides how much the renders are worth.
