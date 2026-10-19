# Does the privileged teacher add anything? The case for and against (2026-10-17, corrected 2026-10-18)

**Correction, 2026-10-18.** The census numbers of 2026-10-17 used a box-against-ring contrast-to-noise that is not
noise-referenced for small targets (`docs/detection_gain_plan.md`, section 1). They are replaced here by the
matched-filter detectability d′ (`eval/detectability.py`). The withdrawn numbers (38 %, 10 %, 9–16 %, cooled LWIR
0.93×) should not be quoted. The design changes of section 6 have been made and measured.

**The question.** Two designs deploy the same two sensors:
- **A (this project):** the teacher trains on all four sensors (RGB, LWIR uncooled, LWIR cooled, MWIR), and the
  student is distilled from it.
- **B:** RGB and LWIR uncooled only, in training and in deployment, with or without distillation from a teacher of
  the same two sensors.

The question is whether A's student beats B, and whether the claim survives a critic.

**Status.** This analysis comes before any M1 result. No model of the four-sensor M1 has been trained, and no number
below is a result of A or B. The evidence is:
- the theory;
- published measurements, with quotes checked against the saved papers (`docs/sources/privileged/readings.json`,
  `tools/source_measurements.privileged_readings()`);
- our earlier pilot;
- a census of what the four cameras see in this project's own scenes (`tools/privileged_census.py`,
  `docs/reference/privileged/data/`).

## Short answer

- **A's gain is not established, and as things stand it can be contested.** Against same-size or same-sensor
  controls, published privileged-sensor gains are 1–3 points and sometimes zero. In our own earlier pilot a control
  student matched A's student exactly.
- **What our scenes show** (d′ census, section 3):
  - **The privileged cameras do see better.** In 71 % of cases, and in 37 % at least twice as well. Cooled LWIR is
    1.23× the uncooled camera's d′, MWIR 1.50×, and all four fused 2.1×.
  - **But the student's own two cameras already carry enough signal for an ideal detector for 95 % of targets.** The
    teacher adds signal an ideal detector could use for only 2 of 126 targets.
  - **The trained detector is far from ideal.** It needs 8–20× more signal (d′ ≈ 43–114 against 5) to find half the
    targets.
  - **At that working point the teacher's views are ahead for 20 % of all targets (32 % of small ones).** That is
    where distillation could help the student learn faster, the real form of the privileged-information claim.
- **What can be defended:**
  - A costs nothing at deployment, and the design cannot leave the project worse off.
  - The case where A can help is identified in advance: targets the student's frames hold, but the student's network
    misses, while the teacher's views are clear.
  - The strongest real-world use is still labelling real, unlabelled four-sensor recordings.
- **To make any claim stick**, the experiment in section 6 is needed:
  - the control student;
  - an equal-budget version of B;
  - several seeds;
  - results at the detector's working point.

## 1. What theory says

- **Learning using privileged information is a learning-speed device.** Vapnik and Izmailov describe "two mechanisms
  that can be used for significantly accelerating the speed of Student's learning using privileged information"
  [VAPNIK2015, p. 1].
  - The bound they build on: "in the separable case, the rate of convergence has the order of magnitude 1/ℓ"
    against 1/√ℓ otherwise (p. 5).
  - It speeds learning towards what the student's inputs allow. It does not raise that ceiling: no teacher adds
    information that is not in the student's own frames.
- **The gain is largest with little data.** Lopez-Paz et al.: "the teacher is most helpful in low data regimes"
  [LOPEZPAZ2016, p. 5].
  - In their MNIST experiment, "the benefits of distillation diminished as we further increased the sample size"
    (p. 8).
  - Their closest analogue to a better sensor uses clean images as privileged and noisy images as the student's.
    There, "Distillation on the 300 labeled samples did not improve the student performance" (p. 8); the gain came
    only from the teacher's soft labels on unlabelled images.
- **What a good teacher is.** Menon et al.: a teacher "providing the true class-probabilities can lower the variance
  of the student objective", and the criterion for a good teacher is "the quality of its probability estimates"
  [MENON2021, p. 1].
  - The probabilities the student needs are p(target | its own frames), x_s.
  - A privileged teacher outputs t = p(target | x_s, privileged frames x_p). Averaged over x_p, t equals the
    student's own p(target | x_s), so it is a valid target.
  - Its variance given x_s is lower than a hard label's only by E[Var(y | x_s, x_p) | x_s], the part of the label's
    uncertainty that **remains** with the privileged view.
  - Where the privileged cameras see the target clearly (the case A is built for), t is close to the label itself
    and adds almost nothing to the response target.
  - The largest reduction would come from a perfect teacher of the student's own sensors, which is what B's
    distillation control approximates.
  - *This is our derivation from Menon et al.'s framework, not their claim about privileged teachers.* It covers the
    teacher's outputs only. Feature-level guidance, the other route used by M1 and by the papers in section 2, is
    not bounded by it.
- **Distillation helps without any new information**, so a student gain does not by itself show transfer:
  - students "parameterized identically to their teachers" improve: "these Born-Again Networks (BANs), outperform
    their teachers significantly" [FURLANELLO2018, p. 1];
  - "a poorly-trained teacher with much lower accuracy than the student can still improve the latter significantly"
    [YUAN2020, p. 1];
  - distillation "does not typically work as it is commonly understood" [STANTON2021, p. 1];
  - "larger models do not often make better teachers" [CHO2019, p. 1].
- **Cross-sensor transfer needs shared cues.** "KD is not a universal cure in crossmodal knowledge transfer"
  [XUE2023, p. 1]. Distillation performance "is dependent on the proportion of modality-general decisive features
  preserved in the teacher network" (p. 5).
  - Here, the cooled LWIR shares the student's band physics, so its cues are largely shared.
  - The MWIR's strongest cue, the CO2 plume, is not. At the source it is 10–70 % of the stubs' contrast in 3.7–4.8 µm
    against ≤ 2 % in 8–14 µm (`docs/thermal_bands.md`, section 5).

## 2. What has been measured elsewhere

| Study (privileged → test sensor, task) | Student's sensor only | Same-size or same-sensor control | Privileged-trained student | Privileged sensor also at test |
|---|---|---|---|---|
| [HOFFMAN2016] depth → RGB, detection, mAP (AlexNet / VGG) | 26.6 / 29.9 | RGB ensembles 29.0 / 32.7 | 30.5 / 34.0 | — |
| [XU2017] thermal → RGB, pedestrians, miss rate (lower is better) | 54.26 % | two RGB networks 52.15 % | 49.55 % | — |
| [GARCIA2018] depth → RGB, actions, accuracy | 66.52 % | none reported | 73.42 % | 79.73 % |
| [LOU2026] IR → RGB, aerial vehicles, mAP50 | 63.9 | — | 65.3 (features), 66.0 (distillation), 67.6 (their method) | 81.4 |
| [XUE2023] RGB+depth → RGB, segmentation, mIoU | 46.36 (no distillation) | RGB teacher 48.00 | RGB+depth teacher 47.78 | — |
| Our pilot (`../fyp`, SWIR → RGB+LWIR, AP50) | baseline 0.224 | control teacher without SWIR 0.322 | 0.321 | teacher 0.220 |

**What the table supports:**
- **Gains exist.** The closest analogues show them: thermal training lowered an RGB detector's miss rate by 2.6 points
  against a same-size control, and IR training raised an aerial RGB detector by 1.4–3.7 mAP.
- **They shrink against fair controls.**
  - Hoffman: +3.9 / +4.1 mAP over one RGB model, but +1.5 / +1.3 over an RGB ensemble of the same size.
  - Xu reports a same-size control and still finds 2.6 points.
  - Garcia reports none.
- **They recover a small share of what the privileged sensor is worth at test.** Lou's students close 8, 12 and 21 %
  of the 17.5-point gap to the model with IR at test.
  - Their own phrasing, "about 85% of the upper-bound" [LOU2026, p. 5], is a ratio of scores and hides that most of
    the gap remains.
- **They can vanish.**
  - Xue's multimodal teacher made a slightly worse student than the single-modality teacher: "+4.64%" for the teacher
    gave "-0.22%" for the student [XUE2023, p. 4].
  - In our own pilot, the student taught with SWIR (0.321) and the control taught without it (0.322) were the same
    (`../fyp/docs/m1_evidence_status.md`).
  - All of that pilot's gain over the supervised baseline came from distillation itself:
    - +0.096, 95 % interval −0.016 to +0.246;
    - one seed, 27 test scenes.
  - The same document records that "0 of 97 training/validation scenes had a target visible only in SWIR".
- **Few published comparisons report seed variance.** Lou et al. "evaluated random seed values in the range 1–10 and
  observed that all settings yielded comparable performance" (p. 5), without numbers.

## 3. What this project's own scenes show

**The census.** `tools/privileged_census.py` → `docs/reference/privileged/data/census.json` and
`census_summary.json`:
- **Scenes:** 126 rendered M1 scenes outside the sealed test split (107 training, 19 validation; 54 day, 72 night).
- **Measure:** each target's matched-filter detectability d′ in each camera's network-input frame
  (`eval/detectability.py`), one noise draw per scene, with the loader as configured. The old ring measure is still
  stored beside it.
- **Who counts as who:** "student" is the better of RGB and uncooled LWIR; "privileged" is the better of cooled LWIR
  and MWIR.
- **Classes:** D ≥ 5 is where an ideal (matched-filter) detector finds half the targets at about 0.01 false alarms per
  frame; F 2–5; N < 2.
- **Intervals:** Wilson 95 %, shown in brackets.

| Share of targets | All (126) | ≤ 16 px (63) |
|---|---|---|
| Privileged camera sees the target better than the student's | 71 % [62–78] | 76 % [64–85] |
| …at least twice as well | 37 % [29–46] | 35 % [24–47] |
| Student detectable by an ideal detector (D) | 95 % [90–98] | 90 % [81–96] |
| Student faint (F) or blind (N), privileged detectable | 2 of 126 | 2 of 63 |
| Ideal fusion of all four detectable | 97 % | 94 % |

- **Camera ratios where they see the target:** cooled LWIR 1.23× the uncooled camera's d′ (1.35× for small targets),
  MWIR 1.50×. Perfect fusion of all four gives 2.1× the student's two.
- **The trained detector against the ideal one** (`tools/detector_vs_ideal.py`, the noise verification's
  RGB + uncooled-LWIR models on the same 55 scenes and the same noise draw):
  - it finds 36 and 27 of 55 targets (two seeds) at each model's threshold frozen on validation for 0.01 false alarms
    per frame, where an ideal detector would find 52. *Corrected 2026-10-19 (independent audit):* those thresholds
    gave 0.20 and 0.064 false alarms per frame on these test frames, not 0.01. At 0.01 itself, with the threshold
    taken from the test frames' own false alarms, they find 23 and 20 of 55;
  - it finds none with d′ below 10, and about half of those with d′ 20–50;
  - half detection is at d′ ≈ 43 and ≈ 114 at those frozen thresholds. At 0.01 false alarms per frame the same models
    need d′ ≈ 224 and ≈ 364, and pilot 1's supervised RGB + LWIR detector, the stronger baseline, needs 51–133
    (three seeds).
- **At that working point (d′ ≈ 43)** the student's own cameras put 63 % of targets above it (46 % of small ones).
  The teacher's fused views put 83 % above it (78 %). The teacher is ahead for 25 of 126 targets (20 %) and for 20 of
  63 small ones (32 %).
- **Zooming the teacher's cameras in does not help in M1's scenes** (measured, `docs/detection_gain_plan.md`,
  section 2). A centred 2× zoom raises their d′ where they see the target, but they then miss 68–75 % of targets.
  The teacher is left ahead above the working point for 6–8 targets instead of 25.

**Cost.** Mean in-Blender time per scene, from the 180 scenes' own timings:

| | Day | Night | All |
|---|---|---|---|
| Time per scene | 75.7 s | 41.1 s | 55.9 s |
| Share spent on cooled LWIR and MWIR | 51 % | 31 % | 42 % |
| RGB + uncooled LWIR scenes for the same render time | 2.03× | 1.44× | 1.73× |

The teacher and the distilled student also train more slowly than the two-sensor model: 0.50 and 0.56 s per step,
against 0.37 s (`docs/m1_v2.md`).

**Resolution of M1 as designed** (from the noise verification, `docs/sensor_noise.md`):
- two seeds of the same training differed by 0.04–0.12 in mean FROC (0.38 in one case);
- paired differences on 55 test scenes had 95 % intervals about ±0.1–0.2 wide;
- with 180 test scenes and one seed per model, M1 resolves differences of roughly 0.1 FROC. M1 chooses checkpoints on
  60 validation scenes, not 9, so its seed spread should be smaller; it has not been measured.

That is several times the 1–3-point gains the literature reports against matched controls. M1 can therefore neither
confirm nor rule out a gain of the size to expect.

## 4. A against B

| | A: four-sensor teacher → RGB + LWIR student | B: RGB + LWIR in training and deployment |
|---|---|---|
| Deployed sensors and cost | RGB + uncooled LWIR | same |
| Render cost per scene | 1× | 0.58× (1.73× scenes for the same time) |
| Training cost | teacher plus distillation (0.50–0.56 s per step) | 0.37 s per step; distillation optional |
| What it can learn that B cannot | faster learning on targets the student's network misses although its frames hold them, while the teacher's views are clear (20–32 % at the trained detector's working point); feature guidance from cleaner thermal views | — |
| What B can do that A does not | — | spend the saved time on 1.7× the scenes; no train/deploy sensor mismatch; simpler causal story |
| Evidence of a gain over the other | none yet in this project (pilot: zero); 1–3 points against controls elsewhere | the pilot's control student matched A's student |
| Main risks | the gain is generic distillation; modality-specific cues (MWIR plume) do not transfer; teacher-only simulation errors propagate; the gain is too small to measure | gives up the faint-target gain and the real-data labelling route |
| Best real-world use | label real, unlabelled four-sensor recordings for the student (Lopez-Paz et al.'s gain came from exactly that) | train on and fine-tune with real RGB + LWIR data |

## 5. Arguments, ready for a contest

**Objections to A, and how each stands:**
- *"Any student gain is just distillation."*
  - **This holds** unless A's student beats the control student, which is distilled from a teacher of the same two
    sensors.
  - M1 has that control (`student_ctrl`).
  - Our pilot and [XUE2023] are exactly this case.
- *"Your privileged cameras don't see more."*
  - **This does not hold.** They see better in 71 % of cases, 2.1× fused.
  - **What does hold:** an ideal detector would hardly need it (95 % already). So the case rests on the trained
    detector's working point, which has to be shown with results.
- *"Synthetic labels already say where every target is."*
  - **This holds for the teacher's outputs** (section 1).
  - It leaves feature guidance and learning speed. Learning speed matters only while data are scarce, and synthetic
    data are not.
- *"The same render time spent on RGB + LWIR would do better."*
  - **This is untested and fair:** 1.73× the scenes is a large data increase.
  - Lopez-Paz et al. found distillation's benefit shrinks as the sample grows.
- *"The effect is too small to measure."*
  - **This holds for M1 as designed.** A faint-target test set and several seeds are needed.
- *"The MWIR teaches a cue the student cannot see."*
  - **Partly holds** for the plume.
  - Feature distillation weighted to the teacher's confidence could push the student towards it.
  - Today's weight, `w = 1 + 2 clip((C_T − C_S)/5, 0, 1)` (`models/cnn_detector.cnr_gap_weight`), is largest where
    the teacher sees far more than the student. That includes targets the student does not see at all (C_S ≈ 0).

**Objections to B, and how each stands:**
- *"B throws away information."*
  - True at training only.
  - For an ideal detector, it concerns 2 % of targets. At a trained detector's working point, 20–32 %.
- *"Distillation improves students; B has no teacher."*
  - B can have one: distillation from a teacher of its own two sensors is generic and available to B (`student_ctrl`).

**A defensible statement of the thesis, before results:**

> A privileged-sensor teacher can help the RGB + uncooled-LWIR student only where the student's frames hold the target
> but its network misses it, while the teacher's views are clear. In this project's scenes that is 20–32 % of targets
> at a trained detector's working point, and 2 % for an ideal detector. The design isolates the effect: a matched
> baseline, a control student distilled without the privileged sensors, and results per signal regime and against
> the ideal detector.

## 6. What would settle it

1. **Arms**, all with the same deployed sensors and test scenes:
   - `rgb_lwir` (B);
   - `student_ctrl` (B with distillation);
   - `student` (A);
   - **B-budget:** `rgb_lwir` trained on 1.73× the scenes, rendered with RGB and uncooled LWIR only in the time A
     spends on its extra bands.
2. **Seeds:** at least 3 per arm. The compact CNN takes 25–45 min per model on this laptop. Report the spread across
   seeds next to the paired scene bootstrap.
3. **Endpoints, fixed beforehand:**
   - *Primary:* `student − student_ctrl` and `student − B-budget` in FROC.
   - *Secondary:* detection probability in the "student faint, privileged detectable" regime, and the detection
     range at which detection probability falls to 0.5.
   - *Context:* the teacher's own gain over `rgb_lwir`, so the share it passes on can be compared with the 8–21 % of
     [LOU2026].
4. **Test scenes at the working point:** about 200 scenes in which the student's d′ is 10–50 (long ranges, haze,
   night). That is enough to resolve a difference of about 0.15 in detection probability. 180 random scenes hold about
   20–40 such targets.
5. **Design changes, made 2026-10-18:**
   - **Zoomed teacher cameras.** Implemented, loader-only (`configs/sensors.yaml teacher_zoom`). Measured to reduce
     the teacher's lead in M1's scenes, so off.
   - **No distillation where the student cannot see.** Implemented and on: the floor is the student's contrast 1,
     still on the ring measure. Below it, the feature weight is 0 and the cells are left out of the response
     distillation. This removed the weight peak on those targets.
6. **Decision rule:**
   - *A's gain is claimed* only if `student` beats both `student_ctrl` and B-budget on the primary endpoint, across
     seeds and with the scene bootstrap.
   - *Otherwise,* B with distillation is the better-justified design, and the four-sensor teacher is kept only for
     labelling real four-sensor recordings, if they are collected.

## 6a. The pilot test, set up 2026-10-18 before any result (`tools/privileged_ablation.py`)

**What it tests:** whether privileged information adds value beyond ordinary distillation, in the current four-sensor
design.
- *Scenes:* the noise verification's split of the rendered scenes outside M1's sealed test split (62 training,
  9 validation, 55 test).
- *Training:* every model is the compact CNN, trained for 2,000 steps and taken at its final step; seeds 29, 31 and
  37.
- *Batches:* identical for every student of a seed.

| Model | What it is |
|---|---|
| `rgb_lwir` | the standard RGB + uncooled LWIR detector, and the ordinary teacher |
| `teacher` | all four sensors, trained on every sensor subset |
| `student` | distilled from `teacher`, all four sensors fed |
| `student_tmask` | distilled from the *same* `teacher`, with the cooled LWIR and MWIR blanked (`m1_det2.TEACHER_FED`): same weights, same batches, only the privileged inputs differ |
| `student_ctrl` | distilled from `rgb_lwir`: ordinary distillation |

**Why the within-teacher control is the strongest test.** A separately trained ordinary teacher differs from the
privileged one in its training as well as its inputs. Blanking the privileged inputs of the same teacher leaves those
inputs as the only difference. The teacher was trained on every sensor subset, so it is valid without them.

What the masked control can and cannot rule out:
- It may still carry some privileged knowledge in its weights, so `student − student_tmask` is a lower bound on the
  privileged effect.
- `student − student_ctrl` covers the whole effect, including the teacher's training.

**Endpoints, fixed now:**
- *Primary:* `student − student_tmask` in detection at 0.01 false alarms per target-free frame (centre match,
  threshold from the test set's own target-free frames). It is the mean over seeds with a 95 % scene-bootstrap
  interval shared by all models and seeds, and the share of resamples with student > control.
- *Secondary:*
  - `student − student_ctrl` (the conventional contrast);
  - `student_ctrl − rgb_lwir` (what ordinary distillation does);
  - `student − rgb_lwir` (the total over the standard detector);
  - each also in FROC;
  - by the target's uncooled-LWIR d′ (< 20, 20–50, ≥ 50), the low thermal-signature end being < 20.
- *Context:* the teacher with all four sensors against itself on RGB + LWIR only, which is the signal the students
  could gain.

**Decision rule, fixed now:**
- **Shown** if the primary contrast is positive in all three seeds and its 95 % interval excludes 0.
- **Suggestive** if its mean is positive and at least 80 % of resamples favour the student, but the interval
  includes 0.
- **Not shown** otherwise.
- The same three grades apply to `student − student_ctrl`.

**Expected beforehand** (section 3 and `docs/detection_gain_plan.md`): +1 to +4 points in detection, possibly 0. With
55 test targets and 3 seeds this pilot resolves several points at best, so "suggestive" or "not shown" would not rule
out a gain of that size. The decisive version is the working-point test set (item 4 above) with the M1 population.

### Result of the pilot (2026-10-18, `out/m1/privileged_ablation/results.md`)

Detection at 0.01 false alarms per frame, mean of seeds 29, 31 and 37 (55 test scenes):

| Model | 29 | 31 | 37 | Mean |
|---|---:|---:|---:|---:|
| `rgb_lwir` (standard RGB + LWIR) | 0.545 | 0.582 | 0.473 | 0.533 |
| `student` (privileged) | 0.655 | 0.582 | 0.145 | 0.461 |
| `student_tmask` (same teacher, privileged inputs blanked) | 0.636 | 0.291 | 0.164 | 0.364 |
| `student_ctrl` (ordinary distillation) | 0.473 | 0.582 | 0.036 | 0.364 |
| `teacher` (four sensors) | 0.309 | 0.691 | 0.291 | 0.430 |

**The contrasts**, judged by the rule fixed beforehand:

| Contrast | Mean | 95 % interval | P(better) | Verdict |
|---|---:|---|---:|---|
| `student − student_tmask` (primary) | +0.097 | [−0.048, +0.200] | 0.93 | suggestive (seed 37 −0.018) |
| `student − student_ctrl` | +0.097 | [−0.061, +0.230] | 0.92 | suggestive (seed 31 ±0) |
| `student − rgb_lwir` | −0.073 | [−0.139, +0.085] | 0.29 | no gain over the standard detector |
| `student_ctrl − rgb_lwir` | −0.170 | [−0.273, +0.030] | 0.05 | ordinary distillation hurt |

**Why the distilled students fell behind the standard detector.**
- *Rankings were intact:* their centre-matched AP was 0.79–0.82, like the baseline's.
- *The false alarms on empty frames rose:* the score needed for 0.01 false alarms per frame rose from 0.49–0.54
  (baseline) to 0.67–0.75 (seed 37 students), while their target scores stayed at 0.51–0.55.
- *The teachers raise such false alarms themselves:* two of the three four-sensor teachers scored 0.29–0.31 at that
  operating point. Distillation passes them on.

**The masked control is valid only if the teacher works without its privileged inputs.** Seed 31's teacher fed RGB and
LWIR only fell to 0.109, so its masked student (0.291) was poor for that reason. That inflates the primary contrast
in that seed.

**What this means:**
- *Privileged against ordinary distillation:* about +10 points, suggestive.
- *Privileged against the standard detector:* none yet. Distillation in its present form costs more at a strict
  false-alarm rate than the privileged information gains.
- *By uncooled-LWIR d′:* too few targets for a conclusion at the faint end (d′ < 20: 12 targets).

**Next:**
1. Stop distilling the teacher's false alarms. The synthetic labels say where every target is, so the response
   distillation can be cut to the teacher's output near true targets and to the labels elsewhere.
2. Make the teacher's false alarms lower, and its use of missing inputs robust.
3. The blind-teacher control, which is running.
4. The low-heat test of section 6b.

**Added after the first seed's interim test result (2026-10-18, secondary).** A blind teacher, `teacher_blind`, and
its student, `student_blind`:
- *What the teacher is:* the four-sensor teacher's network and subset training, with the cooled LWIR and MWIR always
  absent in training, mining, validation and evaluation (`m1_det2.BLANK_BANDS`).
- *Why:* seed 29 suggested that the gain sits in the teacher's training rather than in its inputs at distillation. A
  critic would then ask whether that comes from the privileged cameras or from the teacher's different training
  regime and size.
- *What it answers:*
  - `student_tmask − student_blind` is the privileged cameras' share through the teacher's training;
  - `student − student_blind` is the whole privileged effect with the teacher's network and regime matched;
  - `student_blind − student_ctrl` is what the regime and size alone do.

The decision rule above is unchanged and applies to its own contrasts only.

**Status (2026-10-19):** this run was stopped at step 325 of 2,000 for the independent audit and wrote nothing. The
control is now part of pilot 2 (section 6c), with gated students.

### Corrections after the independent audit (2026-10-19)

The audit recomputed every number of the pilot from its saved outputs and confirmed them all exactly. It also showed
that some statements of section 6a and of the result above do not hold as written. Section 6a stays as it was
pre-registered; the corrections are here.
- **"Identical batches" holds only up to step 400.** The crop positions, kinds, flips and dropout draws stay in step,
  and the students start from B's weights. From the first hard-negative mining on, though, each arm mines its own
  false alarms: the pools held 78 (`student`), 38 (`student_tmask`) and 51 (`student_ctrl`) examples at step 400. The
  contrasts therefore measure a teacher's whole effect under one common adaptive procedure, not the inputs alone.
- **`student − student_tmask` is not a lower bound.** The masked teacher keeps weights learned with the privileged
  cameras and can fail without them: seed 31's teacher fell from 0.691 with all four cameras to 0.109 on RGB +
  LWIR alone. Masking can therefore shrink or inflate the contrast. Its seeds give +0.018, +0.291 and −0.018, so
  seed 31 carries the positive mean.
- **Neither control isolates the privileged cameras.** `student_ctrl`'s teacher also differs in network width and
  subset training. The capacity-matched blind teacher of pilot 2 is the control that does.
- **One scene sets the strict threshold.** At 0.01 false alarms per frame, 2 false alarms are allowed among the 220
  target-free frames, and those 220 frames are four noise draws of each of 55 scenes. In 12 of 18 evaluations, the
  false alarms that set the threshold came from one cloudy daytime scene (`s000046`). Without it, `student −
  rgb_lwir` is +0.025 instead of −0.073, and `teacher − rgb_lwir` +0.037 instead of −0.103.
- **The intervals hold the three trained seeds fixed.** Including seed-to-seed variation (t, 2 degrees of freedom),
  `student − student_ctrl` is +0.097 [−0.130, +0.324]. Between seeds, detection at 0.01 has SD 0.056 for
  `rgb_lwir` and 0.25–0.29 for the distilled students.
- **The thresholds frozen on the 9 validation scenes are not calibrated.** On test they give 1.4–10 % false alarms
  per frame for the two-sensor models and up to 71 % for the teacher. Validation FROC does not rank the runs as test
  does (Spearman −0.28 over the 15 runs).
- **AP as reported uses the 55 frames with a target only**, so it never sees the target-free frames.

### Hard cases of the standard detector (pilot 1; analysis 2026-10-19, `tools/privileged_hardcases.py`)

**The question.** On the scenes where `rgb_lwir` fails, does the privileged student do better? The plan was written
before computing (`docs/reference/privileged/data/hardcases_plan.md`). Selecting B's own misses and comparing with
that same B favours any other model (regression to the mean). So the primary hard set is cross-fitted: for each seed,
the scenes B misses in both other seeds, 18–21 per seed. Results: `docs/reference/privileged/data/hardcases_pilot1.json`.
- **The four-sensor teacher does better exactly where B fails.** On the cross-fitted hard set its JAFROC is +0.116
  over B's (seeds +0.124, +0.110, +0.114; scene interval [−0.033, +0.276]). It detects them at less than half B's
  false-alarm cost (log10 −0.37), and at the scene-level point it finds +0.18 more. The privileged cameras do carry
  signal on the standard detector's hard cases.
- **The student keeps little of it.**
  - `student − rgb_lwir` on the hard set: +0.014 at 0.01 false alarms per frame, +0.019 JAFROC. Not shown.
  - `student − student_ctrl`: positive in all seeds at 0.05 (+0.048, +0.050, +0.056) and at the scene-level point
    (+0.048, +0.100, +0.056), but every interval includes 0.
- **What the student does gain is on large, close targets.**
  - Targets of 32 px and more (19 scenes): 0.72 for the student against 0.61 for `rgb_lwir` and 0.47 for
    `student_ctrl` at 0.01 (0.86 / 0.77 / 0.74 at 0.05).
  - Several large night targets that `rgb_lwir` scores just below its threshold score well above it with the student
    (`s000005`: 0.48 / 0.33 / 0.42 → 0.57 / 0.65 / 0.62).
  - The masked student shows the same gain and `student_ctrl` does not. So it comes from what the four-sensor teacher
    learned, not from its inputs at distillation time.
  - This is post hoc. Pilot 2 tests it on fresh scenes.
- **What the student loses is small targets and daytime:** 0.28 against 0.43 below 8 px, and 0.45 against 0.62 by
  day. That is where the inherited cloud false alarms raise its threshold.
- **Targets the uncooled LWIR cannot see**, but MWIR or the cooled LWIR can (`s000060`: LWIR d′ 0.4, MWIR 45.8), are
  found by the teacher and by neither student. Their own cameras hold no signal there.
- **The mechanism check:** of the 14 targets the teacher found and `rgb_lwir` missed (in-sample, all seeds), the
  student found 8 and `student_ctrl` 4.

## 6b. The low thermal-signature test (next; proposed 2026-10-18)

**Why.** Electric propulsion removes the hot engine and exhaust. In this project's physics the engine is a large part
of the thermal signal (`docs/engine_signature_physics.md`, section 11):
- by day it is 0.5–4× the airframe's LWIR contrast intensity against an ambient-temperature background, by aspect
  and power setting;
- at night it is 5–20×;
- at 2–8 px the engine can be most of the thermal signature.

**Which camera matters then.**
- *Against a clear sky* the airframe keeps a contrast because the LWIR sky is colder than the air, so a cold-engine
  target still shows in uncooled LWIR. Against warmer backgrounds it fades.
- *The MWIR loses the most:* its signal is the hot parts and the plume.
- *The cooled LWIR's lower noise* (about 1.5× in this rig) is the privileged camera that should matter most for a
  low-heat target.

**The test.** The same scenes with the engine's excess temperatures set to 0: a zero-engine-heat bound. A real
electric drone, with its warm motors and battery, lies between this bound and today's engine.

**What it needs:**
1. A signatures overlay (`configs/m1_engine_off.yaml`): the M1 overlay plus one engine profile, `engine_off`, with
   every engine part at 0 K excess and weight 1. It is a variant of this target's engine heat, not the REB 30 of
   `../fyp`.
2. No exhaust plume and no RGB glow when the engine is off. `render/plume_layer.jets` looks up per-setting constants,
   so it must return no jets for `engine_off`. About half a day with tests.
3. A render of the same 126 scene seeds with the overlay, into a separate dataset folder: about 2 h.
4. The census (d′ by camera) and `tools/privileged_ablation.py` on those scenes: about 6 h.

**Decision rule.** The same as section 6a, plus one comparison: the primary contrast on the zero-engine-heat scenes
against the hot-engine scenes. The claim that the method helps especially for low-heat targets needs the contrast to
be larger there.

## 6c. Pilot 2: target-gated distillation, each sensor's contribution, low thermal signatures (set up 2026-10-19 before any pilot-2 result)

**Why.** Pilot 1 and its independent audit (2026-10-18) found:
- *Inherited false alarms:* the distilled students took over the teachers' false alarms. The response distillation
  weights the cells the teacher is confident about, so they lost to the standard detector at a strict false-alarm rate
  while their target ranking matched it.
- *Weak controls:* the masked control failed in one seed (the teacher on RGB + LWIR alone fell to 0.109), and neither
  control isolated privileged information.
- *Seed variance* dominated the uncertainty.
- *A reused test cohort:* the test cohort had been used in development.

**What changes:**
- *Target-gated distillation* (`models/cnn_detector.kd_gate`, `m1_det2.KD_TARGET_ONLY`). Every pilot-2 student is
  distilled only near the labelled targets, in both the response and the feature distillation, so a teacher false
  alarm away from the labels cannot be taught. Nothing else in the student's training changes.
- *One teacher per sensor combination, capacity-matched.* The combinations are: base (RGB + uncooled LWIR), base +
  cooled LWIR, base + MWIR, and all four.
  - All four use the same four-input network, the same subset training, steps and settings; there is no per-arm tuning.
  - A camera a teacher lacks is always blank (`m1_det2.BLANK_BANDS`).
  - Each teacher has a gated student, fed the cameras that teacher was trained with.
  - Also trained: the gated student of B (ordinary distillation); B itself is the supervised reference.
  - Names: `teacher_blind`, `teacher_cool`, `teacher_mwir`, `teacher`, `student_blind_g`, `student_cool_g`,
    `student_mwir_g`, `student_g`, `student_ctrl_g`, `rgb_lwir`.
- *Five seeds:* 29, 31, 37, 41 and 43. For seeds 29–37, B and the four-sensor teacher are pilot 1's.
- *Three test cohorts:*
  - (a) pilot 1's 55 test scenes: secondary, because they were used in development;
  - (b) the same 55 scenes re-rendered with a low thermal signature and with an electric-like signature: a robustness
    test, since every model is trained on today's engine only;
  - (c) a fresh cohort: rendered population scenes 180–299 outside M1's sealed test split, never looked at. This is
    the confirmatory cohort.
- *The procedure, stated exactly.* Every arm follows the same adaptive procedure. Batches are identical up to the
  first hard-negative mining (step 400); after it, each arm mines its own false alarms. A contrast therefore measures
  a teacher's whole effect under a common procedure, not inputs alone.

**The thermal arms** (fixed in their overlay files before any is rendered). Both keep the scene, camera, background,
noise draws and the current engine's geometry, and change only the target's propulsion heat:
- *Low thermal signature* (`configs/m1_low_heat.yaml`): every engine part at the air temperature and no exhaust plume.
  It is a bound below any powered propulsion.
- *Electric-like* (`configs/m1_electric_like.yaml`): the L550E's own cooling surfaces carry an electric drive's loss
  heat at the same shaft power, instead of the combustion engine's cooling load. There is no exhaust gas and no plume.
  The excess temperatures come from the engine heat model's own balance with a sourced motor efficiency
  (`thermal/engine_heat.py`). It is a variant of this engine's heat, not the REB 30 of `../fyp`.

**Endpoints, fixed now:**
- *Primary:* `student_g − rgb_lwir` in detection at 0.01 false alarms per target-free frame (threshold from the
  cohort's own target-free frames), on the fresh cohort (c). It is the mean over the five seeds, with a hierarchical
  bootstrap 95 % interval that resamples seeds and scenes.
- *Co-primary (privileged information):* `student_g − student_blind_g`. Same student procedure and same teacher
  network; the teachers differ only in whether the cooled LWIR and MWIR were present.
- *Secondary:*
  - each extra camera's contribution: `student_cool_g − student_blind_g`, `student_mwir_g − student_blind_g`, and their
    interaction; the same for the teachers;
  - `student_ctrl_g − rgb_lwir`, `student_g − student_ctrl_g`, and `student_g − student` (the gating, seeds 29–37);
  - FROC; detection at the threshold frozen on validation, together with the false-alarm rate it gives; the
    scene-level operating point (at most one target-free scene with a false alarm);
  - the hard-case analysis of `docs/reference/privileged/data/hardcases_plan.md` on each cohort;
  - the primary and co-primary contrasts on the two thermal arms, against the standard scenes;
  - targets of 32 px and more: a hypothesis taken from pilot 1's test cohort, so it is tested on cohort (c) only.

**Decision rule, fixed now** (for the primary and co-primary contrasts):
- **Shown:** the hierarchical 95 % interval excludes 0, and at least 4 of the 5 seeds are positive.
- **Suggestive:** the mean is positive and at least 80 % of the hierarchical resamples are positive.
- **Not shown:** otherwise.

Every seed is reported; none is left out.

**Expected beforehand:**
- If the gate removes the inherited false alarms, `student_g` should at least match B at the strict rate.
- The privileged share `student_g − student_blind_g` is expected to be small. Pilot 1's checks placed the students'
  gains in the teacher's training on large, close targets rather than in signal the student's own cameras lack.
- The cooled LWIR, the privileged camera with less noise, should matter most on the low-signature arms.
- An interim summary after seeds 29–37 on cohort (a) may be reported, labelled interim. The decision is made on the
  full design only.

## 6d. Amendment to pilot 2: the smallest gain that matters, and enough seeds to settle it (2026-10-19, before any pilot-2 result)

**Why.** With five seeds and about 84 fresh scenes, pilot 2 as set up in section 6c can resolve a gain of roughly 10–15
points of detection at the strict false-alarm rate. The literature leads us to expect 1–4. The user therefore fixed the
smallest gain that matters for the thesis, and chose to train on a cloud GPU so that enough seeds can be run.

**Margin.** Δ = 0.03: three percentage points of detection at 0.01 false alarms per target-free frame.

**Final decision for the primary and co-primary contrasts.** This replaces section 6c's grades for the final analysis;
those grades are still reported. [L, U] is the hierarchical 95 % interval (seeds and scenes).
- **A gain of at least 3 points shown:** L ≥ +0.03.
- **A gain shown, smaller than 3 points or not shown to reach them:** 0 < L < +0.03.
- **No gain of 3 points or more:** U < +0.03. This is the definite negative answer (one-sided, 2.5 %).
- **Inconclusive:** otherwise.

**How many seeds.**
- The interim estimates the variance components of detection at 0.01: between seeds and between scenes, pooled over
  the primary arms. It does not use the contrasts' means.
- The final number of seeds n is the smallest with an expected half-width of the hierarchical interval ≤ 0.025 on the
  final cohort, with n between 20 and 40.
- Seeds 101 to 100 + n are trained on one cloud GPU with the same code and settings; only the device differs.
- The laptop's seeds (29–43) are the interim. They are reported, not pooled, so that one kind of hardware carries the
  decision.
- Arms on the GPU:
  - every seed: `rgb_lwir`, `teacher`, `teacher_blind`, `student_g`, `student_blind_g`, `student_ctrl_g`;
  - the first 15 seeds: also `teacher_cool`, `teacher_mwir`, `student_cool_g`, `student_mwir_g`.

**Final cohort.** Every rendered population scene 180–599 outside M1's sealed test split, about 290 scenes. This
replaces 180–299 as the primary cohort. The 55 scenes of pilot 1 and the two thermal arms of them stay secondary.

**Unchanged:** everything else in section 6c.

## 7. Limits of this analysis

- **The census measures the model, not the world.** It uses M1's scenes and camera rig, the band physics of
  `docs/thermal_bands.md`, the cameras' NETD and residual-pattern ranges, and Lahore station hours. How much better a
  real cooled 7.7–9.3 µm camera at the same field of view sees an airborne target against this sky than an uncooled
  8–14 µm one (1.23× here) is the first thing to check with real cameras.
- **d′ is an ideal-observer measure.** It assumes the target's shape is known and the noise roughly Gaussian, with one
  noise draw per scene. The D/F/N thresholds are definitions.
- **The working point is rough.** It comes from the noise verification's small models (62 training scenes, 2,000
  steps): 55 test targets and two seeds. M1's models should sit closer to the ideal detector. M1's evaluation now
  reports the same comparison (`tools/m1_det2_eval.regimes`).
- **The literature differs from this case.** Its gains are on real data, other tasks and other sensors. None tests a
  cooled LWIR or MWIR teacher for an uncooled LWIR + RGB student.
- **Pilot 1 of this design is limited** (section 6a and the corrections above): 55 test scenes, three seeds,
  a strict threshold set by two false alarms, and a cohort used in development. Pilot 2 (section 6c) adds seeds,
  capacity-matched controls, the thermal arms and a fresh cohort.
- **The earlier pilot is not evidence for this design.** It is the earlier REB 30 pilot (`../fyp`): a different, compact detector, SWIR as the
  privileged sensor, one seed and 27 test scenes.
