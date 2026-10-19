# Camera models: from rendered radiance to the video a camera delivers (2026-10-17)

The renderer produces noise-free radiance. The data loader turns it into each camera's video frame on the camera's own
pixels (`data/camera_emulation.py`): the **sensor** (photons or heat to counts, with its noise), then the **camera's own
processing** (defect replacement, filters, gain control, tone mapping). The second half was missing before 2026-10-17,
so the networks saw sensor data that no real camera outputs. Code: `noise/isp.py`, `noise/rgb_noise.apply_camera`,
`noise/lwir_fpn.apply_camera`; parameters: `configs/sensors.yaml` → `camera_pipelines`; readings with quotes:
`docs/sources/cameras/readings.json` (manufacturers) and `docs/sources/sensor_noise/readings.json` (papers).

The settings are loader-only: tuning them never invalidates rendered scenes (`data/provenance.LOADER_ONLY_SENSORS`).
Every training and test frame draws its own setting from the ranges, so the networks learn across the documented
range of cameras and settings instead of one assumed camera.

**Status after the verification (2026-10-17, "Result" below): switched off for training**
(`camera_pipelines.enabled: false`). The loader again uses the previous sensor models (`rgb_noise.apply`,
`lwir_fpn.apply`). Training on these camera models did not pass the decision rule fixed beforehand, every detector was
weakest on their frames, and the test was too small to rank the training noises.

## Why this matters, and what the evidence says

- **Physically modelled camera effects close part of the synthetic-to-real gap.** Carlson et al. (ECCV Workshops 2018)
  trained a detector on synthetic images with physically based sensor effects:
  - chromatic aberration, blur, exposure, Poisson-Gaussian noise and colour shift;
  - real-data AP rose by **+4.3 to +7.3**, and 2,975 augmented images beat 20,000 plain ones;
  - plain additive Gaussian noise did not do this (**−1.3 / +2.2**), while physical sensor noise alone gave **+4.0 / +1.3** (Tables 1–3);
  - the gains were largest on cars small in the image.
- **A full physical noise model is as good as real training data.** Wei et al. (CVPR 2020, Table 1) built training data
  from photon shot noise (Poisson), long-tailed read noise (Tukey lambda), row noise and quantisation:
  - on real low-light images this reached **39.27 / 37.13 / 36.30 dB**, against 38.60 / 37.08 / 36.29 dB for training on real paired data;
  - a Gaussian-only model reached 36.10 / 31.87 / 30.99 dB, and every added term helped;
  - in their journal version, objects found in the denoised images rose to 3.43 per image (3.23 with real training data; TPAMI 2021, Table 9).
- **Thermal fixed pattern learned on simulation works on real cameras.** He et al. (Applied Optics 2018) simulated
  column noise as a per-column cubic of the signal (degree 3 best), and a network trained only on that simulation
  corrected real infrared images.
- **Direct thermal evidence for detection is thinner.** No study found here measures each thermal camera effect's gain
  for detection. The thermal models therefore follow the manufacturers' own description of their processing (S1) and
  the physics papers (S2), extending the RGB evidence above. The real-data check below is what will confirm them.

## How this choice compares with the best alternatives

The strongest noise models in the literature need frames from the actual camera. Without that camera, the best
available choice is the physics model: it matches the real-frame methods within ~0.1 dB in low light. Camera effects
have to be randomized over documented ranges, and wrong ranges can hurt (last-but-one row). All numbers below are
read off the saved papers (`sensor_noise/readings.json`).

| Approach | Needs | Published comparison | Use here |
|---|---|---|---|
| Additive Gaussian noise | nothing | Detection on real data −1.3 / +2.2 AP (Carlson 2018, Table 2) | baseline only (`gaussian` below) |
| Poisson-Gaussian / heteroscedastic (Foi 2008, Brooks 2019) | two parameters | Real low light: 46.96 / 43.84 dB at ×100 / ×200, against 47.77 / 45.46 for the physics model (Zhang 2021, Table 1) | the previous RGB model (`previous`); **the training default again** since the verification |
| Physics noise formation: Poisson, Tukey-lambda read, row, quantisation (Wei 2020) | a calibration of the camera for exact values; ranges otherwise | As good as training on real paired data (Wei 2020, Table 1); within 0.13 dB of real-frame sampling in low light (Zhang 2021) | implemented (`physics`, `camera`); off for training until a better-powered check or real camera data supports it |
| Sampling the camera's real dark frames (Zhang 2021) | dark frames from the actual camera | Best: 53.92 / 47.90 / 45.47 dB against the physics model's 53.43 / 47.77 / 45.46 | next step once the RGB camera is available |
| Learned noise models: Noise Flow 2019, GAN-based | paired real data from the camera | 0.42 nats/pixel better likelihood than the camera's noise functions (Abdelhamed 2019), but "after careful calibration, the physics-based methods outperform the DNN-based methods" (Zhang 2021) | not applicable without camera data; not better than calibrated physics |
| Randomized camera effects (Carlson 2018) | effect models and ranges | +4.3 to +7.3 AP on real data | implemented with the cameras' own documented processing (`camera`); off for training: with histogram AGC the detector fired on clouds (verification) |
| Camera-effect ranges learned from real images (Carlson 2019) | unlabelled real video | +1.7 / +5.4 AP, where hand-chosen ranges gave −2.1 / +4.8 (Table II) | next step with real video: chosen ranges can hurt |
| Learned image translation (CycleGAN, UNIT, MUNIT) | real images | −0.8 to −5.5 AP, worse than no augmentation (Carlson 2019, Table II) | not recommended |
| LWIR: column offsets only (Kuang 2017) | — | A network trained on it "drops significantly" on real column noise (He 2018) | superseded |
| LWIR: per-column polynomial (He 2018), structured smooth offset (Saragadam 2021), the manufacturer's processing chain | — | Trained on the simulation, works on real images (He 2018) | implemented for the thermal cameras; off for training with the rest of the camera models (the previous column, pixel and stripe terms are used) |
| LWIR: sampling real flat-field (shutter) frames | uniform-scene frames from the actual camera | the thermal counterpart of real-frame sampling | next step with the camera |

## RGB zoom camera (`noise/rgb_noise.apply_camera`)

| Stage | Model | Source |
|---|---|---|
| Exposure | Fixed exposure time (it sets motion blur); aperture/ND, then gain up to 50 dB (62 dB in high-sensitivity mode) | FCB-EV7520 manual pp. 10–11 [S1] |
| Colour sampling | Bayer RGGB mosaic | IMX327 product information [S1] |
| Photon shot noise | Poisson on photo-electrons | Wei et al. 2020 eq. 2 [S2] |
| Read noise | Tukey-lambda, λ from 0.14 (Gaussian) down to −0.15 (long tail) | Wei et al. 2020 eq. 4 [S2]; λ range [C] |
| Row (banding) noise | Gaussian offset per row, 0–25 % of read noise | Wei et al. 2020 [S2]; size [C] |
| ADC | 10 or 12 bits, saturation at full well | IMX327 [S1] |
| Demosaicking | Bilinear | Brooks et al. 2019 [S2] |
| White balance, lateral chromatic aberration | Channel gain error ±4 %; red/blue scaled 0–0.15 % about the centre | Carlson et al. 2018 model [S2]; sizes [C] |
| Noise reduction | Off or levels 1–5; acts only as gain rises, none in bright light (Gaussian σ up to 1.3 camera px) | FCB manual p. 11 [S1]; σ [C] |
| Edge enhancement | 16 steps (unsharp mask up to 0.8) | FCB manual [S1]; amount [C] |
| Gamma | Standard (BT.709 curve) or straight (1/2.2) | FCB manual [S1]; curves [C] |

## Uncooled LWIR camera (`noise/lwir_fpn.apply_camera`, uncooled class)

| Stage | Model | Source |
|---|---|---|
| Temporal noise | Scene's NETD; temporal row and column noise up to 35 % of NETD | Boson Table 13 [S1] |
| Residual fixed pattern | Pixel and column offsets, column gain, column drift (existing model) | He et al. 2018 [S2]; sizes [C] |
| Column non-linearity | Per-column quadratic and cubic in the signal, up to 0.2 % of the frame's span | He et al. 2018 (degree 3) [S2]; size [C] |
| Low-frequency shading | Side-to-side gradient plus circularly symmetric term, up to 0.3 K | Boson LFSR description [S1], Saragadam et al. 2021 [S2]; size [C] |
| Image non-uniformities | 0–6 blobs of radius < 10 px and 0–2 larger, up to 0.05 K | Boson sec. 11.4 types A/B [S1]; counts, size [C] |
| Camera filters | Column, row, spatial and shading filters (on by default) remove 0–80 % of the residual pattern | Boson sec. 5.4 [S1]; share [C] |
| Defects | Up to 1.5 % defective pixels, ≤ 2 bad lines, replaced from neighbours; 0–6 missed per frame | Boson Table 15, sec. 11.3 [S1]; missed [C] |
| AGC | Histogram-based (85 %): information-based weighting, plateau 3–15 %, tail 0–2 %, max gain 0.7–3, linear 0–40 %, ACE 0.85–1.15, DDE 0.8–1.3; else linear | Boson sec. 6.8, defaults 7 %, 0 %, 1.38, 20 %, 0.97, 0.95 [S1]; Tau 2 [S1]; spread [C] |

## Cooled LWIR and MWIR cameras (`noise/lwir_fpn.apply_camera`, cooled classes)

| Stage | Model | Source |
|---|---|---|
| Temporal noise, residual pattern | Scene's NETD; residual non-uniformity from the 0.04 % RNU bound | Lynred / SCD datasheets [S1] (existing rig model) |
| Defects | 0.05–0.5 % (cooled LWIR), 0.05–0.2 % (MWIR), replaced in the core; 0–3 missed per frame | Operability 99.5–99.8 % [S1]; replacement listed for the Mini Blackbird [S1] |
| Blinking (RTS) pixels | Up to 0.002 % of pixels jump 2–6 NETD | "only a few blinking pixels", caught by defect detection (Ribet-Mohamed et al. 2019) [S2]; share [C] |
| Shading, non-uniformities | Up to 0.1 K and 0.02 K (drift after NUC, narcissus) | [C] |
| Core processing | Spatial/temporal noise reduction (0–60 % of the residual pattern); AGC or dynamic-range compression (histogram 70 %) | Mini Blackbird datasheet [S1]; ranges [C] |

## Verification on our task (`tools/noise_ablation.py`, set up 2026-10-17 before any result)

Does the camera model hurt the detector, and what does it gain? The same detector (`rgb_lwir`, the student's two
cameras; identical seeds, crops and 2,000 steps) is trained on four noise models and tested on each of them. It is also
tested on a "foreign" camera that none of them models.

| Variant | Training and test frames |
|---|---|
| `gaussian` | Additive white Gaussian noise at the sensor's own level, linear display stretch: plain augmentation |
| `previous` | The models before 2026-10-17: heteroscedastic shot + read noise (RGB); NETD, column/pixel pattern, stripes (LWIR); linear stretch |
| `physics` | The new sensor physics without the camera's processing (no noise reduction, edge enhancement, filters; linear stretch) |
| `camera` | The new default: sensor physics + the camera's processing |
| `foreign` (test only) | The previous sensor, then blur, a display gamma and JPEG compression: processing no variant has |

**Scenes.** Only scenes outside M1's sealed test split, so this choice cannot leak into the M1 evaluation:
- test: the population's validation-split scenes plus every third training-split scene;
- validation: every eighth of the rest;
- training: the remainder.

**Decision rule, fixed before the results.**
1. *No deterioration:* on each other variant's own test frames, the `camera`-trained model must not be significantly
   worse than the model trained on that variant (paired scene bootstrap, 95 %).
2. *Gain:* its FROC on the `camera` and `foreign` frames, against the others.
3. *What suits best:* the training noise with the highest worst-case FROC over the five test sets (the camera that
   will actually be met is unknown).

Two training seeds show how much of a difference is only training randomness.

Limits:
- every test camera here is still a model, so this shows robustness across camera models, not the gain on real
  images;
- the published real-data gains are in the table above.

### Result (2026-10-17)

Files: `out/m1/noise_ablation/results.md` and `results.json`; per-scene scores in `scores.json`. The four-model run as
fixed beforehand is in `results_four_models.*`.

Setup: 62 training, 9 validation and 55 test scenes; two seeds; 2,000 steps. Each test set has 220 target-free
frames.

Seed-averaged FROC (rows: trained on; columns: tested on). `mixed` was added after the first results: every training
frame comes from one of the four models' caches at random.

| Trained on | gaussian | previous | physics | camera | foreign | mean | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian | 0.395 | 0.425 | 0.280 | 0.243 | 0.389 | 0.346 | 0.243 |
| previous | 0.364 | 0.418 | 0.309 | 0.230 | 0.309 | 0.326 | 0.230 |
| physics | 0.243 | 0.355 | 0.275 | 0.198 | 0.339 | 0.282 | 0.198 |
| camera | 0.159 | 0.195 | 0.116 | 0.134 | 0.150 | 0.151 | 0.116 |
| mixed | 0.284 | 0.289 | 0.180 | 0.164 | 0.355 | 0.254 | 0.164 |

**The decision rule, as fixed:**
1. *No deterioration:* not shown. On each other model's own frames, the camera-trained model is below the model
   trained on them:
   - pooled over seeds: −0.236 (gaussian), −0.223 (previous), −0.159 (physics);
   - the same sign in both seeds;
   - none significant at 95 % with 55 scenes. The test does not reject, but it does not show the absence of a loss
     either.
2. *Gain:* none. On camera frames the difference is −0.06 to −0.11; on foreign frames −0.16 to −0.24 (not
   significant).
3. *Best worst case:* gaussian 0.243 (highest in 45 % of scene resamples), previous 0.230 (33 %), physics 0.198
   (11 %), camera 0.116 (10 %), mixed 0.164 (1 %).

**Why** (diagnosis of the saved scores and frames):
- **Targets:** the camera-trained model scores targets like the others (median 0.52; 82–85 % above 0.3).
- **False alarms:** it scores cloud banks in the uncooled LWIR as large targets, 0.77–0.88 (boxes of 120–280 px, the
  4× pyramid level).
  - This happens in two cloudy test scenes: s000046 (day, haze, 4 km, 5° field), in both seeds, and s000013 (night,
    haze, 2 km, 28° field), in seed 31.
  - It holds in every noise draw, and on its own frames and the previous ones alike.
  - The other models' strongest false alarm on any target-free frame scores 0.41–0.67.
  - The budgets allow 1–11 false alarms in 220 frames, so these 4–8 frames set its thresholds.
- **Without those two scenes** (a check after the fact, not the result): camera mean 0.538 and worst case 0.399,
  against 0.415–0.518 and 0.321–0.342 for the others.
- **Likely mechanism (a hypothesis, not tested on its own):**
  - Histogram AGC, the Boson's default, spends the grey levels on the dominant background: "In classic HEQ, the
    background dominates the histogram" (Boson manual, p. 53).
  - It maps a cloud bank and a large warm target to similar output, which removes the intensity difference a linear
    stretch keeps.
  - With 62 training scenes, the model learns "large bright blob".
- **Every model is weakest on the camera frames:** that column is the lowest or second lowest for all ten trained
  models. The camera's processing makes detection harder for every detector.

**What the test could not settle: the ranking of the training noises.**
- The differences between training noises (up to 0.2) are as large as:
  - the difference between two seeds of one noise (mixed: mean FROC 0.062 and 0.446);
  - the effect of one test scene (leaving one out moves a model's worst case by up to 0.18).
- The main cause is checkpoint selection on 9 validation scenes (9 target frames). The chosen step varies from 750 to
  2,000, and the chosen checkpoint can be poor: mixed, seed 29, was kept at step 1,000 with a full-validation FROC of
  0.097. M1 selects on 60 validation scenes.

**Decision (2026-10-17):**
- Training uses the previous models: `camera_pipelines.enabled: false`. They tie for the best worst case with plain
  Gaussian noise. Gaussian noise is not used because it is the weakest model on real images (Carlson 2018, Table 2;
  Wei 2020, Table 1).
- The camera models stay in the code and in the ablation, as the hardest test condition.
- **Deployment.** The uncooled core can deliver its data before the AGC:
  - the AGC converts "16-bit data to an 8-bit output suitable for display" (Boson manual, p. 27);
  - the output can be tapped pre-AGC, post-AGC or post-zoom (p. 30);
  - readings in `cameras/readings.json`.

  A detector that reads the pre-AGC stream avoids the loss measured here, and its matching training noise is the
  sensor model without AGC.

**To settle it** (proposed; needs the go-ahead for the laptop time):
- The same comparison with fixed-step models: the final step, with no checkpoint selection.
  - Three seeds for `previous`, `camera` and `mixed`: about 2.5 h on the 180 scenes.
  - Or, once the 600 M1 scenes are rendered, on about 400 scenes outside M1's test split: about 5 h, with narrower
    scene-to-scene swings.
- Then real camera frames (below).

## Not modelled yet

- Video compression (H.264/H.265) of the delivered stream; it depends on the installation.
- Temporal filters (3D noise reduction, AGC damping, smart frame averaging): one frame at a time here. On video,
  temporal noise reduction can erase small moving targets; the tracker test (`sequences`) is where it would show.
- The RGB camera's night switch to monochrome near-infrared (auto ICR). The renders are visible-band only.
- Rolling shutter of the RGB sensor.

## How to validate with real cameras

From one camera of each class, record:
- a uniform scene (lens cap or sky): temporal noise, row/column noise spectra, shading;
- a still scene over a few minutes: drift of the residual pattern between corrections;
- typical scenes: histograms of the 8-bit output, which tell the AGC mode and max gain.

Then compare these statistics with frames generated for the same camera settings (`noise/isp.py` exposes each stage).
The ranges marked [C] are the ones to narrow first.
