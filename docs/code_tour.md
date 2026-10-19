# Code tour: where everything lives

A map for the defense's code navigation. Follow one frame from the 3D model to a confirmed track.
File paths are clickable in most editors. Line-level detail is in each file's top docstring.

```
target/ ──> render/ ──> data/ + noise/ + thermal/ ──> models/ ──> tools/m1_det2.py ──> eval/ ──> app/
3D KAL      Blender      composite, atmosphere,       CNN         train / evaluate     metrics,   live
model       layers       camera noise                 detector                        tracker    console
```

## 1. The target: `target/`

| File | What it does | Ask yourself |
|---|---|---|
| `kal_geometry.py` | Builds the KAL airframe mesh from `configs/target.yaml` (span 2.0-2.6 m, sweep, fins) and places the engine from `engine_l550e.py` | Where do the dimensions come from? (photos + ranges, graded in the build spec) |
| `engine_l550e.py` | MD-550 / L550E: published envelope and ratings, the visible parts (finned cylinders, case, carburettors, exhaust stubs, drive flange) | Which numbers are published (B) and which are scaled from a photo (C)? |
| `build_target.py` | Blender: mesh + material slots per band (skin, engine heads, fins, case, carburettors, flange, exhaust stubs, propeller disc) | Why does each engine part get its own slot? Each has its own temperature and emissivity |
| `kal_measure.py` | Checks the mesh against the YAML and measures hot-part visibility by aspect | |
| `kal_hero.py` | Presentation-only model for videos (look of photo P1); **not** used for training | |

## 2. Rendering a scene: `render/`

| File | What it does |
|---|---|
| `sampling.py` | Draws one scene from the configs: camera (S1 640x512, 8.78°), target pose/range/aspect, weather, sun/moon, per-band signatures. `assign_splits` fixes train/val/test per scene |
| `placement.py` | Camera geometry (pinhole matching Blender), projections, apparent size in pixels |
| `signatures.py` | Per-part temperatures (LWIR), reflectances (RGB/SWIR), SWIR self-emission of hot parts; CHT-limit caps (250 °C) |
| `scene.py` | Blender: renders sky/ground and each object as **layers** (radiance + coverage), not finished images |
| `lwir.py`, `rgb.py`, `swir.py` | The three band passes. LWIR is emission (Planck radiance x emissivity + reflected surroundings) |
| `track_sequence.py` | Flight sequences (a moving target and a pan-tilt camera) for the video tests and the app |

Why layers? Haze, blur and sensor noise are added later in Python, so each training epoch can redraw noise without re-rendering.

## 3. From layers to camera frames: `data/`, `thermal/`, `noise/`

| File | What it does |
|---|---|
| `data/composite.py` | `I = I·e^(-kd) + L∞(1 - e^(-kd))`: haze over sky/ground/objects; LWIR water-vapour loss |
| `thermal/atmosphere.py`, `thermal/sky.py`, `thermal/planck_lut.py` | Legacy LWIR water-vapour transmission and Idso (1981) sky (M1), band radiance tables for every thermal camera |
| `thermal/gas_bands.py`, `thermal/atmosphere_bands.py` | SP-3080 narrow-band gas model (H2O, CO2, CO, N2O, CH4) and the layered clear atmosphere shared by the three thermal cameras: sky by elevation, path transmittance, direct sun, MWIR scattered sky (`docs/thermal_bands.md`) |
| `thermal/plume.py`, `render/plume_layer.py` | Exhaust composition, the four jets cut at the propeller plane, their radiance in each thermal band as an object layer |
| `render/thermal.py` | Band-generic thermal pass (LWIR uncooled on the layered atmosphere, LWIR cooled, MWIR with sun lamp, glints and scattering sky) |
| `noise/rgb_noise.py` | Shot + read noise, auto-exposure, gain (Brooks et al. 2019 style) |
| `noise/lwir_fpn.py` | Microbolometer: responsivity, column/pixel non-uniformity, NETD 30-60 mK, AGC to 8-bit. `realistic_stripes` adds column-gain, drift, row and temporal-column noise (v2) |
| `data/dataset.py` | `SceneDataset`: layers -> haze -> blur -> registration jitter -> camera noise. Labels = projected geometry boxes (`gt_objects`) |
| `data/provenance.py` | Render contract hash: refuses to mix scenes rendered with different settings |

## 4. The detector: `models/cnn_detector.py`

| Part | Where | Key idea |
|---|---|---|
| Input preprocessing | `preprocess`, `destripe`, `standardise` | 8-bit video -> stripe filter (thermal) -> per-frame robust z-score (removes camera gain/level) |
| Network | `CNNDetector` | Per-band stems (lossless 2x2 space-to-depth keeps every pixel of a 2-4 px target) -> concatenation fusion + band presence flags -> encoder 4/8/16 -> neck back to stride 4 |
| Outputs | `hm`, `reg` | CenterNet-style: target-centre heatmap + sub-cell offset + log width/height |
| Training targets | `build_targets_dense` | Gaussian heatmap; box supervised over the Gaussian core (v2) |
| Losses | `focal_loss`, `detection_loss_v2`, `giou` | Focal loss for the rare target cells; L1 + GIoU for boxes |
| Distillation | `response_kd`, `feature_kd`, `kd_mask`, `cnr_gap_weight` | Student copies the teacher's heatmap and its features near the target, weighted where the teacher sees much better (CNR gap) |
| Inference | `decode`, `predict`, `nms` | Heatmap peaks -> boxes; 3-level image pyramid for large targets |

`models/pilot.py` is the earlier compact pilot (contrast proposals + MLP; completed M1 evidence).
`models/detector.py`, `distill.py` and `train/engine.py` are the full spec-7.1 detector, planned for M2; the teacher (RGB +
LWIR uncooled + LWIR cooled + MWIR) and the distilled student (RGB + LWIR uncooled) are wired and smoke-tested
(`tests/test_teacher_student.py`), not yet trained on a full dataset.

## 5. The experiment: `tools/m1_det2.py` (+ `tools/m1_det2_eval.py`, `tools/member_job.py`)

The M1 preliminary evaluation on the four sensors (`docs/m1_v2.md`): seven compact-CNN models (teacher on RGB + LWIR
uncooled + LWIR cooled + MWIR; student and control student on RGB + LWIR uncooled; the RGB + LWIR baseline; three
single-camera references), run on one laptop (an optional two-laptop split remains).

| Stage | Function | What to explain |
|---|---|---|
| Render | `render`, `pack_part`, `unpack_part` | Each laptop renders its share of the 600 scenes; part zips carry every file's SHA-256 and the render contract |
| Freeze | `manifest` | Scene lists per split, seeds, sensors, camera coverage, training rules, metric definitions, all written **before** training |
| Cache | `cache`, `_scene_frames` | Camera frames per scene: 4 noise draws + 2 target-free frames for training (`train`: four cameras; `train_solo`: thermal cameras for single-camera models); validation 2 + 8; test 1 + 4 |
| Cameras | `data/dataset._camera_sensor`, `noise/isp.py`, `rgb_noise.apply_camera`, `lwir_fpn.apply_camera` | Each camera's sensor noise and own processing: defects, filters, AGC, noise reduction (`docs/sensor_noise.md`) |
| Coverage | `coverage_boxes`, `cnn_detector.preprocess` / `coverage_masks` | A zoom camera that cannot widen to the grid sees its centre: preprocessed there only; its mask is a network input |
| Batches | `Sampler.batch` | 50 % target crops, 25 % background, 25 % target-free; every pyramid level; mined false alarms; each crop's coverage masks |
| Train | `_train_one` | AdamW, cosine LR, 3000 steps; distillation for the students; validation every 250 steps; resumable `last.pt` |
| Select | `validate`, `froc`, `fa_threshold` | Checkpoint by validation FROC on whole frames; thresholds at 0.01 / 0.05 false alarms per target-free frame |
| Evaluate | `m1_det2_eval.evaluate` | Opens the test scenes once (`test_lock.json`); paired scene bootstrap; teacher gate; coverage slices |
| Video | `m1_det2_eval.sequences`, `StreamCamera` | Fixed camera patterns + fresh temporal noise; confirmed false tracks per minute |
| Jobs | `member_job.py`, `RUN_M1_PRELIMINARY.bat`, `member/*.bat` | One-click job on one laptop (default) or split over two; Blender 4.2.23 fetched and checked; resumable |

## 6. Metrics and tracking: `eval/`

| File | What it does |
|---|---|
| `metrics.py` | AP50 (COCO 101-point), operating points, bootstrap intervals, per-bin rules |
| `nwd.py` | IoU, centre-distance and NWD matching (centre match = small-target convention) |
| `pd_fit.py` | Logistic fit of detection vs log2(size) -> px50 per aspect |
| `tracking.py` | `MultiTracker`: confirm 3 of 5 frames, gate, coast 10 frames |

## 7. The live console: `app/` (`RUN_APP.bat` or `python -m app`)

| File | What it does |
|---|---|
| `engine.py` | `Scenario` (live radiance of a rendered flight), `LiveRun` (camera -> preprocess -> detector -> threshold -> tracker, two threads), `evidence_run` (rebuilds a test frame and checks the stored evaluation) |
| `server.py` | Flask JSON API on 127.0.0.1 |
| `static/` | The UI (plain HTML/CSS/JS): camera panels, boxes, magnifier, confidence timeline, controls |

## 8. Configuration and rules

- `configs/*.yaml`: target, render, sensors, signatures, train. `configs/m1.yaml` is the M1 overlay (open sky,
  day/night, narrow camera, clear air).
- `thermal/engine_heat.py` (+ `docs/engine_signature_physics.md`): how every engine temperature in `signatures.yaml`
  is derived from heat transfer; `python thermal/engine_heat.py` prints the derivation. Its sourced anchors (Limbach
  SFC chart, Rotax CHT/EGT limits, AFIT energy balance, NACA 488 fin cooling, FLIR emissivity table, ASTM G173/E490
  and IAU solar values) are listed in `docs/sources/README.md`. `tools/engine_report.py` saves every derived engine
  number to `docs/reference/engine/data/`; `tools/check_glow_glint_blender.py` checks the glow and glint shaders.
- Flight settings: 44 m/s default, 39-56 m/s, min AGL 20 m (`configs/render.yaml`).
- `tests/`: 200+ checks (`python -m pytest tests -q`).
