# Camera models: the four sensors and the camera rig

The project's sensors are RGB, uncooled LWIR, cooled LWIR and MWIR; SWIR is deferred (CLAUDE.md, sensor set rule).
The privileged, training-only teacher sees all four; the deployable student sees RGB and uncooled LWIR. This document
explains how each camera is modelled, where every number comes from, and how the cameras are emulated on the data
pipeline's output grid.

| What | Where |
| --- | --- |
| Camera classes (ranges, sources, grades) | `configs/sensors.yaml` `cameras` |
| Sampling, field of view, rig, cooled-detector physics | `render/cameras.py` |
| Emulation of each camera on the output grid | `data/camera_emulation.py`, used by `data/dataset.py` |
| Thermal noise model (any band) | `noise/lwir_fpn.py` (with `thermal/planck_lut.band_lut`) |
| Manufacturer documents, quotes, numbers | `docs/sources/cameras/` (`readings.json`, `pages/`) |
| Tests | `tests/test_cameras.py`; quotes re-checked by `python tools/source_measurements.py` |

## 1. The classes and their sources

The ranges are the user's camera table (2026-10-12), traced to manufacturer documents. Every number below is read off
a saved document with its quote (`readings.json`, 82 readings from 17 documents); where the table had no traceable
source the value is graded C and says so.

| Camera | Detector and band | Formats (pixel pitch) | Lens | Sensitivity | Frame rate |
| --- | --- | --- | --- | --- | --- |
| **LWIR uncooled** (student; sets the grid) | VOx microbolometer, 8–14 µm (Boson, Dione 1280; Tau 2 7.5–13.5) | 640×512 at 12 µm (Boson), 10 µm (Tenum 640), 17 µm (Tau 2, legacy); 320×256 at 12 µm (Boson); 1280×1024 at 12 µm (Dione 1280) | fixed, 4.9–100 mm (Boson 640 lenses 4.9–72.8 mm; Tau 2 7.5–100 mm), field of view kept within 5–46° | NETD 25–50 mK (sources span 20–60 mK by grade); temporal row and column noise ≤ 0.35 × NETD each (Boson Table 13) | 9, 25, 30, 60 Hz (Boson 8.6/60; Tau 2 <9/25/30) |
| **LWIR cooled** (teacher) | T2SL or MCT, 7.7–9.3 µm (Scorpio LW; Pelican-D LW cut-off 9.3 µm) | 640×512 at 15 µm (both) | zoom 21.7–550 mm = 25–1° on 640×15 µm (**C**: no cooled-LWIR zoom datasheet found) | NETD 10–25 mK (Scorpio LW 22 mK at 210 Hz, 10 mK with 4-frame accumulation; Pelican-D LW 15 mK with 8-frame averaging); residual fixed pattern ≤ 34 mK | 30–100 Hz (detectors to 210/360 Hz) |
| **MWIR** (teacher) | MCT 3.7–4.8 µm (Scorpio MW, Daphnis-HD); InSb 3.6–4.9 µm (Pelican-D); HOT XBn 3.6–4.2 µm (Mini Blackbird) | 640×512 at 15 µm (Scorpio MW, Pelican-D); 1280×1024 at 10 µm (Mini Blackbird); 1280×720 at 10 µm (Daphnis-HD) | zoom 15–300 mm (SupIR MWIR zoom: 36.4–1.8° on 640×15 µm) | NETD 12–25 mK (sources 18–25 mK; 12 mK lower edge **C**); residual fixed pattern ≤ 14–18 mK | 25–120 Hz (detectors 85–300 Hz) |
| **RGB** (student and teacher) | 1/2.8-inch colour CMOS (IMX327) | 1920×1080 at 2.9 µm; 720p output of the zoom block (whole sensor scaled) | zoom 4.3–129 mm (FCB-EV7520: 63.7–2.3°) | read noise 1–3 e⁻ (**C**: Sony publishes sensitivity, not read noise) | 25–60 fps (IMX327 up to 60) |

Weights of formats, detectors and frame rates are coverage choices (C). The MWIR detector sets the band: the HOT blue
band ends at 4.2 µm, below the CO2 4.3 µm absorption band, which matters for the MWIR render and the exhaust plume.

**Field of view from format and lens.** HFOV = 2 atan(N p / 2 f) (pinhole). Sampling the focal length and computing
the field of view keeps resolution, pitch and field of view consistent (no camera that does not exist). The formula
reproduces the datasheets within their lens distortion: Boson 640 with 18 mm 24.1° (table 24.3°), Boson 320 with
13.8 mm 15.8° (16.0°), FCB-EV7520 65.8–2.5° (63.7–2.3°), SupIR zoom on 640×15 µm 35.5–1.83° (36.4–1.8°).

## 2. The rig: one mount, one output grid

The detector is trained and evaluated on one 640×512 frame per band (`configs/sensors.yaml` image). The rig defines
that frame and how each camera maps onto it:

* **The student's uncooled LWIR camera sets the grid.** Its format and lens are drawn (focal length log-uniform over
  the lens range, redrawn until the field of view is 5–46°); the 640×512 frame spans its field of view. A 640-wide
  core maps one pixel to one output pixel; a 320×256 core's pixel spans 2×2 output pixels (r = 2); a 1280×1024 core is
  binned 2×2 (r = 0.5).
* **The zoom cameras share the mount** and zoom to the narrowest setting that still covers the whole grid, within
  their focal range. The RGB zoom always covers it (63.7° wide end against the grid's ≤ 46°). The MWIR zoom (36.4° on
  640×15 µm) cannot cover the widest grids, and the cooled-LWIR zoom (25°) cannot cover grids wider than 25°. There
  they see only the centre (coverage < 1, about 8 % and 32 % of scenes respectively).
* **Pixel scale.** r = (p / f) / (grid pixel angle). RGB is about 2× finer than the grid (r ≈ 0.47 at 1080p). MWIR
  and cooled LWIR match it (r = 1) when their zoom covers the grid exactly, and are finer when HD or zoom-limited.

When the scene sampler cannot place a target in its size bin with the drawn camera (`render/sampling.sample_scene`),
it falls back to the student's longest lens, the narrowest field the class allows. That replaces the earlier fixed
narrow camera. The M1 pilot keeps its single fixed camera (`configs/m1.yaml` `cameras: null`).

## 3. Emulating each camera on the grid

Each band is rendered once on the grid. The loader turns the composited grid image into what that camera records
(`data/camera_emulation.py`):

1. **Optics.** The camera's PSF is given in its own pixels (`psf_sigma_px`) and spans σ × r output pixels; the
   render already applied the floor σ0.
2. **To the camera's pixels.** The covered part of the grid is taken to the camera's own pixel grid: area averaging
   for a coarser camera (its pixel aperture), linear interpolation for a finer one (the render holds no finer detail).
3. **Sensor model on the camera's pixels**, with the scene's own draws from the rig:
   * RGB: shot, read and quantisation noise with auto-exposure (`noise/rgb_noise.py`).
   * Thermal cameras: counts, then fixed pattern, temporal row and column noise, NETD and automatic gain control
     (`noise/lwir_fpn.py`), each in its own band's Planck table.
4. **Back to the grid.** Area averaging for a finer camera (binning: temporal noise falls as r, test), linear
   interpolation for a coarser one (correlated noise, as on a real upscaled frame). Outside the covered region the
   camera sees nothing: zero, with a coverage mask.

**Cooled-detector physics (C+).**
* *Integration time:* the well fills in t = fill × capacity / (π L_photon / (4F²) × p² × QE). Inputs are the
  datasheets' capacities, fills and f-numbers. QE is >50 % for the Pelican-D LW; the rest is C.
* *Results:* cooled LWIR 0.1–0.7 ms; MWIR from about 3.5 ms at f/2 to the frame period at f/5.5 and with the HOT blue
  band.
* *Residual fixed pattern:* the detectors' residual non-uniformity, < 0.04 % of the dynamic range over 10–90 % well
  fill, is converted to kelvin through the band's relative photon-radiance slope. That gives ≤ 34 mK in 7.7–9.3 µm
  and ≤ 14–18 mK in the MWIR bands.

## 4. Where the cameras are used

* **Renders in every band of the rig** (2026-10-13, `docs/thermal_bands.md`): the 7.7-9.3 um cooled-LWIR pass with its
  own band table, and the MWIR pass with reflected sunlight, the MWIR sky and layered atmosphere and the exhaust plume
  (CO2 4.3 um). The loader emulates each camera on the grid from these renders (`data/camera_emulation.py`); the old
  loader-only cooled stand-in of `tools/m1_det2.py` (a noise swap on the 8-14 um render) is gone.
* **Teacher and student** (teacher: all four; student: RGB + uncooled LWIR): wired in the full detector
  (`train/engine.py`, `tests/test_teacher_student.py`) and in the M1 preliminary evaluation (`tools/m1_det2.py`,
  `docs/m1_v2.md`), where each zoom camera's per-pixel coverage mask replaces the band-present flag.

## 5. Grades and how to validate

| Item | Grade | Validation |
| --- | --- | --- |
| Formats, pitches, bands, NETD figures, frame rates, lens and zoom ranges with a quote | B (S1 for the product, S2 for the class) | `readings.json` and the saved documents |
| Uncooled NETD range 25–50 mK (inside the sources' 20–60 mK) | B | the user's table, inside the datasheet span |
| Cooled-LWIR zoom 1–25° | C | a cooled LWIR continuous-zoom lens datasheet |
| MWIR NETD lower edge 12 mK | C | an MWIR detector datasheet quoting it |
| RGB read noise 1–3 e⁻ | C | a photon-transfer measurement of an IMX327-class camera |
| Format, detector and frame-rate weights | C | coverage choices |
| Integration times, residual fixed pattern in kelvin | C+ | physics from datasheet capacities, f-numbers and non-uniformity |
| Rig geometry (co-boresighted, zoom to cover the grid) | C | a description of the target sensor installation |
