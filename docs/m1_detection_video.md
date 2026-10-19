# KAL detection simulation video

Current short demo: `out/video_v2/KAL_detection_6s.mp4` (1920 x 1080, 25 fps, 6 s) of the detection console.

**Accuracy notes (corrected 2026-10-04).** The console runs the v1 CNN `rgb_lwir` checkpoint, which was
validated (validation AP50 0.450) but never evaluated on test scenes; v1 is superseded by M1 detector v2
(`docs/m1_v2.md`). Version 2 of the video uses that model's **F1 threshold (0.35)**; version 1 used its
false-alarm threshold (0.70) and detected the target in 22 % of frames (`out/video/sensor/clip_stats.json`).
At 0.35 the same model gave 0.045 false alarms per empty validation frame. The clip's hit rule (IoU >= .5 or
centre inside the box +2 px) is lenient. Treat the video as a visualisation of one clear-day approach, not a result. Version 2 (2026-10-03)
uses the P1-based presentation look (`kal_hero.build_hero(look="photo")`) with a calmer matte finish
applied in `tools/render_kal_video.py` only, a propeller blur disc instead of frozen blades, an 85 mm
f/2 chase camera, and flies at **54 m/s (194 km/h)**: the user's requested 180-220 km/h band, also
inside the protected 39-56 m/s range (configuration unchanged). The first version at 44 m/s is in
`out/video/`. The earlier flat-shaded flight film (`out/m1/flight_video/`) and contrast-detector demo
(`out/demo/`) stay on disk unchanged.

## What the viewer sees

1. **Title** and two short presentation shots of the high-detail KAL model in flight over a
   desert range (`tools/render_kal_video.py --shot intro|flyby`). These are visualisation only.
2. **Detection console** (26 s, the main part):
   - **S1 RGB** and **S1 LWIR** panels: the actual detector inputs. The 640 x 512 frames come from
     the project renderer and sensor models (`render/track_sequence.py` + `tools/kal_video.py sensor`):
     the same scene sampler, signatures, haze, PSF and noise models as the M1 dataset, the 8.78 deg
     S1 camera, one camera parameter set for the whole clip (fixed column/pixel pattern, fresh
     temporal noise). Orange boxes are the saved CNN's detections at the threshold in the bottom strip
     (v2 video: the validation F1 threshold 0.35; see the accuracy notes above). The magnifier is display only.
   - **EO zoom**: a co-located narrow camera rendered in Eevee with the high-detail model
     (`target/kal_hero.py`), the same airframe variant, sun, sky and haze as the sensor scene. The
     green track box is the detector's sensor-frame box projected into the zoom view through the
     shared viewpoint; amber means the track is coasting without a detection.
   - Bottom strip: detector identity and threshold, simulation-truth range and pixel size, and the
     per-frame best detector confidence against the threshold.
3. **End card**: clip statistics computed against simulation truth only after detection finished
   (`out/video/sensor/clip_stats.json`).

## Scenario (all from the project's protected settings or sampled priors)

- Base scene 7004 of the M1 sampler (overlay `configs/m1.yaml`): clear day, 41 C air, 29 km
  visibility, sun 24.5 deg high and about 90 deg to the side of the view, no cloud layer.
- KAL variant parameters of that scene; level flight at 54 m/s (v2) along a gentle S-turn with
  coordinated bank; range 1.47 km to 0.40 km, 7 to 31 px on the sensor.
- The pan-tilt head stares at a cue for 2.5 s and then follows the aircraft with a lag. Its
  pointing is **scripted from the flight path**; it is not closed-loop tracking on detector
  output. The tracker overlay is display logic and does not feed back into detection.
- Propeller speed 6,400 rpm (the nominal power setting, 5/8 of rated power: 7,500 x 0.625^(1/3)), shown as
  the translucent disc a normal video shutter records.

## What it does not show

- It is one synthetic clip: a demonstration, not an accuracy measurement. Held-out numbers: the compact
  pilot (`out/m1/pilot/results.md`) and, once run, M1 detector v2 (`out/m1/det2/results.md`).
- No birds, other aircraft, terrain clutter, haze/fog or real sensor footage.
- The presentation model's panel seams, fasteners and wear are visual assumptions (level C); its
  dimensions come from `target/kal_geometry.py` and are identical to the dataset model.

## Rebuild

```sh
set KAL_CONFIG_OVERLAY=configs/m1.yaml
blender -b -P render/track_sequence.py -- --out out/video_v2/sensor --start 7004 --speed 54 --duration 21
python tools/kal_video.py sensor --sensor out/video_v2/sensor --model out/m1/cnn/runs/rgb_lwir/best.pt
blender -b -P tools/render_kal_video.py -- --shot zoom --sensor out/video_v2/sensor --out out/video_v2/hero --first 375 --last 525
blender -b -P tools/render_kal_video.py -- --shot intro --seconds 4 --sensor out/video_v2/sensor --out out/video_v2/hero
python tools/kal_video.py console --out out/video_v2 --sensor out/video_v2/sensor --hero out/video_v2/hero --first 375 --last 525 --threshold threshold_f1
python tools/kal_video.py short --out out/video_v2 --sensor out/video_v2/sensor --hero out/video_v2/hero --first 375 --last 525
```
