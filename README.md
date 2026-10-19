# KAL target model

Specification: section 2 of `FYP_Build_Spec_Multispectral_Aerial_Detection.md`. The engine is the
MD-550 / L550E (`target/engine_l550e.py`); how its infrared signature is derived from heat transfer:
`docs/engine_signature_physics.md`. Every source behind those numbers, with its reliability tier, page and
SHA-256: `docs/sources/README.md` (the PDFs sit next to it, not committed); which component uses which source:
`SOURCES_GUIDE.txt`. Every derived engine number (heat balance, cooling, contrast, plume, glow, glints) is saved as data
in `docs/reference/engine/data/` by `python tools/engine_report.py`; every number read off a source is re-derived by
`python tools/source_measurements.py`.

```
pip install -r requirements.txt
python target/kal_geometry.py        # metre summary + closed-mesh check
python target/engine_l550e.py        # engine parts against the published envelope
python target/kal_measure.py         # mesh-vs-YAML invariants + hot-part visibility by aspect
python thermal/engine_heat.py        # derivation of every engine temperature range in configs/signatures.yaml
pytest tests -q                      # full suite, including tests/test_engine_heat.py
python tools/photo_check.py          # overlays in docs/reference/, RMS per photo
python tools/draw_reference.py       # docs/reference/kal_views.png/.svg (top, side, front, rear)
python tools/make_decals.py          # assets/decals/ (already generated)
blender -b -P target/build_target.py -- --variants 5 --out out/phase1
blender -b --factory-startup -P render/scene.py -- --start 0 --count 10 --out out/dataset   # RGB, LWIR uncooled, LWIR cooled, MWIR
blender -b --factory-startup --python-exit-code 1 -P tools/check_thermal_blender.py        # MWIR sun wiring check
python train/train_teacher.py --data out/dataset                                         # teacher: all four sensors
python train/train_student.py --data out/dataset --teacher out/runs/teacher/best.pt       # student: RGB + LWIR uncooled
python tools/m1_det2.py smoke --evaluate                                                  # M1 pipeline, end to end, few minutes
RUN_M1_PRELIMINARY.bat                                                                    # M1 preliminary evaluation, one laptop
```

M1 preliminary evaluation (compact CNN, four sensors): `docs/m1_v2.md`. Models of the cameras' own processing (defect
replacement, filters, AGC, noise reduction), their sources and their verification: `docs/sensor_noise.md`. They are
off for training since that verification (2026-10-17). The case for and against the four-sensor teacher: `docs/privileged_teacher_case.md`. What could give better detection, from the latest work and this
project's own measurements: `docs/detection_gain_plan.md`. An optional two-laptop split (`member/README.md`) remains for
later work.

Sensors (CLAUDE.md sensor set rule): RGB, LWIR uncooled, LWIR cooled, MWIR; SWIR is deferred (its code stays, rendered
only with `--bands ...,swir`). The thermal bands' shared atmosphere, emissivities and exhaust plume:
`docs/thermal_bands.md`; scene conditions from Lahore's station record: `docs/lahore_climate.md`; camera classes:
`docs/camera_models.md`.

Edit airframe dimensions only in `configs/target.yaml` and engine parts only in `target/engine_l550e.py`,
then rerun everything above. After any change to `thermal/engine_heat.py`, copy the YAML lines it prints into
`configs/signatures.yaml` (the tests fail until they agree).
PyYAML is optional: without it (e.g. inside Blender) the bundled `target/_yamlite.py` parses the config.
