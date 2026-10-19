# MD-550 / L550E engine: reference images

All images are generated from the model, so they always match `target/engine_l550e.py` (geometry measured on
Limbach's L550E / L550EF datasheet drawings; sources in `docs/sources/README.md`). Regenerated 2026-10-10.

| File | What it shows | Made by |
| --- | --- | --- |
| `KAL_reference_drawing.png` | Three views with the specification table and evidence levels | `python tools/draw_spec_sheet.py` |
| `kal_views_with_engine.png` | Dimensioned four-view drawing (top, side, front, rear) | `python tools/draw_reference.py` |
| `dataset_mesh_side.png`, `dataset_mesh_rear.png` | The dataset mesh in Blender (what the detector trains on) | `blender -b -P target/build_target.py -- --variants 2` |
| `engine_closeup.png`, `engine_closeup_under.png`, `kal_rear34.png` | Presentation model close-ups: individual fins, head blocks with heat-sink fins, plugs and leads, carburettors, exhaust stacks, drive flange | `blender -b -P tools/hero_preview.py -- --views rear34,engine,engine_under` |

## Derived data (`data/`)

Every derived engine number as a file, re-made by `python tools/engine_report.py` (2026-10-11, model revision
md550-engine-v3); `docs/engine_signature_physics.md` quotes them. The numbers read off the sources themselves are in
`docs/sources/` (`python tools/source_measurements.py`).

| File | What |
| --- | --- |
| `heat_report.txt` | The derivation table of `thermal/engine_heat.py`: sourced inputs, checks, heat balance, envelopes, YAML lines |
| `engine_profiles.csv` | Per power setting and surface: the model's temperature envelope and the `signatures.yaml` range |
| `heat_balance.csv` | Cooling load against the free-air capacity of the barrel fins and heat sinks (centre, with inflow, without heat sinks, best corner) |
| `cooling.csv` | NACA 488 h and fin efficiencies of the barrel fins and heat sinks over the envelope, equivalent test speeds, channel-model cross-check |
| `emissivity.csv` | LWIR emissivity range of each surface |
| `visible_area.csv` | Visible projected area of every slot by aspect (ray cast on the canonical mesh) |
| `lwir_contrast.csv` | Engine vs airframe LWIR contrast by aspect, power setting and thermal state, with the heat sinks' share |
| `plume.csv` | Exhaust plume against the stubs: LWIR contrast (best estimate, ×3 margin); SWIR break-even emissivity |
| `night_glow.csv` | RGB glow of the stubs: thresholds and the share of scenes where it shows (100,000 sampler draws) |
| `sun_glint.csv` | LWIR sun-glint draws by percentile and how much too hot a glint reads on each finish |
| `summary.json` | The headline numbers |
| `blender_glow_glint_check.json` | Blender renders of the glint and glow shaders against their equations (`blender -b --factory-startup -P tools/check_glow_glint_blender.py`) |

`glint_check_sphere.png` (in this folder) shows that check: the rendered LWIR glint on a sphere, the analytic one, and
their difference × 10.

The heads carry Limbach's optional heat sinks, as on the analogue (the published 410 mm width needs them); whether the
KAL's engine has them is not resolved in photo P4, so their finish stays a range in `configs/signatures.yaml`.
