# Thermal bands: LWIR uncooled, LWIR cooled and MWIR (2026-10-13)

The project's sensors are RGB, LWIR uncooled, LWIR cooled and MWIR (CLAUDE.md sensor set rule; SWIR deferred). This
note covers the physics the three thermal cameras share: surface radiance in each camera's band, the clear air between
the scene and the camera (sky, path, sun), and the L550E's exhaust plume. Sources and their tiers:
`docs/sources/README.md`, section "Thermal-band atmosphere and exhaust plume". Code: `thermal/gas_bands.py`,
`thermal/atmosphere_bands.py`, `thermal/plume.py`, `render/thermal.py`, `render/plume_layer.py`.

## 1. Bands and what is shared

| Camera | Band | Planck table | Notes |
| --- | --- | --- | --- |
| LWIR uncooled (student) | 8-14 µm | `planck_lut.lwir_lut` | the existing pass, now on the layered atmosphere (`thermal_atmosphere.enabled`) |
| LWIR cooled (teacher) | 7.7-9.3 µm (Scorpio LW / Pelican-D LW class) | `band_lut((7.7, 9.3))` | own render pass, own noise (render/cameras.py) |
| MWIR (teacher) | the scene's detector: MCT 3.7-4.8, InSb 3.6-4.9 or HOT 3.6-4.2 µm | `band_lut(band)` | own pass; sun, scattered sky and plume matter here |

The three thermal cameras look at the same surfaces at the same instant, so they share the LWIR draw's temperatures
(skin underside and top, the engine power-setting profile and its caps, ground, walls, roofs, clouds) and its finish:
each emissivity sits at the same quantile of its band's range as the LWIR draw (`render/signatures.add_thermal_bands`).
RGB keeps its own draws (spec 7.2 rule 4): the student's two bands stay independent, while the teacher's extra bands are
physically consistent with the student's LWIR band. This is a design choice recorded for review (section 7).

## 2. Surface radiance and emissivities

L = ε L_band(T) + (1 − ε) L_env (sky with the normal up, ground with it down) + sun glint, as in the LWIR pass
(`render/lwir.py`), with the band's Planck table. Emissivity ranges per band (`configs/signatures.yaml`):

* **LWIR uncooled**: FLIR T810252 Table 21.1 LW and total-band rows (unchanged, `thermal/engine_heat.emissivity_ranges`).
* **LWIR cooled**: the same LW rows (they are 8-14 µm averages); painted and oxide surfaces have silicate and oxide
  features near 8-9.5 µm, so painted lower bounds are widened by 0.03 (C).
* **MWIR**: the table's SW (2-5 µm) rows (`engine_heat.emissivity_ranges_mwir`, tied to the YAML by
  `tests/test_engine_heat.py`): paints 0.84-0.97, anodised aluminium 0.61-0.67 (0.95-0.97 in LW), thin lacquer on
  aluminium 0.50-0.53, stainless untreated 0.30; total-band metal rows as in LWIR. Fin packs keep the groove cavity
  effect. Ground 0.80-0.95 (C, to be sourced from a spectral library), walls and roofs from the SW brick, concrete,
  plaster and tile rows.

MWIR by day adds the sun's diffuse reflection: a sun lamp of the band's direct irradiance at the ground (W/m²) lights a
diffuse BSDF of albedo (1 − ε)(1 − f_s), so shadows follow from the geometry; the world is hidden from diffuse rays so
the sky's reflection is not counted twice. `tools/check_thermal_blender.py` checks the wiring in Blender: a sphere under
a 4 W/m² sun renders ε L(T) + (1 − ε) E cos i / π to 0.003 % (`docs/reference/engine/data/blender_thermal_check.json`).
The specular part is the glint lobe of the LWIR model with the band's own irradiance and lobe draws.

## 3. Gas absorption: the narrow-band model

NASA SP-3080's statistical narrow-band model (Goody random band of Lorentz lines, Doppler lines, the combined depth of
eqs. 5-25/26, Curtis-Godson averages along a non-uniform path), 5 cm⁻¹ intervals, as implemented by NIST RADCAL and
ported to NumPy. Data:

| Gas | Source | Check |
| --- | --- | --- |
| H2O | SP-3080 Table A2-35 (RADCAL's copy, five entries corrected to the print) and RADCAL's 1/d fit | the corrected entries carry the printed values |
| CO2 4.3 µm | RADCAL's band model scaled by 273/300 onto SP-3080's Table A2-29 | band integral within 0.4 % at 300, 600, 1200 K; core rms 9, 4, 4 %; 1/d within a factor 1.6 of Table A2-33 in the core |
| CO2 15 and 9.4/10.4 µm | RADCAL (SD15 table, two-band model) | (no printed table read) |
| CO (exhaust) | SP-3080 Tables A2-2 / A2-3, 300-1200 K | reproduces the print |
| CO (air), N2O, CH4 | HITRAN2020 lines → narrow-band k, Godson 1/d, air widths (200-320 K) | HITRAN CO vs SP-3080 300 K: ±20 % in the core, 7 % in band strength |
| H2O continuum | MT_CKD 4.3 (AER), x a scene factor 1.0-1.44 (Roberts et al. 1976 is 1.44 x at 1000 cm⁻¹) | two sources: the range covers both |
| N2 collision-induced | Lafferty et al. (1996) via LBLRTM, 2000-2900 cm⁻¹ | closes most of the 3.8-4.0 µm deficit against G173 |

The arithmetic is checked against the handbook's own worked example (Table A1-1, pure CO2 at 1 atm, 60 cm in ten
sections): X*, a_c, X_c and X within 0.3 %, transmittances within 0.0015, Planck within 0.04 %, the emerging radiance
within 0.5 % (1.7 % at 3600 cm⁻¹, where the printed sum carries a misprinted transmittance). Three misprints were
identified (two k values and two transmittances) and are named in `tests/test_gas_bands.py`.

## 4. The atmosphere between the scene and the camera

Layers from the camera's station to 30 km (`thermal/atmosphere_bands.Profile`), each uniform at its mid-height state:

* temperature: T_sfc − Γ z to a 16 km tropopause (C), then isothermal; Γ drawn in 4.5-7.7 K/km, the 10-90 % range of
  the surface-500 hPa lapse rate in 2,756 radiosondes of Lahore City, Patiala and New Delhi, Oct 2023 - Sep 2026
  (NOAA IGRA 2; Lahore alone has only 46 humidity soundings in the period). Pressure hydrostatic from 988 hPa (ISA at
  Lahore's 214 m, C+ until the climate update).
* water vapour: ρ_v(z) = ρ_v(0) exp(−z / H_w), capped at saturation; ln H_w = 6.003 + 0.487 ln e_sfc ± 0.37 (same
  soundings): ~1.1 km at 8 hPa (winter), 1.7 km at 20 hPa, 2.3 km at 36 hPa (monsoon). The fit returns the soundings'
  precipitable water with a median ratio near 1.
* CO2 419-440 ppm (NOAA GML 2023-2026 global means 419-428, top widened for the urban boundary layer, C); N2O
  336-340 ppb, CH4 1920-2100 ppb (GML; the CH4 top for the South Asian plains, C); CO 150-3000 ppb in the boundary
  layer (C, polluted city) over 100 ppb above.
* aerosol: grey in each band, extinction 3.912 / V x the scene's band ratio (render.yaml weather; the cooled-LWIR and
  MWIR ratios at the LWIR ratio's quantile), 1.5 km scale height; it absorbs and emits a share 1 − ω0 (ω0 drawn per
  band, C).

Rays run through spherical shells (R = 6,371 km), so a horizontal path is finite. The sky radiance at the camera is the
emission sum along each ray (SP-3080 eq. A-3), tabulated at 13 elevations from 0.25° to 90°. Path transmittances are
tabulated for horizontal paths (5 m - 20 km, for the terrain) and computed for each labelled object's own slant path,
averaged over the band with the dB/dT weight of a near-ambient contrast; the plume uses its own spectrum (section 5).
Aerosol stays in the loader (`data/composite.py`), so visibility can still be swept without re-rendering.

**Checks.** ASTM G173-03 (its own atmosphere: US Standard 1976, 1.42 cm water, 370 ppm, AM1.5): direct-beam
transmittance 3.6-4.0 µm, model 0.79 vs G173 0.82 (`tests/test_atmosphere_bands.py`, within 5 %); per 50 nm the
N2O band at 3.9 µm appears in both. Idso (1981): the clear-air 8-14 µm flux emittance is 0.04-0.15 below Idso's
Phoenix fit (no ozone and no aerosol in the clear-air model; Idso fitted desert air with dust). Rather than pick one,
each scene draws a share u of the gap (uniform 0-1) and closes it with a grey window absorber in 714-1300 cm⁻¹ with a
1.5 km scale height (C+); with u = 1 the scene reproduces Idso within 0.005 (test).

**MWIR sky by day.** Thermal emission plus the aerosol's single scattering of the direct beam (plane-parallel layer,
Henyey-Greenstein phase function, g 0.6-0.8, C; Liou 2002 sec. 6.4), as a 2-D table in elevation and scattering angle
in the world shader. Clouds emit at their temperature and in MWIR reflect 5-25 % of the sun (C).

**Direct sun.** ASTM E490 through the gas and aerosol along the sun's path: in 3.7-4.8 µm, 7.90 W/m² at the top of the
atmosphere and typically 2.5-3.5 W/m² at the ground at 40° elevation; in 8-14 µm 1.26 W/m² at the top.

Cost: 0.6 s per scene for the three bands' tables (memoised narrow-band lookups), stored in the metadata
(`extra.thermal.tables`), so the loader never repeats the radiative transfer.

## 5. The exhaust plume

`thermal/plume.py`, rendered analytically into each thermal band's object layer (`render/plume_layer.py`):

* **Gas.** CH_y fuel (y 1.87-2.25) at the scene's equivalence ratio 1.0-1.2 ([AUSSERER]); carbon, hydrogen and
  oxygen balances, and for rich mixtures the water-gas equilibrium K = x_CO x_H2O / (x_CO2 x_H2) = 3.5 (Heywood 1988):
  at φ = 1.1, CO2 11.4 %, H2O 12.5 %, CO 2.9 %, H2 0.9 %. Short-circuited fresh air dilutes it (0.18-1.0 mol per mol
  burnt air, from [AUSSERER]'s 15-50 % short-circuiting).
* **Exit.** The stub barely cools the gas (NTU 0.04-0.07): exit excess = the setting's gas/wall ratio (1.55-2.0) x
  the scene's stub-wall draw, clipped to the setting's port-gas band. Exit velocity / flight speed from the exhaust mass
  flow (7-16 g/s per cylinder), 0.6-1.4.
* **Mixing.** Round-jet similarity: f_c = min(1, K D / x), Gaussian radial profile of half-width S x (K 5-20, S
  0.08-0.11, C; Pope 2000). The axis bends with the free stream. The pusher propeller's plane, 0.13-0.27 m behind the
  outlets, ends the plume: its slipstream (~14 kg/s through the disc) dilutes the ~0.05 kg/s of exhaust some 300-fold.
* **Radiance.** Rays through the four jets on a grid across their projection, the same narrow-band model (CO2, H2O,
  CO), splatted into the image with the base PSF; the layer carries emission P, absorbing coverage A and the plume's
  own gas transmittance along the path, weighted by its emitted spectrum.

At the source (cruise, 0 km), the four jets' contrast intensity relative to the four stubs' is ≤ 2 % in 8-14 and
7.7-9.3 µm (confirming the older bound in `engine_heat.plume_to_stub_lwir_ratio`), but 10-70 % in 3.7-4.8 µm (CO2
4.3 µm). Through the air the plume loses far more than the stubs, because the air's own CO2 absorbs the band centre:
over 1.3 km the plume's gas transmittance was 0.15 against ~0.65 for the airframe in the smoke scene.

## 6. Render passes and the loader

`render/scene.py` renders `rgb,lwir,lwir_cooled,mwir` by default (`--bands`; SWIR deferred). Thermal passes are emission
only with no bounces, as before; the MWIR pass adds its sun lamp by day (direct lighting only). Render noise against
each camera's sensor noise (`tools/render_noise_check.py`, horizon scene 14, ground pixels): LWIR 0.01 x and cooled
LWIR 0.04 x at the 4 background samples; sunlit MWIR 0.87 x at 4 samples, so its backgrounds use 32 (0.18 x, target
< 0.35; `render.mwir_sun_background_samples`). Production cost on a 4-core CPU: ~3 min for a sunlit horizon scene (RGB
background 142 s; thermal bands ~33 s, plus ~40 s for the 32-sample MWIR background). The teacher's cameras do not
always cover the student's frame: the uncooled camera sets the grid (5-46° HFOV), the cooled-LWIR zoom reaches 25° at
most and MWIR 36°. Over 150 sampled scenes the cooled LWIR covers only part of the frame in 35 % of scenes and misses
the target in 14 %; MWIR in 11 % and 3 %. Uncovered pixels are zeros. Thermal layers are
stored in float32 (MWIR contrasts are tiny against its signal). The loader composites every thermal band with its own
gas table and emulates its camera from the scene's rig (`data/camera_emulation.py`): NETD, fixed pattern and integration
on the band's own Planck table. Older scenes without `extra.thermal` keep the legacy LWIR water-vapour model.

## 7. Grades, open items and checks to make

* Shared thermal temperatures and finish quantile across the three thermal cameras (section 1): physics says so; the
  student's RGB + LWIR independence is unchanged. **For review by the user.**
* Uncooled LWIR moved from the Idso sky to the layered atmosphere for new scenes (`thermal_atmosphere.enabled: true`;
  M1's overlay keeps the legacy model), so that the two LWIR cameras share one sky where their bands overlap.
* Not modelled: ozone (9.6 µm, inside 8-14 µm; part of the Idso gap), the water-vapour continuum's temperature
  dependence for the foreign part, fuel vapour in the exhaust (C-H band 3.3-3.5 µm, below the MWIR bands), plume
  occlusion by the airframe, overlap of the four jets.
* C ranges to source: the MWIR and cooled-LWIR aerosol band ratios and ω0, g (OPAC aerosol tables would do), MWIR
  ground emissivity (ECOSTRESS spectral library soils), cloud reflectance at 4 µm, the jet mixing constant K with the
  pusher propeller.
* Validation that would settle the Idso gap: a narrow-field 8-14 µm radiometer at the zenith and at 25° in Lahore,
  clear sky, with a surface weather reading; and an MWIR radiometric image of a running small two-stroke for the
  plume (the plume-to-stub ratio is the number to check).
