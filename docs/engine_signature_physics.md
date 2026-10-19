# Engine infrared signature: physics and maths

This document shows where every engine number in `configs/signatures.yaml` comes from. The code that
does the calculation is `thermal/engine_heat.py`; the tests that hold it together are
`tests/test_engine_heat.py` and `tests/test_glow_and_glints.py`. To reproduce every number below:

    python thermal/engine_heat.py              derivation table, checks and the YAML lines
    python tools/engine_report.py              every derived number as data: docs/reference/engine/data/
    python tools/source_measurements.py        every number read off a source: docs/sources/*/
    python -m pytest tests/test_engine_heat.py tests/test_glow_and_glints.py -q

The data files quoted below (`heat_balance.csv`, `cooling.csv`, `lwir_contrast.csv`, `plume.csv`, `night_glow.csv`,
`sun_glint.csv`, `summary.json`) are in [docs/reference/engine/data/](reference/engine/data/).

Grades follow the project rule (header of `configs/signatures.yaml`): **A** observed in KAL imagery,
**B** strongly inferred or published, **C+** standard engineering practice, **C** plausible assumption,
**D** unknown. Source tiers **S1–S4** follow `CLAUDE.md`; every source, its tier, its relevance and what is taken
from it is listed in [docs/sources/README.md](sources/README.md). No number here is a measurement of the KAL or of an
MD-550 in flight.

## 1. Why these numbers matter, and why the engine data are needed

A detector trained on synthetic images can only work on real images if the synthetic target looks
the way a real one does to each sensor:

* **LWIR (8–14 µm)** records each surface's emissivity × its Planck radiance, plus the reflected
  surroundings: L = ε·L(T) + (1 − ε)·L(T_env). So it depends directly on the surface **temperatures**.
* **SWIR (0.9–1.7 µm)** records reflected light, plus the self-emission of hot parts (Planck's law
  again), which matters from about 50 °C at night (section 9).
* **RGB** records reflected light, plus a dull red glow of the hottest exhaust stubs at night (section 11).

The engine surface temperatures are set by heat transfer: how much heat the engine makes, how much
gas flows through the exhaust, and how fast the air carries heat off the fins and stubs. That is
why the published ratings are needed: **displacement and rated speed** give the engine's air and
exhaust flow (section 3), and **rated power** with the **specific fuel consumption** gives the heat that the fins
must reject. The engine mass plays no part in any band, so it is not used.

## 2. Inputs

| Input | Value | Grade, tier | Source |
| --- | --- | --- | --- |
| Rated power, speed; bore × stroke; displacement | 37 kW (50 HP) at 7,500 rpm; 66 × 40 mm, 4 cylinders → 547.4 cc (published 548 cm³) | B, S1 | [L550E-DS] p. 1; V = π/4 · B² · S · 4 |
| Envelope L × W × H | 300 × 410 × 301 mm | B, S1 | 300: [L550EF-DS] p. 2 label; 410 and 301.7: [L550E-DS] p. 2 drawing (section 2b) |
| Fin pack per cylinder | 12 fins, 2.5 mm thick, pitch 6.67 mm (gap 4.17 mm), rounded rectangle 108 × 114 mm around a 41 mm barrel, 66 → 146 mm out → 0.171 m², mean fin height 20.5 mm | B− (measured on the drawings, ±1 mm); corner and root radius C | [L550E-DS], [L550EF-DS] p. 2 |
| Head end | head block 146 → 157 mm out, ~100 × 93 mm, with the spark plug | B− (outline) | [L550E-DS] 313.6 mm across the heads "without heat sinks" |
| Heat sinks (Limbach option, fitted) | 157 → 182 mm out, ~100 × 93 mm: 12 ± 2 black fins 2 mm thick across the crank axis, 21 mm deep on a 4 mm base, pitch 8.3 mm, split around the plug → 0.034 m² per head | B− (outline, drawing); C+ (fin count by eye, ±2) | [L550E-DS] 363.4 mm across them "with optional heat sinks"; [L550E-DS] / [L550EF-DS] photos (`docs/sources/limbach/visual_readings.json`) |
| Exhaust stack | Ø 31.5 mm outside, 1.5 mm wall, flange 67 mm and end 162.5 mm from the crank axis (95.5 mm), centres 88.5 mm either side | B− (wall C) | [L550E-DS] p. 2 front view |
| Flight envelope | 39–56 m/s, sea level–5,000 m, −20 to +50 °C | spec / press report | `configs/render.yaml` (not re-sourced: the project does not research the target online) |
| Brake specific fuel consumption | 389–590 g/kWh | B, S1 + S2 | [L550E-DS] full-throttle chart 335–434 g/HPh = 449–590 g/kWh; [ROTAX-OM] 503 UL: 486–524 g/kWh at take-off, 389–419 at 75 % (EN 228 density), stated 550 |
| Delivered air/fuel ratio | 12.2–15.1 | B−, S2 + chemistry | Stoichiometric 14.6 (petrol, CH1.87) and 15.1 (iso-octane) from the oxygen balance, over the equivalence ratio 1.0–1.2 of carburetted small two-strokes [AUSSERER] pp. 284–285, 316 |
| Gasoline LHV, heat of vaporisation | 43 MJ/kg; 300–350 kJ/kg | B, S2 | Heywood (1988), app. D (handbook values) |
| Cylinder-head temperature (CHT) bands, over a 15 °C day | low 99–215, nominal 165–215, high 165–245 K | B (analogue), S2 | [ROTAX-OM] p. 10-1: CHT at the spark-plug seat, 447 / 503 normal 190–230 / 180–220 °C, maxima 260 / 250 °C; p. 10-15: can be higher at reduced throttle. Low-power lower end: physics bound (section 5) |
| Exhaust gas temperature (EGT) bands, over 15 °C | low 385–565, nominal 445–605, high 523–635 K | B (analogue), S2; low end C+ | [ROTAX-OM] p. 10-1: 503 normal 460–580 °C (printed as 860–1,000 °F in the same table: the upper end is 538 °C in °F; both kept), 582 normal 500–620 °C, maximum 650 °C; p. 10-15. Low-power lower end 400 °C: estimate |
| CHT cap | 250 °C | B, S1 + S2 | [ROTAX-OM] 503 UL maximum; [EASA-E082] IV.1 Limbach L 1700 limit |
| Cooling load / shaft power | 0.46–0.78 | B−, S2 | [AUSSERER] p. v: cylinder heat loss 10–11 % of the fuel energy / brake efficiency 1 / (BSFC · LHV) = 0.14–0.21 |
| Emissivities | see section 8 | B, S2 | [FLIR-TAB] Table 21.1 entries with their band and temperature |
| Propeller inflow at the engine | 0 up to 0.06–0.41 × flight speed | physics (bound) | Actuator-disc momentum theory, ideal propeller (section 5) |
| Fin heat-transfer coefficient | h = 72.9 W m⁻² K⁻¹ × (V / 34.0 m/s)^0.796 × (gap / 4.45 mm)^0.322, with similitude in diameter and density | B, S2 | [NACA-488] wind-tunnel measurements of finned cylinders across a free air stream, no baffles (section 5) |
| Local air speed / flight speed, before the inflow | stubs 0.7–1.0; at the cylinders 0.5–0.9 | C | Stubs hang near the free stream behind the body; the fins sit partly in the ~45 mm turbulent boundary layer of the 3 m fuselage (δ = 0.37 x Re_x^−0.2) |
| Heat-sink h / NACA 488 h of aligned fins | 0.25–1.0 | C | The heat-sink fins stand across the ram air (section 5) |
| Pulsation factor of exhaust heat transfer | 1.2–2.0 | C+ | Blow-down pulses raise heat transfer over steady pipe flow (Heywood 1988, ch. 12); to be checked against Depcik and Assanis (2002) |
| Cast aluminium conductivity | 120–160 W m⁻¹ K⁻¹ | B | Cast Al alloys |
| Part-to-part fractions | head 0.85–1.0 × CHT; fin root 0.80–0.95 × CHT; case 0.25–0.45 × CHT; flange 0.3–0.8 × case; tail cone 0.1–0.4 × case | C+ / C | No measurement on a comparable engine found (section 6) |
| Sun, for the night glow and the glints | 5772 K, R 695,700 km, 1 au = 149,597,870,700 m; in-band direct sun at air mass 1.5: r 125.9, g 133.8, b 115.1, 0.9–1.7 µm 249.6 W/m²; 8–14 µm above the atmosphere 1.261 W/m² | B, S1 | IAU 2015 Resolution B3 (quoting IAU 2012 B2); ASTM G173-03 and ASTM E490 standard spectra (NREL's tables), integrated by `tools/source_measurements.py` |

Three power settings are drawn by the sampler, with dataset weights 0.25 / 0.50 / 0.25 (a coverage
choice, not duty-cycle data): low_power 30–50 %, nominal 50–75 %, high_power 75–100 % of rated power.

### 2a. Which inputs are sourced, and which are estimates

| Kind | Inputs | Status |
| --- | --- | --- |
| Exact physics | Planck's law, Stefan–Boltzmann, Kirchhoff (ε = 1 − ρ for opaque surfaces), energy and mass balances, momentum theory of the propeller, the oxygen balance of the fuel | Authentic |
| Standard correlations, measurements and property data | NACA Report 488 finned-cylinder heat transfer (the fin h), Sutherland viscosity, air conductivity (tested against Incropera tab. A.4), ISA atmosphere, Churchill–Bernstein, Gnielinski, laminar flat plate and parallel-plate channel (Incropera tab. 8.1, eq. 8.41; a cross-check of NACA 488), fin theory, groove cavity emissivity, Roberts et al. water-vapour continuum | Authentic (S2) |
| Solar constants and spectra | IAU 2015 nominal Sun; ASTM G173-03 (AM1.5) and E490 (AM0) standard spectra | Authentic (S1: standards) |
| Engine data and geometry | Ratings, SFC chart, envelope, fin pack, head end, heat sinks, exhaust stacks | Manufacturer datasheets and drawings (S1); the internal fin, heat-sink and stack dimensions are measured on the drawings, ±1 mm; the heat-sink fin count is read by eye on the photos (±2) |
| Operating limits and energy balance | CHT and EGT bands and cap, BSFC, AFR, cooling load | Closest analogue's operator's manual, a type-certificate data sheet, measured energy balances of small two-strokes (S1/S2), applied to the L550E by the physics above |
| Emissivities | All engine surfaces and the skin | FLIR camera manual's referenced table (S2) |
| Estimates (C+ / C) | Local air-speed factors before the inflow, the heat-sink h factor, the pulsation factor, part-to-part fractions, carburettor coupling, stub wall thickness, the low-power EGT floor, plume area and margin, power-setting fractions and weights, the glint lobe width and specular fraction, the colour camera's bands | Estimates, each with its reasoning below and a way to validate it (section 15) |

### 2b. Sources and their reliability

The full list is [docs/sources/README.md](sources/README.md). In short:

* **Engine data: Limbach's own datasheet replaces Wikipedia.** The L550E datasheet (limflug.de, 2025) confirms
  37 kW at 7,500 rpm, 66 × 40 mm, 548 cm³ and 16 kg. Its p. 2 drawing gives the envelope and, measured
  with its own dimension lines as the scale, the fins, the head end and the exhaust stacks. Its SFC curve is the
  manufacturer's fuel consumption for this engine. The sister L550EF sheet (same cylinders) has legible labels and
  was used to calibrate the measurements: the two sheets agree on the fin height (114 vs 113 mm) and on the width
  across the heat sinks (364 mm).
* **Temperature bands: the Rotax 503 UL operator's manual.** No L550E operating limit is public. The 503 UL is
  the closest engine whose manufacturer publishes them: air cooled, carburetted, aero two-stroke, 497 cm³,
  37 kW take-off (the L550E: 548 cm³, 37 kW). Its CHT is read at the spark-plug seat, where the L550E's optional
  14 mm thermocouple sits, so the bands are comparable. Limbach's own certified air-cooled engine (L 1700, EASA
  TCDS E.082) has the same 250 °C CHT limit as the 503 UL.
* **Energy balance: AFIT's measurements.** Ausserer's dissertation (AFIT, 2016; peer reviewed in *J. Eng. Gas
  Turbines Power*, 2018) measured every energy pathway of three air-cooled two-strokes. They are smaller (28–85 cm³)
  than the L550E's 137 cm³ cylinders, but the dissertation shows that the heat-loss share does not change above
  ~10 cm³, so its 10–11 % cooling share carries over.
* **Fin cooling: NACA's wind-tunnel measurements.** Biermann and Pinkel (NACA Report 488, 1934) measured the heat
  transfer of 18 finned cylinders standing across a free air stream with no baffles, at 30–150 mph. That is how the
  L550E is cooled: Limbach lists no fan or shroud and describes the engine as "air cooled" and "suitable for pusher and
  tractor installations". The L550E's fins (pitch 0.26 in, thickness 0.10 in, height 0.81 in) lie inside the tested
  ranges (pitch 0.10–0.60, thickness 0.04–0.27, width 0.37–1.47 in), so the report's measured h replaces the
  laminar-plate estimate used before; that physics model is kept as an independent cross-check.
* **Solar values: the standards.** The night glow (section 11) and the sun glints (section 12) need the Sun's
  in-band irradiance. The IAU 2015 nominal Sun sets its temperature and size; NREL's tables of the ASTM G173-03
  (air mass 1.5) and ASTM E490 (above the atmosphere) standard spectra give the in-band values. The tables reproduce
  the standards' own totals (G173: 1,000.4 W/m² global, 900.1 W/m² direct; E490: 1,366.1 W/m²), and two independent
  copies of G173 (NREL's workbook via PVEducation, and pvlib's) agree to the last digit. In 8–14 µm the E490 Sun gives
  1.261 W/m², 16 % less than a 5772 K black body (1.508), because the Sun's brightness temperature at 10 µm is only
  ~4,970 K: the standard replaces the earlier black-body estimate.
* **The FLIR note.** *5 factors influencing radiometric temperature measurements* (FLIR, 2016) is an authentic FLIR
  publication, but for emissivities it is a weak source in this context: Table 1 gives single values without
  temperature, waveband or surface state. Its values trace to FLIR's own camera-manual table, which gives those
  conditions and the references (Bramson; Wolfe and Zissis; AGEMA measurements). Stripped of the conditions they
  mislead in LWIR: "anodised aluminium 0.55" is a total emissivity at 100 °C, while dull anodised aluminium
  measures 0.95–0.97 in 8–14 µm. The manual's table is used instead; the note stays a cross-check.

## 3. Operating state (published power ↔ published displacement)

Shaft power P = f · 37 kW. Fuel flow ṁ_f = P · BSFC. Air flow ṁ_a = AFR · ṁ_f. Exhaust flow per
cylinder ṁ_ex = ṁ_f (1 + AFR) / 4. A fixed-pitch propeller absorbs P = C_P ρ n³ D⁵, so at a near-constant
power coefficient the shaft speed is n = 7,500 · f^(1/3) rpm.

**BSFC.** Limbach's chart reads 335 g/HPh at 7,000–7,500 rpm rising to 434 g/HPh at 4,000–5,000 rpm, at full
throttle. The datasheet's "37 kW (50 HP)" fits both metric (0.7355 kW) and mechanical (0.7457 kW) horsepower, so
both are used: 449–590 g/kWh. Part throttle burns less per kWh, because a throttled crankcase-scavenged engine
short-circuits less of its charge ([AUSSERER] pp. 307–311: short-circuiting falls from ~50 % to 15–20 % at half
throttle and efficiency rises 4–6 points): the Rotax 503 UL uses 15 l/h at 75 % power, 389–419 g/kWh. Band:
389–590 g/kWh. The brake efficiency 1 / (BSFC · LHV) is 0.14–0.21, and 0.18 at the L550E's rated point.

**AFR.** Stoichiometric air/fuel ratio from the oxygen balance of CH_y: (1 + y/4) / 0.2095 × 28.96 /
(12.011 + 1.008 y) = 14.6 for petrol (y = 1.87) and 15.1 for iso-octane. Carburetted small two-strokes run rich,
at equivalence ratio 1.1–1.2 ([AUSSERER] p. 316); a simple carburettor swings 0.8–1.2 over the speed range
(pp. 284–285) and aero manuals forbid lean running. With φ = 1.0–1.2: AFR 12.2–15.1.

At the nominal centre (f = 0.625, BSFC 489 g/kWh, AFR 13.7): P = 23.1 kW, n = 6,412 rpm, ṁ_ex = 11.5 g/s
per cylinder.

**Check: delivery ratio.** A two-stroke takes one charge per revolution, so DR = ṁ_a / (ρ_a · V_d · n).
Over all inputs the published power and displacement imply DR = 0.26–1.09. Crankcase-scavenged
two-strokes measured by AFIT ran from 0.3 at quarter throttle to 1.1–1.2 at full throttle ([AUSSERER] pp. 270,
306). **The published ratings are consistent with the measured range.**

## 4. Exhaust stubs

The stub is a thin steel tube with hot gas inside and cross-flow outside. Per unit outer area, the wall
sits where the heat in equals the heat out:

    θ_w = φ · θ_g,    φ = h_i′ / (h_i′ + h_o + h_r),    h_i′ = h_i · D_i / D_o

* **Inside:** Gnielinski pipe-flow correlation, Re_i = 4 ṁ_ex / (π D_i μ), times the entrance factor
  1 + (D/L)^0.7 for a short tube (L/D ≈ 3.4) and the pulsation factor.
* **Outside:** Churchill–Bernstein correlation for a cylinder in cross-flow at the local air speed, with
  film properties at the altitude's ISA pressure.
* **Radiation:** h_r = ε σ (T_w² + T_a²)(T_w + T_a).
* **Along the stub:** the gas cools as θ_g(x) = θ_g0 · exp(−NTU · x/L), NTU = U π D_o L / (ṁ_ex c_p), so
  the visible mean wall excess is φ · θ_g0 · (1 − e^−NTU) / NTU.

**Gas temperature at the port, θ_g0.** The EGT band of the setting, from the Rotax limits (section 2): nominal,
the 503 and 582 normal ranges (460–620 °C); high power, the 503's normal maximum to its maximum. Rotax's table prints
that normal maximum as 580 °C and as 1,000 °F (538 °C); the conflict cannot be resolved from the manual, so the
high-power band covers both (538–650 °C). Low power reaches the normal maximum (the reduced-throttle warning) from a
400 °C floor (estimate). The measured EGT is a mix of burnt gas and short-circuited fresh charge ([AUSSERER] p. 296),
which is what the stub carries.

At the nominal centre: Re_i = 14,100, h_i = 184, h_o = 139, h_r = 12 W m⁻² K⁻¹, φ = 0.52, NTU = 0.057.
The wall sits at about half the gas excess, about 270 K above ambient. Over all inputs and the
flight envelope, the stub wall excess is:

| Setting | Stub wall excess |
| --- | --- |
| low_power | 70–400 K |
| nominal | 115–465 K |
| high_power | 175–515 K |

The low end comes from fast, cold, dense air with the propeller's full inflow and weak pulsation; the high end from
slow, thin air at 5,000 m with strong pulsation. (Full power at 5,000 m is not actually reachable, because a
naturally aspirated engine loses power with air density, so the top of the high_power range is
deliberately generous.)

**Check: energy.** The sensible exhaust share of the fuel energy implied by each port-gas band,
x_ex = θ_g (1 + AFR) c_p / LHV, is 0.14–0.27. AFIT measured 0.14–0.18 at peak power on 28–85 cm³ engines
([AUSSERER] p. v), whose brake efficiency (0.14–0.15) is lower than the L550E's (0.18) because they short-circuit
more fuel. After shaft work, cooling (0.10–0.11) and the unburnt charge, at most ~0.30 remains.
**The port-gas bands are consistent with the measured energy balance.**

## 5. Cylinder heads, fins and heat sinks: free-air cooling

**CHT bands.** The base of the fins sits at the cylinder-head temperature of the setting. Rotax gives the normal
CHT of the 447 / 503 UL as 190–230 / 180–220 °C and the maxima as 260 / 250 °C, read at the spark-plug seat
([ROTAX-OM] p. 10-1). As excesses over a 15 °C day:

* nominal: the two normal ranges, 165–215 K;
* high_power: the 503 normal minimum to the 447 maximum, 165–245 K (capped at 250 °C in the scene);
* low_power: upper end 215 K, because Rotax warns that CHT and EGT can be *higher* at reduced throttle, where more
  hot residual gas stays in the cylinder (p. 10-15). Lower end, a physics bound: at a given cooling flow the head
  excess is proportional to the heat it rejects, θ = Q / (h A η), and the cooling load falls no faster than the
  power, because its share of the fuel energy rises at part throttle ([AUSSERER] pp. 307–311). So
  θ_low ≥ 165 K × 0.30 / 0.50 = 99 K.

The barrel runs a little cooler than the plug seat, so the mean fin-root excess is 0.80–0.95 × CHT (C+).

Each fin is a straight fin of corrected length L_c = L + t/2 with an adiabatic tip:

    m = sqrt(2h / (k t)),    η = tanh(m L_c) / (m L_c),    mean fin excess = η × root excess (exact)

**How the L550E is cooled.** By the air stream alone, over the cast fins of its barrels and the finned heat sinks on
its heads: Limbach lists no fan, blower or shroud, describes the engine as "air cooled" and "suitable for pusher and
tractor installations" ([L550E-DS] p. 1). The model does the same: ram air at the flight speed × an installation
factor, plus the pusher propeller's inflow, over the barrel fins and the heat sinks; no fan, no baffles.

**h on the fins: NACA Report 488.** NACA measured exactly this case: Biermann and Pinkel set 18 finned cylinders
(root diameter 4.66 in) across a free air stream with no baffles, at 30–150 mph, and found that the average
heat-transfer coefficient q of the fin surface depends mainly on the air speed and the fin space (gap):
q ∝ V^0.796 (p. 11, fig. 10) and q ∝ s^0.322 (p. 16, fig. 9). Fig. 9, read against its own gridlines
(`docs/sources/naca/fig9_digitised.csv`; the 162 points read are in `fig9_points.csv`), gives
q = 0.0892 B.t.u./sq.in./°F/hr = 72.9 W m⁻² K⁻¹ at s = 0.175 in (4.45 mm) and 76 mph (34.0 m/s) in sea-level air
(1.176 kg/m³); the digitised lines reproduce the report's exponent and its quoted q = 0.087–0.091 at 76 mph (test).
Other diameters follow the report's similitude, q_x = q_T(J·V, s/J) / J with J = D_x / D_T (eqs. 2–4), and other air
densities its equal-mass-flow rule (eq. 6):

    h = 72.9 × (V_eq / 34.0)^0.796 × (gap / 4.45 mm)^0.322 × J^(0.796 − 0.322 − 1),
    V_eq = u ρ / 1.176,   J = 82 mm / 118.4 mm

with u the local air speed at the cylinders and ρ the air density. The L550E's fins (pitch 6.67 mm = 0.26 in,
thickness 2.5 mm = 0.10 in, height 20.5 mm = 0.81 in) lie inside the tested ranges (pitch 0.10–0.60 in, thickness
0.04–0.27 in, width 0.37–1.47 in). Over the flight envelope the local mass flow equals 9.7–71 m/s at sea level against
the tested 13–67 m/s: the slowest corner (39 m/s at 5,000 m, wake factor 0.5, no inflow) extrapolates 28 % below the
tested speeds, the fastest 5 % above (`cooling.csv`; the test allows −30 % / +10 %). On the barrel fins
h = 32–155 W m⁻² K⁻¹ over the envelope (69 at its centre) and **η = 0.86–0.98**: the fins are nearly isothermal, so
the visible fin temperature is close to the root temperature whatever the exact h.

**Cross-check: an independent physics model.** The laminar channel model used before NACA 488 was found (boundary
layers growing from each leading edge, δ = 5x / √Re_x, Nu = 0.664 Re^0.5 Pr^(1/3), until they meet in mid-gap;
fully developed flow between isothermal plates beyond, Nu = 7.54 on D_h = 2 × gap, Incropera tab. 8.1; the channel
air's own heating, eq. 8.41) gives 54 W m⁻² K⁻¹ at the envelope centre against NACA's 69 (`cooling.csv`): 22 % lower,
as expected of a laminar model without the free-stream turbulence of a real installation. The test holds the two
within 30 %.

**Heat sinks.** Limbach offers heat sinks for the heads: its drawing gives 363.4 mm across them "with optional heat
sinks" and 313.6 mm "without", and the published 410 mm width across the plug caps needs them; both datasheet
photos show them fitted, black (`docs/sources/limbach/crops/`, `visual_readings.json`). Each carries 12 ± 2 fins
(counted by eye on the photos), 2 mm thick, 21 mm deep on a 4 mm base at 8.3 mm pitch, split around the spark plug:
0.034 m² per head, against 0.171 m² of barrel fins. Their fins are plates across the crank axis, so they stand
across the ram air instead of along it and their channels open across the stream: h = (0.25–1.0) × the NACA 488
value at the same local speed (C), 9–178 W m⁻² K⁻¹ over the envelope, η = 0.81–0.99. Their root is the head
(0.85–1.0 × CHT), so their mean excess is η × the head excess: 65–215 K at low power, 110–215 K nominal and
110–245 K at high power (`engine_heatsink`; never hotter than the head, test).

**Local air speed.** Flight speed × (a wake factor, C) + the pusher propeller's inflow. The propeller draws
air through the engine ahead of it. Momentum (actuator-disc) theory with an ideal propeller (thrust T ≤ P / V) gives the
induced velocity at the disc, v_i = (√(V² + 2T / (ρA)) − V) / 2, and ahead of it on the axis
v_i (1 − x / √(x² + R²)) (semi-infinite vortex cylinder). At the nearest cylinder (125 mm ahead of the 0.50–0.64 m
disc) this is up to 0.06–0.25 × V at low power, 0.10–0.33 at nominal and 0.14–0.41 at high power. A real
propeller adds less, so the model samples 0 up to this bound.

**Check: heat balance.** The free-air cooling must reject the cooling load, Q = (0.46–0.78) × shaft power
(section 2), inside the Rotax CHT band. Its capacity is h A η θ of the barrel fins plus that of the heat sinks, at
the band centres (`heat_balance.csv`):

| Setting | Load per cylinder | Barrel fins (h 69) | Heat sink | Load / capacity, envelope centre | with the propeller inflow bound | without the heat sinks | best-cooled corner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_power | 2.31 kW | 1.53 kW | 0.23 kW | 1.31 | 1.12 | 1.51 | 0.61 |
| nominal | 3.60 kW | 1.85 kW | 0.28 kW | 1.69 | 1.36 | 1.95 | 0.77 |
| high_power | 5.04 kW | 1.99 kW | 0.30 kW | 2.20 | 1.68 | 2.53 | 0.98 |

The envelope centre is 47.5 m/s at 2,500 m and 15 °C; the best-cooled corner is 56 m/s at sea level and −20 °C with
the highest wake factor and heat-sink h and the propeller's full inflow (barrel h 145–153 W m⁻² K⁻¹). The heat sinks
carry 13 % of the free-air capacity and lower the ratio by 13 %.

* At the best-cooled corner the free-air cooling carries the whole sourced load inside the CHT band at every setting
  (ratio ≤ 1). Independent sources (Limbach's geometry and SFC, NACA's measured h, Rotax's temperatures, AFIT's
  cooling share) close the balance inside the flight envelope.
* At the envelope centre it carries 45–76 % of the load, 60–89 % with the propeller's inflow. There the engine runs
  at the top of its CHT band or would pass it, more so at high power. That is what AFIT reports for pusher engines
  exposed only to the free stream: they may struggle to cool at full throttle ([AUSSERER] p. 301), and raising the
  cooling air from 15 to 45 m/s moved the head temperature by ~60 °C (pp. 300–301). The 250 °C cap (section 2)
  bounds it, and the CHT band stays the boundary condition because the real engine is operated inside it.
* The remaining factor (≤ 2.2) is inside the combined uncertainty of the cooling share (measured at peak power on
  smaller engines), the wake factor and the centre altitude.

**The fin pack: photo against heat balance, settled by the drawing.** Before Limbach's drawings were measured, the fin
size came from photos, and a heat balance with the laminar model asked for about twice the fin area. The plan then
was to sample the fin size between the two: deeper fins, up to about 100 × 140 mm and about 20 fins, which could
double the visible hot area. Limbach's own drawings now fix the fin pack (S1, ±1 mm; `docs/sources/limbach/
measurements.json`): 12 fins at 6.55–6.67 mm pitch, 2.5 mm thick, 108 × 114 mm around the 41 mm barrel, the outermost
fin 143 mm from the crank axis. Twenty fins would not fit the 80 mm finned length at the measured pitch, and a
140 mm fin would stand 26 mm outside the drawn 114 mm outline. With NACA 488's measured h and the heat sinks the
balance closes inside the flight envelope (table above), so the disagreement that motivated deeper fins is gone.
The fin size is therefore the drawing's and is not sampled: the visible hot area does not double. Whether the MD-550
copies the L550E's fins exactly is the question left; a close-up photo of the MD-550's fins would settle it
(section 15).

**Head block and plug.** They sit at the CHT sensing point (the plug seat) or slightly cooler towards the heat sink,
so they are drawn at 0.85–1.0 × CHT.

## 6. Case, drive flange, tail cone

These parts are heated by conduction and cooled by air. In a crankcase-scavenged two-stroke the
fresh charge, carrying evaporating fuel, flows through the crankcase and cools it from inside. Their
temperatures are fractions of the hotter part next to them (C: no measurement of these parts on a comparable
engine was found; NACA's cylinder surveys are of large radial four-strokes, too different to use):

* case: 0.25–0.45 × CHT;
* drive flange: 0.3–0.8 × case (a spinning disc is cooled hard by the air; it is fed by conduction along the crank);
* tail cone at the mounting face: 0.1–0.4 × case.

## 7. Carburettors: colder than the air

Fuel evaporating into the charge cools it by up to

    ΔT_evap = h_fg / (AFR · c_p) = 20–29 K

for gasoline (h_fg 300–350 kJ/kg, AFR 12.2–15.1). This is the physics behind carburettor icing, which
happens well above 0 °C ambient. The carburettor body sits between the cooled charge (internal
coupling 0.5–0.8) and the ambient air. With 30–60 % of the fuel evaporating inside it, the body can
run 15 K below ambient; heat soak from the case can lift it a little above. Range: −15 to +15…+20 K.

## 8. Emissivity: why the fins look darker than their metal

Surface data from FLIR's camera-manual table ([FLIR-TAB], each entry with its band and temperature):

* cast aluminium: as received 0.09 (total); blast-cleaned casting 0.46 (LW); strongly oxidised 0.2–0.3; polished
  0.04–0.06;
* anodised aluminium, dull black or light grey: 0.95–0.97 (LW);
* black lacquer: shiny 0.87 to dull 0.96–0.98; paints of 8 colours: 0.92–0.94 (LW);
* stainless steel: untreated sheet 0.28 (LW), rolled at 700 °C 0.45, type 18-8 oxidised at 800 °C 0.85; soot 0.95.

Each gap between two fins is a groove. An isothermal diffuse groove whose opening is the fraction F
of its inner surface looks like a surface of emissivity ε_a = ε / (ε + (1 − ε) F) (Modest,
*Radiative Heat Transfer*, ch. 6). Here F = gap / (gap + 2 · fin height) = 0.092. Weighted with the fin
edges (37.5 % of the pitch):

* bare fins (ε = 0.09): apparent emissivity 0.36;
* anodised or painted fins (ε = 0.97): apparent emissivity 0.99.

The heat sinks are black on both datasheet photos: black lacquer 0.87–0.98, in grooves 6.3 mm wide and 21 mm deep
(F = 0.13) with fin edges 24 % of the pitch, gives 0.95–0.99.

| Surface | Range | From |
| --- | --- | --- |
| Head block and plug | 0.09–0.97 | bare cast aluminium to dull anodised or painted (finish not resolved on the KAL) |
| Heat sinks | 0.95–0.99 | black finish, raised by the grooves |
| Fin pack | 0.36–0.99 | groove cavities |
| Case and covers; tail cone; skin | 0.87–0.98 | black paint |
| Carburettors | 0.04–0.30 | bare die-cast aluminium |
| Drive flange | 0.06–0.95 | plated to gold-anodised |
| Exhaust stacks | 0.28–0.95 | stainless, heat-tinted → oxidised → sooted |

## 9. SWIR self-emission

For an opaque surface ε = 1 − ρ (Kirchhoff), so each part adds (1 − ρ) × its Planck radiance in
0.9–1.7 µm, converted to render units with the project's calibration (`signatures.yaml` hot_emission: the
renderer's sun disc of 140 units at 30–60° elevation against 250 W/m² of direct sun in the band, which the ASTM G173-03
air-mass-1.5 standard spectrum confirms: 249.6 W/m²). In render units:

| Surface temperature | Black-body emission | Compared with night airglow (2×10⁻⁵ – 2×10⁻⁴) |
| --- | --- | --- |
| 50 °C | 1.4×10⁻⁶ | < 0.1 × the faintest airglow: skipped |
| 100 °C | 5.7×10⁻⁵ | 0.3–3 × |
| 200 °C | 9.0×10⁻³ | ~45–450 × |
| 500 °C | 17 | larger than sunlit black paint (2.2) |

So a running engine glows in night SWIR, and the hottest stubs are visible even by day. The
renderer therefore adds emission for every target part above 50 °C in every scene
(`render/signatures.swir_self_emission`). The 50 °C threshold is where a black surface drops below
0.1 × the faintest airglow, so nothing visible is cut off. SWIR draws its own temperatures (spec 7.2
rule 4).

## 10. Exhaust plume: small in LWIR, not settled in SWIR, large in MWIR

The plume is not rendered. **LWIR.** The 8–14 µm band is the atmospheric window *because* H2O and CO2 hardly absorb
there, and by Kirchhoff a gas that hardly absorbs hardly emits. The calculation (`thermal/engine_heat.py`
plume_lwir_emissivity, plume_to_stub_lwir_ratio):

    ε_plume = min(1, k [1 − exp(−σ_c(T_g, e) d) + 0.03 (1 − exp(−d / L_sat))]),   L_sat = 50 m × 420 ppm / x_CO2
    plume / stub contrast = (A_plume / A_stub) · ε_plume [L(T_g) − L(T_a)] / (ε_stub [L(T_w) − L(T_a)])

* **Water-vapour continuum** σ_c: the project's own (`thermal/atmosphere.py`, Roberts, Selby and Biberman 1976) at
  the outlet gas temperature and a vapour pressure of 12 % H2O; over the outlet, d = 31.5 mm, its optical depth is of
  order 10⁻⁶ (it weakens with temperature as exp(1800 (1/T − 1/296))).
* **CO2 13–14 µm band edge:** 3 % of the band (`render.yaml` edge_co2), saturating within ~50 m of ambient air at
  420 ppm; at 10 % CO2 it saturates within L_sat = 0.21 m, so over the outlet it reaches ~0.4 %.
* **Plume emissivity:** 0.42 % (k = 1), 1.25 % with a ×3 margin for hot bands absent from both.
* **Area:** A_plume / A_stub = 3.2, the region hotter than half the outlet excess (about 5 × 1.5 outlet diameters for
  a jet bent over by an equal cross-flow; generous).

At the band centres (gas at the outlet 442–553 K above ambient, `plume.csv`), the plume's LWIR contrast as a share
of the stub's:

| Setting | Sooted stub (ε 0.95) | Centre (ε 0.615) | Shiny stub (ε 0.28) |
| --- | --- | --- | --- |
| low_power | 4.7 % (14 % with ×3) | 7.1 % (21 %) | 15 % (46 %) |
| nominal | 3.6 % (11 %) | 5.4 % (16 %) | 11.5 % (34 %) |
| high_power | 3.0 % (9 %) | 4.5 % (13.5 %) | 9.6 % (29 %) |

So in LWIR the plume adds a few per cent to a sooted or weathered stub and up to 15 % (46 % with the margin) to a
shiny one. That is inside the stubs' own uncertainty: their emissivity is only known to 0.28–0.95, a factor of 3.4
in their contrast, so a stub drawn anywhere in that range already covers a stub-plus-plume of the same appearance.
Leaving the plume out makes the engine slightly fainter in LWIR, so detection is if anything harder.

**SWIR: not calculated, and not shown to be small.** An earlier version of this document said the plume is weaker
still in SWIR because only weak overtone bands of H2O and CO2 fall inside 0.9–1.7 µm. That was not calculated, and
Planck's law argues against taking it for granted: the gas leaves the stub 210–250 K hotter than the stub wall, and
in 0.9–1.7 µm a black body at the gas temperature is 41–640 times brighter than one at the wall temperature
(`plume.csv`, swir_gas_over_wall_planck). With the same generous area, the plume would equal the stub if its
band-averaged emissivity reached only 0.02–0.5 % (swir_break_even_plume_eps_pct), and a few centimetres of hot water
vapour may well reach that in its 1.38 µm band. Settling it needs a sourced band model of hot H2O and CO2 (for
example the tables of NASA SP-3080, Ludwig et al. 1973, or the HITEMP line list), which this project has not yet
obtained. Until then the SWIR plume is an open item (D, section 15): its omission could make the engine fainter in
SWIR than it is, which again errs towards harder detection, but by an unknown amount.

**MWIR (3–5 µm)**, not a band of this project: the CO2 4.3 µm band is nearly opaque even over a few centimetres, so
the plume and the hot metal would dominate the signature.

## 11. What this means for detection

* **Visibility by aspect** (ray-cast on the canonical mesh, `target/kal_measure.visible_triangle_area`;
  `visible_area.csv`):
  * level head-on: the forebody hides the engine except the stub ends below it (16 cm²);
  * from the side, behind or below: 500–970 cm² of engine surface shows, 300–580 cm² of it the hottest parts (heads,
    heat sinks, fins, stubs); the heat sinks alone show 45–190 cm², most from the side.
* **Engine vs airframe in LWIR, with the heat sinks** (`tools/engine_report.py`, `lwir_contrast.csv`). Contrast
  intensity I = Σ A_visible · ε · [L(T_a + ΔT) − L(T_a)] at the centres of the signatures.yaml ranges on a 15 °C day,
  the skin blended from its underside to its upper-surface ΔT by the normal as the shader does, and the warm tail
  cone counted with the airframe. Engine / airframe in daytime:

  | Setting | side, below (az 90°, el −30°) | behind, below (az 180°, el −20°) | from below (el −90°) | head-on level |
  | --- | --- | --- | --- | --- |
  | low_power | 1.9 | 2.7 | 1.15 | 0.5 |
  | nominal | 2.4 | 3.5 | 1.4 | 0.7 |
  | high_power | 2.8 | 4.0 | 1.6 | 0.8 |

  Counting the tail cone with the engine instead, the nominal ratios are 2.9 and 4.1. The heat sinks supply 17–35 %
  of the engine's contrast at these aspects (35 % from the side). Compared like for like with the figures quoted
  before the heat sinks (all non-engine surfaces at one underside skin temperature, ε 0.9: 2.7× and 3.8× at nominal),
  the heat sinks raise them to 3.2× and 4.2× (`engine_over_airframe_previous_definition`); the lower figures in the
  table come from counting the airframe more completely, not from a weaker engine. At night the skin sits within
  ~1 K of the air and the engine is about 5–20× the airframe from below, the side and behind. At 2–8 px the engine
  can be most of the thermal signature.
* **Thermal crossover:** the engine setting is independent of time of day, so at dusk the skin loses its contrast
  but the engine does not.
* **Night RGB: the red glow of the hottest stubs (modelled).** Hot metal glows: by Kirchhoff an opaque surface adds
  (1 − albedo) × its Planck radiance in each colour band (r 0.60–0.70, g 0.50–0.60, b 0.40–0.50 µm, a colour camera
  with an IR-cut filter, C+), converted to render units like SWIR: × the renderer's sun disc (140 units) / the direct
  sun in the same band from the ASTM G173-03 standard spectrum (r 125.9, g 133.8, b 115.1 W/m²)
  (`render/signatures.rgb_self_emission`, `render/rgb.py`). Below 370 °C a black surface's red glow is under 0.1 × the
  faintest night sky (`render.yaml` airglow_rgb 3×10⁻⁶), so nothing is added there. A black stub's glow passes the
  faintest night-sky floor at 422 °C and the brightest at 479 °C; the colour is a dull red (r > 20 g, g > 20 b). Only the
  stubs get that hot (heads and fins are capped at 250 °C). The RGB pass draws its own temperatures from the LWIR
  ranges (spec 7.2 rule 4). In the scene sampler's own draws (100,000 Monte Carlo scenes, `night_glow.csv`) the stub
  glow outshines the scene's night sky in **2.6 % of all scenes** (10.6 % of night scenes): the hottest nominal and
  high-power stubs, as expected (about 1–2 % was the earlier rough estimate).

## 12. Sun glints in LWIR (modelled)

FLIR's radiometry note warns that a reflection of the sun reads as a sun glint whose radiometric temperature "can be
hundreds of degrees inaccurate" (16-0539, p. 6, `docs/sources/flir/flir_16-0539_page-06.png`). In 8–14 µm the sun is
a ~4,970 K source: rare, small and very bright glints when sun, surface and camera line up. Every surface with
ε < 1 therefore also reflects the sun's own in-band radiance specularly, in the same emission-only shader
(`render/lwir.py` _glint, `render/signatures.sun_glint_params` and glint_radiance):

    L_glint = (1 − ε) · f_s · E_sun / (π σ²) · exp(−2 (1 − R·S) / σ²),   σ² = lobe² + r_sun²,   R = 2 (N·I) N − I
    E_sun = E_toa · (1 − ε_z)^(1 / sin el)

* E_toa = 1.261 W/m², the 8–14 µm irradiance of the ASTM E490 standard spectrum above the atmosphere (a 5772 K black
  body would give 1.508). r_sun = 695,700 km / 1 au = 0.2665° (IAU 2015 B3, IAU 2012 B2).
* The atmosphere: the scene's own clear sky (Idso 1981 zenith emittance ε_z, Granqvist–Hjortsberg slant law), so the
  sun's in-band transmittance follows the scene's humidity and the sun's elevation (0–0.73; median 0.22).
* The lobe: an energy-conserving Gaussian around the mirror direction R of the view ray I about the normal N, half-width
  lobe = 0.5–6° (near-mirror metal to satin paint, C) widened by the sun's own disc; f_s = 0.3–1.0 is the specular
  share of the reflectance (C). A perfect mirror returns the sun's own radiance; the integral of the lobe over solid
  angle returns (1 − ε) f_s E_sun (tests).
* One draw per scene, on its own random stream; none at night or with the sun below the horizon (66 % of scenes have
  the sun up).

At the exact mirror angle, how much too hot a 300 K surface reads (`sun_glint.csv`, 100,000 scene draws):

| Surface (ε) | median draw | 90th percentile | 99th percentile |
| --- | --- | --- | --- |
| Bare cast aluminium (0.09) | +16 K | +123 K | +429 K |
| Untreated stainless stub (0.28) | +12 K | +103 K | +365 K |
| Stub, centre of its range (0.61) | +7 K | +63 K | +236 K |
| Painted skin (0.93) | +1.4 K | +15 K | +70 K |
| Black heat sink (0.97) | +0.5 K | +6 K | +32 K |

So glints on bare metal read hundreds of degrees too hot in the brightest draws, as FLIR warns, but only within a few
lobe widths (1–6°) of the mirror direction: small, bright points that come and go with the aspect, and nearly absent
on the black heat sinks and the painted skin. In RGB the sun lamp already produces its own specular glints.

**Checked in Blender.** `tools/check_glow_glint_blender.py` renders a unit sphere with the real LWIR material
(ε 0.09, a 6° lobe, the sun at 40°) and compares every pixel with the equation above: the rendered peak equals the
analytic one to 0.005 %, the error over the whole sphere is 2×10⁻⁵ of the glint amplitude and the spot falls on the
same pixel (`docs/reference/engine/glint_check_sphere.png`: rendered | analytic | difference × 10). A plane with the
real RGB material of a 500 °C stub in the dark renders the glow of `rgb_self_emission` to 0.03 % in each channel
(`docs/reference/engine/data/blender_glow_glint_check.json`).

## 13. Propeller

* **Speed** (rotation blur only): 4,000–7,500 rpm. The upper end is the rated speed. The lower end is the cube law at 15 % power: 7,500 × 0.15^(1/3) = 3,985 rpm.
* **Diameter upper bound:** the helical tip Mach number M = sqrt((π n D)² + V²) / sqrt(γ R T) at D = 0.64 m, 7,500 rpm and 56 m/s is 0.76 at 15 °C and 0.81 at −20 °C. Propellers are kept below about 0.8–0.85, where compressibility losses and noise rise steeply. The lower bound (0.50 m) comes from photo P6.

## 14. Did removing the performance figures change the LWIR, SWIR or MWIR output?

* **No pixel changed.** Before this derivation, the renderer drew engine temperatures directly from hand-set ranges and read no engine rating. Removing the ratings changed nothing in any band.
* **The physics does need them.** A defensible derivation of the stub and fin temperatures needs:
  * the engine's air and exhaust flow (displacement × speed);
  * its heat load (rated power and fuel consumption).

  So bore, stroke, rated power, rated speed and the SFC chart are in `target/engine_l550e.py`, used only by this
  derivation and by the propeller blur. The mass stays out: no band depends on it. The tip-speed bound is a check
  on the propeller diameter (section 13).
* **MWIR:** this project has no MWIR band (it renders RGB, LWIR 8–14 µm and SWIR 0.9–1.7 µm). Adding one would need the plume model of section 10.

## 15. Limits, and how each estimate could be validated

* No temperature here is measured on a KAL or an MD-550 in flight. The CHT and EGT bands are the operating limits
  of the closest engine with published limits (Rotax 503 UL), applied to the L550E.
* The flight envelope (speed, altitude, air temperature) comes from the spec and a press report and is not
  re-sourced here.
* Not modelled: the exhaust plume (section 10; small in LWIR, an open question in SWIR) and radiative exchange
  between engine parts (ignoring it errs slightly hot for the stubs). The night RGB glow (section 11) and the LWIR sun
  glints (section 12) are modelled.

**The temperature bands: what would replace them.** The authentic source for the L550E's own CHT and EGT limits is
Limbach's operating manual for the engine. It is not public, and by the project's scope rule it is not searched for
online. Limbach's datasheets, which are saved (`docs/sources/limbach/`), list only the optional 14 mm CHT thermocouple
at the plug seat, with no limits. If the manual, or any Limbach document with the limits, is placed in
`docs/sources/limbach/`, its bands replace the Rotax ones in `thermal/engine_heat.py` (ROTAX and POWER_SETTINGS);
`python thermal/engine_heat.py` then prints the new YAML lines, and `tests/test_engine_heat.py` fails until
`configs/signatures.yaml` carries them. Independently, any thermal camera pointed at a running small air-cooled
two-stroke (a model-aircraft or generator engine) would validate the temperature *ratios*: head, fins, case and stubs
in one image.

| Estimate | How to validate it |
| --- | --- |
| CHT / EGT bands (analogue limits) | Limbach's L550E operating manual (not public; ask Limbach), or a CHT thermocouple under a plug and an EGT probe on any running L550E-class engine |
| Part-to-part fractions, fin-root and head ratios | Any thermal camera, even a phone attachment, on a running small air-cooled two-stroke (model-aircraft or generator engine): head, fin, case and stub temperatures together |
| Local air-speed factors, propeller inflow | A pitot or hot-wire probe at the cylinders of a pusher installation in a wind tunnel or on a car roof test |
| Heat-sink h factor (fins across the stream) | The same car-roof or wind-tunnel test with a heated finned plate across the flow, against one along it |
| Pulsation factor | Depcik and Assanis (2002, SAE 2002-01-0372) correlation for exhaust heat transfer |
| MD-550 fin pack and heat-sink fins (copy of the L550E?) | A close-up photo of the MD-550's cylinders and heads, or the 3D model of the L550E head and heat sink |
| Stub emissivity state | Look at the stubs on the KAL photos at higher resolution (heat tint vs soot) |
| SWIR plume | A sourced band model of hot H2O and CO2 (NASA SP-3080 tables or HITEMP) through the section 10 equations; or an InGaAs camera on a running small two-stroke's exhaust |
| Glint lobe width and specular fraction | LWIR images of bare cast aluminium, stainless tube and painted panels swept through the sun's mirror angle |
| Colour-camera bands for the glow | The spectral response of the RGB camera used |

## References

* ASTM E490-00a (reapproved 2019). *Standard Solar Constant and Zero Air Mass Solar Spectral Irradiance Tables*. ASTM International (table as distributed by NREL, workbook AM0AM1_5.xls).
* ASTM G173-03 (reapproved 2020). *Standard Tables for Reference Solar Spectral Irradiances: Direct Normal and Hemispherical on 37° Tilted Surface*. ASTM International (table as distributed by NREL).
* Ausserer, J. K. (2016). *The Scaling of Loss Pathways and Heat Transfer in Small Scale Internal Combustion Engines*. PhD dissertation AFIT-ENY-DS-16-S-055, Air Force Institute of Technology. Peer-reviewed companion: Ausserer, J. K., Polanka, M. D., Litke, P. J. and Baranski, J. A. (2018). *J. Eng. Gas Turbines Power* 140(11), 112803.
* Biermann, A. E. and Pinkel, B. (1934). Heat transfer from finned metal cylinders in an air stream. NACA Report 488.
* BRP-Rotax (1999). *Operator's Manual for engine types 447 UL SCDI, 503 UL DCDI, 582 UL DCDI*. Initial issue 1999-05-01.
* Churchill, S. W. and Bernstein, M. (1977). A correlating equation for forced convection from gases and liquids to a circular cylinder in crossflow. *J. Heat Transfer* 99, 300–306.
* EASA (2006). Type-certificate data sheet E.082, Limbach L 1700 series engines, issue 01.
* EN 228. Automotive fuels: unleaded petrol, requirements and test methods.
* FLIR Systems (2016). *5 factors influencing radiometric temperature measurements* (16-0539). Sun glints and reflections (p. 6); Table 1 as a cross-check only (`docs/sources/flir/`).
* FLIR Systems (2017). *User's manual FLIR ETS3xx series* (T810252), ch. 21, Table 21.1 emissivity tables.
* Gnielinski, V. (1976). New equations for heat and mass transfer in turbulent pipe and channel flow. *Int. Chem. Eng.* 16, 359–368.
* Granqvist, C. G. and Hjortsberg, A. (1981). Radiative cooling to low temperatures: general considerations and application to selectively emitting SiO films. *J. Appl. Phys.* 52, 4205–4220.
* Hansen, J. E. and Travis, L. D. (1974). Light scattering in planetary atmospheres. *Space Sci. Rev.* 16, 527–610 (Rayleigh optical depth; cross-check only).
* Heywood, J. B. (1988). *Internal Combustion Engine Fundamentals*. McGraw-Hill.
* Idso, S. B. (1981). A set of equations for full spectrum and 8- to 14-µm and 10.5- to 12.5-µm thermal radiation from cloudless skies. *Water Resources Research* 17, 295–304.
* IAU (2015). Resolution B3 on recommended nominal conversion constants for selected solar and planetary properties (quoting IAU 2012 Resolution B2 on the astronomical unit). Prša, A. et al. (2016), *Astron. J.* 152, 41.
* Incropera, F. P., DeWitt, D. P., Bergman, T. L. and Lavine, A. S. (2011). *Fundamentals of Heat and Mass Transfer*, 7th ed. Wiley.
* Limbach Flugmotoren (2025). *LIMBACH L550E Technical Specification* and *LIMBACH L550EF Technical Specification* (datasheets, limflug.de).
* Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed. Academic Press.
* Roberts, R. E., Selby, J. E. A. and Biberman, L. M. (1976). Infrared continuum absorption by atmospheric water vapor in the 8–12 µm window. *Applied Optics* 15, 2085–2090.
