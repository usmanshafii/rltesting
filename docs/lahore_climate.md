# Scene conditions from Lahore's climate record (2026-10-13)

CLAUDE.md sensor set rule 6 (user, 2026-10-12): scene conditions are those of Lahore and comparable sites in the
region, from the past three years' records. This note says where the numbers come from, what they show and how the
renderer uses them. Reader: `tools/source_measurements.lahore_climate()`; data: `docs/sources/noaa_isd/`.

## Source

NOAA NCEI Integrated Surface Database (ISD), station **LAHORE CITY** (USAF 416400, WBAN 99999; 31.55 N, 74.333 E,
214 m; ICAO OPLH), hourly records (NCEI Access Data Service, dataset `global-hourly`: temperature, dew point,
visibility, manual present weather MW1-3, total sky cover GF1, layer cover GA1, station pressure MA1), and the daily
summaries (GSOD) since 1995. Lahore's airport (416410, OPLA) has only 521 synoptic reports in the window and reports the
same visibilities, so the city station is used. Field definitions: the ISD format document (visibility p. 10, sky cover
pp. 59-62, present weather pp. 103-105, which uses WMO code table 4677). Tier S1: measurements at the place the rule
names.

NOAA's archive for both Lahore stations stops on **2025-08-24**, so "the past three years" is the last three years on
record: **2022-08-25 to 2025-08-24**, 16,881 usable hours (synoptic report kept where an hour has two).

## What the record shows

| | Day (sun ≥ 15°) | Dusk (−6° to 15°) | Night (sun ≤ −18°) |
| --- | --- | --- | --- |
| Hours | 6,415 | 2,455 | 6,565 |
| Air temperature, 1 / 50 / 99 % | 8.5 / 30.0 / 43.0 °C | 6.0 / 25.0 / 42.0 °C | 7.0 / 23.0 / 36.5 °C |
| Relative humidity, 5 / 50 / 95 % | 0.22 / 0.54 / 0.88 | 0.28 / 0.67 / 0.93 | 0.40 / 0.75 / 0.93 |
| Visibility, 5 / 50 / 95 % | 2.0 / 4.0 / 5.0 km | 1.0 / 4.0 / 5.0 km | 1.0 / 4.0 / 5.0 km |
| Weather: haze or smoke / fog / rain / clear | 93 / 2 / 3 / 2 % | 92 / 4 / 2 / 1 % | 93 / 5 / 2 / 0.4 % |
| Clear sky (0 okta) / 6-8 oktas | 32 % / 20 % | 39 % / 13 % | 52 % / 10 % |

* **Smoke and haze in almost every hour.** The commonest present-weather codes are 04 (visibility reduced by smoke,
  8,518 reports) and 05 (haze, 4,488), then 10 (mist) and 44/45 (fog). Reported visibility is 2-5 km most of the
  time; the station reports more than 5 km in 1.4 % of hours (8 km at most). This is Lahore's smog; the old generic
  "clear 20-60 km" weather does not occur in the record.
* **Seasons** (monthly means): January 11.6 °C, vapour pressure 10.4 hPa, median visibility 2 km, haze or fog in 99 %
  of hours; May-June 32-34 °C, dry (RH 0.43-0.49); July-September 30-31 °C at 30-33 hPa (monsoon); November-December
  haze or fog 99 % of hours.
* **Trend.** The three 12-month blocks average 24.80, 24.56 and 25.59 °C (vapour pressure 20.7, 19.4, 20.9 hPa; haze or
  fog in 95, 96 and 98 % of hours): no steady change inside the window beyond year-to-year swings. Over 1995-2024 (26
  complete years of GSOD daily means) the station's annual mean has no warming trend (−0.12 °C per decade; the
  late-1990s gap and station changes make this a weak estimate). Averaging the three years is therefore the right
  climatology for the scenes, as the user asked.

## How the renderer uses it

`configs/render.yaml` `climate` (enabled; M1's overlay keeps its earlier conditions): each scene draws one station hour
in its own lighting state, every hour of the window equally likely, so the scenes follow the climatology jointly
(cold smoggy winter mornings stay cold, humid and murky; dry pre-monsoon afternoons stay hot and dry). From that hour:

* air temperature (replaces the spec's −20..+50 °C draw and the "desert day" option);
* humidity: the hour's own RH (no climatological dew-point cap), which drives the LWIR sky and the thermal atmosphere's
  vapour (`docs/thermal_bands.md`);
* weather kind: fog (codes 40-49 or visibility < 1 km), haze (smoke, haze, mist, dust, rain, or visibility < 5 km),
  clear otherwise; the band extinction ratios follow the kind;
* visibility: the reported value; an unobscured hour at the 5 km reporting ceiling is drawn log-uniformly up to 15 km
  (C: the station almost never reports beyond 5 km);
* cloud cover: the total-sky oktas mapped to the procedural cloud mask's coverage as oktas / 8 (C: the mask's threshold
  is not calibrated against oktas); 0 oktas: no clouds;
* station pressure for the thermal atmosphere.

* the RGB (and SWIR) sky's aerosol: Blender's Nishita dust density set from the hour's visibility, dust = (3.912 / V)
  / 2.22e-5 m-1 (Blender 4.2 `intern/sky/source/sky_nishita.cpp`: aerosol scattering 2e-5 m-1 per unit dust,
  extinction 1.11 x, 1.2 km scale height), so the sky carries the same haze the loader applies to the target and the
  ground (4 km gives 44; the old 0.5-6 range was a 29-350 km sky). The training loader therefore only jitters
  visibility by 0.7-1.4 x around the scene's own (the sky is baked at render time).

The sun's position stays the sampler's own draw (geometry, sun-near-line-of-sight coverage); the hour's lighting state
matches it. The engine's heat model was derived over −20..+50 °C air, which contains Lahore's 1-99 % range.

## Limits and checks

* The record is one urban station; the region's other stations in NOAA's archive (Patiala, Amritsar, Sialkot,
  Faisalabad) were not merged. Their radiosondes are used for the vertical profiles (`docs/thermal_bands.md`).
* Visibility is an observer's estimate against landmarks, coarse (whole kilometres above 1 km) and capped near 5 km.
  A transmissometer or sun-photometer record (aerosol optical depth, e.g. AERONET Lahore) would calibrate the
  extinction directly, including its wavelength dependence for the thermal bands.
* Rain hours (2-3 %) are rendered as haze (no rain model).
