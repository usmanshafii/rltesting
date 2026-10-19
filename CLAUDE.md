# Project rules for AI assistants

This folder is the MD-550 / L550E engine version of the KAL multispectral detection FYP (README.md). The
REB 30 electric version lives in `../fyp` and is not to be reintroduced here.

## Evidence rule (user, 2026-10-08; sources rule added 2026-10-10)

1. **Look up authentic sources first; estimate only what has none.** Before estimating any number, search for
   an authentic, verifiable source for it: manufacturer datasheets, manuals and drawings, standards, regulator
   documents, peer-reviewed papers and theses with their method, standard textbooks and handbooks, measurements,
   and documents the user supplies. Cite each next to the number (document, table or page), and add a test or a
   validation note wherever the value can be checked. The register of sources in use is `docs/sources/README.md`.
2. **Judge every source's reliability and its relevance to this context, and do not use weak sources.**
   - Tiers: **S1** authoritative and about the exact item (its manufacturer's datasheet, manual or drawing; a
     standard; a regulator's document; a peer-reviewed measurement of the item). **S2** authoritative, about a
     close analogue or general physics (a comparable product's manufacturer manual, peer-reviewed work, textbooks,
     handbooks, government research reports). **S3** secondary (encyclopedias such as Wikipedia, vendor white
     papers or application notes that quote values without their conditions, trade press): only to find an S1/S2
     source or as a cross-check. **S4** weak, never used for values (forums, blogs, marketing and distributor
     listings, search-engine snippets, AI-generated text, unsourced specifications).
   - The project's own build spec (`FYP_Build_Spec_*.md`) records decisions, not evidence: a range taken from it
     is not sourced until it is traced to an S1/S2 source.
   - Relevance: note what the source actually measured or specifies (item, condition, temperature, waveband,
     power setting, measuring point) and how close that is to this use. An authoritative source about something
     different (another engine type, another waveband, an unspecified surface state) is a cross-check, not a
     value. Example: FLIR's 12-page radiometry note (16-0539) is a genuine FLIR document but S3 for emissivities,
     because its Table 1 omits waveband and temperature; FLIR's camera-manual Table 21.1 gives both and is used.
   - Check the document itself, not only where it is hosted: a mirror copy of a manufacturer's manual is usable
     once its content and identifiers are verified; record its URL and SHA-256 in `docs/sources/README.md`.
3. **Save every authentic source used to build anything (user, 2026-10-10).** Keep, under
   `docs/sources/<source>/`: the document itself (PDF, or a saved copy of a web page), images of the pages cited,
   any images taken from it (photos, drawings, charts), and every number read off it as data (CSV/JSON). Keep the
   code that read the numbers (`tools/source_measurements.py`) so the reading can be repeated, register the source
   in `docs/sources/README.md` (URL, retrieval date, pages, tier, SHA-256), and list which component uses it in
   `SOURCES_GUIDE.txt`. Third-party PDFs and images are git-ignored (kept locally, not redistributed); the data,
   register and guide are committed.
4. **Least delta where no source exists.** Choose the value with the least expected error: the centre of
   independent bounds (the geometric mean when the uncertainty is a factor), cross-checked with physics
   (energy, mass and heat balances, momentum theory, Planck's law). Keep a range wide enough to cover the
   plausible truth so the detector trains across it, label it (grade C or C+), and write down how it could be
   validated.
5. **Never present an estimate as a measurement.** Grades A–D as in the `configs/signatures.yaml` header.
   When an authentic source and an estimate disagree, the source wins and the range is widened to cover both
   until the conflict is explained. When two sources disagree, prefer the more relevant one and widen the range
   to cover both.
6. **Traceable physics.** Every signature number must be traceable to an equation and its inputs
   (`thermal/engine_heat.py`, `docs/engine_signature_physics.md`); configs are tied to the derivations and the
   sourced anchors by tests (`tests/test_engine_heat.py`).

## Sensor set rule (user, 2026-10-12)

1. **The current sensors are four: RGB, LWIR uncooled, LWIR cooled and MWIR.**
2. **SWIR is deferred until further notice.** Leave the existing SWIR code, configs, data and documents in place
   (do not delete them), but do not extend, render, calibrate or source new SWIR work unless the user re-enables it.
   Where a document or result lists SWIR, mark it as deferred rather than as a current band.
3. **Any work is done for all four sensors.** Rendering, signatures, physics derivations, sources, sensor and noise
   models, datasets, training, evaluation, tests and documents cover RGB, LWIR uncooled, LWIR cooled and MWIR
   together. If a change truly does not apply to one of them, say so and why (for example: the night glow of hot
   parts is an RGB effect, but the same hot parts must still be checked in MWIR, where they are far brighter).
4. **LWIR uncooled and LWIR cooled are separate sensors.** They share the scene physics where their wavebands
   overlap, but each has its own waveband, sensor and noise model and results are reported for each. A loader-side
   noise swap on the uncooled render (as in `tools/m1_det2.py` COOLED) is not enough where the cooled detector's
   band differs.
5. **Teacher and student (user, 2026-10-12).** The privileged, training-only teacher sees all four sensors: RGB,
   LWIR uncooled, LWIR cooled and MWIR. The deployable student sees RGB and LWIR uncooled. This replaces the build
   spec's SWIR-teacher design (`FYP_Build_Spec_*.md` carries a dated note until its sections are revised).
6. **Scene conditions are those of Lahore and comparable sites in the region (user, 2026-10-12):** air temperature,
   humidity, haze, smog and dust, visibility, cloud and season, sourced per the evidence rule (for example national
   meteorological normals and station records). Ranges taken from the spec (such as −20 to +50 °C) are re-derived
   from those sources.

## Scope

- Synthetic-data detection research for the FYP: the externally observable shape and signatures of the
  target for the project's sensors (RGB, LWIR uncooled, LWIR cooled, MWIR; SWIR deferred, see the sensor set rule).
- Use local information about the target. Do not research the target munition online, and do not select or
  model the equipment of a specific armed force; sensor models stay generic and openly documented
  (`configs/sensors.yaml`). Online look-ups are for civil sources: the L550E's manufacturer data, comparable civil
  engines, physics and material data, published detector and camera datasheets (by class, not by operator), and
  meteorological and climate records.
- Geometry changes to `target/engine_l550e.py` are followed by `python tools/draw_reference.py` and the Blender
  previews (`target/build_target.py`, `tools/hero_preview.py`) so the reference images match the model.
