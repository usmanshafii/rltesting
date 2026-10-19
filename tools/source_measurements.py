"""Re-derive every number read off a source image, from the saved copies in docs/sources/.

    python tools/source_measurements.py      -> docs/sources/limbach/measurements.json,
                                                docs/sources/limbach/sfc_chart_digitised.csv,
                                                docs/sources/naca/fig9_digitised.csv and fig9_points.csv,
                                                docs/sources/astm/band_integrals.csv, e490_am0_model_grid.csv,
                                                docs/sources/iau/constants.json, docs/sources/hirth/readings.json;
                                                checks docs/sources/cameras/readings.json (quotes, page images)
                                                and docs/sources/sensor_noise/readings.json and
                                                docs/sources/privileged/readings.json and
                                                docs/sources/detection_methods/readings.json (quotes, SHA-256)

Each drawing is scaled by its own labelled dimension lines; each chart by its own labelled gridlines. The images
are the manufacturer's / NACA's (git-ignored, see docs/sources/README.md); the JSON and CSV outputs are committed and
tied to the model by tests/test_engine_heat.py and tests/test_glow_and_glints.py. The spectral tables (ASTM G173-03,
E490) are integrated over the renderer's bands; two text readings use the PDFs' text layers. NumPy + Pillow, and
PyMuPDF for the text readings.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "docs", "sources")
LIM = os.path.join(SRC, "limbach")


def _grey(path):
    return np.asarray(Image.open(path).convert("L")).astype(float)


def _runs(mask, gap=0):
    """(start, end) index pairs of True runs, bridging gaps of up to `gap` False samples."""
    out, cur, last = [], None, None
    for i, v in enumerate(mask):
        if v:
            if cur is None:
                cur = i
            last = i
        elif cur is not None and i - last > gap:
            out.append((cur, last))
            cur = None
    if cur is not None:
        out.append((cur, last))
    return out


def _longest_row_run(a, rows, cols, thr, min_len):
    """Rows (in `rows`) whose dark pixels (< thr) form one run longer than min_len: (row, x0, x1)."""
    found = []
    for y in rows:
        for s, e in _runs(a[y, cols[0]:cols[1]] < thr, gap=3):
            if e - s >= min_len:
                found.append((y, s + cols[0], e + cols[0]))
    return found


# ============================================================================ Limbach L550EF front view
def ef_front_view() -> dict:
    """[L550EF-DS] p. 2 front view: 392 mm plug-to-plug line and the 128 + 235 mm height lines set the scale;
    the fins of the left bank are measured on rows above the barrel."""
    a = _grey(os.path.join(LIM, "images", "L550EF_drawing_front_view.png"))
    top = _longest_row_run(a, range(30, 80), (0, a.shape[1]), 235, 500)
    y_line, x0, x1 = top[0]
    px_mm_w = (x1 - x0) / 392.0
    vert = [(x, s, e) for x in range(60, 110) for s, e in _runs(a[:, x] < 235, gap=3) if e - s > 500]
    xv, yv0, yv1 = vert[0]
    px_mm_h = (yv1 - yv0) / (128.0 + 235.0)
    scale = 0.5 * (px_mm_w + px_mm_h)
    xc, y_axis = 0.5 * (x0 + x1), yv0 + 128.0 * scale
    fins = []
    for dz in (45.0, 50.0, 55.0):
        y = int(round(y_axis - dz * scale))
        cen = [0.5 * (s + e) + 120 for s, e in _runs(a[y, 120:470] < 160)]
        mm = sorted(((xc - c) / scale for c in cen), reverse=True)
        outer = [m for m in mm if m > 115.0]                       # outside the starter ring gear
        pairs = [(outer[i], outer[i + 1]) for i in range(len(outer) - 1) if outer[i] - outer[i + 1] < 3.5]
        centres = [0.5 * (p + q) for p, q in pairs]
        fins.append(dict(z_mm=dz, centres_mm=[round(c, 2) for c in centres],
                         face_spacing_mm=[round(p - q, 2) for p, q in pairs]))
    allc = sorted({round(c, 1) for f in fins for c in f["centres_mm"]}, reverse=True)
    pitches = [p - q for p, q in zip(allc, allc[1:]) if 5.5 < p - q < 8.0]
    thick = [t for f in fins for t in f["face_spacing_mm"]]
    return dict(source="[L550EF-DS] p. 2 front view", scale_px_per_mm=round(scale, 4),
                scale_check=dict(width_392=round(px_mm_w, 4), height_363=round(px_mm_h, 4)),
                fin_pitch_mm=round(float(np.median(pitches)), 2), fin_face_spacing_mm=round(float(np.median(thick)), 2),
                outermost_fin_mm=round(max(allc), 1), rows=fins)


# ============================================================================ Limbach L550EF side view
def ef_side_view() -> dict:
    """[L550EF-DS] p. 2 side view: the 300 + 204 mm line sets the scale; the outlines of the two cylinder ends of
    one bank give their width along the crank axis and their spacing."""
    a = _grey(os.path.join(LIM, "images", "L550EF_drawing_side_view.png"))
    top = _longest_row_run(a, range(60, 90), (0, a.shape[1]), 235, 700)
    _, x0, x1 = top[0]
    scale = (x1 - x0) / 504.0
    frac = (a[230:380, 120:500] < 170).mean(axis=0)
    cols = [i + 120 for i, f in enumerate(frac) if f > 0.45]
    edges = [0.5 * (s + e) + cols[0] for s, e in _runs(np.isin(np.arange(cols[0], cols[-1] + 1), cols), gap=2)]
    left, mid, right = edges[0], edges[len(edges) // 2], edges[-1]
    return dict(source="[L550EF-DS] p. 2 side view", scale_px_per_mm=round(scale, 4),
                outline_edges_px=[round(e, 1) for e in edges],
                cylinder_end_width_mm=round((right - left) / 2.0 / scale, 1),
                cylinder_pitch_mm=round((right - left) / 2.0 / scale, 1))


# ============================================================================ Limbach L550E front view
def e_front_view() -> dict:
    """[L550E-DS] p. 2 front view: four nested width lines (410 plug caps, heat sinks, no heat sinks, stack centres);
    scale from the 410 mm line, cross-checked by the 364 mm across the heat sinks measured on the L550EF sheet. The
    exhaust stacks (straight tube walls) and the vertical dimension lines (from the crank axis) are measured."""
    a5 = _grey(os.path.join(LIM, "images", "L550E_drawing_strip2_front_side_views.png"))
    lines = {}
    for y in range(125, 178):
        for s, e in _runs(a5[y, :870] < 235, gap=3):
            if e - s > 250:
                lines.setdefault((s, e), y)
    widths = sorted({e - s for s, e in lines}, reverse=True)
    w = [widths[0]] + [x for x in widths[1:] if all(abs(x - y) > 10 for y in widths[:widths.index(x)])]
    scale = w[0] / 410.0
    x_outer = [k for k in lines if k[1] - k[0] == w[0]][0]
    xc, y_axis = 0.5 * (x_outer[0] + x_outer[1]), 456.0
    walls = []
    for y in (205, 215, 230, 250, 270):
        for lo, hi in ((330, 420), (610, 700)):
            cen = [lo + 0.5 * (s + e) for s, e in _runs(a5[y, lo:hi] < 200)]
            walls.append([(c - xc) / scale for c in cen])
    od = [abs(p[1] - p[0]) for p in walls if len(p) == 2]
    centre = [abs(0.5 * (p[0] + p[1])) for p in walls if len(p) == 2]
    stacked = np.vstack([_grey(os.path.join(LIM, "images", f)) for f in
                         ("L550E_drawing_strip1.png", "L550E_drawing_strip2_front_side_views.png",
                          "L550E_drawing_strip3_chart_top.png")])
    y0 = 596 + y_axis
    vdims = {}
    for x in range(100, 180):
        for s, e in _runs(stacked[600:1500, x] < 235, gap=3):
            if e - s > 60:
                vdims[(round((y0 - (e + 600)) / scale, 1), round((y0 - (s + 600)) / scale, 1))] = x
    return dict(source="[L550E-DS] p. 2 front view", scale_px_per_mm=round(scale, 4),
                width_lines_mm=[round(x / scale, 1) for x in w[:4]],
                stack_od_mm=round(float(np.median(od)), 1), stack_centre_mm=round(float(np.median(centre)), 1),
                vertical_dims_mm=sorted(vdims), note="z from the crank axis, + towards the exhaust stacks")


# ============================================================================ Limbach L550E SFC chart
def sfc_chart() -> list[tuple]:
    """[L550E-DS] p. 2 full-throttle chart: x 4,000-7,500 rpm and y 0-600 g/HPh set by the chart's own gridlines
    (4,000 / 7,500 rpm at x 151 / 508.5; 20 g/HPh per 30.05 px, 500 g/HPh at y 361). The SFC curve is the light-grey
    dash-dot line: its pixels (150-222) clear of the gridlines and of the black power curve, binned per 250 rpm,
    between 320 and 450 g/HPh. Where the curve runs along a gridline (above ~6,600 rpm, the 340 line) no pixel is
    free of it and the bin is reported on that gridline (+-3 g/HPh)."""
    a = _grey(os.path.join(LIM, "crops", "L550E_chart_stitched.png"))
    x4000, x7500, y500, px_unit = 151.0, 508.5, 361.0, 1.5025
    rpm_of = lambda x: 4000 + (x - x4000) * 3500 / (x7500 - x4000)  # noqa: E731
    val_of = lambda y: 500 - (y - y500) / px_unit  # noqa: E731
    hgrid = {int(round(y500 + k * 20 * px_unit)) + d for k in range(-12, 26) for d in range(-3, 4)}
    vgrid = {int(round(x4000 + k * (x7500 - x4000) / 14)) + d for k in range(15) for d in range(-3, 4)}
    black = a < 120
    near_black = np.zeros_like(black)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            near_black |= np.roll(np.roll(black, dy, axis=0), dx, axis=1)
    y_lo, y_hi = int(y500 + (500 - 450) * px_unit), int(y500 + (500 - 320) * px_unit)
    bins = {}
    for x in range(int(x4000) + 4, int(x7500) - 3):
        if x in vgrid:
            continue
        for y in range(y_lo, y_hi):
            if y in hgrid or near_black[y, x] or not 150 <= a[y, x] <= 222:
                continue
            b = int(round((rpm_of(x) - 4125) / 250.0)) * 250 + 4125
            bins.setdefault(b, []).append(val_of(y))
    out = []
    for b in range(4125, 7500, 250):
        v = bins.get(b, [])
        out.append((b, round(float(np.median(v)), 1) if len(v) >= 3 else "on 340 gridline"))
    return out


# ============================================================================ NACA Report 488 fig. 9
def naca_fig9(points_out: list | None = None) -> list[dict]:
    """NACA Report 488 fig. 9 (average q vs average fin space, 6 air speeds), scaled by its own labelled gridlines;
    each speed line is fitted on log axes and evaluated at s = 0.175 in. points_out collects the raw readings."""
    a = _grey(os.path.join(SRC, "naca", "naca488_fig9_300dpi.png"))
    gx = np.array([(102, .02), (302, .04), (415, .06), (497, .08), (758, .2), (879, .3), (958, .4), (1022, .5)])
    gy = np.array([(76, .40), (273, .20), (470, .10), (535, .08), (616, .06), (667, .05), (733, .04), (813, .03),
                   (929, .02), (1126, .01)])
    bx = np.polyfit(np.log10(gx[:, 1]), gx[:, 0], 1)
    by = np.polyfit(np.log10(gy[:, 1]), gy[:, 0], 1)
    speeds = [150, 110, 76, 56, 40, 30]
    pts = {v: [] for v in speeds}
    for x in range(370, 890, 10):
        cen = [0.5 * (s + e) + 150 for s, e in _runs(a[150:900, x] < 90) if e - s >= 1]
        if len(cen) == 6:
            for v, y in zip(speeds, sorted(cen)):
                pts[v].append((np.log10(10 ** ((x - bx[1]) / bx[0])), (y - by[1]) / by[0]))
    if points_out is not None:
        points_out.extend(dict(air_speed_mph=v, fin_space_in=round(10 ** ls, 4), q_btu_sqin_f_hr=round(10 ** lq, 5))
                          for v in speeds for ls, lq in pts[v])
    rows = []
    for v in speeds:
        p = np.array(pts[v])
        m, c = np.polyfit(p[:, 0], p[:, 1], 1)
        rows.append(dict(air_speed_mph=v, fitted_slope=round(m, 3), n_points=len(p),
                         **{"q_at_0.175in": round(10 ** (c + m * np.log10(0.175)), 4),
                            "q_at_0.164in": round(10 ** (c + m * np.log10(0.164)), 4)}))
    return rows


# ============================================================================ ASTM G173-03 and E490 spectra
ASTM_BANDS_NM = {"rgb_r": (600, 700), "rgb_g": (500, 600), "rgb_b": (400, 500), "swir": (900, 1700),
                 "lwir": (8000, 14000), "lwir_cooled": (7700, 9300), "mwir_mct": (3700, 4800),
                 "mwir_insb": (3600, 4900), "mwir_hot": (3600, 4200), "g173_mwir_window": (3600, 4000)}


def _spectrum(name):
    """Two-column (wavelength nm, W m-2 nm-1) rows of a table exported verbatim from NREL's AM0AM1_5.xls."""
    with open(os.path.join(SRC, "astm", name), encoding="utf-8") as f:
        rows = list(csv.reader(f))[2:]
    return np.array([[float(x) for x in r] for r in rows])


def _band_integral(lam, e, lo, hi):
    """Trapezoid integral over [lo, hi] nm on log-log interpolation (the E490 table steps 1 um above 10 um, where
    the spectrum falls as a power law)."""
    g = np.linspace(lo, hi, int(hi - lo) + 1)
    return float(np.trapz(np.exp(np.interp(np.log(g), np.log(lam), np.log(np.maximum(e, 1e-300)))), g))


def e490_model_grid() -> list[dict]:
    """ASTM E490 AM0 spectral irradiance on the thermal model's wavenumber grid (thermal/atmosphere_bands.W_GRID,
    700-2800 cm-1 every 5 cm-1), per cm-1: log-log interpolation of the table in wavelength, E_w = E_lambda lambda^2
    / 1e7. The renderer's MWIR sun reads this table (docs/sources/astm/e490_am0_model_grid.csv); the workbook export
    it comes from is git-ignored. Values are written with full double precision (repr), so the renderer gets the same
    numbers as from the workbook itself."""
    e490 = _spectrum("ASTM_E490_AM0_from_NREL_workbook.csv")
    lam_nm, e_nm = e490[:, 0], e490[:, 1]
    w = np.arange(700.0, 2800.0 + 0.1, 5.0)
    lam = 1e7 / w
    e = np.exp(np.interp(np.log(lam), np.log(lam_nm), np.log(np.maximum(e_nm, 1e-300))))
    e_w = e * lam ** 2 / 1e7
    return [dict(wavenumber_cm1=repr(float(a)), wavelength_um=repr(float(1e4 / a)), e490_w_m2_per_cm1=repr(float(b)))
            for a, b in zip(w, e_w)]


def astm_spectra() -> list[dict]:
    """In-band irradiances of the ASTM G173-03 AM1.5 spectra (extraterrestrial, global on a 37 deg tilt, direct +
    circumsolar) and of the ASTM E490 AM0 spectrum, over the bands the renderer calibrates on. First rows: the
    tables' totals, which must be the standards' own (G173: 1000.4 global, 900.1 direct; E490: 1366.1 W/m2)."""
    g173 = _spectrum("ASTM_G173-03_from_NREL_workbook.csv")
    e490 = _spectrum("ASTM_E490_AM0_from_NREL_workbook.csv")
    rows = [dict(band="g173_total", lo_nm=280, hi_nm=4000,
                 **{k: round(float(np.trapz(g173[:, j], g173[:, 0])), 2)
                    for j, k in ((1, "g173_extraterrestrial"), (2, "g173_global_tilt"), (3, "g173_direct"))},
                 e490_am0=""),
            dict(band="e490_total", lo_nm=119.5, hi_nm=1e6, g173_extraterrestrial="", g173_global_tilt="",
                 g173_direct="", e490_am0=round(float(np.trapz(e490[:, 1], e490[:, 0])), 2))]
    for name, (lo, hi) in ASTM_BANDS_NM.items():
        r = dict(band=name, lo_nm=lo, hi_nm=hi)
        for j, k in ((1, "g173_extraterrestrial"), (2, "g173_global_tilt"), (3, "g173_direct")):
            r[k] = round(_band_integral(g173[:, 0], g173[:, j], lo, hi), 3) if hi <= 4000 else ""
        r["e490_am0"] = round(_band_integral(e490[:, 0], e490[:, 1], lo, hi), 4)
        rows.append(r)
    return rows


# ============================================================================ IAU 2015 Resolution B3, Hirth datasheet
def _pdf_text(path, page=None):
    import fitz  # PyMuPDF, only for the two text readings below
    doc = fitz.open(path)
    return "".join(p.get_text() for i, p in enumerate(doc) if page is None or i == page)


def iau_b3() -> dict:
    """Nominal solar values of IAU 2015 Resolution B3 and the astronomical unit of IAU 2012 Resolution B2 (quoted in
    the same document), read from its text."""
    import re
    t = _pdf_text(os.path.join(SRC, "iau", "IAU2015_Resolution_B3_nominal_solar_planetary_values.pdf"))
    num = lambda pat: re.search(pat, t).group(1).replace(" ", "")  # noqa: E731
    return dict(source="IAU 2015 Resolution B3 (iau.org)",
                solar_radius_m=float(num(r"R\s*N\s*⊙\s*=\s*([\d.]+)\s*×\s*10")) * 1e8,
                total_solar_irradiance_w_m2=float(num(r"S\s*N\s*⊙\s*=\s*(\d+)\s*W")),
                effective_temperature_k=float(num(r"T\s*N\s*eﬀ⊙\s*=\s*(\d+)\s*K")),
                astronomical_unit_m=float(num(r"equal to ([\d ]+) m")))


def hirth_2301() -> dict:
    """Hirth 2301 (23 series) datasheet p. 2: specific fuel consumption, read from the text layer. The table's values
    follow its labels in order; the two (g/kWh) entries are the two consumptions (page image hirth_2301_pdfpage2)."""
    lines = [s.strip() for s in _pdf_text(os.path.join(SRC, "hirth", "Hirth_2301_23-series_datasheet_2025.pdf"), 1)
             .splitlines()]
    i = lines.index("Speed")
    vals = [float(v.replace(",", ".")) for v in lines[i + 1:i + 5]]
    return dict(source="[HIRTH-23] p. 2 performance table", sfc_g_kwh_at_5500rpm=vals[2], sfc_g_kwh_best_point=vals[3],
                note="best point: 70-80 % load, half rated speed; cross-check of the BSFC band only")


# ============================================================================ camera and detector documents
def _norm_text(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKC", s).replace("µ", "μ")
    return re.sub(r"[\s|]+", " ", s).strip()


def camera_readings() -> dict:
    """docs/sources/cameras/readings.json holds, for each camera class, the numbers read off the manufacturer
    documents with the quote they come from. Re-check every quote against the saved copy (on the cited page of a
    PDF; saved web pages are decoded from their HTML) and re-render the cited pages as images (pages/)."""
    import html
    import re
    import fitz
    cam = os.path.join(SRC, "cameras")
    with open(os.path.join(cam, "readings.json"), encoding="utf-8") as f:
        data = json.load(f)
    pages, checked = {}, 0
    for key, doc in data["documents"].items():
        path = os.path.join(cam, doc["file"])
        if doc["file"].endswith(".pdf"):
            d = fitz.open(path)
            pages[key] = [_norm_text(p.get_text()) for p in d]
            os.makedirs(os.path.join(cam, "pages"), exist_ok=True)
            for p in (doc.get("pages") or range(1, len(d) + 1)):
                if p <= len(d):
                    d[p - 1].get_pixmap(dpi=110).save(os.path.join(cam, "pages", f"{key}_p{p:03d}.png"))
        else:
            s = open(path, encoding="utf-8", errors="ignore").read()
            s = s.replace("\\u003C", "<").replace("\\u003E", ">").replace('\\"', '"').replace("\\r\\n", " ").replace("\\t", " ")
            pages[key] = [_norm_text(html.unescape(re.sub(r"<[^>]+>", " | ", s)))]
    for r in data["readings"]:
        q = _norm_text(r["quote"])
        where = [i + 1 for i, t in enumerate(pages[r["doc"]]) if q in t]
        if not where or (r["page"] and r["page"] not in where):
            raise ValueError(f"{r['doc']}: quote not found as cited: {r['quote']!r}")
        checked += 1
    return dict(documents=len(data["documents"]), readings_checked=checked)


def sensor_noise_readings() -> dict:
    """docs/sources/sensor_noise/readings.json holds what the noise-model papers state and measured (noise terms,
    distributions, ablation results), with quotes. Re-check every quote against the saved copy (PDF, or a saved
    record's text) on its cited page, and the SHA-256 of each document."""
    return _paper_readings("sensor_noise")


def privileged_readings() -> dict:
    """docs/sources/privileged/readings.json: what the papers on learning with privileged information and on
    distillation state and measured (docs/privileged_teacher_case.md), checked as sensor_noise_readings."""
    return _paper_readings("privileged")


def detection_method_readings() -> dict:
    """docs/sources/detection_methods/readings.json: measured gains of detection methods that could give this project
    better detection (docs/detection_gain_plan.md), checked as sensor_noise_readings."""
    return _paper_readings("detection_methods")


def electric_drive_readings() -> dict:
    """docs/sources/electric_drive/readings.json: electric motor and controller efficiencies (EU 2019/1781 Annex I,
    Hepperle 2012) behind the electric-like signature arm (thermal/engine_heat.MOTOR_EFFICIENCY), checked as
    sensor_noise_readings."""
    return _paper_readings("electric_drive")


def _paper_readings(folder) -> dict:
    import hashlib
    import fitz
    src = os.path.join(SRC, folder)
    with open(os.path.join(src, "readings.json"), encoding="utf-8") as f:
        data = json.load(f)
    pages = {}
    for key, doc in data["documents"].items():
        path = os.path.join(src, doc["file"])
        if hashlib.sha256(open(path, "rb").read()).hexdigest() != doc["sha256"]:
            raise ValueError(f"{key}: {doc['file']} differs from the registered copy")
        if path.endswith(".pdf"):
            pages[key] = [_norm_text(p.get_text()) for p in fitz.open(path)]
        else:
            pages[key] = [_norm_text(open(path, encoding="utf-8").read())]
    for r in data["readings"]:
        q = _norm_text(r["quote"])
        where = [i + 1 for i, t in enumerate(pages[r["doc"]]) if q in t]
        if not where or r["page"] not in where:
            raise ValueError(f"{r['doc']}: quote not found as cited: {r['quote']!r}")
    return dict(documents=len(data["documents"]), readings_checked=len(data["readings"]))


# ============================================================================ NASA SP-3080 band data via NIST RADCAL
# Five entries of RADCAL's copy of SP-3080 Table A2-35 (H2O k at STP) differ from the printed table; the printed
# values (checked on the page images, docs/sources/nasa_sp3080/pages/sp3080_check_cells*.png and
# sp3080_p472_top_zoom.png) replace them. (wavenumber cm-1, temperature K): (RADCAL, SP-3080 printed, PDF page)
SP3080_H2O_CORRECTIONS = {(100, 300): (3.68, 3.86, 472), (6500, 1000): (2.47e-3, 2.17e-3, 481),
                          (7200, 1500): (2.25e-2, 2.55e-2, 482), (7600, 300): (1.85e-2, 1.85e-4, 483),
                          (9125, 2000): (7.48e-4, 7.84e-4, 485)}
H2O_TEMPS_K = (300, 600, 1000, 1500, 2000, 2500)       # RADCAL SD rows (SP-3080 Table A2-35 without 3000 K)
CO2_15UM_TEMPS_K = (300, 600, 1200, 1500, 1800, 2400)  # RADCAL SD15 rows


def _fortran_reshape_blocks(text, name):
    import re
    out = {}
    pat = re.compile(rf"^{name}\(1:6,(\d+):(\d+)\)\s*=\s*RESHAPE\s*\(\(/(.*?)/\)", re.S | re.M)
    for m in pat.finditer(text):
        body = re.sub(r"!.*", "", m.group(3))
        vals = [float(v.replace("_EB", "")) for v in re.findall(r"[-+]?\.?\d[\d.]*E[-+]?\d+_EB|[-+]?\d+\._EB", body)]
        out[(int(m.group(1)), int(m.group(2)))] = vals
    return out


def radcal_tables() -> dict:
    """From NIST RADCAL (docs/sources/nist_radcal/rcal.f90): the H2O narrow-band absorption coefficients k(T, w) at
    STP, 50-9425 cm-1 every 25 cm-1 (SP-3080 Table A2-35, five entries corrected to the printed table) and the CO2
    15 um band table SD15 (500-895 cm-1 every 5 cm-1). Written to h2o_k_sp3080.csv and co2_15um_k.csv."""
    rad = os.path.join(SRC, "nist_radcal")
    text = open(os.path.join(rad, "rcal.f90"), encoding="utf-8", errors="ignore").read()
    res = {}
    for name, n, w0, dw, temps, fname in (("SD", 376, 50, 25, H2O_TEMPS_K, "h2o_k_sp3080.csv"),
                                          ("SD15", 80, 500, 5, CO2_15UM_TEMPS_K, "co2_15um_k.csv")):
        table = np.full((6, n), np.nan)
        for (a, b), v in _fortran_reshape_blocks(text, name).items():
            table[:, a - 1:b] = np.array(v[:6 * (b - a + 1)]).reshape(b - a + 1, 6).T
        if np.isnan(table).any():
            raise ValueError(f"{name}: incomplete table")
        wn = w0 + dw * np.arange(n)
        if name == "SD":
            for (w, t), (old, new, _pg) in SP3080_H2O_CORRECTIONS.items():
                j, i = int((w - w0) // dw), H2O_TEMPS_K.index(t)
                if not np.isclose(table[i, j], old):
                    raise ValueError(f"RADCAL SD at {w} cm-1 {t} K is {table[i, j]}, expected {old}")
                table[i, j] = new
        with open(os.path.join(rad, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wavenumber_cm-1"] + [f"k_{t}K_per_cm_atm_STP" for t in temps])
            for j in range(n):
                w.writerow([int(wn[j])] + [repr(float(x)) for x in table[:, j]])
        res[name] = dict(rows=n, file=fname)
    return res


# SP-3080 Table A2-29, CO2 4.3 um fundamental band, k at STP [cm-1 atm-1], PDF pages 455-457 (document pages 445-447).
# The scan's text layer carries the numbers; each word is assigned to the nearest column header and row label. The
# columns used are 300, 600 and 1200 K (the atmosphere and the exhaust plume); 1500-3000 K are not transcribed. Cells
# the text layer garbles are re-read from the page images (pages/sp3080_pdfpage455-457.png); printed blanks (2150 cm-1)
# are left empty. tests/test_gas_bands.py compares thermal/gas_bands.py against this table.
SP3080_A229_PAGES = (455, 456, 457)
SP3080_A229_TEMPS_K = (300, 600, 1200)


def _sp3080_cell(text):
    import re
    t = "".join(c if ord(c) < 128 else "?" for c in text).replace(" ", "")
    t = t.replace(",", ".").replace("O", "0").replace("l", "1").replace("I", "1").replace("i", "1").lstrip("_.:;'`")
    if re.fullmatch(r"-*_?-*0[.?o]*", t):
        return 0.0                                                   # printed "-0."
    for cand in (t, t.replace("!", ""), t.replace("e", ".", 1)):
        m = re.fullmatch(r"(\d)[.?](\d{3})E([-+?])(\d{2})", cand)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}E{m.group(3).replace('?', '+')}{m.group(4)}")
    return None


def sp3080_co2_43um() -> dict:
    import re
    import fitz
    folder = os.path.join(SRC, "nasa_sp3080")
    d = fitz.open(os.path.join(folder, "NASA_SP-3080_Ludwig_1973.pdf"))
    all_temps = (300, 600, 1200, 1500, 1800, 2400, 3000)
    table, unread = {}, []
    for pg in SP3080_A229_PAGES:
        page = d[pg - 1]
        page.get_pixmap(dpi=130).save(os.path.join(folder, "pages", f"sp3080_pdfpage{pg}.png"))
        words = page.get_text("words")
        heads = {}
        for x in words:
            m = re.fullmatch(r"(\d{3,4})K\.?", x[4].replace("O", "0"))
            if m and int(m.group(1)) in all_temps:
                heads[int(m.group(1))] = (x[0] + x[2]) / 2, x[3]
        y_head = max(v[1] for v in heads.values())
        rows = {int(x[4]): (x[1] + x[3]) / 2 for x in words
                if re.fullmatch(r"\d{4}", x[4]) and x[0] < 100 and x[1] > y_head}
        cells = {}
        for x in words:
            if x[1] < y_head or x[0] < 105 or x[0] > 700:
                continue
            T = min(heads, key=lambda k: abs(heads[k][0] - (x[0] + 20)))
            w = min(rows, key=lambda k: abs(rows[k] - (x[1] + x[3]) / 2))
            if abs(rows[w] - (x[1] + x[3]) / 2) <= 7:
                cells.setdefault((w, T), []).append((x[0], x[4]))
        for w in rows:
            for T in SP3080_A229_TEMPS_K:
                if (w, T) not in cells:
                    table[(w, T)] = None                             # printed blank
                    continue
                txt = "".join(t for _, t in sorted(cells[(w, T)]))
                v = _sp3080_cell(txt)
                if v is None:
                    unread.append((w, T, txt, pg))
                table[(w, T)] = v
    if unread:
        raise ValueError(f"SP-3080 Table A2-29: unread cells {unread}")
    wns = sorted({w for w, _ in table})
    with open(os.path.join(folder, "co2_43um_k_tableA2-29.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["wavenumber_cm-1"] + [f"k_{t}K_per_cm_atm_STP" for t in SP3080_A229_TEMPS_K])
        for w in wns:
            wr.writerow([w] + ["" if table[(w, t)] is None else repr(table[(w, t)]) for t in SP3080_A229_TEMPS_K])
    return dict(rows=len(wns), first=wns[0], last=wns[-1], blank=sum(v is None for v in table.values()))


# ============================================================================ NOAA IGRA 2 radiosondes (regional profiles)
# Integrated Global Radiosonde Archive v2.2 (NOAA NCEI), period-of-record files per station, format igra2-data-format.txt.
# Lahore City (PKM00041640) has few humidity soundings after 2020, so the nearest Punjab-plain stations with a full
# record over the same period stand in for the vertical structure (CLAUDE.md sensor set rule 6: Lahore and comparable
# sites): Patiala (INM00042101, 230 km ESE, 251 m) and New Delhi Safdarjung (INM00042182, 420 km SE, 211 m).
IGRA_STATIONS = {"PKM00041640": "Lahore City", "INM00042101": "Patiala", "INM00042182": "New Delhi/Safdarjung"}
IGRA_PERIOD = ((2023, 10), (2026, 9))           # the past three years (user, 2026-10-12)
R_D, R_V, G0 = 287.05, 461.5, 9.80665


def _magnus_hpa(t_c):
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def _igra_soundings(path):
    import zipfile
    z = zipfile.ZipFile(path)
    lines = z.read(z.namelist()[0]).decode("ascii", "replace").splitlines()
    i = 0
    while i < len(lines):
        h = lines[i]
        n = int(h[32:36])
        yield h, lines[i + 1:i + 1 + n]
        i += 1 + n


def _igra_value(r, a, b):
    v = r[a:b].strip()
    return None if v in ("", "-9999", "-8888") else int(v)


def igra2_profiles() -> dict:
    """Per sounding: surface state, precipitable water, vapour scale height and lower-troposphere lapse rate, from every
    sounding of IGRA_STATIONS in IGRA_PERIOD with temperature and dew point at the surface and at 850, 700 and 500 hPa.
    Heights: hypsometric integration of the virtual temperature upward from the station elevation (IGRA station list),
    which the reported geopotential of the standard levels checks. Precipitable water = integral of vapour density
    (Magnus over water, dew point = T - dew-point depression) from the surface to the highest level with humidity up to
    300 hPa; H_w = PW / surface vapour density; H_fit: least-squares e-folding height of vapour density below 500 hPa.
    Written to regional_soundings_<period>.csv and the monthly summary regional_profiles_<period>.json."""
    folder = os.path.join(SRC, "noaa_igra2")
    elev = {}
    for line in open(os.path.join(folder, "igra2-station-list.txt"), encoding="ascii", errors="replace"):
        if line[:11] in IGRA_STATIONS:
            elev[line[:11]] = float(line[31:37])
    rows, zcheck = [], []
    (y0, m0), (y1, m1) = IGRA_PERIOD
    for st in IGRA_STATIONS:
        for h, recs in _igra_soundings(os.path.join(folder, f"{st}-data.txt.zip")):
            y, m = int(h[13:17]), int(h[18:20])
            if not (y0, m0) <= (y, m) <= (y1, m1):
                continue
            lev = []
            for r in recs:
                p, t, dd = _igra_value(r, 9, 15), _igra_value(r, 22, 27), _igra_value(r, 34, 39)
                if p is None or t is None or dd is None:
                    continue
                lev.append(dict(p=p / 100.0, t=t / 10.0, td=(t - dd) / 10.0, surface=r[1] == "1",
                                gph=_igra_value(r, 16, 21) if r[0] == "1" else None))
            sfc = [v for v in lev if v["surface"]]
            have = {round(v["p"]) for v in lev}
            if not sfc or not {850, 700, 500} <= have:
                continue
            lev = sorted([v for v in lev if v["p"] <= sfc[0]["p"] + 1e-6], key=lambda v: -v["p"])
            z, out = elev[st], []
            for k, v in enumerate(lev):
                e = float(_magnus_hpa(v["td"]))
                tv = (v["t"] + 273.15) * (1.0 + 0.61 * 0.622 * e / v["p"])
                if k:
                    z += R_D / G0 * 0.5 * (tv + out[-1]["tv"]) * math.log(out[-1]["p"] / v["p"])
                out.append(dict(v, z=z, e=e, tv=tv, rho_v=e * 100.0 / (R_V * (v["t"] + 273.15))))
                if v["gph"] is not None and v["p"] >= 499.0:
                    zcheck.append(z - v["gph"])
            wet = [v for v in out if v["p"] >= 300.0]
            pw = sum(0.5 * (a["rho_v"] + b["rho_v"]) * (b["z"] - a["z"]) for a, b in zip(wet, wet[1:]))   # kg m-2
            low = [v for v in out if v["p"] >= 500.0 and v["rho_v"] > 0]
            zz = np.array([v["z"] - out[0]["z"] for v in low])
            slope = np.polyfit(zz, np.log([v["rho_v"] for v in low]), 1)[0]
            v500 = next(v for v in out if round(v["p"]) == 500)
            s0 = out[0]
            rows.append(dict(station=st, year=y, month=m, hour_utc=int(h[24:26]), p_sfc_hpa=s0["p"], t_sfc_c=s0["t"],
                             td_sfc_c=s0["td"], e_sfc_hpa=round(s0["e"], 3), pw_mm=round(pw, 2),
                             h_w_m=round(pw / s0["rho_v"], 1), h_fit_m=round(-1.0 / slope, 1) if slope < 0 else "",
                             lapse_sfc_500_k_per_km=round((s0["t"] - v500["t"]) / ((v500["z"] - s0["z"]) / 1000.0), 3),
                             t_500_c=v500["t"], z_500_m=round(v500["z"], 1), levels=len(out)))
    tag = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
    with open(os.path.join(folder, f"regional_soundings_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for mo in range(1, 13):
        sel = [r for r in rows if r["month"] == mo]
        def q(key, rs=sel):
            v = np.array([float(r[key]) for r in rs if r[key] != ""])
            return [round(float(x), 3) for x in np.percentile(v, [10, 25, 50, 75, 90])] if len(v) else None
        summary[mo] = dict(n=len(sel), by_station={st: sum(r["station"] == st for r in sel) for st in IGRA_STATIONS},
                           h_w_m=q("h_w_m"), h_fit_m=q("h_fit_m"), pw_mm=q("pw_mm"), e_sfc_hpa=q("e_sfc_hpa"),
                           t_sfc_c=q("t_sfc_c"), lapse_k_per_km=q("lapse_sfc_500_k_per_km"))
    allq = {k: [round(float(x), 3) for x in np.percentile([float(r[k]) for r in rows if r[k] != ""], [10, 25, 50, 75, 90])]
            for k in ("h_w_m", "h_fit_m", "pw_mm", "lapse_sfc_500_k_per_km")}
    e = np.array([float(r["e_sfc_hpa"]) for r in rows])
    hw = np.array([float(r["h_w_m"]) for r in rows])
    ok = (e > 0.5) & (hw > 100.0)
    slope_b, icept_a = np.polyfit(np.log(e[ok]), np.log(hw[ok]), 1)
    resid = np.log(hw[ok]) - (icept_a + slope_b * np.log(e[ok]))
    fit = dict(form="ln(H_w / m) = a + b ln(e_sfc / hPa)", a=round(float(icept_a), 3), b=round(float(slope_b), 3),
               sd=round(float(resid.std()), 3), n=int(ok.sum()),
               h_1_99_percent_m=[round(float(x), 1) for x in np.percentile(hw[ok], [1, 99])],
               lahore_median_residual=round(float(np.median(resid[np.array([r["station"] for r in rows])[ok]
                                                                   == "PKM00041640"])), 3))
    res = dict(period=tag, stations=IGRA_STATIONS, soundings=len(rows), vapour_scale_height_fit=fit,
               height_check_m=dict(mean=round(float(np.mean(zcheck)), 1), sd=round(float(np.std(zcheck)), 1),
                                   n=len(zcheck)),
               all_months_percentiles_10_25_50_75_90=allq, monthly=summary)
    with open(os.path.join(folder, f"regional_profiles_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


# ============================================================================ AER continua (MT_CKD 4.3, LBLRTM)
# Water-vapour self and foreign continuum, MT_CKD 4.3 (AER; Mlawer et al. 2012 and later; github.com/AER-RC/MT_CKD
# data/absco-ref_wv-mt-ckd.nc, commit 679f0a4 of 2024-05-08): reference coefficients at 296 K and 1013 mb with the
# self-continuum temperature exponent, as used by mt_ckd_h2o_module.f90. Nitrogen collision-induced fundamental near
# 4.3 um (Lafferty et al. 1996, Appl. Opt. 35:5911, with MT_CKD's O2 and H2O broadening efficiencies): LBLRTM
# src/contnm.f90 (commit c91f83c of 2024-12-03), BLOCK DATA bn2f, at 272 and 228 K.
def aer_continua() -> dict:
    import re
    from scipy.io import netcdf_file
    res = {}
    f = netcdf_file(os.path.join(SRC, "aer_mt_ckd", "absco-ref_wv-mt-ckd.nc"), "r", mmap=False)
    wn = f.variables["wavenumbers"][:].copy()
    cols = {k: f.variables[k][:].copy() for k in ("self_absco_ref", "for_absco_ref", "self_texp")}
    ref = (float(f.variables["ref_press"].getValue()), float(f.variables["ref_temp"].getValue()))
    title = f._attributes["Title"].decode().strip()
    f.close()
    keep = (wn >= 600.0) & (wn <= 3000.0)
    with open(os.path.join(SRC, "aer_mt_ckd", "mt_ckd_4.3_h2o_600-3000cm.csv"), "w", newline="", encoding="utf-8") as g:
        g.write(f"# {title}; reference pressure {ref[0]} mb, temperature {ref[1]} K; coefficients in cm2 molec-1 "
                "(cm-1)-1, to be multiplied by the radiation term v tanh(hcv / 2kT). Copyright (c) Atmospheric and "
                "Environmental Research, Inc., 2022, all rights reserved; redistributed for scientific and research "
                "purposes with this notice, acknowledging AER (MT_CKD licence terms).\n")
        w = csv.writer(g)
        w.writerow(["wavenumber_cm-1", "self_absco_ref", "for_absco_ref", "self_texp"])
        for i in np.where(keep)[0]:
            w.writerow([repr(float(wn[i]))] + [repr(float(cols[k][i])) for k in cols])
    res["mt_ckd_rows"] = int(keep.sum())
    text = open(os.path.join(SRC, "aer_lblrtm", "contnm.f90"), encoding="utf-8", errors="ignore").read()
    blk = text[text.index("BLOCK DATA bn2f\n"):]
    blk = blk[:blk.index("end block data bn2f")]
    v1, v2, dv, n = [float(x) for x in re.search(r"/\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+)\s*/", blk).groups()]
    arrays = {}
    for name in ("xn2_272", "xn2_228", "a_h2o"):
        body = blk[blk.index(f"DATA {name}/") + len(f"DATA {name}/"):]
        body = body[:body.index("/")]
        arrays[name] = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", body.replace("&", " "))]
        if len(arrays[name]) != int(n):
            raise ValueError(f"{name}: {len(arrays[name])} values, expected {int(n)}")
    with open(os.path.join(SRC, "aer_lblrtm", "n2_cia_fundamental_lafferty1996.csv"), "w", newline="",
              encoding="utf-8") as g:
        g.write("# LBLRTM contnm.f90 BLOCK DATA bn2f: N2-N2 collision-induced absorption, fundamental band (Lafferty et "
                "al. 1996), cm-1 amagat-2 with the radiation field removed (multiply by tanh(hcv / 2kT)); a_h2o: "
                "relative N2-H2O efficiency (x 9/7 in n2_ver_1); O2 efficiency 1.294 - 0.4545 T / 296.\n")
        w = csv.writer(g)
        w.writerow(["wavenumber_cm-1", "k_272K", "k_228K", "a_h2o"])
        for i in range(int(n)):
            w.writerow([round(v1 + dv * i, 6), arrays["xn2_272"][i], arrays["xn2_228"][i], arrays["a_h2o"][i]])
    res["n2_cia_rows"] = int(n)
    res["n2_cia_grid"] = (v1, v2, dv)
    return res


# ============================================================================ HITRAN narrow-band tables
# HITRAN line list (Gordon et al. 2022, JQSRT 277:107949, HITRAN2020) from hitran.org's line-by-line API (par_line
# 160-character records, intensities at 296 K weighted by natural abundance), for the trace gases that absorb in the
# MWIR band and that SP-3080 does not cover: N2O (2nu1 at 3.9 um, nu3 at 4.5 um), CO (4.7 um, the air's own CO) and
# CH4 (near 3.6 um). Each 5 cm-1 interval of thermal/atmosphere_bands.W_GRID gets the statistical narrow-band
# parameters of thermal/gas_bands.py: k = sum S / dv (per cm atm at STP), the Godson line-spacing parameter
# 1/d = (sum sqrt S)^2 / (dv sum S), and the S-weighted air-broadened half-width and its temperature exponent.
# S(T) = S(296) Q(296)/Q(T) exp(-c2 E'' (1/T - 1/296)) (1 - exp(-c2 v / T)) / (1 - exp(-c2 v / 296)); Q(T): rigid
# rotor (T for linear molecules, T^1.5 for CH4) times harmonic vibrational factors (about 1 % from HITRAN's TIPS
# over 200-320 K).
HITRAN_SETS = {"N2O": dict(files=("hitran_n2o_1150-1350.par", "hitran_n2o_2000-2800.par"), rot=1.0,
                           modes=((1285.0, 1), (589.0, 2), (2224.0, 1))),
               "CO": dict(files=("hitran_co_1900-2350.par",), rot=1.0, modes=((2143.0, 1),)),
               "CH4": dict(files=("hitran_ch4_1150-1350.par", "hitran_ch4_2450-2850.par"), rot=1.5,
                           modes=((2917.0, 1), (1534.0, 2), (3019.0, 3), (1306.0, 3)))}
HITRAN_TEMPS_K = tuple(range(200, 321, 10))
C2_CM_K = 1.4387769


def _q_ratio(T, rot, modes):
    """Q(296) / Q(T)."""
    def q(t):
        v = 1.0
        for w0, g in modes:
            v /= (1.0 - math.exp(-C2_CM_K * w0 / t)) ** g
        return t ** rot * v
    return q(296.0) / q(T)


def hitran_snb() -> dict:
    folder = os.path.join(SRC, "hitran")
    res = {}
    grid = np.arange(700.0, 2800.0 + 0.1, 5.0)
    for mol, cfg in HITRAN_SETS.items():
        nu, S, gair, eld, nair = [], [], [], [], []
        for fname in cfg["files"]:
            with open(os.path.join(folder, fname), encoding="ascii") as f:
                for line in f:
                    if len(line.rstrip("\r\n")) < 60:
                        continue
                    nu.append(float(line[3:15]))
                    S.append(float(line[15:25]))
                    gair.append(float(line[35:40]))
                    eld.append(float(line[45:55]))
                    nair.append(float(line[55:59]))
        nu, S, gair, eld, nair = map(np.array, (nu, S, gair, eld, nair))
        idx = np.floor((nu - (grid[0] - 2.5)) / 5.0).astype(int)
        ok = (idx >= 0) & (idx < len(grid))
        rows = []
        for j in np.unique(idx[ok]):
            m = ok & (idx == j)
            w0 = grid[j]
            ks, ds = [], []
            for T in HITRAN_TEMPS_K:
                st = S[m] * _q_ratio(T, cfg["rot"], cfg["modes"]) * np.exp(-C2_CM_K * eld[m] * (1.0 / T - 1.0 / 296.0)) \
                     * (1.0 - np.exp(-C2_CM_K * nu[m] / T)) / (1.0 - np.exp(-C2_CM_K * nu[m] / 296.0))
                tot = st.sum()
                ks.append(tot / 5.0 * 2.6867811e19)                        # per cm atm at STP (Loschmidt, 273.15 K)
                ds.append(np.sqrt(st).sum() ** 2 / (5.0 * tot) if tot > 0 else 1.0)
            wts = S[m] / S[m].sum()
            rows.append([w0, round(float((gair[m] * wts).sum()), 5), round(float((nair[m] * wts).sum()), 4),
                         int(m.sum())] + [float(f"{x:.5g}") for x in ks] + [float(f"{x:.5g}") for x in ds])
        out = os.path.join(folder, f"snb_{mol.lower()}_hitran2020.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            f.write(f"# HITRAN2020 {mol} lines ({', '.join(cfg['files'])}) -> statistical narrow-band parameters per 5 cm-1 interval "
                    "(tools/source_measurements.hitran_snb); k per cm atm at STP; invd in cm; gamma_air at 296 K and "
                    "1 atm in cm-1, temperature exponent n_air.\n")
            w = csv.writer(f)
            w.writerow(["wavenumber_cm-1", "gamma_air_296", "n_air", "lines"] + [f"k_{t}K" for t in HITRAN_TEMPS_K]
                       + [f"invd_{t}K" for t in HITRAN_TEMPS_K])
            w.writerows(rows)
        res[mol] = dict(lines=int(len(nu)), intervals=len(rows), band_k_296=round(float(
            sum(r[4 + HITRAN_TEMPS_K.index(300)] for r in rows) * 5.0), 2))
    return res


# ============================================================================ Lahore surface climate (NOAA ISD / GSOD)
# NOAA NCEI Integrated Surface Database, station LAHORE CITY (USAF 416400, WBAN 99999; 31.55 N 74.333 E, 214 m), hourly
# records (global-hourly, NCEI Access Data Service) and daily summaries (GSOD). NOAA's archive for both Lahore stations
# ends on 2025-08-24, so "the past three years" (user, 2026-10-12) is the last three years on record. Field formats:
# isd-format-document.pdf (VIS p. 10, GA1 p. 59, MW1 pp. 103-105: WMO present-weather code table 4677); GSOD_readme.txt.
LAHORE_ISD = dict(station="41640099999", lat=31.55, lon=74.333, elev_m=214.0, utc_offset_h=5.0)
CLIMATE_WINDOW = ("2022-08-25", "2025-08-24")
ISD_BAD_QC = set("2367")                       # suspect / erroneous quality codes


def _isd_value(field, scale=10.0, missing=9999):
    if not field:
        return None
    parts = field.split(",")
    try:
        v = int(parts[0])
    except ValueError:
        return None
    if abs(v) == missing or (len(parts) > 1 and parts[1] in ISD_BAD_QC):
        return None
    return v / scale


def sun_elevation_deg(utc, lat, lon):
    """Solar elevation (NOAA's general solar position formulae: fractional-year declination and equation of time),
    accurate to ~0.1 deg, enough to bin hours into the renderer's day / dusk / night ranges."""
    import datetime as dt
    doy = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60.0
    g = 2 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60.0 + eqt + 4.0 * lon
    ha = math.radians(tst / 4.0 - 180.0)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, cz))))


def _weather_class(codes, vis_m):
    """WMO 4677 present weather -> the renderer's weather kinds: fog (40-49, or mist / shallow fog / any obscuration
    below 1 km), dust (06-09, 30-35), haze (04 smoke, 05 haze, 10 mist; or visibility under 5 km), rain (50-69, 80-99),
    otherwise clear."""
    cs = [c for c in codes if c is not None]
    if any(40 <= c <= 49 for c in cs) or (vis_m is not None and vis_m < 1000.0):
        return "fog"
    if any(6 <= c <= 9 or 30 <= c <= 35 for c in cs):
        return "dust"
    if any(50 <= c <= 69 or 80 <= c <= 99 for c in cs):
        return "rain"
    if any(c in (4, 5, 10) for c in cs) or (vis_m is not None and vis_m < 5000.0):
        return "haze"
    return "clear"


def lahore_climate() -> dict:
    import datetime as dt
    folder = os.path.join(SRC, "noaa_isd")
    st = LAHORE_ISD
    recs = {}
    for fname in sorted(os.listdir(folder)):
        if not (fname.startswith(f"isd_hourly_{st['station']}_") and fname.endswith(".csv")):
            continue
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = r["DATE"]
                if key in recs and recs[key]["REPORT_TYPE"] == "FM-12":
                    continue                                   # keep the synoptic report of an hour
                recs[key] = r
    lo, hi = CLIMATE_WINDOW
    rows = []
    for key in sorted(recs):
        if not (lo <= key[:10] <= hi):
            continue
        r = recs[key]
        t, td = _isd_value(r.get("TMP")), _isd_value(r.get("DEW"))
        if t is None or td is None or td > t + 0.5:
            continue
        vis = None
        if r.get("VIS"):
            vp = r["VIS"].split(",")
            if vp[0].isdigit() and int(vp[0]) != 999999 and vp[1] not in ISD_BAD_QC:
                vis = float(int(vp[0]))
        codes = []
        for k in ("MW1", "MW2", "MW3"):
            if r.get(k):
                c = r[k].split(",")[0]
                codes.append(int(c) if c.isdigit() else None)
        okta = None                                            # total sky cover (GF1, p. 62), else the layer (GA1)
        for fld in ("GF1", "GA1"):
            if okta is None and r.get(fld):
                c = r[fld].split(",")[0]
                okta = int(c) if c.isdigit() and int(c) <= 8 else None
        p_st = None
        if r.get("MA1"):
            mp = r["MA1"].split(",")
            if len(mp) >= 4 and mp[2].isdigit() and int(mp[2]) != 99999 and mp[3] not in ISD_BAD_QC:
                p_st = int(mp[2]) / 10.0
        utc = dt.datetime.fromisoformat(key)
        el = sun_elevation_deg(utc, st["lat"], st["lon"])
        light = "day" if el >= 15.0 else ("dusk" if el >= -6.0 else ("night" if el <= -18.0 else "twilight"))
        e = float(_magnus_hpa(td))
        rows.append(dict(utc=key, local_hour=int(utc.hour + st["utc_offset_h"]) % 24, month=utc.month,
                         sun_elevation_deg=round(el, 2), lighting=light, t_air_c=t, dew_point_c=td,
                         rh=round(min(1.0, e / float(_magnus_hpa(t))), 4), vapour_pressure_hpa=round(e, 3),
                         visibility_m="" if vis is None else vis, weather=_weather_class(codes, vis),
                         ww_codes=";".join(str(c) for c in codes if c is not None),
                         cloud_okta="" if okta is None else okta, station_pressure_hpa="" if p_st is None else p_st))
    out_csv = os.path.join(folder, f"lahore_hourly_conditions_{lo}_{hi}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pct(vals, q=(1, 5, 25, 50, 75, 95, 99)):
        v = np.array([float(x) for x in vals if x != "" and x is not None])
        return [round(float(x), 2) for x in np.percentile(v, q)] if len(v) else None
    summary = dict(station=st, window=CLIMATE_WINDOW, hours=len(rows), percentiles=[1, 5, 25, 50, 75, 95, 99])
    by_light = {}
    for light in ("day", "dusk", "night"):
        sel = [r for r in rows if r["lighting"] == light]
        n = len(sel)
        by_light[light] = dict(
            hours=n, t_air_c=pct(r["t_air_c"] for r in sel), rh=pct(r["rh"] for r in sel),
            vapour_pressure_hpa=pct(r["vapour_pressure_hpa"] for r in sel),
            visibility_m=pct(r["visibility_m"] for r in sel),
            weather_share={k: round(sum(r["weather"] == k for r in sel) / max(n, 1), 4)
                           for k in ("clear", "haze", "fog", "dust", "rain")},
            cloud_okta_share={str(k): round(sum(r["cloud_okta"] == k for r in sel) / max(1, sum(r["cloud_okta"] != "" for r in sel)), 4)
                              for k in range(9)})
    summary["by_lighting"] = by_light
    monthly = {}
    for mo in range(1, 13):
        sel = [r for r in rows if r["month"] == mo]
        monthly[mo] = dict(hours=len(sel), t_mean_c=round(float(np.mean([r["t_air_c"] for r in sel])), 2),
                           t_p5_p95=pct((r["t_air_c"] for r in sel), (5, 95)),
                           rh_mean=round(float(np.mean([r["rh"] for r in sel])), 3),
                           e_mean_hpa=round(float(np.mean([r["vapour_pressure_hpa"] for r in sel])), 2),
                           vis_median_m=pct((r["visibility_m"] for r in sel), (50,)),
                           haze_or_fog_share=round(sum(r["weather"] in ("haze", "fog") for r in sel) / max(1, len(sel)), 3))
    summary["monthly"] = monthly
    # the three years one by one (12-month blocks ending on the last day on record): the trend inside the window
    end = dt.date.fromisoformat(hi)
    blocks = []
    for k in (2, 1, 0):
        b_hi = end.replace(year=end.year - k)
        b_lo = (b_hi.replace(year=b_hi.year - 1) + dt.timedelta(days=1))
        sel = [r for r in rows if b_lo.isoformat() <= r["utc"][:10] <= b_hi.isoformat()]
        blocks.append(dict(start=b_lo.isoformat(), end=b_hi.isoformat(), hours=len(sel),
                           t_mean_c=round(float(np.mean([r["t_air_c"] for r in sel])), 2),
                           e_mean_hpa=round(float(np.mean([r["vapour_pressure_hpa"] for r in sel])), 2),
                           vis_median_m=pct((r["visibility_m"] for r in sel), (50,)),
                           haze_or_fog_share=round(sum(r["weather"] in ("haze", "fog") for r in sel) / max(1, len(sel)), 3)))
    summary["three_years"] = blocks
    # long-term trend from the daily summaries: annual mean of daily mean temperature, years with >= 300 days
    years = {}
    with open(os.path.join(folder, f"gsod_{st['station']}_1995-01-01_2025-08-24.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tf = float(r["TEMP"])
            if tf < 9000:
                years.setdefault(int(r["DATE"][:4]), []).append((tf - 32.0) / 1.8)
    full = {y: float(np.mean(v)) for y, v in years.items() if len(v) >= 300}
    ys = np.array(sorted(full))
    slope = float(np.polyfit(ys, [full[y] for y in ys], 1)[0]) * 10.0 if len(ys) > 5 else None
    summary["gsod_annual_mean_c"] = {int(y): round(full[y], 2) for y in ys}
    summary["gsod_trend_c_per_decade"] = round(slope, 3) if slope is not None else None
    summary["gsod_years_used"] = [int(ys[0]), int(ys[-1]), int(len(ys))] if len(ys) else None
    with open(os.path.join(folder, f"lahore_climate_{lo}_{hi}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    print("Camera documents:", camera_readings())
    print("Sensor-noise papers:", sensor_noise_readings())
    print("Privileged-information papers:", privileged_readings())
    print("Detection-method papers:", detection_method_readings())
    print("Electric-drive documents:", electric_drive_readings())
    print("RADCAL tables:", radcal_tables())
    print("SP-3080 Table A2-29:", sp3080_co2_43um())
    print("AER continua:", aer_continua())
    print("HITRAN narrow-band tables:", hitran_snb())
    print("Lahore climate:", {k: v for k, v in lahore_climate().items() if k in ("hours", "three_years", "gsod_trend_c_per_decade")})
    print("IGRA 2 regional profiles:", {k: v for k, v in igra2_profiles().items() if k != "monthly"})
    meas = dict(ef_front=ef_front_view(), ef_side=ef_side_view(), e_front=e_front_view())
    with open(os.path.join(LIM, "measurements.json"), "w", encoding="utf-8") as f:
        json.dump(meas, f, indent=1)
    sfc = sfc_chart()
    with open(os.path.join(LIM, "sfc_chart_digitised.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rpm", "sfc_g_per_hph"])
        w.writerows(sfc)
    points = []
    fig9 = naca_fig9(points)
    with open(os.path.join(SRC, "naca", "fig9_digitised.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fig9[0]))
        w.writeheader()
        w.writerows(fig9)
    with open(os.path.join(SRC, "naca", "fig9_points.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)
    spectra = astm_spectra()
    with open(os.path.join(SRC, "astm", "band_integrals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(spectra[0]))
        w.writeheader()
        w.writerows(spectra)
    e490 = e490_model_grid()
    with open(os.path.join(SRC, "astm", "e490_am0_model_grid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(e490[0]))
        w.writeheader()
        w.writerows(e490)
    iau, hirth = iau_b3(), hirth_2301()
    for path, data in ((os.path.join(SRC, "iau", "constants.json"), iau),
                       (os.path.join(SRC, "hirth", "readings.json"), hirth)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    print(json.dumps(meas, indent=1))
    print("SFC chart:", sfc)
    print("NACA 488 fig. 9:", fig9, f"({len(points)} raw points)")
    print("ASTM spectra:", spectra)
    print("IAU:", iau)
    print("Hirth:", hirth)


if __name__ == "__main__":
    main()
