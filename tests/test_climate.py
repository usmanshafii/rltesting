"""Scene conditions from Lahore's station record (CLAUDE.md sensor set rule 6; docs/lahore_climate.md)."""
import csv
import json
import os

import numpy as np
import pytest

from render import sampling

CFGS = sampling.load_configs()
CLIM = CFGS["render"]["climate"]
ROOT = sampling.ROOT
SUMMARY = os.path.join(ROOT, "docs", "sources", "noaa_isd", "lahore_climate_2022-08-25_2025-08-24.json")


def test_record_and_summary_agree():
    path = os.path.join(ROOT, CLIM["file"])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    js = json.load(open(SUMMARY))
    assert js["hours"] == len(rows) and js["window"] == ["2022-08-25", "2025-08-24"]
    assert all("2022-08-25" <= r["utc"][:10] <= "2025-08-24" for r in rows)
    for light, v in js["by_lighting"].items():
        sel = [float(r["t_air_c"]) for r in rows if r["lighting"] == light]
        assert v["hours"] == len(sel)
        assert v["t_air_c"][3] == pytest.approx(float(np.percentile(sel, 50)), abs=0.01)
    # the lighting bins are the renderer's: day >= 15 deg, dusk -6..15, night <= -18 (render.yaml sky)
    sky = CFGS["render"]["sky"]["sun_elevation_deg"]
    for r in rows[:2000]:
        el = float(r["sun_elevation_deg"])
        lo, hi = {"day": (sky["day"][0], 91), "dusk": (sky["dusk"][0], sky["dusk"][1]),
                  "night": (-91, sky["night"][1]), "twilight": (sky["night"][1], sky["dusk"][0])}[r["lighting"]]
        assert lo - 1e-9 <= el <= hi + 1e-9
    # every hour is physically consistent
    t = np.array([float(r["t_air_c"]) for r in rows])
    td = np.array([float(r["dew_point_c"]) for r in rows])
    assert np.all(td <= t + 0.5) and 0.0 < t.min() and t.max() < 50.0


def test_weather_classes_follow_wmo_codes():
    from tools.source_measurements import _weather_class
    assert _weather_class([44], 4000.0) == "fog" and _weather_class([], 800.0) == "fog"
    assert _weather_class([4], 4000.0) == "haze" and _weather_class([5], 6000.0) == "haze"
    assert _weather_class([], 3000.0) == "haze" and _weather_class([], 6000.0) == "clear"
    assert _weather_class([7], 6000.0) == "dust" and _weather_class([61], 6000.0) == "rain"


def test_scenes_take_station_hours():
    rows = sampling.climate_records(CLIM["file"])
    for idx in range(12):
        spec, _ = sampling.sample_scene(idx, CFGS, 12)
        c = spec["climate"]
        assert c is not None and c["utc"]
        rec = next(r for r in rows[spec["lighting"]] if r["utc"] == c["utc"])
        assert spec["air_temp_c"] == pytest.approx(float(rec["t_air_c"]))
        assert spec["signatures"]["environment"]["lwir"]["sky"]["rh"] == pytest.approx(float(rec["rh"]))
        if rec["weather"] == "fog":
            assert spec["weather"]["kind"] == "fog"
        if rec["visibility_m"] and rec["weather"] != "clear":
            assert spec["weather"]["visibility_km"] == pytest.approx(max(float(rec["visibility_m"]) / 1000.0, 0.05))
        if rec["cloud_okta"] == "0.0" or rec["cloud_okta"] == "0":
            assert spec["clouds"] is None


def test_m1_overlay_uses_lahore_conditions(monkeypatch):
    """The four-sensor M1 (2026-10-15) takes its conditions from the station record like every other scene."""
    monkeypatch.setenv(sampling.OVERLAY_ENV, "configs/m1.yaml")
    cfgs = sampling.load_configs()
    spec, _ = sampling.sample_scene(1, cfgs, 600)
    assert spec["climate"] is not None and spec["lighting"] in ("day", "night")
    assert "416400" in spec["climate"]["source"] or "41640" in spec["climate"]["source"]
    assert spec["climate"]["utc"] and "visibility_m" in spec["climate"]
