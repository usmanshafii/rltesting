"""docs/privileged_teacher_case.md and docs/detection_gain_plan.md: their census numbers are the saved census's, and their
quotes are in the saved papers."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]
DATA = ROOT / "docs" / "reference" / "privileged" / "data"
CASE = " ".join((ROOT / "docs" / "privileged_teacher_case.md").read_text(encoding="utf-8").split())   # one line
PLAN = " ".join((ROOT / "docs" / "detection_gain_plan.md").read_text(encoding="utf-8").split())


def _pct(v):
    return f"{100 * v:.0f} %"


def test_census_summary_is_the_census_and_the_documents_quote_it():
    import privileged_census as P
    rows = list(json.loads((DATA / "census.json").read_text()).values())
    summary = json.loads((DATA / "census_summary.json").read_text())
    assert not any(r["split"] == "test" for r in rows)                 # M1's test split stays sealed
    small = [r for r in rows if (r["size_px"] or 1e9) <= P.MAX_PX]
    assert P._regimes(rows) == summary["all"] and P._regimes(small) == summary["small"]       # the old ring measure
    assert P._dregimes(rows) == summary["dprime_all"] and P._dregimes(small) == summary["dprime_small"]
    a, s = summary["dprime_all"], summary["dprime_small"]
    for v in (a["privileged_better"], a["privileged_2x"], a["student_detectable"], s["student_detectable"]):
        assert _pct(v) in CASE and _pct(v) in PLAN
    assert _pct(a["privileged_better"]) == "71 %" and _pct(a["student_detectable"]) == "95 %"
    assert f"{a['lwir_cooled_over_lwir_median_where_seen']:.2f}×" in CASE
    assert f"{a['mwir_over_lwir_median_where_seen']:.2f}×" in CASE
    assert f"{summary['render_time_s']['all']['student_only_scenes_per_four_band_scene']:.2f}×" in CASE
    assert f"{100 * summary['render_time_s']['all']['privileged_share']:.0f} %" == "42 %"


def test_working_point_and_detector_numbers_are_the_saved_ones():
    wp = json.loads((DATA / "working_point.json").read_text())
    assert wp["matched all"]["43.0"][1] == 25 and wp["matched small"]["43.0"][1] == 20
    assert "25 of 126" in CASE and "20 of 63" in CASE and "6–8 targets" in CASE
    assert wp["_z2 all"]["43.0"][1] == 6 and wp["_z1to3 all"]["43.0"][1] == 8
    det = json.loads((DATA / "detector_vs_ideal.json").read_text())["models"]
    found = sorted(m["detected"] for m in det.values())
    assert found == [27, 36] and all(m["ideal_detectable"] == 52 and m["n"] == 55 for m in det.values())
    assert "36 and 27 of 55" in CASE and "52" in CASE
    d50 = sorted(round(m["dprime50"]) for m in det.values())
    assert d50 == [43, 114] and "≈ 43" in CASE and "≈ 114" in CASE


def test_paper_quotes_are_in_the_saved_copies():
    import source_measurements as S
    for folder, check, n_docs in (("privileged", S.privileged_readings, 13),
                                  ("detection_methods", S.detection_method_readings, 9)):
        readings = json.loads((ROOT / "docs/sources" / folder / "readings.json").read_text(encoding="utf-8"))
        if not all((ROOT / "docs/sources" / folder / d["file"]).exists() for d in readings["documents"].values()):
            pytest.skip("the papers are git-ignored copies; fetch them from the URLs in readings.json")
        assert check() == dict(documents=n_docs, readings_checked=len(readings["readings"]))


def test_plan_quotes_the_measured_baseline_and_projections():
    ap = json.loads((DATA / "baseline_ap.json").read_text())
    for m in ap.values():
        for k in ("ap50", "ap_centre", "ap_centre_small"):
            assert f"{m[k]:.3f}" in PLAN
    pr = json.loads((DATA / "projections.json").read_text())["projections"]
    gains = [round(100 * (v["five_frames_static"] - v["one_frame"])) for k, v in pr.items() if k.endswith(" all")]
    bound = [round(100 * (v["five_frames_moving_bound"] - v["one_frame"])) for k, v in pr.items() if k.endswith(" all")]
    assert f"+{min(gains)} to +{max(bound)} (all targets)" in PLAN


def test_the_preregistered_rule_is_unchanged_in_the_document():
    import hashlib
    raw = (ROOT / "docs" / "privileged_teacher_case.md").read_text(encoding="utf-8")
    for name in ("preregistration.json", "preregistration_pilot2.json", "preregistration_pilot2_amendment.json"):
        pre = json.loads((DATA / name).read_text(encoding="utf-8"))
        assert hashlib.sha256(pre["text"].encode("utf-8")).hexdigest() == pre["sha256"]
        assert pre["text"].rstrip() in raw                   # written before any result, still there word for word
