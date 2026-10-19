"""Bundle the rendered flight scenarios for the console on another laptop.

    python -m app.export            -> out/m1/app_scenarios.zip   (plan, sky panoramas, target layers)

On the other laptop, put the zip in the fyp folder or Downloads; the console unpacks it on first start.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "out/video_v2/sensor"] + sorted((ROOT / "out/m1/det2/flights").glob("*"))
KEEP = ("plan.json", "pano_rgb.npz", "pano_lwir.npz")


def export(path=ROOT / "out/m1/app_scenarios.zip"):
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for d in SOURCES:
            if not (d / "plan.json").exists():
                continue
            for f in [d / k for k in KEEP] + sorted((d / "layers").rglob("*.npz")):
                if f.exists():
                    z.write(f, f.relative_to(ROOT))
                    n += 1
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {n} files)")
    return path


def unpack_if_present():
    """First start on another laptop: unpack app_scenarios.zip from the fyp folder or Downloads."""
    from app.engine import scenarios
    if scenarios():
        return
    for place in (ROOT, ROOT / "member", Path.home() / "Downloads"):
        for z in sorted(place.glob("app_scenarios*.zip")):
            print(f"unpacking {z} ...")
            with zipfile.ZipFile(z) as f:
                f.extractall(ROOT)
            return


if __name__ == "__main__":
    export()
