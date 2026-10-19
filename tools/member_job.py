"""One-click M1 four-sensor job (no Claude needed). RUN_M1_PRELIMINARY.bat (repository root) runs everything on one
laptop (--role solo, the default plan since 2026-10-17). For a two-laptop split, member/RUN_ME.bat starts laptop B's
share (the member's Vivobook) and member/RUN_LAPTOP_A.bat laptop A's (Usman's). Every step is skipped when already done, so running
it again resumes after a crash, restart or sleep, and carries on once the other laptop's file has arrived.

Laptop B (about twice as fast: two thirds of the rendering and the four distillation-chain models)
   1. check the computer                 2. Blender 4.2.23 (downloaded and verified once, about 380 MB)
   3. render scenes 200-599 (~3-4 h)     4. pack them: out/m1/m1_part_B.zip   -> SEND IT TO USMAN
   5. wait for Usman's m1_part_A.zip; verify and unpack it
   6. freeze the experiment manifest     7. camera-frame cache: distillation set + validation (~40 min)
   8. train rgb_lwir, teacher, student, student_ctrl (~2 h)
   9. results zip: out/m1/det2_results_B_<PC>_<date>.zip   -> SEND IT TO USMAN
Laptop A
   1-2 as B                              3. render scenes 0-199 (~3-4 h)
   4. pack them: out/m1/m1_part_A.zip   -> SEND IT TO THE MEMBER
   5. wait for m1_part_B.zip; unpack     6. manifest
   7. cache: single-camera set, validation, test (~1 h)
   8. train lwir_only, lwir_cooled_only, mwir_only (~1.5 h)
   9. wait for the member's results zip; import it
  10. evaluate once on the sealed test scenes -> out/m1/det2/results.md

    python tools/member_job.py --role B            the full job of laptop B (A for laptop A)
    python tools/member_job.py --role B --check    setup check only (~15 min): packages, Blender, a 2-scene quick
                                                   render and a short training test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]
LOG = ROOT / "member" / "job_log.txt"
PINS = {"numpy": "1.26.0", "scipy": "1.11.3", "torch": "2.1.2"}
# Blender 4.2 LTS, the release the population is rendered with on laptop A (SHA-256 from download.blender.org's
# blender-4.2.23.sha256, checked 2026-10-15)
BLENDER = dict(version="4.2.23", url="https://download.blender.org/release/Blender4.2/blender-4.2.23-windows-x64.zip",
               sha256="82e791475779a7342424a480bdde9a20b43710da9264c60346125aa16cd910cb", bytes=383193007)
NAMES = {"A": "Usman", "B": "the member"}
DISK_GB = {"A": 25, "B": 30, "solo": 25}          # solo: scenes ~3 GB, frame caches ~19 GB, runs and reports


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def say(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else "", flush=True)


def banner(*lines):
    say()
    say("*" * 78)
    for line in lines:
        say("  " + line)
    say("*" * 78)
    say()


def step(n, title):
    say()
    say(f"===== STEP {n}: {title} =====")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ computer and Blender
def check_computer(role):
    import numpy
    import scipy
    import torch
    say(f"Python {platform.python_version()}, numpy {numpy.__version__}, scipy {scipy.__version__}, torch {torch.__version__}")
    for name, want in PINS.items():
        got = {"numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__.split("+")[0]}[name]
        if got != want:
            say(f"WARNING: {name} {got} installed, {want} expected; results may not match the other laptop exactly")
    say(f"CPU: {platform.processor()} | logical cores {os.cpu_count()} | training threads {max(2, (os.cpu_count() or 4) // 2)}")
    free = shutil.disk_usage(ROOT).free / 1e9
    say(f"Free disk: {free:.0f} GB (this job needs about {DISK_GB[role]} GB)")
    if free < DISK_GB[role]:
        raise SystemExit(f"Not enough free disk space: free at least {DISK_GB[role]} GB on this drive and run again.")
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
        ms = MS()
        ms.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        say(f"Memory: {ms.avail / 1e9:.1f} GB free of {ms.total / 1e9:.1f} GB (close browsers and games while it runs)")
    except Exception:
        pass


def download(url, dest, size=None):
    """HTTP download with progress, resuming a partial .part file when the server allows it."""
    import urllib.request
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KAL-M1-member-job"}   # the CDN refuses Python's default
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0                                        # no resume support: start again
        total = (size or (have + int(r.headers.get("Content-Length", 0)))) or None
        mode = "ab" if have else "wb"
        got, shown = have, -1
        with open(part, mode) as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                got += len(chunk)
                pct = int(100 * got / total) if total else -1
                if pct != shown and pct % 5 == 0:
                    say(f"  downloaded {got / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB ({pct} %)" if total else ""))
                    shown = pct
    os.replace(part, dest)


def ensure_blender():
    """Blender 4.2 installed on this PC, or the portable 4.2.23 in .blender/ (downloaded once and checked against the
    official SHA-256)."""
    import m1_det2 as D
    try:
        exe, ver = D.find_blender()
        say(f"Blender {ver}: {exe}")
        return exe
    except FileNotFoundError:
        pass
    dest = ROOT / ".blender"
    dest.mkdir(exist_ok=True)
    z = dest / Path(BLENDER["url"]).name
    if not z.exists() or z.stat().st_size != BLENDER["bytes"]:
        say(f"Downloading Blender {BLENDER['version']} (about {BLENDER['bytes'] / 1e6:.0f} MB, one time) from download.blender.org ...")
        try:
            download(BLENDER["url"], z, BLENDER["bytes"])
        except OSError as e:
            raise SystemExit(f"Could not download Blender ({e}). Check the internet connection and run again.")
    say("Verifying the Blender download ...")
    if sha256(z) != BLENDER["sha256"]:
        z.unlink()
        raise SystemExit("The Blender download was damaged (checksum mismatch) and was deleted. Run again to download it again.")
    say("Checksum OK. Unpacking Blender (a few minutes) ...")
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    z.unlink()
    exe, ver = D.find_blender()
    say(f"Blender {ver} ready: {exe}")
    return exe


# ============================================================================ files from the other laptop
def search_places():
    return [ROOT, ROOT / "member", ROOT / "out" / "m1", Path.home() / "Downloads", Path.home() / "Desktop"]


def find_zip(pattern, check):
    """Newest file matching the pattern in the usual places (browsers add ' (1)' to repeated downloads) that passes
    `check` (a function raising on a wrong or damaged file)."""
    found = sorted({p for place in search_places() if place.exists() for p in place.glob(pattern)},
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in found:
        try:
            check(p)
            return p
        except Exception as e:                  # noqa: BLE001 - report and try the next candidate
            say(f"skipping {p}: {e}")
    return None


def other_part(D, role):
    other = "B" if role == "A" else "A"
    lo, hi = D.PARTS[other]
    src = ROOT / D.SOURCE
    if all(D.scene_done(src, i) for i in range(lo, hi)):
        say(f"Scenes {lo}-{hi - 1} from {NAMES[other]} are already unpacked.")
        return True

    def check(p):
        info = D.part_info(p)
        if info.get("part") != other:
            raise ValueError(f"it holds part {info.get('part')}, not {other}")
    z = find_zip(f"m1_part_{other}*.zip", check)
    if z is None:
        banner(f"WAITING for {NAMES[other]}'s file  m1_part_{other}.zip",
               "When it arrives, put it in your Downloads folder (or the fyp folder),",
               "then double-click the same .bat file again: the job continues from here.")
        return False
    say(f"Found {z} ({z.stat().st_size / 1e9:.2f} GB). Verifying and unpacking ...")
    D.unpack_part(z)
    return True


def results_from_b(D):
    """Laptop A: import laptop B's trained runs once they arrive."""
    if all((D.OUT / "runs" / n / "best.pt").exists() for n in D.ROLES["B"]["models"]):
        say("The member's trained models are already imported.")
        return True

    def check(p):
        with zipfile.ZipFile(p) as z:
            if not any(n.startswith("det2/runs/teacher/") for n in z.namelist()):
                raise ValueError("it does not hold the member's runs")
    z = find_zip("det2_results_B_*.zip", check)
    if z is None:
        banner("WAITING for the member's results file  det2_results_B_<PC>_<date>.zip",
               "When it arrives, put it in your Downloads folder (or the fyp folder),",
               "then run this job again: it imports the models and evaluates.")
        return False
    D.import_results(D.OUT, z)
    return True


# ============================================================================ the job
def check_mode(D, role):
    step(3, "quick render of 2 scenes at low quality (scratch folder out/m1/check_dataset)")
    lo = D.PARTS[role][0] if role in D.PARTS else 0
    D.render(start=lo, count=2, source="out/m1/check_dataset", samples=16)
    step(4, "short training test (scratch folder out/m1/det2_smoke)")
    models = {"A": ["lwir_only", "lwir_cooled_only"], "B": ["rgb_lwir", "teacher", "student"],
              "solo": ["rgb_lwir", "teacher", "student", "lwir_cooled_only"]}[role]
    D.smoke(source=str(ROOT / "out" / "m1" / "check_dataset"), models=models, steps=20)
    banner("CHECK PASSED.", {"B": "Now double-click RUN_ME.bat for the full job.", "A": "Now run RUN_LAPTOP_A.bat for the full job.",
                             "solo": "Now run RUN_M1_PRELIMINARY.bat for the full job."}[role])


def solo_job(D):
    """One laptop does everything: the preliminary evaluation without exchanging files."""
    step(3, f"render all {D.N_POP} scenes (hours; resumes where it stopped; skips finished scenes)")
    D.render(start=0, count=D.N_POP)
    step(4, "frozen experiment manifest")
    D.manifest(D.OUT)
    step(5, "camera-frame cache, every set (hours; resumes where it stopped)")
    D.cache(D.OUT, set(D.ROLES["solo"]["cache"]))
    step(6, f"training {', '.join(D.ROLES['solo']['models'])} (progress and time left are printed)")
    D.train(D.OUT, D.ROLES["solo"]["models"])
    step(7, "evaluation on the sealed test scenes (once)")
    if (D.OUT / "report.json").exists() and (D.OUT / "test_lock.json").exists():
        say(f"Already evaluated: {D.OUT / 'results.md'}")
    else:
        from m1_det2_eval import evaluate
        evaluate(D.OUT)
    z = D.package_results(D.OUT)
    banner(f"ALL DONE. Results: {D.OUT / 'results.md'} (examples in {D.OUT / 'gallery'})", f"Everything zipped: {z}")


def full_job(D, role):
    if role == "solo":
        return solo_job(D)
    other = "B" if role == "A" else "A"
    lo, hi = D.PARTS[role]
    step(3, f"render scenes {lo}-{hi - 1} (hours; progress and time left are printed; resumes where it stopped)")
    D.render(part=role)
    step(4, "pack the rendered scenes for the other laptop")
    own = ROOT / "out" / "m1" / f"m1_part_{role}.zip"
    ok = False
    if own.exists():
        try:
            ok = D.part_info(own).get("part") == role
        except Exception:                       # noqa: BLE001 - an interrupted zip: pack again
            ok = False
    if not ok:
        own = D.pack_part(role)
    banner(f"SEND THIS FILE TO {NAMES[other].upper()} (Google Drive, OneDrive or USB):", f"    {own}",
           f"({own.stat().st_size / 1e9:.2f} GB.) The job now continues with the other laptop's scenes.")
    step(5, f"scenes from {NAMES[other]}")
    if not other_part(D, role):
        return
    step(6, "frozen experiment manifest")
    D.manifest(D.OUT)
    step(7, "camera-frame cache (resumes where it stopped)")
    D.cache(D.OUT, set(D.ROLES[role]["cache"]))
    step(8, f"training {', '.join(D.ROLES[role]['models'])} (progress and time left are printed)")
    D.train(D.OUT, D.ROLES[role]["models"])
    if role == "B":
        step(9, "results zip")
        z = D.package_results(D.OUT, "B")
        banner("ALL DONE. SEND THIS FILE TO USMAN (Google Drive, WhatsApp or USB):", f"    {z}",
               f"and also {LOG}")
        return
    step(9, "the member's trained models")
    if not results_from_b(D):
        return
    step(10, "evaluation on the sealed test scenes (once)")
    if (D.OUT / "report.json").exists() and (D.OUT / "test_lock.json").exists():
        say("Already evaluated: out/m1/det2/results.md")
    else:
        from m1_det2_eval import evaluate
        evaluate(D.OUT)
    z = D.package_results(D.OUT)
    banner(f"ALL DONE. Results: {D.OUT / 'results.md'} (examples in {D.OUT / 'gallery'})", f"Everything zipped: {z}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["A", "B", "solo"], required=True,
                    help="solo: one laptop does everything; A / B: Usman's and the member's laptops (two-laptop split)")
    ap.add_argument("--check", action="store_true", help="setup check and a short training test only")
    a = ap.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    logf = open(LOG, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logf)
    sys.stderr = Tee(sys.__stderr__, logf)
    say(f"KAL M1 four-sensor job, {'one laptop' if a.role == 'solo' else 'laptop ' + a.role}, started on {platform.node()} "
        f"({'setup check' if a.check else 'full job'})")
    bundle = ROOT / "member" / "code_bundle.json"
    if bundle.exists():
        import json
        info = json.loads(bundle.read_text(encoding="utf-8"))
        say(f"Code bundle of {info.get('created')} (render contract {info['render_contract']['config_sha256'][:12]})")
    import m1_det2 as D
    D.keep_awake(True)
    try:
        step(1, "check this computer")
        check_computer(a.role)
        step(2, "Blender")
        ensure_blender()
        if a.check:
            check_mode(D, a.role)
        else:
            full_job(D, a.role)
    finally:
        D.keep_awake(False)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            say(f"STOPPED: {e.code}")
            sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        say(f"STOPPED WITH AN ERROR. Send {LOG} to Usman.")
        sys.exit(1)
