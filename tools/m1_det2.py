"""M1 detector, four-sensor revision (2026-10-15): the preliminary evaluation of the privileged-teacher design.

The M1 detector v2 pipeline (its validation diagnosis of 2026-10-04 is in docs/m1_v2.md: background and target-free
crops, mined false alarms, LWIR destriping, 8-bit video, per-frame standardisation, dense box targets, FROC checkpoint
selection) on the four sensors of CLAUDE.md's sensor set rule, each rendered in its own band and emulated as its own
camera of the scene's rig (render/cameras.py, data/camera_emulation.py):

    teacher        RGB + LWIR uncooled + LWIR cooled + MWIR        privileged, training only
    student        RGB + LWIR uncooled, distilled from the teacher  the deployable model
    student_ctrl   RGB + LWIR uncooled, distilled from rgb_lwir     control: the same distillation, no privileged sensor
    rgb_lwir       RGB + LWIR uncooled, supervised                  the matched baseline
    lwir_only, lwir_cooled_only, mwir_only                          single-camera references

Coverage. A zoom camera that cannot widen to the uncooled camera's field sees only the centre of the grid. Its per-pixel
coverage mask enters the network as a plane at stride 4 (models/cnn_detector, replacing the constant band-present
flag), its frame is preprocessed inside the covered box only, and a single-camera model is not taught targets its
camera cannot see.

Crops and false alarms. Training uses 128 px crops (4-7x cheaper per step than whole frames): half hold the target, a
quarter are sky from the same frames, a quarter come from target-free renders of the scene, and up to a third of the
sky crops are the model's own false alarms, mined on whole target-free frames. Every false-alarm number is measured on
whole frames: checkpoints are chosen by FROC on whole validation frames (2 present + 8 target-free per scene), the
thresholds come from those frames' scores, and the test reports the rate the frozen thresholds give on 4 target-free
frames per test scene.

Two laptops (docs/m1_v2.md). A = i7-6500U (2 cores), B = Ryzen 5 7520U-class (4 cores, about twice as fast). B renders
scenes 200-599 and trains the distillation chain; A renders scenes 0-199, trains the single-camera references and runs
the evaluation. Rendered scenes travel as part zips, trained runs as results zips; tools/member_job.py runs either role.

    python tools/m1_det2.py render --part A|B          this laptop's share of the 600 scenes (resumable)
    python tools/m1_det2.py pack-part --part A|B       zip it for the other laptop (out/m1/m1_part_<P>.zip)
    python tools/m1_det2.py unpack-part --zip Z        verify and unpack the other laptop's scenes
    python tools/m1_det2.py manifest                   freeze scenes, cohort, sensors and training rules
    python tools/m1_det2.py cache --part train,val     camera frames (sets: train, train_solo, val, test)
    python tools/m1_det2.py train [--models ...]       resumable; checkpoints chosen on validation only
    python tools/m1_det2.py evaluate                   opens the sealed test cohort once (test_lock.json)
    python tools/m1_det2.py sequences                  false alarms after track confirmation, on video
    python tools/m1_det2.py package-results --role R   zip trained runs (import-results --zip Z on the other laptop)
    python tools/m1_det2.py export-code                the code bundle for laptop B (no git needed there)
    python tools/m1_det2.py smoke [--source DIR]       a few minutes, end to end, on a few rendered scenes
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "target"), str(ROOT / "tools")]

import numpy as np  # noqa: E402

from data.dataset import BAND_SALT, SceneDataset, gt_objects, scene_camera  # noqa: E402
from data.provenance import atomic_json  # noqa: E402
from render.sampling import load_configs, load_yaml  # noqa: E402

OUT = ROOT / "out" / "m1" / "det2"
SOURCE = "out/m1/dataset"
OVERLAY = "configs/m1.yaml"
N_POP = 600
BANDS = ("rgb", "lwir", "lwir_cooled", "mwir")        # CLAUDE.md sensor set rule (SWIR deferred)
PARTIAL = ("lwir_cooled", "mwir")                     # zoom cameras that may not widen to the uncooled camera's field
PARTS = {"A": (0, 200), "B": (200, 600)}              # B is about twice as fast: two thirds of the rendering
BLENDER_SERIES = "4.2"                                # the whole population is rendered with Blender 4.2 LTS
TRAIN_SEED = 29
VAL_PRESENT = (17, 1017)            # two sensor-noise draws per validation scene
VAL_ABSENT = tuple(17 + 1000 * k for k in range(8))
TEST_PRESENT = (17,)
TEST_ABSENT = (17, 1017, 2017, 3017)
PREPROCESS = "8bit+destripe+standardise"              # models/cnn_detector.PREPROCESS_V2

# Uncooled commodity LWIR: stripes that are not constant down a column, on top of the rig camera's own NETD and
# row/column temporal noise. Residual row/column noise after shutter correction that grows between corrections is
# reported by FLIR (US 9,235,876 "Row and column noise reduction in thermal images"); amplitudes are [C]. The uncooled
# camera only: data/dataset.py drops these terms for the cooled FPAs, whose residual non-uniformity is the rig's.
LWIR_REALISM = dict(column_gain_frac=[0.001, 0.005],   # [C] residual gain non-uniformity after calibration
                    column_drift_k=[0.0, 0.05],         # [C] 1/f column-offset drift along the readout, K
                    row_noise_k=[0.0, 0.02],            # [C] row offsets redrawn every frame, K (the rig's draw wins)
                    column_temporal_k=[0.0, 0.02])      # [C] column offsets redrawn every frame, K (the rig's draw wins)

# name: (inference bands, teacher, modality dropout: False, True (RGB / LWIR dropout) or "subsets" (any sensor subset))
SPECS = {
    "lwir_only": (["lwir"], None, False),
    "lwir_cooled_only": (["lwir_cooled"], None, False),
    "mwir_only": (["mwir"], None, False),
    "rgb_lwir": (["rgb", "lwir"], None, True),
    "teacher": (["rgb", "lwir", "lwir_cooled", "mwir"], None, "subsets"),
    "student": (["rgb", "lwir"], "teacher", True),
    "student_ctrl": (["rgb", "lwir"], "rgb_lwir", True),
}
# Bands the teacher is given while a student is distilled (default: all of the teacher's). The within-teacher control of
# tools/privileged_ablation.py distils from the same four-sensor teacher with only RGB and uncooled LWIR fed, so the
# privileged cameras are the only difference (docs/privileged_teacher_case.md, section 6).
TEACHER_FED = {}
# Bands a model never sees in training (always absent: zeros, flag 0). The "blind" teacher of tools/privileged_ablation.py
# keeps the four-sensor teacher's architecture and subset training but never sees the privileged cameras, so it controls
# for the training regime and the network size (added 2026-10-18 after the first seed's test results).
BLANK_BANDS = {}


def blank_present(model):
    """The `present` flags for a model trained with bands always absent (BLANK_BANDS), or None."""
    blank = getattr(model, "blank_bands", ())
    return {b: float(b not in blank) for b in model.bands} if blank else None


# Students distilled only near the labelled targets (models/cnn_detector.kd_gate): the response and feature distillation
# leave out every other cell, so the teacher's false alarms are not taught. Pilot 1 found that the students inherited them
# (2026-10-19, docs/privileged_teacher_case.md 6c).
KD_TARGET_ONLY = set()


ORDER = list(SPECS)
ROLES = {"A": dict(models=["lwir_only", "lwir_cooled_only", "mwir_only"], cache=["train_solo", "val", "test"]),
         "B": dict(models=["rgb_lwir", "teacher", "student", "student_ctrl"], cache=["train", "val"]),
         "solo": dict(models=list(SPECS), cache=["train", "train_solo", "val", "test"])}     # one laptop does everything
# The teacher's sensor subsets (all 15 non-empty subsets of the four sensors): with SUBSET_ALL it sees all four, otherwise
# one of the 14 proper subsets uniformly, so it learns every sensor alone, every pair and every triple.
SUBSETS = [tuple(b for k, b in enumerate(BANDS) if m >> k & 1) for m in range(1, 2 ** len(BANDS))]
PROPER_SUBSETS = [t for t in SUBSETS if len(t) < len(BANDS)]

TRAINING = dict(steps=3000, batch=16, crop=128, lr=2e-3, weight_decay=1e-4, warmup=100,
                realisations=4, absent_realisations=2,
                crop_mix=dict(positive=0.5, background=0.25, absent=0.25),
                levels=[1, 2, 4], level_probs=[0.5, 0.25, 0.25],
                hard_negatives=dict(steps=[500, 1000, 1500, 2000, 2500], frames=48, min_score=0.15,
                                    share=0.33, pool_max=600),
                val_every=250, dropout=dict(both=0.5, lwir=0.3, rgb=0.2), subset_all=0.4,
                loss=dict(w_off=1.0, w_size=1.0, w_giou=1.0, region=0.5),
                kd=dict(lambda_resp=1.0, lambda_feat=1.0, temperature=2.0, mask_floor=0.05, beta=2.0,
                        student_floor=1.0, unseen_weight=0.0),   # no distillation on targets the student cannot see
                fa_budgets=[0.005, 0.01, 0.02, 0.05], operating_fa=[0.01, 0.05], f1_grid_step=0.01)
METRICS = dict(primary="FROC: mean centre-matched detection probability at 0.005, 0.01, 0.02 and 0.05 false "
                       "alarms per target-free frame (thresholds from those frames' scores)",
               detection_match="predicted centre within max(2 px, half the larger label side) of the label "
                               "centre (eval/nwd.py centre_ok_matrix); the infrared small-target convention",
               secondary="AP50 (IoU >= .5), AP50:95, IoU recall, per size bin, aspect, lighting, coverage, px50 fits",
               gate="distillation is claimed only if the teacher beats rgb_lwir on validation FROC and AP50; a student "
                    "gain is credited to the cooled LWIR and MWIR cameras only if the student also beats student_ctrl")


def planes(name: str) -> bool:
    """Coverage planes: models trained with modality dropout, and models with a camera that may not cover the grid."""
    bands, _, dropout = SPECS[name]
    return bool(dropout) or any(b in PARTIAL for b in bands)


# ============================================================================ small helpers
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# This file as the process loaded it. A hash taken when a run is saved can name code edited after the process started
# (the 2026-10-18 audit found that for seed 31 of tools/privileged_ablation.py).
TOOL_SHA256 = sha(Path(__file__))
MODEL_CODE_SHA256 = sha(ROOT / "models" / "cnn_detector.py")
# Training and inference device: "cpu" (default, the laptop) or "cuda" (a cloud GPU, docs/privileged_teacher_case.md 6d).
# Only where the tensors live changes; on the CPU every .to(device) below is a no-op.
DEVICE = os.environ.get("KAL_DEVICE", "cpu")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def eval_cfg():
    c = load_yaml(str(ROOT / "configs/train.yaml"))["eval"]
    c["bootstrap"] = 1000
    return c


_CFG = {}


def cfgs():
    """Repository configs with the M1 overlay, plus the uncooled camera's stripe realism. Loader-side only: the render
    contract (configs/*.yaml) is untouched. Built once per process."""
    if "c" not in _CFG:
        C = copy.deepcopy(load_configs(overlay=OVERLAY))
        C["sensors"]["lwir_noise"].update(LWIR_REALISM)
        _CFG["c"] = C
    return _CFG["c"]


def meta_of(sid):
    """A scene's metadata with the loader-only rig settings applied (configs/sensors.yaml teacher_zoom)."""
    from data.dataset import apply_teacher_zoom
    return apply_teacher_zoom(read(ROOT / SOURCE / "meta" / f"{sid}.json"), cfgs()["sensors"])


def aspect_of(meta):
    for o in meta["extra"]["objects"]:
        if o.get("cls") == "target":
            return float(o.get("aspect_deg", -1.0))
    return -1.0


def clean(o):
    """JSON-safe copy: NaN and infinities (empty size bins, no px50 crossing) become null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def threads():
    n = os.environ.get("KAL_THREADS")
    return int(n) if n else max(2, (os.cpu_count() or 4) // 2)


def keep_awake(on=True):
    """Stop Windows from sleeping during a long job (no effect elsewhere)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def cnr(img, box):
    """|box mean - 2 px annulus mean| / annulus std (spec 7.2 / 8); applied to the network-input frame."""
    y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32) if img.shape[-1] == 3 else img[..., 0]
    H, W = y.shape
    x0, y0, x1, y1 = box
    i0, j0 = int(max(0, math.floor(y0))), int(max(0, math.floor(x0)))
    i1, j1 = int(min(H, max(i0 + 1, math.ceil(y1)))), int(min(W, max(j0 + 1, math.ceil(x1))))
    a0, b0, a1, b1 = max(0, i0 - 2), max(0, j0 - 2), min(H, i1 + 2), min(W, j1 + 2)
    inner = y[i0:i1, j0:j1]
    ring = y[a0:a1, b0:b1].copy()
    mask = np.ones(ring.shape, bool)
    mask[i0 - a0:i1 - a0, j0 - b0:j1 - b0] = False
    r = ring[mask]
    if r.size < 4:
        return 0.0
    return float(abs(inner.mean() - r.mean()) / max(r.std(), 1e-6))


def coverage_boxes(meta) -> dict:
    """band -> (y0, y1, x0, x1): the part of the output grid the band's camera sees (data/camera_emulation)."""
    from data import camera_emulation as ce
    img = cfgs()["sensors"]["image"]
    H, W = int(img["height"]), int(img["width"])
    out = {}
    for b in BANDS:
        cam = scene_camera(meta, b)
        out[b] = tuple(int(v) for v in ce.covered_box(H, W, cam["coverage"] if cam else (1.0, 1.0)))
    return out


def centre_inside(box, cov) -> bool:
    """Is the label's centre inside a covered box (y0, y1, x0, x1)?"""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return bool(cov[2] <= cx < cov[3] and cov[0] <= cy < cov[1])


def target_in_view(meta, cov=None) -> dict:
    """band -> does the band's camera see the target's centre?"""
    cov = cov or coverage_boxes(meta)
    g = gt_objects(meta)
    return {b: bool(g) and centre_inside(g[0]["box"], cov[b]) for b in BANDS}


class _Absent(SceneDataset):
    """The same scene composited without any labelled object (nor its exhaust plume): same sky, haze and sensors."""

    def _layers(self, path):
        layer, _ = super()._layers(path)
        return layer, []


# ============================================================================ rendering and the two laptops' parts
def blender_version(exe) -> str:
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in r.stdout.splitlines():
        if line.startswith("Blender "):
            return line.split()[1]
    return ""


def find_blender(explicit=None):
    """(path, version) of a Blender 4.2: --blender, $BLENDER, the portable copy the member job installs (.blender/),
    the standard install folders, then PATH. Other versions are refused: both laptops must render alike."""
    cands = [explicit, os.environ.get("BLENDER")]
    cands += sorted(glob.glob(str(ROOT / ".blender" / "blender-4.2*" / "blender.exe")), reverse=True)
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender 4.2*\blender.exe"), reverse=True)
    cands += [shutil.which("blender")]
    seen = []
    for c in cands:
        if c and os.path.exists(c) and c not in seen:
            seen.append(c)
            v = blender_version(c)
            if v.startswith(BLENDER_SERIES + "."):
                return c, v
    raise FileNotFoundError(f"Blender {BLENDER_SERIES} not found (checked: {seen or 'nothing installed'})")


def scene_done(src, i) -> bool:
    """A scene is complete when its metadata, target parameters and the layers of all four bands exist."""
    src = Path(src)
    try:
        m = read(src / "meta" / f"s{i:06d}.json")
    except (OSError, ValueError):
        return False
    mods = m.get("modalities", {})
    return (set(BANDS) <= set(mods) and all((src / mods[b]["layers"]).is_file() for b in BANDS)
            and (src / "params" / f"s{i:06d}.json").is_file())


def render(part=None, start=None, count=None, source=None, samples=None, blender=None, chunk=10, scenes=None, overlay=None):
    """Render scenes of the population with the M1 overlay: the part's share (A: 0-199, B: 200-599) or
    [start, start + count). Resumable: finished scenes are skipped; each chunk is one Blender process
    (tools/generate_dataset.py, which checks every scene already in the folder against the render contract).
    scenes: population indices instead of a range; overlay: another overlay (a thermal arm of
    tools/privileged_ablation.py, which extends M1's), rendered into its own folder `source`."""
    overlay = overlay or OVERLAY
    if overlay != OVERLAY and (source is None or source == SOURCE):
        raise ValueError(f"{overlay} renders into its own folder: give --source (not {SOURCE})")
    if scenes:
        idx = sorted({int(i) for i in scenes})
        runs, a = [], idx[0]
        for prev, cur in zip(idx, idx[1:] + [None]):
            if cur != prev + 1:
                runs.append((a, prev + 1))
                a = cur
    else:
        runs = [PARTS[part] if part else (int(start), int(start) + int(count))]
    src = ROOT / (source or SOURCE)
    exe, ver = find_blender(blender)
    log(f"Blender {ver}: {exe}")
    src.mkdir(parents=True, exist_ok=True)
    lock = src / ".render_lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            pid = -1
        while pid > 0 and pid != os.getpid() and _pid_alive(pid):          # never two renders into one folder
            log(f"another render (process {pid}) is writing {src}; waiting for it to finish ...")
            time.sleep(300)
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M')}")
    keep_awake(True)
    try:
        for lo, hi in runs:
            _render_chunks(lo, hi, src, exe, samples, chunk, overlay)
    finally:
        keep_awake(False)
        lock.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Is a process with this id still running? (Windows: query its exit code; never signals it.)"""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))           # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _render_chunks(lo, hi, src, exe, samples, chunk, overlay=OVERLAY):
    env = dict(os.environ, KAL_CONFIG_OVERLAY=overlay)
    left = [i for i in range(lo, hi) if not scene_done(src, i)]
    log(f"scenes {lo}-{hi - 1}: {hi - lo - len(left)} already rendered, {len(left)} to render into {src}")
    t0, done = time.time(), 0
    for s in range(lo, hi, chunk):
        c = min(chunk, hi - s)
        todo = [i for i in range(s, s + c) if not scene_done(src, i)]
        if not todo:
            continue
        cmd = [sys.executable, str(ROOT / "tools" / "generate_dataset.py"), "--out", str(src), "--n", str(N_POP),
               "--start", str(s), "--count", str(c), "--chunk", str(c), "--bands", ",".join(BANDS), "--blender", exe]
        if samples:
            cmd += ["--samples", str(samples)]
        r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode or not all(scene_done(src, i) for i in todo):
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
            raise RuntimeError(f"rendering scenes {s}-{s + c - 1} failed (Blender logs: {src / 'logs'}). Run again to "
                               f"resume.\n{tail}")
        done += len(todo)
        n_left = sum(1 for i in range(s + c, hi) if not scene_done(src, i))
        per = (time.time() - t0) / done
        log(f"rendered scenes {s}-{s + c - 1}: {done} this run, {n_left} left, about {per * n_left / 3600:.1f} h left "
            f"({per:.0f} s per scene)")
    log(f"scenes {lo}-{hi - 1} are all rendered in {src}")


def expected_contract() -> dict:
    """The render contract this code and these configs produce (both laptops must agree on it)."""
    from data.provenance import render_contract
    return render_contract(load_configs(overlay=OVERLAY), N_POP, BANDS)


def _contract_matches(c: dict) -> bool:
    exp = expected_contract()
    return all(c.get(k) == exp[k] for k in ("config_sha256", "n_scenes", "bands"))


def pack_part(part, source=None, dest=None) -> Path:
    """Zip this laptop's rendered share (metadata, target parameters, layers; no previews) with a list of every file's
    SHA-256, for the other laptop."""
    lo, hi = PARTS[part]
    src = ROOT / (source or SOURCE)
    missing = [i for i in range(lo, hi) if not scene_done(src, i)]
    if missing:
        raise ValueError(f"part {part} is not finished: {len(missing)} scenes missing (first {missing[:5]}); render first")
    metas = [read(src / "meta" / f"s{i:06d}.json") for i in range(lo, hi)]
    contracts = {json.dumps(m["extra"]["render_contract"], sort_keys=True) for m in metas}
    if len(contracts) != 1:
        raise ValueError(f"part {part} mixes render contracts")
    contract = json.loads(contracts.pop())
    if not _contract_matches(contract):
        raise ValueError(f"part {part} was rendered with other configs or code than this copy")
    files = []
    for m in metas:
        sid = m["scene_id"]
        files += [src / "meta" / f"{sid}.json", src / "params" / f"{sid}.json"] + [src / m["modalities"][b]["layers"] for b in BANDS]
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    info = dict(schema=1, part=part, scenes=[lo, hi], render_contract=contract, files=entries, machine=platform.node(),
                created=time.strftime("%Y-%m-%d %H:%M"))
    dest = Path(dest) if dest else ROOT / "out" / "m1" / f"m1_part_{part}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:      # npz layers are already compressed
        z.writestr("m1_part.json", json.dumps(info, indent=1))
        for arc in entries:
            z.write(ROOT / arc, arc)
    os.replace(tmp, dest)
    log(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB, {len(files)} files, scenes {lo}-{hi - 1})")
    return dest


def part_info(zip_path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read("m1_part.json"))


def unpack_part(zip_path) -> dict:
    """Unpack the other laptop's part after checking it was rendered with the same configs and code, then verify every
    file's SHA-256."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = json.loads(z.read("m1_part.json"))
        if not _contract_matches(info["render_contract"]):
            raise ValueError(f"{zip_path.name} was rendered with different configs or code: both laptops need the same "
                             "code bundle")
        names = list(info["files"])
        bad_names = [n for n in names if not n.startswith(SOURCE + "/") or ".." in n.split("/")]
        if bad_names:
            raise ValueError(f"{zip_path.name}: unexpected paths {bad_names[:3]}")
        z.extractall(ROOT, names)
    bad = [n for n, h in info["files"].items() if sha(ROOT / n) != h]
    if bad:
        raise ValueError(f"{len(bad)} files of {zip_path.name} are damaged (first {bad[:3]}): download it again")
    log(f"unpacked part {info['part']} (scenes {info['scenes'][0]}-{info['scenes'][1] - 1}, {len(names)} files, "
        f"rendered on {info['machine']}): every checksum matches")
    return info


def export_code(dest=None) -> Path:
    """The code bundle for laptop B: every file a commit would hold (git's tracked and untracked, not-ignored files)
    except out/, plus the derived thermal tables in thermal/cache. It unzips to a folder named fyp."""
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                       capture_output=True, check=True)
    files = sorted({f for f in r.stdout.decode("utf-8").split("\0")
                    if f and not f.startswith("out/") and (ROOT / f).is_file()})
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "thermal" / "cache").glob("*.npz"))
    stamp = time.strftime("%Y%m%d_%H%M")
    dest = Path(dest) if dest else ROOT / "out" / "m1" / f"kal_code_{stamp}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    info = dict(created=stamp, machine=platform.node(), git_head=head, uncommitted_changes=dirty, files=len(files),
                render_contract=expected_contract())
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, "fyp/" + f)
        z.writestr("fyp/member/code_bundle.json", json.dumps(info, indent=1))
    os.replace(tmp, dest)
    log(f"wrote {dest} ({dest.stat().st_size / 1e6:.0f} MB, {len(files)} files; render contract "
        f"{info['render_contract']['config_sha256'][:12]})")
    return dest


# ============================================================================ manifest
def manifest(out, scenes=None, smoke=False):
    src = ROOT / SOURCE
    metas = {}
    for p in sorted((src / "meta").glob("s*.json")):
        idx = int(p.stem[1:])
        if idx < N_POP and (scenes is None or p.stem in scenes):
            metas[p.stem] = read(p)
    if not smoke:
        missing = [i for i in range(N_POP) if f"s{i:06d}" not in metas]
        if missing:
            raise ValueError(f"population incomplete: {len(missing)} scenes missing (first: {missing[:5]})")
    if not metas:
        raise ValueError(f"no rendered scenes in {src}")
    contracts = {m["extra"]["render_contract"]["config_sha256"] for m in metas.values()}
    if len(contracts) != 1:
        raise ValueError("mixed render contracts in the source population")
    for m in metas.values():
        for band in BANDS:
            if band not in m["modalities"] or not (src / m["modalities"][band]["layers"]).is_file():
                raise FileNotFoundError(f"{m['scene_id']}: missing {band} layers")
    split = {s: sorted(k for k, m in metas.items() if m["split"] == s) for s in ("train", "val", "test")}
    img = cfgs()["sensors"]["image"]
    grid = (0, int(img["height"]), 0, int(img["width"]))
    cover = {}
    for s, ids in split.items():
        cov = {k: coverage_boxes(metas[k]) for k in ids}
        inside = {k: target_in_view(metas[k], cov[k]) for k in ids}
        cover[s] = {b: dict(scenes=len(ids), whole_grid=sum(cov[k][b] == grid for k in ids),
                            target_in_view=sum(inside[k][b] for k in ids)) for b in PARTIAL}
    m = dict(schema=2, experiment="m1-det2-four-sensor", created=time.strftime("%Y-%m-%d"),
             selection=f"M1 population scenes 0..{N_POP - 1}; split = the population's 60/10/30 assignment",
             source=SOURCE, overlay=OVERLAY, config_sha256=next(iter(contracts)),
             signature_model=next(iter(metas.values()))["extra"]["render_contract"]["signature_model"],
             splits=dict(train=split["train"], val=split["val"]), cohorts=dict(test=split["test"]),
             cohort_notes=dict(test="the population's test split; read only by `evaluate`, after every run is frozen"),
             seeds=dict(train=TRAIN_SEED, val_present=VAL_PRESENT, val_absent=VAL_ABSENT,
                        test_present=TEST_PRESENT, test_absent=TEST_ABSENT),
             sensors=dict(bands=list(BANDS), cameras="each scene's camera rig (meta extra.cameras, configs/sensors.yaml), "
                          "emulated on the uncooled camera's grid by data/camera_emulation.py",
                          uncooled_lwir_stripes=LWIR_REALISM, coverage=cover),
             preprocess=PREPROCESS, architecture="m1-cnn-centernet-v2",
             models={k: dict(bands=v[0], teacher=v[1], modality_dropout=v[2], coverage_planes=planes(k))
                     for k, v in SPECS.items()},
             roles={r: dict(scenes=list(PARTS.get(r, (0, N_POP))), **ROLES[r]) for r in ROLES},
             training=TRAINING, metrics=METRICS,
             test_policy="the test cohort is read only by `evaluate`, after all runs and thresholds are frozen")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        old = read(path)
        new = json.loads(json.dumps(m))            # tuples -> lists, as stored on disk
        if _strip(old) != _strip(new):
            raise ValueError(f"{path} already frozen with different content; use a fresh --out directory")
        log("manifest unchanged")
        return
    atomic_json(path, m)
    log(f"froze {path}: {len(split['train'])} train, {len(split['val'])} validation, {len(split['test'])} test scenes")


def _strip(m: dict) -> dict:
    """A manifest without its creation date (two laptops freeze the same manifest on different days)."""
    return {k: v for k, v in m.items() if k != "created"}


# ============================================================================ cache
KD_CH = dict(rgb=slice(0, 3), lwir=slice(3, 4), lwir_cooled=slice(4, 5), mwir=slice(5, 6))
SOLO_CH = dict(lwir=slice(0, 1), lwir_cooled=slice(1, 2), mwir=slice(2, 3))
EVAL_CH = KD_CH
SETS = {"train": KD_CH, "train_solo": SOLO_CH}


def _prep(frames, coverage=None):
    from models import cnn_detector as cd
    return cd.preprocess(frames, cd.PREPROCESS_V2, coverage=coverage)


def _scene_frames(meta, seed, train, epoch, absent, kind="eval"):
    """Camera frames of one scene, preprocessed for the network. kind 'kd': training frames of all four cameras, each
    thermal camera with its own registration jitter against RGB (the label reference); 'solo': training frames of the
    three thermal cameras in label coordinates (single-camera models); 'eval': all four, no jitter. Noise streams are
    salted by band name (data/dataset.py). Returns (frames, boxes, coverage boxes)."""
    cls = _Absent if absent else SceneDataset
    bands = [b for b in BANDS if b != "rgb"] if kind == "solo" else list(BANDS)
    ds = cls(str(ROOT / SOURCE), None, bands, cfgs(), train=train, seed=seed, metas=[meta],
             aug=dict(hflip=0.0, scale_prob=0.0))
    ds.set_epoch(epoch)
    vis, ratio = ds._weather(meta, ds._rng(0))
    reg = train and kind == "kd"
    frames = {b: ds.band_image(meta, b, ds._rng(0, BAND_SALT[b]), vis, ratio, reg=reg and b != "rgb") for b in bands}
    cov = coverage_boxes(meta)
    boxes = [] if absent else [o["box"] for o in gt_objects(meta)]
    return _prep(frames, cov), boxes, cov


def _cnr(frames, boxes, cov):
    """Per band, each label's contrast-to-noise on the network input; 0 where the band's camera does not see it."""
    return {b: [cnr(frames[b], bx) if centre_inside(bx, cov[b]) else 0.0 for bx in boxes] for b in frames}


def _save(path, frames, order):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.concatenate([frames[b] for b in order], -1).astype(np.float16))
    os.replace(tmp, path)


def cache(out, parts, limit=None):
    """Camera-frame cache, resumable. Sets: 'train' (the distillation chain: four cameras with registration jitter),
    'train_solo' (single-camera models: the thermal cameras without jitter), 'val' and 'test' (whole frames)."""
    m = read(out / "manifest.json")
    T = m["training"]
    unknown = set(parts) - {"train", "train_solo", "val", "test"}
    if unknown:
        raise ValueError(f"unknown cache sets {sorted(unknown)}")
    for part in ("train", "train_solo"):
        if part not in parts:
            continue
        kind, order = ("kd" if part == "train" else "solo"), list(SETS[part])
        dest = out / "cache" / part
        dest.mkdir(parents=True, exist_ok=True)
        ids = m["splits"]["train"][:limit] if limit else m["splits"]["train"]
        t0, built = time.time(), 0
        for n, sid in enumerate(ids):
            meta = meta_of(sid)
            for k in range(T["realisations"]):
                info_p = dest / f"{sid}_r{k}.json"
                if not info_p.exists():
                    fr, boxes, cov = _scene_frames(meta, TRAIN_SEED, True, k, absent=False, kind=kind)
                    _save(dest / f"{sid}_r{k}.npy", fr, order)
                    atomic_json(info_p, dict(scene_id=sid, k=k, boxes=boxes, coverage={b: list(cov[b]) for b in BANDS},
                                             visible={b: [centre_inside(bx, cov[b]) for bx in boxes] for b in BANDS},
                                             cnr=_cnr(fr, boxes, cov)))
                    built += 1
            for j in range(T["absent_realisations"]):
                p = dest / f"{sid}_a{j}.npy"
                if not p.exists():
                    fr, _, _ = _scene_frames(meta, TRAIN_SEED, True, 100 + j, absent=True, kind=kind)
                    _save(p, fr, order)
                    built += 1
            if built and ((n + 1) % 10 == 0 or n + 1 == len(ids)):
                per = (time.time() - t0) / (n + 1)
                log(f"{part} cache {n + 1}/{len(ids)} scenes (~{per * (len(ids) - n - 1) / 60:.0f} min left)")
    groups = []
    if "val" in parts:
        groups.append(("val", m["splits"]["val"], VAL_PRESENT, VAL_ABSENT))
    if "test" in parts:
        groups.append(("test", m["cohorts"]["test"], TEST_PRESENT, TEST_ABSENT))
    order = list(EVAL_CH)
    for name, ids, present, absent in groups:
        dest = out / "cache" / "eval"
        dest.mkdir(parents=True, exist_ok=True)
        t0, built = time.time(), 0
        for n, sid in enumerate(ids):
            meta = meta_of(sid)
            info_p = dest / f"{sid}.json"
            info = read(info_p) if info_p.exists() else dict(scene_id=sid, cnr={})
            for j, seed in enumerate(present):
                p = dest / f"{sid}_p{j}.npy"
                if not p.exists() or f"p{j}" not in info["cnr"]:
                    fr, boxes, cov = _scene_frames(meta, seed, False, 0, absent=False)
                    _save(p, fr, order)
                    info["cnr"][f"p{j}"] = _cnr(fr, boxes, cov)
                    info["coverage"] = {b: list(cov[b]) for b in BANDS}
                    atomic_json(info_p, info)
                    built += 1
            for j, seed in enumerate(absent):
                p = dest / f"{sid}_a{j}.npy"
                if not p.exists():
                    fr, _, _ = _scene_frames(meta, seed, False, 0, absent=True)
                    _save(p, fr, order)
                    built += 1
            if built and ((n + 1) % 10 == 0 or n + 1 == len(ids)):
                per = (time.time() - t0) / (n + 1)
                log(f"{name} cache {n + 1}/{len(ids)} scenes (~{per * (len(ids) - n - 1) / 60:.0f} min left)")
    fingerprint(out)


def fingerprint(out):
    """Per-channel mean and spread (and the SHA-256) of a fixed list of cache files: the first 5 training scenes of each
    training set, 8 validation and 5 test scenes. Laptops that built the same inputs agree on the statistics; the
    SHA-256 can differ in the last bit of a float between processors. Files a laptop has not built are absent."""
    m = read(out / "manifest.json")
    want = [f"{s}/{sid}_r0.npy" for s in ("train", "train_solo") for sid in m["splits"]["train"][:5]]
    want += [f"eval/{sid}_p0.npy" for sid in m["splits"]["val"][:8]]
    want += [f"eval/{sid}_p0.npy" for sid in m["cohorts"]["test"][:5]]
    fp = {}
    for w in want:
        p = out / "cache" / w
        if p.exists():
            a = np.load(p).astype(np.float64)
            fp[w] = dict(sha256=sha(p), mean=[round(float(v), 4) for v in a.mean((0, 1))],
                         std=[round(float(v), 4) for v in a.std((0, 1))])
    atomic_json(out / "cache_fingerprint.json", dict(files=fp, numpy=np.__version__, python=platform.python_version(),
                                                     machine=platform.node()))
    return fp


def compare_fingerprints(theirs: dict, mine: dict, tol=2e-3):
    """(compared, identical, differing) over the files both laptops built."""
    common = sorted(set(theirs) & set(mine))
    same = [k for k in common if theirs[k]["sha256"] == mine[k]["sha256"]]
    diff = [k for k in common if k not in same and not (
        np.allclose(theirs[k]["mean"], mine[k]["mean"], atol=tol) and np.allclose(theirs[k]["std"], mine[k]["std"], atol=tol))]
    return common, same, diff


# ============================================================================ data access
def eval_frames(out, sid, bands, kind="p", j=0):
    a = np.load(out / "cache" / "eval" / f"{sid}_{kind}{j}.npy", mmap_mode="r")
    return {b: np.asarray(a[..., EVAL_CH[b]], np.float32) for b in bands}


def train_frames(path, bands, ch):
    a = np.load(path, mmap_mode="r")
    return {b: np.asarray(a[..., ch[b]], np.float32) for b in bands}


class Sampler:
    """Deterministic crop sampler shared by every run of a cache set (same seed -> same crops, flips, dropout draws and
    crop kinds), so the baseline, the students and the control see identical batches. Each crop carries every band's
    coverage mask (the covered box cut to the crop window, pooled and flipped with the image)."""

    def __init__(self, out, m, solo=False):
        self.set = "train_solo" if solo else "train"
        self.ch = SETS[self.set]
        self.dir = out / "cache" / self.set
        self.T = m["training"]
        self.items = [read(self.dir / f"{sid}_r{k}.json") for sid in m["splits"]["train"]
                      for k in range(self.T["realisations"])]
        self.absent = [self.dir / f"{sid}_a{j}.npy" for sid in m["splits"]["train"]
                       for j in range(self.T["absent_realisations"])]
        self.cov = {it["scene_id"]: {b: tuple(v) for b, v in it["coverage"].items()} for it in self.items}
        self.pool = []                       # mined false alarms: (path, cx, cy, level)

    def _window(self, rng, H, W, R, centre=None, box=None):
        if box is not None:                  # keep the whole box inside the crop
            x0, y0, x1, y1 = box
            lo_x, hi_x = max(0.0, x1 - R + 2), min(W - R, x0 - 2)
            lo_y, hi_y = max(0.0, y1 - R + 2), min(H - R, y0 - 2)
            ox = rng.uniform(lo_x, hi_x) if hi_x > lo_x else np.clip((x0 + x1 - R) / 2, 0, W - R)
            oy = rng.uniform(lo_y, hi_y) if hi_y > lo_y else np.clip((y0 + y1 - R) / 2, 0, H - R)
        elif centre is not None:
            cx, cy = centre
            ox = np.clip(cx - rng.uniform(0.15, 0.85) * R, 0, W - R)
            oy = np.clip(cy - rng.uniform(0.15, 0.85) * R, 0, H - R)
        else:
            ox, oy = rng.uniform(0, W - R), rng.uniform(0, H - R)
        return int(ox), int(oy)

    def batch(self, rng, model_bands, dropout, teacher_bands=(), visible_only=False):
        """visible_only (single-camera models): labels whose centre the camera does not see are dropped, so a model is
        never taught to find a target its camera cannot see."""
        import torch
        import torch.nn.functional as F
        from models import cnn_detector as cd
        T = self.T
        crop, B = T["crop"], T["batch"]
        want = list(dict.fromkeys(list(model_bands) + list(teacher_bands)))
        out = {b: np.zeros((B, cd.BAND_CHANNELS[b], crop, crop), np.float32) for b in want}
        masks = {b: np.zeros((B, 1, crop, crop), np.float32) for b in want}
        boxes, cnrs, kinds = [], [], []
        present = {b: np.ones(B, np.float32) for b in model_bands}
        mix = T["crop_mix"]
        for i in range(B):
            u = rng.uniform()
            kind = "pos" if u < mix["positive"] else ("absent" if u < mix["positive"] + mix["absent"] else "bg")
            hard = kind != "pos" and bool(self.pool) and rng.uniform() < T["hard_negatives"]["share"]
            it, idx = None, np.zeros(0, int)
            if hard:
                path, cx, cy, f = self.pool[int(rng.integers(len(self.pool)))]
                path = Path(path)
            elif kind == "absent":
                path = self.absent[int(rng.integers(len(self.absent)))]
                f = int(rng.choice(T["levels"], p=T["level_probs"]))
            else:
                it = self.items[int(rng.integers(len(self.items)))]
                path = self.dir / f"{it['scene_id']}_r{it['k']}.npy"
                idx = np.arange(len(it["boxes"]))
                if visible_only and len(idx):
                    idx = idx[np.asarray(it["visible"][model_bands[0]], bool)]
                if kind == "pos" and len(idx):
                    b0 = np.asarray(it["boxes"][int(idx[0])], np.float32)
                    f = cd.level_for(float(np.max(b0[2:] - b0[:2])))
                else:
                    f = int(rng.choice(T["levels"], p=T["level_probs"]))
            bx = np.asarray(it["boxes"], np.float32).reshape(-1, 4)[idx] if it is not None else np.zeros((0, 4), np.float32)
            cov = self.cov[path.stem.split("_")[0]]
            arr = np.load(path, mmap_mode="r")
            H, W = arr.shape[:2]
            R = crop * f
            if hard:
                ox, oy = self._window(rng, H, W, R, centre=(cx, cy))
            elif kind == "pos" and len(bx):
                ox, oy = self._window(rng, H, W, R, box=bx[0])
            else:
                ox, oy = self._window(rng, H, W, R)
            flip = rng.uniform() < 0.5
            for b in want:
                t = torch.from_numpy(np.ascontiguousarray(arr[oy:oy + R, ox:ox + R, self.ch[b]], np.float32).transpose(2, 0, 1))[None]
                mk = np.zeros((1, 1, R, R), np.float32)
                y0, y1, x0, x1 = cov[b]
                mk[..., max(0, y0 - oy):max(0, min(R, y1 - oy)), max(0, x0 - ox):max(0, min(R, x1 - ox))] = 1.0
                mt = torch.from_numpy(mk)
                if f > 1:
                    t, mt = F.avg_pool2d(t, f), F.avg_pool2d(mt, f)
                if flip:
                    t, mt = torch.flip(t, dims=[3]), torch.flip(mt, dims=[3])
                out[b][i] = t[0].numpy()
                masks[b][i] = mt[0].numpy()
            b2 = (bx - np.array([ox, oy, ox, oy], np.float32)) / f
            if flip and len(b2):
                b2 = np.c_[crop - b2[:, 2], b2[:, 1], crop - b2[:, 0], b2[:, 3]]
            cb = np.clip(b2, 0, crop)
            area = lambda q: np.clip(q[:, 2] - q[:, 0], 0, None) * np.clip(q[:, 3] - q[:, 1], 0, None)  # noqa: E731
            keep = area(cb) > 0.5 * area(b2) if len(b2) else np.zeros(0, bool)
            boxes.append(cb[keep])
            c = {}
            if it is not None and len(bx):
                c = {b: np.asarray(v, float)[idx][keep] for b, v in it["cnr"].items()}
            cnrs.append(c)
            kinds.append("hard" if hard else kind)
            u = rng.uniform()                # drawn for every model, so all runs keep the same crop sequence
            k = int(rng.integers(len(PROPER_SUBSETS)))
            if dropout == "subsets":         # teacher: all four sensors, or one of the 14 proper subsets
                if u >= T["subset_all"]:
                    for b in model_bands:
                        present[b][i] = float(b in PROPER_SUBSETS[k])
            elif dropout:
                p = T["dropout"]
                if u >= p["both"]:
                    present["rgb" if u < p["both"] + p["lwir"] else "lwir"][i] = 0.0
        imgs = {b: torch.from_numpy(out[b]) for b in want}
        mks = {b: torch.from_numpy(masks[b]) for b in want}
        return imgs, mks, boxes, cnrs, {b: torch.from_numpy(v) for b, v in present.items()}, kinds


# ============================================================================ validation metrics
def records(out, model, sids, kind, draws, present=None, metas=None):
    """Predictions for the given scenes and cached draws; kind 'p' (present) or 'a' (target-free)."""
    from models import cnn_detector as cd
    rows = []
    for sid in sids:
        meta = metas[sid] if metas else meta_of(sid)
        cov = coverage_boxes(meta)
        gts = [] if kind == "a" else [dict(o, cls_name="target") for o in gt_objects(meta)]
        in_view = target_in_view(meta, cov) if kind == "p" else {}
        for j in range(draws):
            frames = eval_frames(out, sid, model.bands, kind, j)
            masks = cd.coverage_masks(cov, frames[model.bands[0]].shape[:2], model.bands) if model.presence else None
            rows.append(dict(scene_id=f"{sid}" if kind == "p" and draws == 1 else f"{sid}_{kind}{j}",
                             scene=sid, gts=gts, camera=meta["camera"], conditions=meta["conditions"],
                             target_aspect_deg=aspect_of(meta), target_in_view=in_view,
                             preds=cd.predict(model, frames, present=present if present is not None else blank_present(model),
                                              masks=masks)))
    return rows


def best_match_scores(present_recs, cfg):
    """Per present frame: the highest score of any prediction that centre-matches the target (or 0)."""
    from eval.nwd import centre_ok_matrix
    out = []
    for r in present_recs:
        G = np.array([o["box"] for o in r["gts"]], float).reshape(-1, 4)
        P = np.asarray(r["preds"]["boxes"], float).reshape(-1, 4)
        S = np.asarray(r["preds"]["scores"], float)
        if not len(G) or not len(P):
            out.append(0.0)
            continue
        ok = centre_ok_matrix(P, G, cfg["centre_match"]["abs_px"], cfg["centre_match"]["rel"]).any(1)
        out.append(float(S[ok].max()) if ok.any() else 0.0)
    return np.asarray(out)


def fa_threshold(absent_recs, budget, floor=0.01):
    """Lowest threshold leaving at most floor(budget x frames) false alarms on target-free frames."""
    scores = np.sort(np.concatenate([np.asarray(r["preds"]["scores"], float) for r in absent_recs] + [np.zeros(0)]))[::-1]
    allowed = int(math.floor(budget * len(absent_recs)))
    if allowed >= len(scores):
        return float(floor)
    return float(np.nextafter(scores[allowed], 2.0))


def froc(present_recs, absent_recs, budgets, cfg, best=None):
    """FROC score: mean detection probability (centre match) at each false-alarm budget per target-free
    frame, with every threshold taken from those frames' own scores. Also returns the per-budget values."""
    best = best_match_scores(present_recs, cfg) if best is None else best
    pd = {}
    for b in budgets:
        t = fa_threshold(absent_recs, b)
        pd[str(b)] = float(np.mean(best >= t)) if len(best) else float("nan")
    return float(np.mean(list(pd.values()))), pd


def f1_threshold(present_recs, cfg, step):
    from eval import metrics
    best = (-1.0, 0.5)
    for t in np.arange(step, 1.0, step):
        op = metrics.operating_point(present_recs, dict(cfg, score_thresh=float(t)))
        f1 = 2 * op["precision"] * op["recall"] / max(op["precision"] + op["recall"], 1e-9)
        if f1 >= best[0]:
            best = (f1, round(float(t), 4))
    return best[1], best[0]


def validate(out, m, model, val_ids, n_present, n_absent, metas):
    from eval import metrics
    cfg = eval_cfg()
    pres = records(out, model, val_ids, "p", n_present, metas=metas)
    absn = records(out, model, val_ids, "a", n_absent, metas=metas)
    score, per = froc(pres, absn, m["training"]["fa_budgets"], cfg)
    return dict(froc=score, pd_at=per, ap50=metrics.ap(pres, cfg=cfg)), pres, absn


# ============================================================================ training
def lr_at(step, T):
    if step < T["warmup"]:
        return T["lr"] * (step + 1) / T["warmup"]
    p = (step - T["warmup"]) / max(1, T["steps"] - T["warmup"])
    return T["lr"] * 0.5 * (1 + math.cos(math.pi * p))


def mine(out, model, sampler, rng, T):
    """Run the current model on random target-free training frames; keep its false alarms."""
    from models import cnn_detector as cd
    H = T["hard_negatives"]
    picks = rng.choice(len(sampler.absent), size=min(H["frames"], len(sampler.absent)), replace=False)
    found = 0
    for i in picks:
        path = sampler.absent[int(i)]
        frames = train_frames(path, model.bands, sampler.ch)
        masks = None
        if model.presence:
            masks = cd.coverage_masks(sampler.cov[path.stem.split("_")[0]], frames[model.bands[0]].shape[:2], model.bands)
        p = cd.predict(model, frames, masks=masks, present=blank_present(model))
        for (x0, y0, x1, y1), s in zip(p["boxes"], p["scores"]):
            if s >= H["min_score"]:
                sampler.pool.append((str(path), (x0 + x1) / 2, (y0 + y1) / 2, cd.level_for(max(x1 - x0, y1 - y0))))
                found += 1
    sampler.pool = sampler.pool[-H["pool_max"]:]
    return found


def _state(model, adapters, opt, step, best, history, rng, sampler, elapsed):
    import torch
    return dict(model=model.state_dict(), adapters=adapters.state_dict() if adapters is not None else None,
                opt=opt.state_dict(), step=step, best=best, history=history, rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), pool=[(str(p), cx, cy, f) for p, cx, cy, f in sampler.pool],
                elapsed=elapsed)


def train(out, names):
    import torch
    from models import cnn_detector as cd
    m = read(out / "manifest.json")
    T = m["training"]
    torch.set_num_threads(threads())
    val_ids = m["splits"]["val"]
    metas = {s: meta_of(s) for s in val_ids}
    keep_awake(True)
    try:
        for name in names:
            _train_one(out, m, T, name, val_ids, metas, cd, torch)
    finally:
        keep_awake(False)


def _train_one(out, m, T, name, val_ids, metas, cd, torch):
    dev = torch.device(DEVICE)
    bands, teacher_name, dropout = SPECS[name]
    use_planes, solo = planes(name), len(bands) == 1
    run = out / "runs" / name
    if (run / "best.pt").exists():
        log(f"{name}: finished run exists, skipped")
        return
    run.mkdir(parents=True, exist_ok=True)
    teacher = None
    if teacher_name:
        tp = out / "runs" / teacher_name / "best.pt"
        if not tp.exists():
            raise FileNotFoundError(f"{name} needs the finished {teacher_name} run first")
        teacher, _ = cd.load(tp)
        teacher.requires_grad_(False)
        teacher.to(dev)
    sampler = Sampler(out, m, solo=solo)
    torch.manual_seed(TRAIN_SEED)
    model = cd.CNNDetector(bands, presence=use_planes)
    model.blank_bands = tuple(BLANK_BANDS.get(name, ()))      # never seen in training, so also absent in mining and validation
    initial = copy.deepcopy(model.state_dict())
    model.to(dev)
    adapters = None
    if teacher is not None:
        adapters = torch.nn.ModuleDict({k: torch.nn.Conv2d(c, ct, 1) for k, c, ct in
                                        (("n4", model.widths[0], teacher.widths[0]), ("n8", model.widths[1], teacher.widths[1]))})
        adapters.to(dev)
    params = list(model.parameters()) + (list(adapters.parameters()) if adapters is not None else [])
    opt = torch.optim.AdamW(params, lr=T["lr"], weight_decay=T["weight_decay"])
    rng = np.random.default_rng([TRAIN_SEED, 1])            # same batches for every run of a cache set
    mine_rng = np.random.default_rng([TRAIN_SEED, 2])
    history, best, start_step, elapsed0 = [], dict(froc=-1.0, ap50=-1.0, step=-1, state=None), 0, 0.0
    last = run / "last.pt"
    if last.exists():                                     # resume after a crash, reboot or sleep
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        if adapters is not None and st["adapters"] is not None:
            adapters.load_state_dict(st["adapters"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        sampler.pool = [(p, cx, cy, f) for p, cx, cy, f in st["pool"]]
        start_step, best, history, elapsed0 = st["step"], st["best"], st["history"], st["elapsed"]
        mine_rng = np.random.default_rng([TRAIN_SEED, 2, start_step])
        log(f"{name}: resuming at step {start_step}")
    kd, L = T["kd"], T["loss"]
    t0 = time.time()
    log(f"{name}: training on {', '.join(bands)}" + (f", distilled from {teacher_name}" if teacher_name else "")
        + (" (coverage planes)" if use_planes else ""))
    for step in range(start_step, T["steps"]):
        model.train()
        if step in T["hard_negatives"]["steps"]:
            model.eval()
            n = mine(out, model, sampler, mine_rng, T)
            model.train()
            log(f"{name}: mined {n} false alarms at step {step} (pool {len(sampler.pool)})")
        imgs, masks, boxes, cnrs, present, kinds = sampler.batch(rng, bands, dropout,
                                                                 teacher.bands if teacher is not None else (), visible_only=solo)
        for b in BLANK_BANDS.get(name, ()):
            present[b] = torch.zeros_like(present[b])
        imgs = {b: v.to(dev) for b, v in imgs.items()}          # the batch on the training device; `present` stays
        masks = {b: v.to(dev) for b, v in masks.items()}        # on the CPU for the per-target checks below
        pres_d = {b: v.to(dev) for b, v in present.items()}
        x = {b: imgs[b] * pres_d[b].view(-1, 1, 1, 1) for b in bands}
        hm_t, reg_t, pos, gauss, reg_w, gt_box = cd.build_targets_dense(boxes, T["crop"], T["crop"], L["region"])
        hm_t, reg_t, pos, reg_w, gt_box = (t.to(dev) for t in (hm_t, reg_t, pos, reg_w, gt_box))
        o = model(x, pres_d if use_planes else None, {b: masks[b] for b in bands} if use_planes else None)
        loss, parts = cd.detection_loss_v2(o, hm_t, pos, reg_t, reg_w, gt_box, L["w_off"], L["w_size"], L["w_giou"])
        if teacher is not None:
            fed = TEACHER_FED.get(name, teacher.bands)     # a band left out is absent: zeros, flag 0, no coverage
            with torch.no_grad():
                tin = {b: imgs[b] if b in fed else torch.zeros_like(imgs[b]) for b in teacher.bands}
                if teacher.presence:
                    to = teacher(tin, {b: torch.full((len(boxes),), float(b in fed), device=dev) for b in teacher.bands},
                                 {b: masks[b] if b in fed else torch.zeros_like(masks[b]) for b in teacher.bands})
                else:
                    to = teacher(tin)
            weights, unseen = [], []
            floor = kd.get("student_floor")         # targets the student's own frames do not show: no distillation
            for i, c in enumerate(cnrs):
                if not len(boxes[i]) or not c:
                    weights.append([])
                    unseen.append([])
                    continue
                ct = np.max([np.abs(c[b]) for b in teacher.bands if b in fed], axis=0)
                got = [b for b in bands if float(present[b][i]) > 0]
                cs = np.max([np.abs(c[b]) for b in got], axis=0) if got else np.zeros_like(ct)
                weights.append([cd.cnr_gap_weight(a, b_, kd["beta"], floor, kd.get("unseen_weight", 1.0))
                                for a, b_ in zip(ct, cs)])
                unseen.append([floor is not None and b_ < floor for b_ in cs])
            gate = cd.kd_gate(gauss) if name in KD_TARGET_ONLY else None
            mask = cd.kd_mask(gauss, weights, kd["mask_floor"], gate=gate).to(dev)
            keep = cd.kd_keep(gauss, unseen) if floor is not None else None
            if gate is not None:
                keep = gate if keep is None else keep * gate
            keep = keep.to(dev) if keep is not None else None
            l_resp = cd.response_kd(o, to, kd["temperature"], keep=keep)
            l_feat = cd.feature_kd(o["n4"], to["n4"], adapters["n4"], mask) + cd.feature_kd(o["n8"], to["n8"], adapters["n8"], mask)
            loss = loss + kd["lambda_resp"] * l_resp + kd["lambda_feat"] * l_feat
            parts.update(resp=float(l_resp), feat=float(l_feat))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, T)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        row = dict(step=step + 1, loss=float(loss), **parts)
        if (step + 1) % T["val_every"] == 0 or step + 1 == T["steps"]:
            model.eval()
            v, _, _ = validate(out, m, model, val_ids, 1, 4, metas)
            row.update(val_froc=v["froc"], val_ap50=v["ap50"])
            if (v["froc"], v["ap50"]) > (best["froc"], best["ap50"]):
                best = dict(froc=v["froc"], ap50=v["ap50"], step=step + 1, state=copy.deepcopy(model.state_dict()))
            el = elapsed0 + time.time() - t0
            per = (time.time() - t0) / (step + 1 - start_step)
            log(f"{name} step {step + 1}/{T['steps']}: loss {float(loss):.3f} val FROC {v['froc']:.3f} AP50 {v['ap50']:.3f} "
                f"| {per:.2f} s/step, ~{per * (T['steps'] - step - 1) / 60:.0f} min left")
            history.append(row)
            torch.save(_state(model, adapters, opt, step + 1, best, history, rng, sampler, el), last)
        else:
            history.append(row)
            if (step + 1) % 25 == 0:
                per = (time.time() - t0) / (step + 1 - start_step)
                log(f"{name} step {step + 1}/{T['steps']}: loss {float(loss):.3f} {per:.2f} s/step")
    model.load_state_dict(best["state"])
    model.eval()
    v, pres, absn = validate(out, m, model, val_ids, len(VAL_PRESENT), len(VAL_ABSENT), metas)
    cfg = eval_cfg()
    thr = {f"threshold_fa{str(b).replace('0.', '')}": fa_threshold(absn, b) for b in T["operating_fa"]}
    thr_f1, f1 = f1_threshold(pres, cfg, T["f1_grid_step"])
    meta = dict(name=name, bands=bands, teacher=teacher_name, modality_dropout=dropout, coverage_planes=use_planes,
                teacher_fed=list(TEACHER_FED.get(name, teacher.bands)) if teacher is not None else None,
                blank_bands=list(BLANK_BANDS.get(name, ())), kd_target_only=name in KD_TARGET_ONLY,
                cache_set=sampler.set, teacher_sha256=sha(out / "runs" / teacher_name / "best.pt") if teacher_name else None,
                step=best["step"], val_froc=v["froc"], val_pd_at=v["pd_at"], val_ap50=v["ap50"], **thr,
                threshold_f1=thr_f1, val_f1=f1, preprocess=PREPROCESS,
                manifest_sha256=sha(out / "manifest.json"), model_code_sha256=MODEL_CODE_SHA256, device=str(dev),
                tool_sha256=TOOL_SHA256, noise_code_sha256=sha(ROOT / "noise/lwir_fpn.py"),
                seconds=elapsed0 + time.time() - t0, threads=threads(), machine=platform.node(),
                processor=platform.processor(), python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, train_scenes=len(m["splits"]["train"]))
    cd.save(run / "best.pt", model, arch=cd.ARCH_V2, **meta)
    untrained = cd.CNNDetector(bands, presence=use_planes)
    untrained.load_state_dict(initial)
    cd.save(run / "untrained.pt", untrained, arch=cd.ARCH_V2, **dict(meta, step=0))
    atomic_json(run / "history.json", history)
    atomic_json(run / "training.json", meta)
    atomic_json(run / "validation.json", dict(present=pres, absent=absn, **{k: meta[k] for k in meta if k.startswith(("val_", "threshold"))}))
    last.unlink(missing_ok=True)
    log(f"SAVED {run / 'best.pt'}: step {best['step']}, val FROC {v['froc']:.3f}, AP50 {v['ap50']:.3f}, "
        f"thresholds {thr} F1 {thr_f1:.2f}")


# ============================================================================ packaging
RESULT_FILES = ("manifest.json", "cache_fingerprint.json", "test_lock.json", "report.json", "predictions.json",
                "results.md", "learning_curves.png")
_NOT_IMPORTED = ("det2/manifest.json", "det2/cache_fingerprint.json", "det2/environment.json")


def package_results(out, role=None) -> Path:
    """Trained runs (weights, histories, validation) of a role, or all, plus any evaluation outputs. Entries are stored
    under det2/ whatever the folder is called here."""
    stamp = time.strftime("%Y%m%d_%H%M")
    names = ROLES[role]["models"] if role else ORDER
    path = out.parent / f"det2_results_{role or 'all'}_{platform.node()}_{stamp}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        files = [p for n in names for p in sorted((out / "runs" / n).rglob("*"))] + sorted((out / "gallery").rglob("*"))
        for p in files:
            if p.is_file() and p.name != "last.pt":
                z.write(p, "det2/" + p.relative_to(out).as_posix())
        for name in RESULT_FILES:
            if (out / name).exists():
                z.write(out / name, "det2/" + name)
        z.writestr("det2/environment.json", json.dumps(dict(machine=platform.node(), processor=platform.processor(),
                                                            python=platform.python_version(), numpy=np.__version__,
                                                            threads=threads(), role=role, models=names, created=stamp), indent=1))
    log(f"results bundle: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def import_results(out, zip_path):
    """Unpack runs from the other laptop. Refuses a different manifest, cache inputs whose statistics differ, an
    evaluation opened with other weights, and finished runs here with different weights."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if _strip(json.loads(z.read("det2/manifest.json"))) != _strip(read(out / "manifest.json")):
            raise ValueError("the bundle was trained on a different manifest")
        if "det2/cache_fingerprint.json" in names and (out / "cache_fingerprint.json").exists():
            theirs = json.loads(z.read("det2/cache_fingerprint.json"))["files"]
            mine_fp = read(out / "cache_fingerprint.json")["files"]
            common, same, diff = compare_fingerprints(theirs, mine_fp)
            log(f"cache fingerprint: {len(common)} files compared, {len(same)} identical, "
                f"{len(common) - len(same) - len(diff)} equal to rounding, {len(diff)} differ")
            if diff:
                raise ValueError(f"the other laptop's camera frames differ ({diff[:3]}): its inputs are not the same")
        if "det2/test_lock.json" in names and (out / "test_lock.json").exists():
            if json.loads(z.read("det2/test_lock.json")) != read(out / "test_lock.json"):
                raise ValueError("this laptop already opened the test cohort with different weights")
        for n in names:
            if n.startswith("det2/runs/") and n.endswith("best.pt") and (out / n[5:]).exists():
                if sha(out / n[5:]) != hashlib.sha256(z.read(n)).hexdigest():
                    raise ValueError(f"{n} already exists here with different weights")
        keep = out / "imports" / zip_path.stem
        keep.mkdir(parents=True, exist_ok=True)
        for n in names:
            if not n.startswith("det2/") or n.endswith("/"):
                continue
            dest = keep / Path(n).name if n in _NOT_IMPORTED else out / n[5:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dest, "wb") as f:
                shutil.copyfileobj(src, f)
    log(f"imported {zip_path}")


# ============================================================================ smoke check
def smoke(source=None, models=None, steps=30, evaluate=False):
    """A few minutes, end to end, on the first rendered scenes (scratch folder out/m1/det2_smoke): manifest, every
    cache set, training with mining, coverage planes and distillation, validation and thresholds."""
    global SOURCE
    if source:
        SOURCE = Path(source).resolve().relative_to(ROOT).as_posix()
    out = ROOT / "out" / "m1" / "det2_smoke"
    if out.exists():
        shutil.rmtree(out)
    src = ROOT / SOURCE
    ids = sorted(p.stem for p in (src / "meta").glob("s*.json") if scene_done(src, int(p.stem[1:])))[:6]
    if len(ids) < 2:
        raise ValueError(f"smoke needs at least 2 finished scenes in {src}")
    manifest(out, scenes=set(ids), smoke=True)
    m = read(out / "manifest.json")
    m["splits"] = dict(train=ids[:-1], val=ids[-1:])           # smoke only: whatever splits the first scenes have
    m["cohorts"] = dict(test=ids[-1:])
    m["training"] = dict(m["training"], steps=steps, val_every=max(1, steps // 2), realisations=1, absent_realisations=1,
                         batch=8, hard_negatives=dict(m["training"]["hard_negatives"], steps=[max(1, steps // 3)], frames=2))
    atomic_json(out / "manifest.json", m)
    names = models or ORDER
    sets = {"val"} | ({"test"} if evaluate else set())
    sets |= {"train_solo"} if any(len(SPECS[n][0]) == 1 for n in names) else set()
    sets |= {"train"} if any(len(SPECS[n][0]) > 1 for n in names) else set()
    cache(out, sets)
    train(out, list(names))
    if evaluate:
        from m1_det2_eval import evaluate as run_eval
        run_eval(out)
    log(f"SMOKE OK: manifest, cache ({', '.join(sorted(sets))}), training of {', '.join(names)}"
        + (", evaluation" if evaluate else "") + " ran")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["render", "pack-part", "unpack-part", "manifest", "cache", "train", "evaluate",
                                        "sequences", "package-results", "import-results", "export-code", "fingerprint",
                                        "smoke"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--part", default=None, help="render / pack-part: A or B; cache: comma list of train, train_solo, "
                                                 "val, test (default train,val)")
    ap.add_argument("--role", default=None, choices=["A", "B"], help="package-results: only that laptop's models")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--zip", default=None, help="unpack-part / import-results: the zip")
    ap.add_argument("--limit", type=int, default=None, help="cache: only the first N training scenes")
    ap.add_argument("--start", type=int, default=None, help="render: first scene (instead of --part)")
    ap.add_argument("--count", type=int, default=None, help="render: number of scenes (with --start)")
    ap.add_argument("--samples", type=int, default=None, help="render: override the path-tracing samples (checks only)")
    ap.add_argument("--source", default=None, help="render / smoke: dataset folder (default out/m1/dataset)")
    ap.add_argument("--scenes", type=int, nargs="*", default=None, help="render: population indices (instead of a range)")
    ap.add_argument("--overlay", default=None, help="render: another config overlay, into its own --source folder")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--steps", type=int, default=30, help="smoke: training steps per model")
    ap.add_argument("--evaluate", action="store_true", help="smoke: also run the evaluation")
    ap.add_argument("--log", default=None, help="also append everything printed to this file")
    a = ap.parse_args()
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        logf = open(a.log, "a", encoding="utf-8")

        class _Tee:
            def __init__(self, stream):
                self.stream = stream

            def write(self, s):
                self.stream.write(s)
                logf.write(s)
                self.stream.flush()
                logf.flush()

            def flush(self):
                self.stream.flush()
                logf.flush()
        sys.stdout, sys.stderr = _Tee(sys.__stdout__), _Tee(sys.__stderr__)
    out = Path(a.out)
    if a.command == "render":
        if a.part not in PARTS and (a.start is None or a.count is None) and not a.scenes:
            ap.error("render needs --part A|B, --start and --count, or --scenes")
        render(a.part if a.part in PARTS else None, a.start, a.count, a.source, a.samples, a.blender,
               scenes=a.scenes, overlay=a.overlay)
    elif a.command == "pack-part":
        if a.part not in PARTS:
            ap.error("pack-part needs --part A|B")
        pack_part(a.part, a.source)
    elif a.command == "unpack-part":
        unpack_part(a.zip)
    elif a.command == "manifest":
        manifest(out)
    elif a.command == "cache":
        cache(out, set((a.part or "train,val").split(",")), a.limit)
    elif a.command == "train":
        train(out, a.models or ORDER)
    elif a.command == "evaluate":
        from m1_det2_eval import evaluate
        evaluate(out)
    elif a.command == "sequences":
        from m1_det2_eval import sequences
        sequences(out, a.models)
    elif a.command == "package-results":
        package_results(out, a.role)
    elif a.command == "import-results":
        import_results(out, a.zip)
    elif a.command == "export-code":
        export_code(a.zip)
    elif a.command == "fingerprint":
        print(json.dumps(fingerprint(out), indent=1))
    else:
        smoke(a.source, a.models, a.steps, a.evaluate)


if __name__ == "__main__":
    main()
