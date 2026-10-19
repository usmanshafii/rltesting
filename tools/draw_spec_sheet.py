"""Proposal reference drawing: three views in metres plus a specification table with evidence levels.

    python tools/draw_spec_sheet.py      -> out/proposal/KAL_reference_drawing.{png,pdf}

The outlines are projections of the canonical kal_geometry mesh (configs/target.yaml defaults), so
every drawn dimension is the model's. The tip-fin outline uses the presentation model's broad tips
(kal_hero PHOTO_FIN_TIP, after the P1 image) so the drawing matches the proposal render; the dataset
samples narrower P2 tips. Values are labelled with the project's evidence levels:
A observed in KAL imagery, B strongly inferred, C modelling assumption, D unknown; the user's engine
decision (MD-550 / L550E, the closest drone analogue) and its published data are labelled as such.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.path import Path as MPath

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [os.path.join(ROOT, "target"), HERE]
import engine_l550e as el  # noqa: E402
import kal_geometry as kg  # noqa: E402

OUT = os.path.join(ROOT, "out", "proposal")
PHOTO_FIN_TIP = 0.75          # target/kal_hero.py (kept here so this tool needs no Blender)
INK, DIM, MUTED = "#15202b", "#3d4a57", "#6b7884"
GREYS = {"skin": "#d9dee3", "control_surface": "#c3cbd3", "fin": "#c9d0d7", "fuselage": "#b8c1ca",
         "tail_cone": "#b8c1ca", "markings": "#f4f6f8", "engine_head": "#8e99a3", "engine_barrel": "#a3acb5",
         "engine_case": "#4b525a", "engine_intake": "#c9d0d7", "engine_flange": "#a9a08f", "exhaust_stub": "#6f655c",
         "engine_heatsink": "#2b3036",
         "propeller": "#4b525a", "propeller_disc": "#cfd5db", "antenna": "#3a4048"}
SKIP = {"decal"}


def project(mesh, view):
    V = mesh["vertices_span"] * mesh["derived"]["span_m"]          # metres, internal axes (x aft)
    if view == "top":
        P, depth, n_view = V[:, [0, 1]], V[:, 2], np.array([0, 0, 1.0])       # nose left, starboard up
    elif view == "side":
        P, depth, n_view = V[:, [0, 2]], -V[:, 1], np.array([0, -1.0, 0])
    else:                                                            # front: from ahead, port on the right
        P, depth, n_view = np.c_[-V[:, 1], V[:, 2]], -V[:, 0], np.array([-1.0, 0, 0])
    polys, depths, cols, solid = [], [], [], []
    for f, s in zip(mesh["faces"], mesh["face_slots"]):
        slot = kg.SLOTS[s]
        if slot in SKIP:
            continue
        pts = V[list(f)]
        n = sum(np.cross(pts[k] - pts[0], pts[k + 1] - pts[0]) for k in range(1, len(f) - 1))
        nn = np.linalg.norm(n)
        if nn == 0 or np.dot(n, n_view) <= 0:
            continue
        shade = 0.72 + 0.28 * np.dot(n / nn, n_view)
        c = np.array(matplotlib.colors.to_rgb(GREYS.get(slot, "#c0c7ce"))) * shade
        disc = slot == "propeller_disc"
        polys.append(P[list(f)])
        depths.append(depth[list(f)].mean())
        cols.append((*np.clip(c, 0, 1), 0.30 if disc else 1.0))
        solid.append(not disc)
    order = np.argsort(depths)
    return [polys[i] for i in order], [cols[i] for i in order], [solid[i] for i in order]


def draw_view(ax, mesh, view, res=1600):
    polys, cols, solid = project(mesh, view)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2))
    # crisp silhouette: rasterise the solid faces and contour the mask
    pts = np.concatenate([p for p, s in zip(polys, solid) if s])
    lo, hi = pts.min(0) - 0.02, pts.max(0) + 0.02
    nx = res
    ny = int(res * (hi[1] - lo[1]) / (hi[0] - lo[0])) + 2
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
    q = np.c_[gx.ravel(), gy.ravel()]
    mask = np.zeros(len(q), bool)
    for p, s in zip(polys, solid):
        if not s:
            continue
        bmin, bmax = p.min(0), p.max(0)
        sel = np.where(~mask & (q[:, 0] >= bmin[0]) & (q[:, 0] <= bmax[0]) & (q[:, 1] >= bmin[1]) & (q[:, 1] <= bmax[1]))[0]
        if len(sel):
            mask[sel[MPath(p).contains_points(q[sel], radius=1e-9)]] = True
    ax.contour(gx, gy, mask.reshape(gx.shape).astype(float), levels=[0.5], colors=INK, linewidths=0.9)


def dim(ax, p0, p1, text, off=0.06, fs=8.5, side=1, ext=None):
    """Dimension line with arrows and extension lines; text centred on the line, offset normal to it."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d) * side
    if ext is not None:
        for p, e in zip((p0, p1), ext):
            ax.plot([e[0], p[0] + 0.02 * nrm[0] * 1.5], [e[1], p[1] + 0.02 * nrm[1] * 1.5], color=MUTED, lw=0.5)
    ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=DIM, shrinkA=0, shrinkB=0,
                                            mutation_scale=7))
    mid = (p0 + p1) / 2 + nrm * off
    rot = math.degrees(math.atan2(d[1], d[0]))
    if rot > 90 or rot < -90:
        rot += 180
    ax.text(mid[0], mid[1], text, ha="center", va="center", fontsize=fs, color=INK, rotation=rot,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))


def scale_bar(ax, x0, y0, fs=7.5):
    for k in range(4):
        ax.add_patch(plt.Rectangle((x0 + 0.25 * k, y0), 0.25, 0.025, fc=INK if k % 2 == 0 else "white", ec=INK, lw=0.6))
    for v in (0, 0.5, 1.0):
        ax.text(x0 + v, y0 - 0.035, f"{v:g}", ha="center", va="top", fontsize=fs, color=INK)
    ax.text(x0 + 1.05, y0 + 0.012, "m", ha="left", va="center", fontsize=fs, color=INK)


def spec_rows(g, cfg):
    """(parameter, value, range at the default span, basis); None value = section header."""
    S = g["span_m"]
    pr = cfg["propeller"]
    r = lambda lo, hi, k=1.0, fmt="{:.2f}": f"{fmt.format(lo * S * k)}–{fmt.format(hi * S * k)}"  # noqa: E731
    return [
        ("AIRFRAME", None, None, None),
        ("Configuration", "Tailless cropped delta, mid wing", "", "A  P1–P8"),
        ("Fins / propulsor", "Tip fins up and down; pusher prop", "", "A  P1–P8"),
        ("Wingspan", f"{S:.2f} m", "2.00–2.60 m", "B  P2, P5 scale"),
        ("Length, nose to hub", f"{g['length'] * S:.2f} m ({(g['length'] + g['probe_len']) * S:.2f} m with probe)",
         "P5: 3.1–3.5 m", "B  P5"),
        ("Root chord", f"{g['te_x'] * S:.2f} m", r(0.90, 1.12) + " m", "C  ratio-derived"),
        ("Tip chord", f"{g['tip_chord'] * S:.2f} m", r(0.13, 0.21) + " m", "B  P1, P6, P7"),
        ("Leading-edge sweep", f"{g['sweep_deg']:.1f}°", "54–63°", "B  derived"),
        ("Forebody diameter", f"{g['forebody_dia'] * S * 1000:.0f} mm", r(0.125, 0.155, 1000, "{:.0f}") + " mm", "B  P6, P7"),
        ("Wing thickness, root", f"{g['wing_tc'] * 100:.0f} % chord", "4–9 %", "C  not visible"),
        ("Tip fins, above / below", f"{g['fin_up_height'] * S:.2f} m / {g['fin_down_height'] * S:.2f} m", r(0.12, 0.18) + " m",
         "B  P8"),
        ("Control surfaces", "2 elevons per wing", "", "A  P1, P8"),
        ("Propeller", f"2 blades, Ø {g['prop_dia'] * S:.2f} m",
         "{:.2f}–{:.2f} m".format(*pr["prop_dia_m"]["range"]), "B  P6; tip Mach ≤ 0.81"),
        ("Nose", "Pitot probe, 2 blade antennas", "", "A  P2"),
        ("ENGINE  (closest drone analogue, user decision)", None, None, None),
        ("Engine", "MD-550 (Limbach L550E copy)", "KAL engine unpublished", "D → analogue (G1)"),
        ("Type", "4-cyl. two-stroke boxer, air cooled", "", "B  Limbach datasheet"),
        ("Envelope", f"{el.LENGTH_M * 1000:.0f} × {el.WIDTH_M * 1000:.0f} × {el.HEIGHT_M * 1000:.0f} mm", "",
         "B  Limbach drawing"),
        ("Displacement / rating", f"{el.displacement_m3() * 1e6:.0f} cc / {el.RATED_POWER_W / 1e3:.0f} kW, "
         f"{el.RATED_RPM:,.0f} rpm", "IR heat balance only", "B  Limbach datasheet"),
        ("Fins / exhaust stacks", f"{el.FIN_PACK['fins']} fins per cylinder; Ø {2 * el.STUB['r'] * 1000:.1f} mm stacks",
         "4 stubs pointing down", "B− measured on drawing"),
        ("PERFORMANCE", None, None, None),
        ("Cruise speed", "44 m/s (160 km/h)", "39–56 m/s", "B  press report"),
        ("Operating altitude", "up to 5,000 m", "", "B  press report"),
        ("Range, endurance, payload", "not published", "", "D  unknown"),
    ]


def main():
    cfg = kg.load_config()
    p = kg.defaults(cfg)
    p["fin_tip_chord"] = PHOTO_FIN_TIP * kg.derive(p)["fin_root_chord"]       # fin root chord is derived
    mesh = kg.build_mesh(p)
    g = mesh["derived"]
    S = g["span_m"]
    if kg.check(g):
        raise SystemExit(kg.check(g))

    FW, FH = 16.54, 11.69                                  # A3 landscape, inches
    SC = 1.95                                              # inches per metre
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    def axes_in(x0, y0, xr, yr):
        a = fig.add_axes([x0 / FW, y0 / FH, (xr[1] - xr[0]) * SC / FW, (yr[1] - yr[0]) * SC / FH])
        a.set_xlim(*xr); a.set_ylim(*yr); a.set_aspect("equal"); a.axis("off")
        return a

    nx, te, hub = g["nose_x"] * S, g["te_x"] * S, g["hub_x"] * S
    tle, zw = g["tip_le_x"] * S, g["wing_z"] * S
    X = (nx - g["probe_len"] * S - 0.25, hub + 0.75)
    TOPY, SIDEY = (-1.75, 1.55), (-0.55, 0.55)
    side = axes_in(0.45, 1.35, X, SIDEY)
    top = axes_in(0.45, 1.35 + (SIDEY[1] - SIDEY[0]) * SC + 0.45, X, TOPY)
    FRX, FRY = (-1.55, 1.55), (-0.55, 0.62)
    xr = 0.45 + (X[1] - X[0]) * SC + 0.35
    front = axes_in(xr, FH - 0.75 - (FRY[1] - FRY[0]) * SC, FRX, FRY)

    # ---------------- top view (nose left, starboard down)
    draw_view(top, mesh, "top")
    half = S / 2
    yd = -half - 0.42
    dim(top, (nx, yd), (hub, yd), f"length, nose to hub {g['length'] * S:.2f} m", off=-0.11,
        ext=[(nx, -0.25), (hub, -0.1)])
    dim(top, (0, -half - 0.16), (te, -half - 0.16), f"root chord {te:.2f} m", off=-0.10, ext=[(0, -0.17), (te, -half)])
    dim(top, (X[1] - 0.35, -half), (X[1] - 0.35, half), f"span {S:.2f} m", off=0.12, side=-1,
        ext=[(te + 0.15, -half), (te + 0.15, half)])
    dim(top, (tle, half + 0.22), (te, half + 0.22), f"tip chord {g['tip_chord'] * S:.2f} m", off=0.11,
        ext=[(tle, half), (te, half)])
    # sweep: angle between the leading edge and the spanwise line through the apex
    top.plot([0, 0], [0, 0.95], color=MUTED, lw=0.6, ls=(0, (6, 3)))
    a_le = math.atan2(half, tle)
    th = np.linspace(a_le, math.pi / 2, 30)
    top.plot(0.7 * np.cos(th), 0.7 * np.sin(th), color=DIM, lw=0.7)
    top.text(0.36, 0.86, f"LE sweep {g['sweep_deg']:.1f}°", fontsize=8.5, color=INK)
    b = g["elevon_breaks"]
    for y0_, y1_ in ((b[0], b[1]), (b[1], b[2])):
        top.annotate("elevon", (te - g["elevon_c"] * S / 2, (y0_ + y1_) / 2 * S), (te + 0.45, (y0_ + y1_) / 2 * S + 0.1),
                     fontsize=7.5, color=MUTED, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    top.text(X[0], TOPY[1] - 0.12, "TOP VIEW", fontsize=12, weight="bold", color=INK)
    scale_bar(top, X[0] + 0.05, TOPY[0] + 0.12)

    # ---------------- side view (from port, nose left)
    draw_view(side, mesh, "side")
    hu, hd = g["fin_up_height"] * S, g["fin_down_height"] * S
    xf = hub + 0.32
    dim(side, (xf, zw), (xf, zw + hu), f"{hu:.2f} m", off=0.17, side=-1, ext=[(te, zw), (te, zw + hu)])
    dim(side, (xf, zw - hd), (xf, zw), f"{hd:.2f} m", off=0.17, side=-1, ext=[(te, zw - hd), (te, zw)])
    Rf = g["R_fore"] * S
    xd = nx + 0.45
    dim(side, (xd, -Rf), (xd, Rf), f"Ø {2 * Rf * 1000:.0f} mm", off=0.0, fs=7.5)
    side.annotate("pitot probe + 2 blade antennas", (nx + 0.12, Rf + 0.03), (0.35, 0.40), fontsize=7.5, color=MUTED,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    side.annotate(f"MD-550 / L550E engine\n{el.LENGTH_M * 1000:.0f} x {el.WIDTH_M * 1000:.0f} x {el.HEIGHT_M * 1000:.0f} mm",
                  (g["eng0"] * S + 0.08, -0.10), (1.35, -0.47), fontsize=7.5, color=MUTED,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    side.annotate(f"2-blade pusher propeller, Ø {g['prop_dia'] * S:.2f} m", (hub + 0.01, 0.22), (hub - 0.55, 0.47),
                  fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    side.text(X[0], SIDEY[1] - 0.1, "SIDE VIEW  (from port)", fontsize=12, weight="bold", color=INK)

    # ---------------- front view
    draw_view(front, mesh, "front")
    dim(front, (-half, -0.45), (half, -0.45), f"span {S:.2f} m", off=-0.08, ext=[(-half, -0.05), (half, -0.05)])
    hfin = hu + hd
    dim(front, (half + 0.18, zw - hd), (half + 0.18, zw + hu), f"fin {hfin:.2f} m", off=0.14, side=-1)
    front.text(FRX[0], FRY[1] - 0.1, "FRONT VIEW", fontsize=12, weight="bold", color=INK)

    # ---------------- specification table
    rows = spec_rows(g, cfg)
    tx0, ty = xr / FW, (FH - 0.95 - (FRY[1] - FRY[0]) * SC - 0.25) / FH
    fig.text(tx0, ty, "SPECIFICATIONS", fontsize=12, weight="bold", color=INK, va="top")
    ty -= 0.030
    cols = (0.0, 0.118, 0.262, 0.345)                      # figure fractions: parameter, value, range, basis
    for k, lab in enumerate(("Parameter", "Value", "Range*", "Basis")):
        fig.text(tx0 + cols[k], ty, lab, fontsize=7.6, weight="bold", color=MUTED, va="top")
    ty -= 0.016
    fig.add_artist(plt.Line2D([tx0, tx0 + 0.425], [ty + 0.004, ty + 0.004], color=INK, lw=0.8))
    for name, val, rng, basis in rows:
        if val is None:
            ty -= 0.005
            fig.text(tx0, ty, name, fontsize=8, weight="bold", color=INK, va="top")
            ty -= 0.0185
            continue
        fig.text(tx0 + cols[0], ty, name, fontsize=7.6, color=INK, va="top")
        fig.text(tx0 + cols[1], ty, val, fontsize=7.6, color=INK, va="top")
        fig.text(tx0 + cols[2], ty, rng, fontsize=7.2, color=MUTED, va="top")
        fig.text(tx0 + cols[3], ty, basis, fontsize=7.2, color=MUTED, va="top")
        ty -= 0.0172
    ty -= 0.006
    legend = ("Evidence: A observed in KAL imagery; B strongly inferred (photo measurement; press report, Indian Masterminds,\n"
              "21 Sep 2026); C modelling assumption; D unknown. *Ranges are photo-measurement uncertainty at the 2.30 m span.\n"
              "Airframe dimensions are photo-based estimates; no published KAL drawing exists. The engine is the closest drone\n"
              "analogue (user decision; Limbach L550E datasheet and drawing), not a published KAL fact. Fin tips drawn after P1.")
    fig.text(tx0, ty, legend, fontsize=6.5, color=MUTED, va="top", linespacing=1.35)

    # ---------------- title block
    fig.add_artist(plt.Rectangle((0.02, 0.02), 0.96, 0.07, transform=fig.transFigure, fc="none", ec=INK, lw=1.0))
    fig.text(0.035, 0.068, "KAL UAV  —  REFERENCE DRAWING AND SPECIFICATIONS", fontsize=14, weight="bold", color=INK,
             va="center")
    fig.text(0.035, 0.038, "Simulation target model for the multispectral (RGB / LWIR / SWIR) aerial detection project.  "
             "Units: metres unless stated.  All views at the same scale.",
             fontsize=8.2, color=DIM, va="center")
    fig.text(0.965, 0.068, "Sources: photos P1-P8 (docs/reference/photos), configs/target.yaml, target/engine_l550e.py, docs/sources",
             fontsize=7.4, color=MUTED, va="center", ha="right")
    fig.text(0.965, 0.038, "Drawn from kal_geometry (canonical defaults)", fontsize=7.4, color=MUTED,
             va="center", ha="right")
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "KAL_reference_drawing.png")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    print("wrote", png)


if __name__ == "__main__":
    main()
