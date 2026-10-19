"""Preview stills of the high-detail KAL presentation model (Blender 4.2, Eevee).

    blender -b -P tools/hero_preview.py -- --out out/hero_preview [--views all] [--samples 32]
"""
import argparse
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "target")]
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import kal_hero  # noqa: E402

VIEWS = {   # name: camera position (m, model frame), look-at, lens mm
    "front34": ((5.0, 3.2, 1.4), (0.2, 0, 0), 50),
    "top34": ((-2.5, 3.5, 3.6), (0.0, 0, 0), 45),
    "rear34": ((-4.6, -2.8, 1.0), (-0.9, 0, 0), 50),
    "under": ((2.0, -2.5, -3.2), (0.0, 0, 0), 40),
    "engine": ((-2.6, -1.1, 0.75), (-1.45, 0, 0.05), 85),
    "engine_under": ((-2.9, -0.9, -0.55), (-1.45, 0, -0.05), 70),
    "nose": ((2.8, 1.2, 0.6), (1.4, 0, 0), 85),
}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/hero_preview")
    ap.add_argument("--views", default="front34,rear34,engine")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--prop", default="disc")
    ap.add_argument("--look", default="photo")
    ap.add_argument("--res", default="1280x720")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.eevee.taa_render_samples = a.samples
    sc.eevee.use_shadows = True
    sc.eevee.use_raytracing = True
    w, h = map(int, a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    sc.world = world
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(38)
    sky.sun_rotation = math.radians(140)
    sky.sun_disc = False
    world.node_tree.links.new(sky.outputs[0], world.node_tree.nodes["Background"].inputs["Color"])
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.0
    sun.data.angle = math.radians(0.6)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(52), 0, math.radians(140 + 90))
    root, hd = kal_hero.build_hero(prop_mode=a.prop, look=a.look)
    kal_hero.set_spin(hd, math.radians(23))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.clip_start = 0.05
    names = list(VIEWS) if a.views == "all" else a.views.split(",")
    for n in names:
        pos, look, lens = VIEWS[n]
        cam.location = pos
        d = Vector(look) - Vector(pos)
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = lens
        sc.render.filepath = os.path.join(out, f"{n}.png")
        bpy.ops.render.render(write_still=True)
        print("HERO_VIEW", n, flush=True)


main()
