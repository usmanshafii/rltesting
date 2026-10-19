"""Decal atlas for the presentation model's "photo" look (target/kal_hero.py, look="photo").

    python tools/make_hero_decals.py      -> assets/decals/hero_decal_atlas.png (+ .json UV regions)

Lettering after the overhead render P1 and the exhibition close-up P3: "KAL" in classical Roman
capitals with a crossbar-less A (drawn as an inverted V), the fin wordmark "IG DEFENCE" in a bold
italic sans after a ring-and-arrowhead logo, the flag, and the same logo as a small spine emblem.
The logo is a simplified drawing, not the manufacturer's artwork. The dataset decals
(assets/decals/decal_atlas.png, tools/make_decals.py) are unchanged.
"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "decals")
N = 2048
FONT_DIRS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"), "/usr/share/fonts/truetype/dejavu"]
ROMAN = ["PERTILI.TTF", "pala.ttf", "times.ttf", "DejaVuSerif.ttf"]       # Perpetua Titling ~ P1/P3
WORDMARK = ["arialbi.ttf", "segoeuiz.ttf", "DejaVuSans-BoldOblique.ttf"]
WING_ASPECT = 0.14 / 0.045          # configs/target.yaml markings.wing_text_size (length / height)
FIN_ASPECT = 6.0                    # logo + wordmark, laid along the fin (kal_hero_tex, look="photo")
INK = (238, 238, 236, 255)


def font(names, size):
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def glyph(ch, f, flip=False):
    """One glyph on its own transparent tile, cropped to its ink."""
    l, t, r, b = f.getbbox(ch)
    im = Image.new("RGBA", (r - l + 8, b - t + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4 - l, 4 - t), ch, font=f, fill=INK)
    im = im.crop(im.getbbox())
    return im.rotate(180) if flip else im


def fit_paste(dst, src, box, fill=0.9):
    x0, y0, x1, y1 = box
    k = min((x1 - x0) * fill / src.width, (y1 - y0) * fill / src.height)
    src = src.resize((max(1, int(src.width * k)), max(1, int(src.height * k))), Image.LANCZOS)
    dst.alpha_composite(src, ((x0 + x1 - src.width) // 2, (y0 + y1 - src.height) // 2))


def kal_word(size=520):
    """K, an inverted V for the crossbar-less A (its thick stroke then falls on the right, as on an
    A), L; baseline-aligned with wide Roman spacing."""
    f = font(ROMAN, size)
    parts = [glyph("K", f), glyph("V", f, flip=True), glyph("L", f)]
    gap = int(size * 0.16)
    h = max(p.height for p in parts)
    im = Image.new("RGBA", (sum(p.width for p in parts) + gap * 2, h), (0, 0, 0, 0))
    x = 0
    for p in parts:
        im.alpha_composite(p, (x, h - p.height))
        x += p.width + gap
    return im


def logo(size):
    """Ring with a swept arrowhead pointing right whose tip breaks out of the ring (P1 fin)."""
    im = Image.new("RGBA", (int(size * 1.35), size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = size / 2
    cx, cy = r, r
    w = max(2, int(size * 0.07))
    d.ellipse((cx - r + w / 2, cy - r + w / 2, cx + r - w / 2, cy + r - w / 2), outline=INK, width=w)
    tip = (size * 1.32, cy - size * 0.02)
    rear_up, rear_lo = (cx - r * 0.55, cy - r * 0.62), (cx - r * 0.55, cy + r * 0.62)
    notch = (cx - r * 0.05, cy + r * 0.05)
    d.polygon([tip, rear_up, notch, rear_lo], fill=INK)
    # a dark slot along the arrow's spine, as on the P1 mark
    d.line([(cx - r * 0.2, cy + r * 0.02), (tip[0] - size * 0.35, tip[1] + size * 0.01)], fill=(0, 0, 0, 0), width=max(1, w // 2))
    return im


def wordmark(height):
    f = font(WORDMARK, int(height * 0.8))
    l, t, r, b = f.getbbox("IG DEFENCE")
    lg = logo(height)
    im = Image.new("RGBA", (lg.width + int(height * 0.25) + (r - l) + 8, height), (0, 0, 0, 0))
    im.alpha_composite(lg, (0, 0))
    ImageDraw.Draw(im).text((lg.width + int(height * 0.25) - l, (height - (b - t)) / 2 - t), "IG DEFENCE", font=f, fill=INK)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    regions = {}
    y = 0

    def region(name, h, w=N):
        nonlocal y
        box = (0, y, w, y + h)
        regions[name] = [0.0, 1 - (y + h) / N, w / N, 1 - y / N]      # PIL rows run down, v runs up
        y += h + 16
        return box

    fit_paste(img, kal_word(), region("wing_text", int(N / WING_ASPECT)))
    fit_paste(img, wordmark(256), region("fin_text", int(N / FIN_ASPECT)), fill=0.94)
    # flag: same layout as tools/make_decals.py (aspect from markings.flag_size)
    fl, fw = 0.050, 0.034
    x0, y0, x1, y1 = region("flag", int(N / 2 / (fl / fw)), N // 2)
    d = ImageDraw.Draw(img)
    s = (y1 - y0) // 3
    d.rectangle((x0, y0, x1, y0 + s), fill=(255, 153, 51, 255))
    d.rectangle((x0, y0 + s, x1, y0 + 2 * s), fill=(255, 255, 255, 255))
    d.rectangle((x0, y0 + 2 * s, x1, y1), fill=(19, 136, 8, 255))
    cx, cy, r = (x0 + x1) // 2, y0 + int(1.5 * s), int(s * 0.42)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 128, 255), width=max(2, r // 9))
    for k in range(24):
        a = 2 * math.pi * k / 24
        d.line([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a))], fill=(0, 0, 128, 255), width=max(1, r // 30))
    # spine emblem: the logo alone, beside the flag
    ex0, ew, eh = N // 2 + 32, 540, 400
    regions["emblem"] = [ex0 / N, 1 - (y0 + eh) / N, (ex0 + ew) / N, 1 - y0 / N]
    fit_paste(img, logo(400), (ex0, y0, ex0 + ew, y0 + eh), fill=0.96)
    img.save(os.path.join(OUT, "hero_decal_atlas.png"))
    with open(os.path.join(OUT, "hero_decal_atlas.json"), "w") as f:
        json.dump(regions, f, indent=2)
    print("wrote", os.path.join(OUT, "hero_decal_atlas.png"), regions)


if __name__ == "__main__":
    main()
