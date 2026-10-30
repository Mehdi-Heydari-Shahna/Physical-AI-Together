"""
Joins the frames that build.py rendered into a video with the bundle's
captions, and writes the book's stills.

Physical AI, Together, Chapters 13 to 15 (the videos of their runs, path
traced).

    python compose.py --bundle DIR --frames DIR --out FILE.mp4 [--no-captions]
    python compose.py --bundle DIR --frames DIR --stills DIR

The captions are drawn as render/film.py draws them on the MuJoCo videos
(the same boxes, fonts and places); the video is H.264 in an MP4 file
through ffmpeg (CRF 18, yuv420p). --stills writes each still of the bundle
that build.py rendered with --frames stills as a 1600 x 900 JPEG without
captions, the size of the book's stills. Needs Pillow, ffmpeg on the PATH
and, for the video's captions, the DejaVu Sans fonts (render/film.py's;
matplotlib's copy will do), else Liberation Sans or Arial, or --font and
--font-bold.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _matplotlib_fonts():
    """DejaVu Sans as matplotlib ships it, if matplotlib is installed."""
    import importlib.util
    spec = importlib.util.find_spec("matplotlib")
    if spec is None or not spec.submodule_search_locations:
        return ()
    ttf = os.path.join(list(spec.submodule_search_locations)[0], "mpl-data", "fonts", "ttf")
    return (os.path.join(ttf, "DejaVuSans.ttf"),)


# DejaVu Sans first, the font of render/film.py; then fallbacks that look alike
_DEJAVU = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
           "C:/Windows/Fonts/DejaVuSans.ttf", "/Library/Fonts/DejaVuSans.ttf", "DejaVuSans.ttf")
FONTS = _DEJAVU + _matplotlib_fonts() + (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf")
FONTS_BOLD = tuple(f.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                   for f in _DEJAVU + _matplotlib_fonts()) + (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf")
_TOLD = set()


def font(paths, size):
    for p in paths:
        try:
            f = ImageFont.truetype(p, size)
        except OSError:
            continue
        if "DejaVu" not in os.path.basename(p) and p not in _TOLD:
            _TOLD.add(p)
            print(f"DejaVu Sans not found: the captions use {p}")
        return f
    raise SystemExit("no font found for the captions: give --font and --font-bold (TrueType files)")


def caption(img, title, lines, regular, bold, corner="bottom-left", scale=1.0):
    """A title and lines of text in a translucent box (render/film.py's)."""
    w, h = img.size
    s = scale * h / 720.0
    big = font(bold, int(22 * s))
    small = font(regular, int(18 * s))
    pad = int(12 * s)
    rows = [(title, big)] if title else []
    rows += [(line, small) for line in lines]
    sizes = [ImageDraw.Draw(img).textbbox((0, 0), t, font=f) for t, f in rows]
    bw = max(b[2] - b[0] for b in sizes) + 2 * pad
    bh = sum(b[3] - b[1] for b in sizes) + pad * (len(rows) + 1)
    x0 = pad if "left" in corner else w - bw - pad
    y0 = h - bh - pad if "bottom" in corner else pad
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(8 * s), fill=(12, 16, 20, 150))
    y = y0 + pad
    for (text, f), b in zip(rows, sizes):
        d.text((x0 + pad, y - b[1]), text, font=f, fill=(240, 240, 236, 255))
        y += b[3] - b[1] + pad
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def inset(img, panel, regular, corner="bottom-right", scale=1.0):
    """A panel seen from above in a translucent box (render/film.py's)."""
    w, h = img.size
    s = scale * h / 720.0
    size, pad = int(190 * s), int(12 * s)
    small = font(regular, int(15 * s))
    text = panel.get("text", "")
    tb = ImageDraw.Draw(img).textbbox((0, 0), text or "Ag", font=small)
    inner = max(size, tb[2] - tb[0])
    bw, bh = inner + 2 * pad, size + 3 * pad + (tb[3] - tb[1])
    x0 = w - bw - pad if "right" in corner else pad
    y0 = h - bh - pad if "bottom" in corner else pad
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(8 * s), fill=(12, 16, 20, 150))
    half = float(panel.get("half", 0.25))
    cx, cy = x0 + pad + inner / 2, y0 + pad + size / 2

    def px(p):
        return (cx + p[0] / half * size / 2, cy - p[1] / half * size / 2)

    for points, outline, fill in panel.get("polys", []):
        pts = [px(p) for p in points]
        if fill is not None:
            d.polygon(pts, fill=tuple(int(255 * c) for c in fill))
        d.line(pts + [pts[0]], fill=tuple(int(255 * c) for c in outline), width=max(1, int(2 * s)))
    for p, rgba in panel.get("dots", []):
        x, y = px(p)
        r = 4.5 * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(255 * c) for c in rgba))
    if text:
        d.text((x0 + pad, y0 + 2 * pad + size - tb[1]), text, font=small, fill=(240, 240, 236, 255))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="the path-traced frames as a video, and the stills")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--frames", required=True, help="the folder that build.py rendered into")
    ap.add_argument("--out", help="the video to write (.mp4)")
    ap.add_argument("--stills", help="the folder for the book's stills (JPEG)")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-bold", default=None)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    regular = (args.font,) if args.font else FONTS
    bold = (args.font_bold,) if args.font_bold else FONTS_BOLD
    with open(os.path.join(args.bundle, "scene.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(args.bundle, "captions.json"), encoding="utf-8") as fh:
        captions = json.load(fh)
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        for k, name in meta["stills"].items():
            src = os.path.join(args.frames, f"{name}.png")
            if not os.path.exists(src):
                src = os.path.join(args.frames, f"frame_{int(k):05d}.png")
            if not os.path.exists(src):
                print(f"missing: the still {name} (frame {k})")
                continue
            img = Image.open(src).convert("RGB").resize((1600, 900), Image.LANCZOS)
            dst = os.path.join(args.stills, f"{name}-cycles.jpg")
            img.save(dst, quality=92, subsampling=0, optimize=True)
            print(dst)
    if not args.out:
        return
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg is not on the PATH")
    n = int(meta["frames"])
    first = Image.open(os.path.join(args.frames, "frame_00000.png"))
    w, h = first.size
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    proc = subprocess.Popen([exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264",
                             "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    missing = 0
    for i in range(n):
        path = os.path.join(args.frames, f"frame_{i:05d}.png")
        if not os.path.exists(path):
            missing += 1
            continue
        img = Image.open(path).convert("RGB")
        cap = captions[i]
        if not args.no_captions and cap.get("panel") is not None:
            img = inset(img, cap["panel"], regular)
        if not args.no_captions and cap["caption"] is not None:
            img = caption(img, cap["caption"][0], cap["caption"][1], regular, bold)
            if cap["caption2"] is not None:
                img = caption(img, cap["caption2"][0], cap["caption2"][1], regular, bold,
                              corner="top-right", scale=0.8)
        proc.stdin.write(np.asarray(img, np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {args.out}")
    print(f"{args.out}: {n - missing} frames" + (f", {missing} missing" if missing else ""))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
