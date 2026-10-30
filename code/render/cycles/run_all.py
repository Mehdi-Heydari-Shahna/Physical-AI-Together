"""
The path-traced versions of the book's videos, from export to video, in one
command: for each video, its script writes the bundle (--export), Blender
renders the stills and the frames with Cycles (build.py), and compose.py
writes the stills as JPEG and the video as MP4. Everything lands in a work
folder; output/ in it holds what the book uses.

Physical AI, Together, Chapters 13 to 15 (the videos of their runs, path
traced).

    python run_all.py --blender BLENDER [--clearpath DIR] [--ur DIR]
                      [--menagerie DIR] [--team-impedance DIR]
                      [--only husky,ur5e,hdmm,excavator,go2,apollo]
                      [--smoke] [--stills-only] [--samples N] [--still-samples N]
                      [--device OPTIX] [--work DIR]

--smoke renders one still of each video at 640 x 360 with 32 samples, to
try the chain (minutes); --stills-only renders the book's four stills of
each video and no video. A video whose maker's checkout is not given is
left out. The humanoid's video draws the record that
code/ch15/humanoid_carry.py wrote, results/humanoid-frames.npz, which the
repository holds. --humanoid-run also runs that script again with daqp,
the repository's QP solver, and --check, into the work folder (humanoid-daqp/),
leaving the book's results as they are; it needs daqp, and its outcome, as
expected or not, goes into output/.

Runs with the Python that runs it (the book's environment: MuJoCo 3.16.0,
NumPy, SciPy, Pillow) for the exports and the compose step, and with the
Blender given (4.2 or later) for the render.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))

VIDEOS = {   # name: (script, the option that names a maker's checkout, its argument)
    "husky": ("ch13/husky_video.py", "--clearpath", "clearpath"),
    "ur5e": ("ch14/arm_video.py", "--ur", "ur"),
    "hdmm": ("ch14/hdmm_video.py", None, None),
    "excavator": ("ch14/excavator_video.py", None, None),
    "go2": ("ch15/quad_video.py", "--menagerie", "menagerie"),
    "apollo": ("ch15/humanoid_video.py", "--team-impedance", "team_impedance"),
}


def run(cmd, log, cwd=None):
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"failed ({p.returncode}) after {time.time() - t0:.0f} s; see {log}")
    print(f"    {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="the book's videos, path traced with Cycles")
    ap.add_argument("--blender", required=True, help="Blender's executable (4.2 or later)")
    ap.add_argument("--clearpath", help="a checkout of clearpath_common (the Husky)")
    ap.add_argument("--ur", help="a checkout of Universal_Robots_ROS2_Description (the UR5e)")
    ap.add_argument("--menagerie", help="a checkout of MuJoCo Menagerie (the Go2)")
    ap.add_argument("--team-impedance", help="a checkout of Team-Impedance-Control (the Apollo)")
    ap.add_argument("--only", default=",".join(VIDEOS))
    ap.add_argument("--work", default=os.path.join(os.getcwd(), "cycles-work"))
    ap.add_argument("--smoke", action="store_true", help="one still a video, small and quick")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--samples", type=int, default=128, help="for the videos' frames")
    ap.add_argument("--still-samples", type=int, default=1024, help="for the book's stills")
    ap.add_argument("--device", default="OPTIX")
    ap.add_argument("--humanoid-run", action="store_true",
                    help="run code/ch15/humanoid_carry.py again with daqp and --check, into the "
                         "work folder (needs daqp)")
    ap.add_argument("--fresh", action="store_true", help="export the bundles again")
    ap.add_argument("--font", help="a TrueType font for the captions, if DejaVu Sans is missing")
    ap.add_argument("--font-bold", help="its bold face")
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "run_all.log")
    back = os.path.join(work, "output")
    os.makedirs(back, exist_ok=True)
    py = sys.executable
    daqp_out = os.path.join(work, "humanoid-daqp")
    if args.humanoid_run:
        if not args.team_impedance:
            raise SystemExit("--humanoid-run needs --team-impedance")
        print("the humanoid's run with daqp (code/ch15/humanoid_carry.py --qp daqp --check)")
        try:   # a check that fails is an outcome to report, not a reason to stop
            run([py, os.path.join(CODE, "ch15", "humanoid_carry.py"), "--team-impedance",
                 args.team_impedance, "--qp", "daqp", "--check", "--out", daqp_out], log,
                cwd=os.path.join(CODE, "ch15"))
        except SystemExit as e:
            print(f"    {e}")
    summary = {}
    for name in [v.strip() for v in args.only.split(",") if v.strip()]:
        script, option, attr = VIDEOS[name]
        checkout = getattr(args, attr) if attr else None
        if option and not checkout:
            print(f"{name}: left out ({option} not given)")
            continue
        if name == "apollo" and not os.path.exists(os.path.join(CODE, "ch15", "results",
                                                                "humanoid-frames.npz")):
            print("apollo: left out (no results/humanoid-frames.npz; run with --humanoid-run)")
            continue
        print(f"{name}:")
        folder = os.path.join(work, name)
        bundle = os.path.join(folder, "bundle")
        if args.fresh and os.path.isdir(bundle):
            shutil.rmtree(bundle)
        if not os.path.exists(os.path.join(bundle, "scene.json")):
            cmd = [py, os.path.join(CODE, script), "--export", bundle]
            if option:
                cmd += [option, checkout]
            run(cmd, log, cwd=os.path.dirname(os.path.join(CODE, script)))
        build = [args.blender, "-b", "-P", os.path.join(HERE, "build.py"), "--", "--bundle", bundle,
                 "--device", args.device]
        if args.smoke:
            with open(os.path.join(bundle, "scene.json"), encoding="utf-8") as fh:
                first = sorted(int(k) for k in json.load(fh)["stills"])[0]
            out = os.path.join(folder, "smoke")
            run(build + ["--out", out, "--frames", str(first), "--size", "640x360", "--samples", "32",
                         "--overwrite"], log)
            summary[name] = os.path.join(out, f"frame_{first:05d}.png")
            continue
        stills_png = os.path.join(folder, "stills-png")
        run(build + ["--out", stills_png, "--frames", "stills", "--samples",
                     str(args.still_samples)], log)
        run([py, os.path.join(HERE, "compose.py"), "--bundle", bundle, "--frames", stills_png,
             "--stills", back], log)
        shutil.copy(os.path.join(stills_png, "render.json"), os.path.join(back, f"{name}-stills-render.json"))
        if args.stills_only:
            summary[name] = "stills"
            continue
        frames = os.path.join(folder, "frames")
        run(build + ["--out", frames, "--samples", str(args.samples)], log)
        with open(os.path.join(bundle, "scene.json"), encoding="utf-8") as fh:
            video = json.load(fh)["name"]
        mp4 = os.path.join(back, f"{video}-cycles.mp4")
        fonts = (["--font", args.font] if args.font else []) + \
            (["--font-bold", args.font_bold] if args.font_bold else [])
        run([py, os.path.join(HERE, "compose.py"), "--bundle", bundle, "--frames", frames, "--out", mp4]
            + fonts, log)
        shutil.copy(os.path.join(frames, "render.json"), os.path.join(back, f"{name}-video-render.json"))
        summary[name] = mp4
    if args.humanoid_run and os.path.isdir(daqp_out):
        for f in ("humanoid-summary.txt", "humanoid-record.txt"):
            if os.path.exists(os.path.join(daqp_out, f)):
                shutil.copy(os.path.join(daqp_out, f), os.path.join(back, f.replace("humanoid-", "humanoid-daqp-")))
    shutil.copy(log, back)
    if not args.smoke:
        zpath = os.path.join(work, "cycles-output.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(back)):
                z.write(os.path.join(back, f), f)
        print(f"the stills, the videos, their render settings and the log: {zpath}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
