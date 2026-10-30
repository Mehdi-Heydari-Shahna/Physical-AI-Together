"""
Drawing a recorded run as a video, in parallel, and its stills.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

A shot is an object that the worker processes rebuild: build() returns the
visual model, pose(model, data, i) sets frame i's state into data (from the
record, nothing simulated), view(i) gives the camera, overlay(i, model,
data) an optional function that adds geoms to the scene, caption(i) the
text on frame i, or None, and panel(i, model, data), if the shot has it, a
drawing from above in a corner of the frame (film.inset). render() splits
the frames between processes, each with its own OpenGL context, writes one
H.264 segment each and joins them without re-encoding; join() joins the
videos of parts drawn separately; still() draws single frames for the book.
"""
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from . import film


def _worker(args):
    shot, frames, path, size, ss, fps, crf, look = args
    m = shot.build()
    d = mujoco.MjData(m)
    f = film.Film(m, size[0], size[1], supersample=ss)
    video = film.Video(path, size[0], size[1], fps=fps, crf=crf)
    for i in frames:
        shot.pose(m, d, i)
        lookat, dist, az, el = shot.view(i)
        if hasattr(shot, "focus"):
            shot.focus(m, lookat)
        img = f.shoot(d, lookat, dist, az, el, overlay=shot.overlay(i, m, d))
        img = film.grade(img, **look)
        panel = shot.panel(i, m, d) if hasattr(shot, "panel") else None
        if panel is not None:
            img = film.inset(img, panel)
        cap = shot.caption(i)
        if cap is not None:
            img = film.caption(img, *cap)
            extra = getattr(shot, "caption2", None)
            if extra is not None and extra(i) is not None:
                img = film.caption(img, *extra(i), corner="top-right", scale=0.8)
        video.add(img)
    video.close()
    return path


def render(shot, n_frames, out, size=(1920, 1080), supersample=2, fps=30, crf=18,
           workers=2, look=None, frames=None):
    """Draws frames 0 .. n_frames-1 of shot (or the frames listed) into the
    MP4 file out. Each process writes one segment; a process that fails, or
    is killed (for example for want of memory), stops the render with an
    error, and every segment's frame count is checked before they are
    joined."""
    look = look or {}
    tmp = tempfile.mkdtemp(prefix="video-")
    chunks = np.array_split(np.arange(n_frames) if frames is None else np.asarray(frames), workers)
    jobs = [(shot, list(map(int, c)), os.path.join(tmp, f"part{k}.mp4"), size, supersample,
             fps, crf, look) for k, c in enumerate(chunks) if len(c)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(job,)) for job in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [k for k, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"render: segments {failed} failed (exit codes "
                           f"{[procs[k].exitcode for k in failed]}); their files are in {tmp}")
    parts = [job[2] for job in jobs]
    for job in jobs:
        if probe(job[2])[0] != len(job[1]):
            raise RuntimeError(f"render: {job[2]} holds {probe(job[2])[0]} frames, not "
                               f"{len(job[1])}")
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run([shutil.which("ffmpeg"), "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", out],
                   check=True)
    shutil.rmtree(tmp)
    return out


def join(parts, out):
    """Joins MP4 files of the same size and rate into out, without
    re-encoding."""
    tmp = tempfile.mkdtemp(prefix="join-")
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w") as fh:
        fh.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run([shutil.which("ffmpeg"), "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp)
    return out


def still(shot, i, path, size=(1920, 1080), supersample=2, captions=False, quality=92,
          look=None):
    """Draws frame i alone into an image file (JPEG or PNG by its suffix)."""
    look = look or {}
    m = shot.build()
    d = mujoco.MjData(m)
    f = film.Film(m, size[0], size[1], supersample=supersample)
    shot.pose(m, d, i)
    lookat, dist, az, el = shot.view(i)
    if hasattr(shot, "focus"):
        shot.focus(m, lookat)
    img = f.shoot(d, lookat, dist, az, el, overlay=shot.overlay(i, m, d))
    img = film.grade(img, **look)
    if captions and shot.caption(i) is not None:
        img = film.caption(img, *shot.caption(i))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith((".jpg", ".jpeg")):
        img.save(path, quality=quality, subsampling=0, optimize=True)
    else:
        img.save(path)
    return path


def fixed(value, digits):
    """value printed with digits decimals, without a negative zero (for the
    frames files)."""
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def probe(path):
    """The video's frame count, frame rate and size, as ffprobe reads them."""
    out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-count_frames", "-show_entries",
                          "stream=nb_read_frames,r_frame_rate,width,height", "-of",
                          "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h, rate, n = out.stdout.strip().split(",")
    num, den = rate.split("/")
    return int(n), float(num) / float(den), int(w), int(h)
