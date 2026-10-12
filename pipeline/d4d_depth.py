"""Estimate depth on the D4D frame nearest to each side's scan, for the front/behind relation.

For each side of the population, the stage runs DA3 on a window of `--window` frames around the frame nearest
to the scan, and keeps that frame's depth alone. The window gives DA3 the context of neighbouring frames. A
scan lies outside its clip, so the frame sits at the window's start for a `start` side and at its end for an
`end` side. The stage writes one file per side, `<key, / as __>__<side>.npz`. It skips a side whose file exists
once the file's frame, window and resolution are this run's, and refuses one whose differ.

Usage:
    python -m pipeline.d4d_depth --root /path/to/D4D/extracted --population d4d_meta/population.json \\
        --out /path/to/d4d_depth [--clips <key> ...] [--shard 0 --n-shards 1] [--gpu N]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import surgical_core.d4d as d4d
from recon3d_wrapper.da3 import DA3, DEFAULT_PROCESS_RES

# The frames DA3 sees at once around a side's frame: as many as a CholecSeg8k clip holds.
WINDOW = 30

# The fields that say which frame and which settings made a side's file.
SETTINGS = ("frame_index", "window", "process_res")


def side_stem(key: str, side: str) -> str:
    """Return the name of a side's files without their suffix: `<key, / as __>__<side>`."""
    return f"{key.replace('/', '__')}__{side}"


def window(n_frames: int, index: int, size: int) -> tuple[int, int]:
    """Return the first frame and one past the last frame of the window of `size` frames around `index`.

    The window is centred on `index` and moved inside the clip at its ends. A clip shorter than `size` is
    taken whole.
    """
    if n_frames < size:
        return 0, n_frames
    lo = max(0, min(index - size // 2, n_frames - size))
    return lo, lo + size


def population_sides(population: dict, clips: list[str] | None) -> list[tuple[str, str]]:
    """Return the population's sides as `(key, side)`, sorted, those of `clips` alone when it is given.

    Raises:
        ValueError: a clip of `clips` has no side in the population.
    """
    sides = sorted((s["key"], s["side"]) for s in population["sides"])
    if clips is None:
        return sides
    unknown = sorted(set(clips) - {k for k, _ in sides})
    if unknown:
        raise ValueError(f"{len(unknown)} clip(s) with no side in the population: {unknown}")
    return [s for s in sides if s[0] in clips]


def side_frame(clip, side: str) -> tuple[int, float]:
    """Return the index of the frame nearest to the side's scan, and that frame's time less the scan's, in s.

    Raises:
        ValueError: the side has no scan.
    """
    scan = clip.pointclouds.get(side)
    if not scan:
        raise ValueError(f"{d4d.clip_key(clip)}: no {side} scan")
    return d4d.nearest_frame(d4d.frame_times(clip), d4d.name_time(scan))


def settings(clip, side: str, size: int, process_res: int) -> dict:
    """Return the fields of `SETTINGS` that a run with these settings writes for the side."""
    index, _ = side_frame(clip, side)
    lo, hi = window(len(clip.left_img_paths), index, size)
    return {"frame_index": index, "window": hi - lo, "process_res": process_res}


def check_done(path: Path, expected: dict) -> None:
    """Refuse a side's file whose frame, window or resolution differ from this run's.

    Raises:
        ValueError: a field of `SETTINGS` differs, or is missing.
    """
    with np.load(path, allow_pickle=False) as z:
        held = {k: int(z[k]) if k in z.files else None for k in SETTINGS}
    if held != expected:
        raise ValueError(f"{path.name} holds {held}, where this run makes {expected}; remove it to make it again")


def run_side(clip, side: str, model, size: int, process_res: int, dst: Path) -> None:
    """Estimate depth on the window around the frame nearest to the side's scan, and write the frame's.

    The file is written under a temporary name and then renamed, so that a run stopped midway leaves no file
    that a later run would skip.

    Raises:
        ValueError: the side has no scan, or the model returns another number of frames than the window holds.
    """
    # Find the frame nearest to the scan, and the window around it.
    index, gap = side_frame(clip, side)
    lo, hi = window(len(clip.left_img_paths), index, size)
    paths = [str(p) for p in clip.left_img_paths[lo:hi]]

    # Estimate depth on the window, and keep the side's frame.
    rec = model.reconstruct(paths)
    if len(rec.depth) != len(paths):
        raise ValueError(f"{d4d.clip_key(clip)}: the model returned {len(rec.depth)} frames for {len(paths)}")
    at = index - lo
    tmp = dst.with_name(dst.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, depth=rec.depth[at].astype(np.float32), conf=rec.conf[at].astype(np.float32),
                            intrinsics=rec.intrinsics[at].astype(np.float32),
                            extrinsics=rec.extrinsics[at].astype(np.float32), frame_index=np.int32(index),
                            frame_minus_gt_s=np.float32(gap), window=np.int32(len(paths)),
                            process_res=np.int32(process_res), image_path=np.str_(paths[at]))
    os.replace(tmp, dst)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="the dataset's root, which holds the specimen_* directories")
    ap.add_argument("--population", required=True, help="the population file, d4d_meta/population.json")
    ap.add_argument("--out", required=True, help="the directory the sides' files go to")
    ap.add_argument("--clips", nargs="*", help="clip keys whose sides alone are run; default: every side")
    ap.add_argument("--window", type=int, default=WINDOW, help="the frames DA3 sees at once")
    ap.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES,
                    help="the longest side each frame is resized to")
    ap.add_argument("--shard", type=int, default=0, help="run the sides whose position modulo --n-shards is this")
    ap.add_argument("--n-shards", type=int, default=1, help="the number of processes the sides are split over")
    ap.add_argument("--gpu", type=int, default=None, help="the index of the GPU to use")
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard {args.shard} is not in 0..{args.n_shards - 1}")

    # List this shard's sides, and refuse a side whose clip the dataset does not hold.
    population = json.loads(Path(args.population).read_text(encoding="utf-8"))
    sides = population_sides(population, args.clips)[args.shard::args.n_shards]
    clips = {d4d.clip_key(c): c for c in d4d.clips(args.root)}
    missing = sorted({k for k, _ in sides} - set(clips))
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) of the population are not in {args.root}: {missing[:5]}")

    # Skip the sides already made with these settings, and refuse those made with others, before the model loads.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for key, side in sides:
        dst = out / f"{side_stem(key, side)}.npz"
        if not dst.is_file():
            todo.append((key, side, dst))
            continue
        check_done(dst, settings(clips[key], side, args.window, args.process_res))
    print(f"shard {args.shard}/{args.n_shards}: {len(sides)} sides, {len(todo)} to run")
    if not todo:
        return

    model = DA3(gpu=args.gpu, process_res=args.process_res)
    for i, (key, side, dst) in enumerate(todo, 1):
        run_side(clips[key], side, model, args.window, args.process_res, dst)
        print(f"  {i}/{len(todo)} {key} {side}", flush=True)


if __name__ == "__main__":
    main()
