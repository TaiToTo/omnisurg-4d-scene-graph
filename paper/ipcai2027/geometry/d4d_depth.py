"""Score each D4D side's depth against its structured-light scan: AbsRel of every answer, on one pixel set.

For each side of the population, the scan is rendered from the endoscope's pose as `d4d_front_behind` renders
it, at the frame nearest the scan. Every answer is scored on the pixels outside the instrument mask where the
scan, the half scan, the dataset's stereo depth and every model have a depth; a side needs `MIN_PIXELS` of them.
Each answer is scaled by the ratio of the medians over those pixels before AbsRel is taken. The answers are the
production DA3 depth, the whole-clip depth of DA3, Pi3X and MapAnything, the half scan and a flat plane. Writes
one row per side, which `d4d_depth_table.py` reads.

Usage:
    python paper/ipcai2027/geometry/d4d_depth.py --root /path/to/D4D/extracted --population d4d_meta/population.json \\
        --production /path/to/d4d_depth --da3-whole DIR --pi3x-whole DIR --mapanything-whole DIR \\
        --out d4d_sides.json [--jobs 12]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import surgical_core.d4d as d4d
from evalkit.tools import d4d_front_behind as fb
from stereomis_summary import write_json

# The models' depth, one file per side, by the answer each gives.
MODELS = ("production", "da3_whole", "pi3x_whole", "mapanything_whole")
ANSWERS = (*MODELS, "half_scan", "flat")
# The fewest pixels a side needs on the common set to be scored, as many as a region needs of the scan.
MIN_PIXELS = fb.MIN_GT_PIX


def absrel(pred: np.ndarray, ref: np.ndarray) -> float:
    """Return AbsRel of `pred` against `ref` after scaling `pred` by the ratio of the medians."""
    p = pred * (np.median(ref) / np.median(pred))
    return float(np.mean(np.abs(p - ref) / ref))


def measure_side(task: tuple) -> dict:
    """Return one side's row: its session, whether the camera moved, its pixels and each answer's AbsRel.

    Raises:
        ValueError: the side has no scan, pose or calibration, or a depth file was made for another frame.
        FileNotFoundError: a model's depth file is missing.
    """
    clip, side, moved, depth_dirs = task
    key, params = d4d.clip_key(clip), clip.endoscope_params
    scan, pose = clip.pointclouds.get(side), clip.poses.get(side)
    if not scan or pose is None or params is None:
        raise ValueError(f"{key} {side}: no scan, pose or calibration")

    # Render the scan, find the frame and its instruments, and read every answer's depth.
    points = d4d.read_scan(scan)
    gt = d4d.scan_depth(points, pose, params)
    index, _ = d4d.nearest_frame(d4d.frame_times(clip), d4d.name_time(scan))
    h, w = int(params["height"]), int(params["width"])
    valid = ~d4d.instrument_mask(d4d.frame_files(clip, "left_images_rect_masks", ".png")[index], (w, h))
    maps = {m: fb.answer_map(clip, index, "mono", points, pose, params,
                             z=fb.side_depth(depth_dirs[m], key, side, index)) for m in MODELS}
    maps["half_scan"] = fb.answer_map(clip, index, "gt_half", points, pose, params)
    maps["flat"] = np.ones((h, w), np.float32)
    # The stereo depth bounds the pixel set and is not scored, as in the measurement the paper reports.
    stereo = fb.answer_map(clip, index, "stereo", points, pose, params)

    # Score every answer on the pixels all of them, the stereo depth and the scan hold.
    common = valid & np.isfinite(gt) & (gt > 0) & np.isfinite(stereo) & (stereo > 0)
    for a in ANSWERS:
        common &= np.isfinite(maps[a]) & (maps[a] > 0)
    row = dict(key=key, side=side, session="/".join(key.split("/")[:2]), moved_camera=moved, n_pixels=int(common.sum()))
    if row["n_pixels"] >= MIN_PIXELS:
        row["absrel"] = {a: absrel(maps[a][common], gt[common]) for a in ANSWERS}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The dataset's root, which holds the specimen_* directories.")
    ap.add_argument("--population", required=True, help="The population, d4d_meta/population.json.")
    ap.add_argument("--production", required=True, help="The depth `pipeline.d4d_depth` wrote.")
    for m in MODELS[1:]:
        ap.add_argument(f"--{m.replace('_', '-')}", required=True, help=f"The {m} depth, one file per side.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    ap.add_argument("--jobs", type=int, default=1, help="Sides measured at once.")
    args = ap.parse_args()

    # List the population's sides in the loader's order, with whether the camera moved.
    population = json.loads(Path(args.population).read_text(encoding="utf-8"))
    moved = {(s["key"], s["side"]): s["moved_camera"] for s in population["sides"]}
    depth_dirs = {m: Path(getattr(args, m)) for m in MODELS}
    tasks = [(t[0], t[1], moved[(d4d.clip_key(t[0]), t[1])], depth_dirs)
             for t in fb.population_tasks(d4d.clips(args.root), population, "mono", "ours", "all", None, None)]
    print(f"{len(tasks)} sides", flush=True)

    # Measure each side and write the rows.
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(measure_side, tasks))
    print(f"{sum('absrel' in r for r in rows)} sides scored", flush=True)
    write_json(args.out, dict(answers=ANSWERS, min_pixels=MIN_PIXELS, sides=rows))


if __name__ == "__main__":
    main()
