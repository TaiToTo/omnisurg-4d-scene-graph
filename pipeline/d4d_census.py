"""Measure the inputs of every D4D side, the census that `pipeline.select_d4d_population` selects from.

`d4d_meta/README.md` defines a side and the fields. For each side with a scan, a pose and the endoscope's
calibration, the step projects the scan into the endoscope's view and leaves out the instruments of the
frame nearest to the scan. It reads `moved_camera` from the dataset's `info/session_details.txt`. It
refuses a session that the details do not list once as `Y` or `N`, a clip without frames or whose
instrument masks are not named by its frames' times, and a side it cannot measure.

Usage:
    python -m pipeline.d4d_census --root /path/to/D4D/extracted --out /path/to/census.json [--jobs 8]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import surgical_core.d4d as d4d

# The side of a block of the view, in pixels. A block counts when more than half of its pixels are free of
# instruments, and is covered when at least `BLOCK_MIN_HITS` of the scan's points fall in it.
BLOCK = 16
BLOCK_MIN_HITS = 5


def read_moved_camera(root: str) -> dict[str, bool]:
    """Return whether the endoscope moves, per session `<specimen>/<session>`, from `info/session_details.txt`.

    Raises:
        ValueError: a line is not `<specimen>/<session>,Y` or `,N`, or names a session twice.
    """
    lines = (Path(root) / "info" / "session_details.txt").read_text(encoding="utf-8").splitlines()
    moved: dict[str, bool] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        session, _, flag = line.rpartition(",")
        if flag.strip() not in ("Y", "N") or len(session.split("/")) != 2 or session in moved:
            raise ValueError(f"session_details.txt: {line!r} is not a new `<specimen>/<session>,Y` or `,N`")
        moved[session] = flag.strip() == "Y"
    return moved


def block_cover(hits: np.ndarray, free: np.ndarray) -> float:
    """Return the share of the blocks mostly free of instruments in which the scan's points cover the block.

    Args:
        hits: (H, W) the scan's points in each pixel.
        free: (H, W) true on a pixel free of instruments.
    """
    hb, wb = hits.shape[0] // BLOCK, hits.shape[1] // BLOCK
    per_block = hits[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3))
    free_block = free[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3))
    counted = free_block > (BLOCK * BLOCK) // 2
    return float((per_block[counted] >= BLOCK_MIN_HITS).mean()) if counted.any() else 0.0


def measure_side(clip, side: str, times: np.ndarray, masks: list[Path]) -> dict:
    """Return the census's fields of one side.

    A side without a scan, a pose or the endoscope's calibration is not present.
    """
    scan, pose, params = clip.pointclouds.get(side), clip.poses.get(side), clip.endoscope_params
    if not scan or pose is None or params is None:
        return {"present": False}
    hits = d4d.pixel_hits(d4d.read_scan(scan), pose, params)
    index, gap = d4d.nearest_frame(times, d4d.name_time(scan))
    free = ~d4d.instrument_mask(masks[index], (int(params["width"]), int(params["height"])))
    return {"present": True, "frame_minus_gt_s": gap, "gt_blk_frac": block_cover(hits, free)}


def measure_clip(task: tuple) -> dict:
    """Return one clip's census record, from `(clip, moved_camera)`."""
    clip, moved = task
    times = d4d.frame_times(clip)
    masks = d4d.frame_files(clip, "left_images_rect_masks", ".png")
    return {"key": d4d.clip_key(clip), "moved_camera": moved} | {
        side: measure_side(clip, side, times, masks) for side in d4d.SIDES}


def tasks(clips: list, moved: dict[str, bool]) -> list[tuple]:
    """Pair each clip with its session's `moved_camera`.

    Raises:
        ValueError: a clip's session is not in `moved`.
    """
    sessions = [d4d.clip_key(c).rsplit("/", 1)[0] for c in clips]
    missing = sorted(set(sessions) - set(moved))
    if missing:
        raise ValueError(f"session_details.txt does not list {len(missing)} sessions, the first being {missing[0]}")
    return [(c, moved[s]) for c, s in zip(clips, sessions)]


def main(argv: list[str] | None = None) -> None:
    """Measure every side of the dataset and write the census, in the loader's order, as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="the dataset's root, which holds the specimen_* directories")
    ap.add_argument("--out", required=True, help="the JSON file to write")
    ap.add_argument("--jobs", type=int, default=8, help="the clips measured at once")
    args = ap.parse_args(argv)

    work = tasks(d4d.clips(args.root), read_moved_camera(args.root))
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        census = list(ex.map(measure_clip, work))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(census, indent=1), encoding="utf-8")
    print(f"{len(census)} clips, {sum(c[s]['present'] for c in census for s in d4d.SIDES)} sides with a scan")


if __name__ == "__main__":
    main()
