"""Find when the tissue moves in each D4D clip, from the clip's stereo depth.

Every `SAMPLE_S` seconds, the step compares each pixel's depth with its depth `LAG_S` seconds later, leaving
out the pixels near an instrument in either frame. The 90th percentile of the change, `p90_mm`, is the
tissue's motion at `t`, in seconds after the clip's first frame. A clip's quiet level is its 10th percentile
of `p90_mm`. One threshold serves every clip: `THRESHOLD_FACTOR` times the median quiet level of the clips
whose camera stays still. A clip's `intervals` are the stretches where `p90_mm` exceeds the threshold. The
step refuses a clip whose depth maps or masks are not named by its frames' times, and a clip it cannot measure.

Usage:
    python -m pipeline.d4d_tissue_motion --root /path/to/D4D/extracted --out /path/to/tissue_motion.json \\
        [--jobs 8]
"""

import argparse
import functools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import surgical_core.d4d as d4d

# The time between the two depth maps compared, and between two samples, in seconds. Over half a second the
# tissue's motion builds up, and the stereo noise of about 0.2 mm a frame does not.
LAG_S = 0.5
SAMPLE_S = 0.25
# A sample needs this many pixels with depth in both frames and no instrument near them.
MIN_PIXELS = 1000
# A clip needs this many samples.
MIN_SAMPLES = 3
QUIET_QUANTILE = 0.10
THRESHOLD_FACTOR = 3.0
MIN_THRESHOLD_MM = 1.2
# Two stretches closer than `MERGE_GAP_S` become one; a stretch shorter than `MIN_INTERVAL_S` is dropped.
MERGE_GAP_S = 0.6
MIN_INTERVAL_S = 0.3
# The camera stays still when its per-frame positions span no more than this, in metres.
STILL_SPAN_M = 0.005
# The dataset folder's `detect_tissue_motion.py` passed `cv2.INTER_NEAREST` as `cv2.resize`'s `dst`, so it
# resized a mask of another size bilinearly.
MASK_INTERPOLATION = cv2.INTER_LINEAR


def camera_moves(clip) -> bool:
    """Return whether the clip's per-frame camera positions span more than `STILL_SPAN_M`.

    Raises:
        ValueError: the clip has no `pose_bounds.npy`, or one that does not hold 17 numbers a frame.
    """
    bounds = clip.load_pose_bounds()
    if bounds is None or bounds.ndim != 2 or bounds.shape[1] != 17:
        raise ValueError(f"{d4d.clip_key(clip)}: no per-frame poses of 17 numbers in pose_bounds.npy")
    positions = bounds[:, :15].reshape(-1, 3, 5)[:, :, 3]
    return bool(np.linalg.norm(positions.max(0) - positions.min(0)) > STILL_SPAN_M)


def series(depth: list[Path], masks: list[Path]) -> tuple[list[float], list[float]]:
    """Return the times `t` and the motion `p90_mm` of one clip, from its depth maps and instrument masks.

    Raises:
        ValueError: the depth maps are not evenly timed forward, or fewer than `MIN_SAMPLES` samples have
            `MIN_PIXELS` pixels to compare.
    """
    # Turn the lag and the sampling interval into frames.
    times = np.asarray([d4d.name_time(p) for p in depth])
    dt = np.median(np.diff(times)) if len(times) > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{depth[0].parent}: no frame interval")
    lag = max(1, int(round(LAG_S / dt)))
    step = max(1, int(round(SAMPLE_S / dt)))
    if len(depth) <= lag + 1:
        lag = max(1, len(depth) // 3)

    @functools.lru_cache(maxsize=64)
    def frame(i: int) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(depth[i])
        return d, d4d.instrument_mask(masks[i], (d.shape[1], d.shape[0]), MASK_INTERPOLATION)

    # Compare each sampled frame with the frame `lag` later.
    t, p90 = [], []
    for i in range(0, len(depth) - lag, step):
        (d0, m0), (d1, m1) = frame(i), frame(i + lag)
        valid = ~m0 & ~m1 & np.isfinite(d0) & np.isfinite(d1)
        if valid.sum() < MIN_PIXELS:
            continue
        t.append(float(times[i] - times[0]))
        p90.append(float(np.percentile(np.abs(d1 - d0)[valid] * 1000.0, 90)))
    if len(t) < MIN_SAMPLES:
        raise ValueError(f"{depth[0].parent}: {len(t)} samples, fewer than {MIN_SAMPLES}")
    return t, p90


def intervals(t: list[float], p90: list[float], threshold: float) -> list[list[float]]:
    """Return the stretches `[start, end]`, in seconds, where `p90` exceeds `threshold`.

    A stretch ends at the sample after its last one above the threshold. Stretches closer than `MERGE_GAP_S`
    are merged, and those shorter than `MIN_INTERVAL_S` dropped.
    """
    # Find the runs of samples above the threshold.
    hot = np.asarray(p90) > threshold
    runs, i = [], 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hot) and hot[j + 1]:
            j += 1
        runs.append([float(t[i]), float(t[min(j + 1, len(t) - 1)])])
        i = j + 1
    # Merge the runs across short gaps, and drop the short ones.
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [iv for iv in merged if iv[1] - iv[0] >= MIN_INTERVAL_S]


def measure(task: tuple[str, bool, list[Path], list[Path]]) -> dict:
    """Return one clip's record without its threshold and intervals, from `(key, camera_moves, depth, masks)`."""
    key, moves, depth, masks = task
    t, p90 = series(depth, masks)
    return {"clip": key, "camera_moves": moves, "t": t, "p90_mm": p90,
            "quiet_level_mm": float(np.quantile(p90, QUIET_QUANTILE))}


def add_intervals(records: list[dict]) -> float:
    """Compute the threshold from every clip's quiet level, add it and each clip's intervals, and return it."""
    still = [r["quiet_level_mm"] for r in records if not r["camera_moves"]]
    threshold = max(THRESHOLD_FACTOR * float(np.median(still or [r["quiet_level_mm"] for r in records])),
                    MIN_THRESHOLD_MM)
    for r in records:
        r["threshold_mm"] = threshold
        r["intervals"] = intervals(r["t"], r["p90_mm"], threshold)
    return threshold


def main(argv: list[str] | None = None) -> None:
    """Measure every clip of the dataset and write the records, in the loader's order, as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="the dataset's root, which holds the specimen_* directories")
    ap.add_argument("--out", required=True, help="the JSON file to write")
    ap.add_argument("--jobs", type=int, default=8, help="the clips measured at once")
    args = ap.parse_args(argv)

    tasks = [(d4d.clip_key(c), camera_moves(c), d4d.frame_files(c, "stereo_depth", ".npy"),
              d4d.frame_files(c, "left_images_rect_masks", ".png")) for c in d4d.clips(args.root)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        records = list(ex.map(measure, tasks))
    threshold = add_intervals(records)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records), encoding="utf-8")
    print(f"{len(records)} clips, threshold {threshold:.2f} mm, "
          f"{sum(bool(r['intervals']) for r in records)} with a stretch of motion")


if __name__ == "__main__":
    main()
