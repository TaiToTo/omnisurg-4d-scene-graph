"""Compare SAM 3's tracking IoU from first-frame reference masks with the reference mask kept fixed, by class type.

A track's samples are read by `track_samples`; the tracks counted are the picked ones with a later frame.
A clip's value is the mean of its tracks' samples, and a level is the mean over clips. The difference,
tracked minus fixed, pairs clips; its 95 % interval resamples videos (`paired_stats.boot_ci`), and
`paired_stats.mark_of` gives its mark. The rows are the geometric view and the all-class view by class
type. In the all-class view, the script also counts the first frames' class regions, the ones the tracker
picked, and the distinct classes among them against the classes the table labels.

Usage:
    python paper/ipcai2027/nodes/tracking_iou.py --dataset atlas120k --data-root DIR --tracks-root DIR \\
        --tag track_rgb_rgb_gt_forward --clips LIST --out tracking_iou_atlas120k.json [--workers 12]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from evalkit.classes import load_table
from evalkit.tools.paired_stats import MIN_VIDEOS_FOR_MARK, boot_ci, mark_of
from track_samples import clip_tracks

VIEWS = ("all", "geometric")
# The rows: a view, the class types it pools (None for every type the view keeps), and the name printed.
ROWS = (
    ("geometric", None, "tissue"),
    ("all", None, "all types"),
    ("all", "tool", "instruments"),
    ("all", "expert", "vessels and ducts"),
    ("all", "tissue", "tissue"),
    ("all", "backdrop", "backdrop"),
    ("all", "appearance", "appearance-defined"),
)


def measure(clip: str, dataset: str, data_root: str, tracks_root: str, tag: str) -> dict:
    """Return one clip's tracks in `VIEWS`; the table is loaded in the worker that measures the clip."""
    return clip_tracks(data_root, tracks_root, tag, clip, load_table(dataset), VIEWS)


def interval(values: dict[str, float], video: dict[str, str]) -> list[float] | None:
    """Return the video-level 95 % interval of the mean of clip values, None on fewer than two videos."""
    clips = sorted(values)
    groups = np.array([video[c] for c in clips])
    if len(set(groups)) < 2:
        return None
    return list(boot_ci(np.array([values[c] for c in clips]), groups))


def row(clips: list[dict], view: str, kind: str | None) -> dict:
    """Return one row: its tracks, clips and videos, both levels, and tracked minus fixed with its mark."""
    # Each clip's mean over the samples of its picked tracks of the row's type.
    tracked, fixed, n_tracks = {}, {}, 0
    for c in clips:
        picked = [t for t in c["views"][view] if t["picked"] and t["tracked"] and kind in (None, t["type"])]
        if picked:
            tracked[c["clip"]] = float(np.mean([v for t in picked for v in t["tracked"]]))
            fixed[c["clip"]] = float(np.mean([v for t in picked for v in t["fixed"]]))
            n_tracks += len(picked)

    # The levels, and the difference paired over clips with its interval over videos.
    video = {c["clip"]: c["video"] for c in clips}
    names = sorted(tracked)
    n_videos = len({video[c] for c in names})
    diff = {c: tracked[c] - fixed[c] for c in names}
    ci = interval(diff, video)
    return {"tracks": n_tracks, "clips": len(names), "videos": n_videos,
            "tracked": float(np.mean(list(tracked.values()))), "tracked_ci95": interval(tracked, video),
            "fixed": float(np.mean(list(fixed.values()))), "fixed_ci95": interval(fixed, video),
            "tracked_minus_fixed": float(np.mean(list(diff.values()))), "tracked_minus_fixed_ci95": ci,
            "mark": mark_of(ci, +1, n_videos=n_videos), "sign_only": n_videos < MIN_VIDEOS_FOR_MARK}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=("atlas120k", "cholecseg8k"))
    ap.add_argument("--data-root", required=True, help="The extracted clips, one directory each.")
    ap.add_argument("--tracks-root", required=True, help="The tracked labels, one directory per clip.")
    ap.add_argument("--tag", required=True, help="The GT-seeded condition's directory under each clip.")
    ap.add_argument("--clips", required=True, type=Path, help="The clip names, one per line.")
    ap.add_argument("--out", required=True, type=Path, help="The JSON to write.")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    # Every clip's tracks, measured in parallel.
    names = a.clips.read_text(encoding="utf-8").split()
    job = partial(measure, dataset=a.dataset, data_root=a.data_root, tracks_root=a.tracks_root, tag=a.tag)
    with ProcessPoolExecutor(a.workers) as ex:
        clips = list(ex.map(job, names, chunksize=1))

    # The first frames' class regions, the picked ones and the distinct classes, then each row.
    first = [t for c in clips for t in c["views"]["all"]]
    out = {"dataset": a.dataset, "tag": a.tag, "n_clips": len(clips), "n_videos": len({c["video"] for c in clips}),
           "first_frame_regions": len(first), "first_frame_regions_picked": sum(t["picked"] for t in first),
           "first_frame_classes": len({t["class"] for t in first}),
           "labelled_classes": len(load_table(a.dataset).ids_in_view("all")),
           "rows": {f"{view}/{name}": row(clips, view, kind) for view, kind, name in ROWS}}
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{a.dataset}: {out['n_clips']} clips, {out['n_videos']} videos; first-frame class regions picked "
          f"{out['first_frame_regions_picked']} of {out['first_frame_regions']}; classes on a first frame "
          f"{out['first_frame_classes']} of the {out['labelled_classes']} labelled")
    print(f"{'row':32s} {'tracks':>6s} {'clips/videos':>12s} {'tracked':>7s} {'fixed':>6s}  tracked - fixed")
    for key, r in out["rows"].items():
        ci = r["tracked_minus_fixed_ci95"]
        span = "" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        note = " (sign)" if r["sign_only"] else ""
        print(f"{key:32s} {r['tracks']:6d} {r['clips']:>5d}/{r['videos']:<6d} {r['tracked']:7.3f} {r['fixed']:6.3f}  "
              f"{r['tracked_minus_fixed']:+.3f}{span} {r['mark']}{note}")


if __name__ == "__main__":
    main()
