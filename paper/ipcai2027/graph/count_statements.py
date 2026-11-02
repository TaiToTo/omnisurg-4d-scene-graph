"""Count the statements per ATLAS-120k clip that the reference masks and the estimated geometry give over a span.

The nodes are the classes of the first scored frame's reference masks. On each clip that reaches `--span-s` seconds
from that frame, the rules of `statements` count the first frame's 2D statements and the temporal statements over
the span, from the reference masks. They count the depth-order, approach and camera statements from the depth and
poses of the clip's bundle. A depth-order statement is a first-frame "in front of" whose front node is higher in
the image. A clip shorter than the span is left out and counted. The ratio of the reference masks' statements to
the first frame's 2D statements is a ratio of means, with an interval from `N_BOOT` resamples of the videos.

Usage:
    python paper/ipcai2027/graph/count_statements.py --clips CLIPS --data-root DIR --tracks-root DIR \\
        --tag track_rgb_rgb_gt_forward --out OUT.json [--span-s 5.4] [--workers 12]
"""
from __future__ import annotations

import argparse
import functools
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evalkit.classes import load_table
from evalkit.evaluate import read_population
from evalkit.inputs import DEPTH_FILE, read_clip
from evalkit.objects import gt_objects
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth
from evalkit.tools.paired_stats import video_of
from surgical_core.clip_time import frame_times

from statements import (KINDS_2D, approach_count, camera_statements, frame_statements, states_of,
                        temporal_statements, windows)
from video_bootstrap import interval, resampled_sums

DATASET = "atlas120k"
VIEW = "all"
COUNTS = ("2d", "temporal", "depth_order", "front_all", "approach", "camera")
# The supplement's statistics resample videos this many times for the ratio, from seed 0.
N_BOOT = 2000
SEED = 0
# Frame times are read from floats; a span reached to within this many seconds counts as reached.
TIME_TOL = 1e-6


def frames_in_span(frames: list[int], times: np.ndarray, span_s: float) -> list[int] | None:
    """Return the frames within `span_s` seconds of the first, or None when the last frame falls short of the span."""
    t0 = times[frames[0]]
    if times[frames[-1]] - t0 < span_s - TIME_TOL:
        return None
    return [i for i in frames if times[i] - t0 <= span_s + TIME_TOL]


def count_clip(clip: str, data_root: str, tracks_root: str, tag: str, span_s: float) -> dict:
    """Return one clip's statement counts over its first `span_s` seconds, or mark it short."""
    # The scored reference frames in time order, cut to the span.
    table = load_table(DATASET)
    inputs = read_clip(data_root, tracks_root, tag, clip, table)
    frames = [i for i in inputs.order if i in inputs.gt and not frame_is_excluded(inputs.gt[i], table)]
    times = frame_times(str(data_root), clip)
    frames = frames_in_span(frames, times, span_s)
    if frames is None:
        return {"clip": clip, "video": video_of(clip), "short": True}

    # The nodes' states on each frame, from the reference masks and the depth.
    scored = [scored_pixels(inputs.gt[i], table, VIEW, valid_depth(inputs.depth[i])) for i in frames]
    depth = [inputs.depth[i] for i in frames]
    names = list(gt_objects(scored[0].classes, scored[0].mask).ids)
    states = states_of([{c: (s.classes == c) & s.mask for c in names} for s in scored], depth)

    # The poses of the same frames, from the bundle `read_clip` read the depth from.
    with np.load(Path(data_root) / clip / DEPTH_FILE) as z:
        E = np.asarray(z["extrinsics"], dtype=np.float64)[frames]
    spans = windows(times[frames])
    scene = np.array([float(np.median(d[np.isfinite(d) & (d > 0)])) for d in depth])

    # The counts of each kind.
    first = frame_statements(states[0], names, 0)
    front = [s for s in first if s[0] == "front"]
    counts = {"2d": sum(s[0] in KINDS_2D for s in first),
              "temporal": len(temporal_statements(states, names)),
              "depth_order": sum(states[0][a][1] < states[0][b][1] for _, a, b, _ in front),
              "front_all": len(front),
              "approach": approach_count(states, names, spans),
              "camera": len(camera_statements(E, scene, spans))}
    return {"clip": clip, "video": video_of(clip), "short": False, "counts": counts}


def summarise(rows: list[dict], span_s: float) -> dict:
    """Return the clips kept, the mean count of each kind, and the reference masks' ratio with its interval."""
    kept = [r for r in rows if not r["short"]]
    by_video = defaultdict(lambda: np.zeros(2))
    for r in kept:
        by_video[r["video"]] += [r["counts"]["2d"] + r["counts"]["temporal"], r["counts"]["2d"]]
    per_video = np.array([by_video[v] for v in sorted(by_video)])
    total = per_video.sum(axis=0)
    draws = resampled_sums(per_video, N_BOOT, SEED)
    return {"span_s": span_s, "clips_listed": len(rows), "clips_kept": len(kept), "clips_short": len(rows) - len(kept),
            "videos": len(by_video),
            "mean_per_clip": {k: float(np.mean([r["counts"][k] for r in kept])) for k in COUNTS},
            "reference_ratio": float(total[0] / total[1]),
            "reference_ratio_ci95_video": interval(draws[:, 0] / draws[:, 1]),
            "n_boot": N_BOOT, "seed": SEED}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", required=True, help="the clip list, one name a line")
    ap.add_argument("--data-root", required=True, help="the clips with reference masks and the DA3 bundle")
    ap.add_argument("--tracks-root", required=True,
                    help="the tracks `read_clip` reads with the masks; the statements use none of their regions")
    ap.add_argument("--tag", required=True, help="the condition's directory under each clip of --tracks-root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--span-s", type=float, default=5.4)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    clips = read_population(a.clips)
    job = functools.partial(count_clip, data_root=a.data_root, tracks_root=a.tracks_root, tag=a.tag, span_s=a.span_s)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, clips, chunksize=1))
    out = summarise(rows, a.span_s)
    a.out.write_text(json.dumps(out, indent=1))
    m = out["mean_per_clip"]
    lo, hi = out["reference_ratio_ci95_video"]
    print(f"{out['clips_kept']} of {out['clips_listed']} clips reach {a.span_s:g} s ({out['videos']} videos)")
    print(f"per clip: 2D {m['2d']:.1f}, temporal +{m['temporal']:.1f}, depth order +{m['depth_order']:.1f} "
          f"({m['front_all']:.1f} with every 'in front of'), approach +{m['approach']:.1f}, camera +{m['camera']:.1f}")
    print(f"ratio {out['reference_ratio']:.3f} [{lo:.3f}, {hi:.3f}] ({N_BOOT} resamples of videos)")


if __name__ == "__main__":
    main()
