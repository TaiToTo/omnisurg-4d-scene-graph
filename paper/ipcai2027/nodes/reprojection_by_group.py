"""Compare by class group the IoU 2.7 s ahead of Pi3X's reprojection, a fixed mask and tracking on ATLAS-120k.

For each pair of frames (t, t + 5) of a clip, `warp_labels` carries the reference labels of frame t to
frame t + 5 with Pi3X's depth, intrinsics and poses. A class's IoU is taken with its reference mask on
t + 5, over the pixels labelled on both frames. The fixed mask pastes frame t's labels onto t + 5. A
(clip, class) row holds the mean over the clip's pairs, and needs `MIN_PAIRS` pairs. Its tracked value
is the tracking IoU on the fifth later frame that holds the class (`track_samples`, all-class view).
Rows are pooled by class group. Each mean has a 95 % interval over videos (`paired_stats.boot_ci`), and
`paired_stats.mark_of` marks reprojection minus fixed.

Usage:
    python paper/ipcai2027/nodes/reprojection_by_group.py --data-root DIR --tracks-root DIR \\
        --tag track_rgb_rgb_gt_forward --clips LIST --out reprojection_by_group.json [--workers 12]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evalkit.classes import ClassTable, ClassType, load_table
from evalkit.fill import fill_from_nearest
from evalkit.inputs import ANCHORED_DATASETS, GT_FLAG, MANIFEST, MASK_DIR, MASK_SUFFIX, gt_frames
from evalkit.scored import frame_is_excluded
from evalkit.tools.paired_stats import boot_ci, mark_of, video_of
from surgical_core.clip_time import frame_times
from surgical_core.geometry.camera import backproject_depth, cam_to_world
from surgical_core.geometry.project import project_world_to_frame
from track_samples import clip_tracks

DATASET = "atlas120k"
PI3X_BUNDLE = Path("exports", "mini_npz", "results__pi3x.npz")
GAP = 5
MIN_PAIRS = 3
# The class group of each class type that the all-class view scores.
GROUP = {ClassType.EXPERT: "vessels and ducts", ClassType.TISSUE: "other tissue", ClassType.BACKDROP: "other tissue",
         ClassType.APPEARANCE: "other tissue", ClassType.TOOL: "instruments"}
MEASURES = ("tracked", "reprojection", "fixed")


def reference_labels(clip_dir: Path, table: ClassTable, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return every frame's reference class ids at `shape`, 0 for background, and which frames have a reference mask.

    A frame that the excluded marker takes out of the score has no reference mask here either.
    """
    frames = json.loads((clip_dir / MANIFEST).read_text(encoding="utf-8"))["frames"]
    suffix = MASK_SUFFIX[DATASET]
    with_mask = {i for i in range(len(frames)) if (clip_dir / MASK_DIR / f"{i:06d}{suffix}").is_file()}
    labels = np.zeros((len(frames), *shape), dtype=np.int32)
    have = np.zeros(len(frames), dtype=bool)
    for i in gt_frames(frames, GT_FLAG[DATASET], with_mask, clip_dir.name, anchored=DATASET in ANCHORED_DATASETS):
        with Image.open(clip_dir / MASK_DIR / f"{i:06d}{suffix}") as image:
            ids, _ = fill_from_nearest(table.mask_ids(image), table.filled_mask_ids)
        if frame_is_excluded(ids, table):
            continue
        ids[np.isin(ids, sorted(table.background_mask_ids))] = 0
        labels[i] = cv2.resize(ids, shape[::-1], interpolation=cv2.INTER_NEAREST) if ids.shape != shape else ids
        have[i] = True
    return labels, have


def warp_labels(labels_t: np.ndarray, depth_g: np.ndarray, K_g: np.ndarray, E_g: np.ndarray,
                K_t: np.ndarray, E_t: np.ndarray) -> np.ndarray:
    """Return, for each pixel of frame t + g, the label of frame t that it comes from.

    The warp is backward, so it leaves no holes. Each pixel of frame t + g with depth is back-projected with
    that frame's depth, intrinsics and world-to-camera pose `E_g`, projected into frame t with `K_t` and
    `E_t`, and takes the label at the nearest pixel there. A pixel is 0 when it has no depth, or when it
    lands outside frame t or behind its camera.
    """
    h, w = depth_g.shape
    m = np.isfinite(depth_g) & (depth_g > 0)
    out = np.zeros((h, w), dtype=np.int32)
    if not m.any():
        return out
    points = cam_to_world(backproject_depth(depth_g, K_g)[m.reshape(-1)], E_g[:, :3], E_g[:, 3])
    u, v, z = project_world_to_frame(points, K_t, E_t)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    ok = np.isfinite(u) & np.isfinite(v) & (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    values = np.zeros(len(u), dtype=np.int32)
    values[ok] = labels_t[vi[ok], ui[ok]]
    out[m] = values
    return out


def class_ious(carried: np.ndarray, target: np.ndarray, classes: np.ndarray) -> dict[int, float]:
    """Return each class's IoU over the pixels labelled on both maps; a class absent from both there has none."""
    both = (carried > 0) & (target > 0)
    out = {}
    for c in classes:
        a, b = (carried == c) & both, (target == c) & both
        union = int((a | b).sum())
        if union:
            out[int(c)] = int((a & b).sum()) / union
    return out


def pair_ious(labels: np.ndarray, have: np.ndarray, depth: np.ndarray, K: np.ndarray,
              E: np.ndarray) -> dict[str, dict[int, list[float]]]:
    """Return, for the reprojection and the fixed mask, each class's IoU on every pair (t, t + `GAP`) of the clip."""
    out: dict[str, dict[int, list[float]]] = {"reprojection": {}, "fixed": {}}
    for t in range(len(labels) - GAP):
        g = t + GAP
        if not (have[t] and have[g]):
            continue
        classes = np.unique(labels[t])
        classes = classes[classes > 0]
        fixed = labels[t].copy()
        fixed[~(np.isfinite(depth[g]) & (depth[g] > 0))] = 0
        carried = {"reprojection": warp_labels(labels[t], depth[g], K[g], E[g], K[t], E[t]), "fixed": fixed}
        for kind, labels_g in carried.items():
            for c, iou in class_ious(labels_g, labels[g], classes).items():
                out[kind].setdefault(c, []).append(iou)
    return out


def clip_rows(per_pair: dict[str, dict[int, list[float]]]) -> dict[int, dict]:
    """Return a row per class with `MIN_PAIRS` reprojected pairs: the mean IoU of the reprojection and the fixed mask.

    Raises:
        ValueError: A row's class has no pair under the fixed mask, so the difference would be undefined.
    """
    rows = {}
    for c, ious in per_pair["reprojection"].items():
        if len(ious) < MIN_PAIRS:
            continue
        if c not in per_pair["fixed"]:
            raise ValueError(f"class {c} has {len(ious)} reprojected pairs and none under the fixed mask")
        rows[c] = {"reprojection": float(np.mean(ious)), "fixed": float(np.mean(per_pair["fixed"][c]))}
    return rows


def measure(clip: str, data_root: Path, tracks_root: Path, tag: str) -> dict:
    """Return one clip's rows, each with its class, group and tracked value, and the seconds between paired frames."""
    table = load_table(DATASET)
    clip_dir = data_root / clip

    # The reprojection and the fixed mask, per class, from Pi3X's bundle and the reference masks.
    with np.load(clip_dir / PI3X_BUNDLE) as z:
        depth, K, E = z["depth"], z["intrinsics"], z["extrinsics"]
    labels, have = reference_labels(clip_dir, table, depth.shape[1:])
    rows = clip_rows(pair_ious(labels, have, depth, K, E))

    # The tracked value: the tracking IoU on the fifth later frame that holds the class.
    tracks = clip_tracks(data_root, tracks_root, tag, clip, table, ("all",))["views"]["all"]
    tracked = {t["class"]: t["tracked"][GAP - 1] for t in tracks if t["picked"] and len(t["tracked"]) >= GAP}

    # The time between the frames of each pair, for the gap in seconds.
    times = frame_times(str(data_root), clip)
    return {"clip": clip, "video": video_of(clip), "pair_seconds": (times[GAP:] - times[:-GAP]).tolist(),
            "step_seconds": float(np.mean(times[1:] - times[:-1])),
            "rows": [{"class": c, "group": GROUP[table.type_of(c)], "tracked": tracked.get(c), **r}
                     for c, r in sorted(rows.items())]}


def level(rows: list[dict], key: str) -> dict:
    """Return the mean of one measure over the rows that have it, with its 95 % interval over videos."""
    have = [r for r in rows if r[key] is not None]
    values = np.array([r[key] for r in have])
    return {"mean": float(values.mean()), "ci95": list(boot_ci(values, np.array([r["video"] for r in have]))),
            "rows": len(have)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path, help="The extracted clips, with Pi3X's bundles.")
    ap.add_argument("--tracks-root", required=True, type=Path, help="The tracked labels, one directory per clip.")
    ap.add_argument("--tag", required=True, help="The GT-seeded condition's directory under each clip.")
    ap.add_argument("--clips", required=True, type=Path, help="The clip names, one per line.")
    ap.add_argument("--out", required=True, type=Path, help="The JSON to write.")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    # Every clip's rows, measured in parallel.
    names = a.clips.read_text(encoding="utf-8").split()
    job = partial(measure, data_root=a.data_root, tracks_root=a.tracks_root, tag=a.tag)
    with ProcessPoolExecutor(a.workers) as ex:
        clips = list(ex.map(job, names, chunksize=1))

    # Each group's rows pooled: the three levels, and reprojection minus fixed paired by row.
    groups = {}
    for name in dict.fromkeys(GROUP.values()):
        rows = [{"clip": c["clip"], "video": c["video"], **r} for c in clips for r in c["rows"] if r["group"] == name]
        videos = np.array([r["video"] for r in rows])
        diff = np.array([r["reprojection"] - r["fixed"] for r in rows])
        ci = list(boot_ci(diff, videos))
        n_videos = len(set(videos))
        groups[name] = {"rows": len(rows), "clips": len({r["clip"] for r in rows}), "videos": n_videos,
                        **{m: level(rows, m) for m in MEASURES},
                        "reprojection_minus_fixed": {"mean": float(diff.mean()), "ci95": ci,
                                                     "mark": mark_of(ci, +1, n_videos=n_videos)}}
    pairs = [s for c in clips for s in c["pair_seconds"]]
    out = {"dataset": DATASET, "tag": a.tag, "n_clips": len(clips), "gap_frames": GAP, "min_pairs": MIN_PAIRS,
           "pair_seconds_mean": float(np.mean(pairs)),
           "step_seconds_median": float(np.median([c["step_seconds"] for c in clips])), "groups": groups}
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{DATASET}: {len(clips)} clips; frames {GAP} apart, {out['pair_seconds_mean']:.3f} s on average "
          f"(median step {out['step_seconds_median']:.3f} s)")
    print(f"{'group':18s} {'rows/clips/videos':>17s}  {'tracked':>22s}  {'reprojection':>22s}  {'fixed':>22s}  "
          "reprojection - fixed")
    for name, g in groups.items():
        cells = "  ".join(f"{g[m]['mean']:.3f} [{g[m]['ci95'][0]:.3f}, {g[m]['ci95'][1]:.3f}]" for m in MEASURES)
        d = g["reprojection_minus_fixed"]
        print(f"{name:18s} {g['rows']:>5d}/{g['clips']}/{g['videos']:<7d}  {cells}  "
              f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] {d['mark']}")


if __name__ == "__main__":
    main()
