"""Measure reprojection consistency on each usable StereoMIS clip: next-frame IoU of the tracked regions.

For each frame pair (t, t + 1), each pixel of frame t + 1 with depth is back-projected with the model's depth of
that frame and its pose, projected into frame t with frame t's pose, and takes the region tracked there. The
warped map is compared with the regions tracked on frame t + 1, on the pixels where both hold a region: the mean
IoU over frame t's regions. A clip's value is the mean over its pairs. Each model is measured under its own poses,
a still camera, and a straight line from its first camera centre to its last at constant speed without rotation.
Writes one row per clip with the IoU of each model and pose set.

Usage:
    python paper/ipcai2027/geometry/reprojection.py --clips stereomis_meta/clips.json --clip-root /path/to/clips \\
        --tracks-root /path/to/tracks --out reprojection_rows.json [--jobs 12]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import surgical_core.stereomis as stereomis
from stereomis_summary import write_json
from surgical_core.geometry.camera import backproject_depth, cam_to_world
from surgical_core.geometry.project import project_world_to_frame

# The tracked regions: label-free SAM regions, 4 points per side, tracked forward from the first frame on RGB.
TRACK_DIR = "track_rgb_rgb_pps4_sam_forward"
BUNDLES = {"da3": "exports/mini_npz/results.npz", "pi3x": "exports/mini_npz/results__pi3x.npz",
           "mapanything": "exports/mini_npz/results__mapanything.npz"}
POSE_SETS = ("own", "still", "line")


def load_labels(track_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n, H, W) region labels and which frames have a label file; a frame without one is all 0.

    Raises:
        FileNotFoundError: no frame has a label file.
    """
    labels = [np.load(p) if p.is_file() else None for p in (track_dir / f"label_{i:04d}.npy" for i in range(n))]
    have = np.array([x is not None for x in labels])
    if not have.any():
        raise FileNotFoundError(f"no label under {track_dir}")
    shape = next(x for x in labels if x is not None).shape
    return np.stack([x if x is not None else np.zeros(shape, np.int32) for x in labels]), have


def line_extrinsics(E: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return world-to-camera poses moving straight from `E`'s first camera centre to its last, at constant speed,
    with the first camera's rotation."""
    c = np.einsum("nji,nj->ni", E[:, :3, :3], -E[:, :3, 3])
    f = (times - times[0]) / (times[-1] - times[0])
    out = np.empty_like(E)
    out[:, :3, :3] = E[0, :3, :3]
    out[:, :3, 3] = -np.einsum("ij,nj->ni", E[0, :3, :3], c[0] + f[:, None] * (c[-1] - c[0]))
    return out


def warp_labels(lab_t: np.ndarray, depth_g: np.ndarray, K_g, E_g, K_t, E_t) -> np.ndarray:
    """Return, for each pixel of frame t + 1, the region of frame t it comes from; 0 where none."""
    H, W = depth_g.shape
    m = np.isfinite(depth_g) & (depth_g > 0)
    out = np.zeros((H, W), dtype=np.int32)
    if not m.any():
        return out
    Pw = cam_to_world(backproject_depth(depth_g, K_g)[m.reshape(-1)], E_g[:, :3], E_g[:, 3])
    u, v, Z = project_world_to_frame(Pw, K_t, E_t)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    ok = np.isfinite(u) & np.isfinite(v) & (Z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    vals = np.zeros(len(u), dtype=np.int32)
    vals[ok] = lab_t[vi[ok], ui[ok]]
    out[m] = vals
    return out


def mean_iou(warped: np.ndarray, target: np.ndarray, ids: np.ndarray) -> float | None:
    """Return the mean IoU over `ids` on the pixels where both maps hold a region, or None when there is none."""
    dom = (warped > 0) & (target > 0)
    ious = []
    for r in ids:
        a, b = (warped == r) & dom, (target == r) & dom
        union = (a | b).sum()
        if union > 0:
            ious.append((a & b).sum() / union)
    return float(np.mean(ious)) if dom.any() and ious else None


def measure_clip(job: tuple) -> dict:
    """Return a clip's mean next-frame IoU for each model and pose set.

    Raises:
        ValueError: the manifest lists other frames than the clip list, or a bundle another number of frames.
    """
    clip, clip_root, tracks_root = job
    name, n = clip["name"], len(clip["frames"])
    manifest = json.loads((clip_root / name / "frame_manifest.json").read_text())
    if [f["native_frame"] for f in manifest["frames"]] != list(clip["frames"]):
        raise ValueError(f"{name}: the manifest lists other frames than the clip list")
    times = np.array([f["time_sec"] for f in manifest["frames"]], dtype=float)
    labels_raw, have = load_labels(tracks_root / name / TRACK_DIR, n)
    row = dict(clip=name, seq=clip["seq"], iou={})
    for model, bundle in BUNDLES.items():
        with np.load(clip_root / name / bundle) as z:
            depth, K, E = z["depth"], z["intrinsics"], z["extrinsics"]
        if len(depth) != n:
            raise ValueError(f"{name} {model}: {len(depth)} frames, the clip has {n}")
        # The labels are read at the depth's size, the grid the warp runs on.
        labels = np.stack([cv2.resize(x.astype(np.int32), depth.shape[:0:-1], interpolation=cv2.INTER_NEAREST)
                           for x in labels_raw]) if labels_raw.shape[1:] != depth.shape[1:] else labels_raw
        poses = {"own": E, "still": None, "line": line_extrinsics(E, times)}
        for pose_set, Ec in poses.items():
            values = []
            for t in range(n - 1):
                if not (have[t] and have[t + 1]):
                    continue
                ids = np.unique(labels[t])
                ids = ids[ids > 0]
                if Ec is None:
                    # A still camera pastes frame t's regions where frame t + 1 has depth.
                    warped = labels[t].copy()
                    warped[~(np.isfinite(depth[t + 1]) & (depth[t + 1] > 0))] = 0
                else:
                    warped = warp_labels(labels[t], depth[t + 1], K[t + 1], Ec[t + 1], K[t], Ec[t])
                v = mean_iou(warped, labels[t + 1], ids)
                if v is not None:
                    values.append(v)
            row["iou"][f"{model}_{pose_set}"] = float(np.mean(values)) if values else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    ap.add_argument("--clip-root", required=True, help="The clips holding the three models' bundles.")
    ap.add_argument("--tracks-root", required=True, help=f"The tracks, <clip>/{TRACK_DIR}/label_NNNN.npy.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    ap.add_argument("--jobs", type=int, default=1, help="Clips measured at once.")
    args = ap.parse_args()
    clips = stereomis.usable_clips(stereomis.read_clip_list(Path(args.clips)))
    jobs = [(c, Path(args.clip_root), Path(args.tracks_root)) for c in clips]
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(measure_clip, jobs))
    for r in rows:
        print(f"  {r['clip']:17s} " + " ".join(f"{k} {v:.3f}" for k, v in r["iou"].items() if v is not None))
    write_json(args.out, dict(gap_frames=1, track_dir=TRACK_DIR, clips=rows))


if __name__ == "__main__":
    main()
