"""Find the StereoMIS clips whose kinematics disagree with their own stereo depth, and print them.

The rule reads the reference alone, never a model's output. For each usable clip, the stereo depth of frame
t + 1 is carried into the camera of frame t twice: with the kinematic poses, and with the camera held still.
A pixel agrees when the carried depth is within 2 % of frame t's own depth there. A clip is excluded when the
kinematics' share of agreeing pixels is more than `MARGIN` below the still camera's. The kinematics then
explain the images worse than no motion does. The other StereoMIS commands take the printed clips as
`--exclude`.

Usage:
    python paper/ipcai2027/geometry/excluded_clips.py --root /path/to/StereoMIS --clips stereomis_meta/clips.json \\
        --stereo-depth /path/to/stereo_depth --out excluded.json [--jobs 12]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import surgical_core.stereomis as stereomis

# The frames between the two depth maps compared, one clip frame (about 0.2 s).
GAP = 1
# The pairs compared start every `PAIR_STEP` frames, and every `PIXEL_STEP`-th pixel of a row and a column is
# carried, as in the measurement the paper reports.
PAIR_STEP = 5
PIXEL_STEP = 2
# A pair is compared only when this many carried pixels land on a depth.
MIN_PIXELS = 200
# The relative depth difference within which a carried pixel agrees.
AGREE = 0.02
# How far the kinematics' share may fall below the still camera's before the clip is excluded.
MARGIN = 0.05


def agreeing_share(depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> float:
    """Return the share of carried pixels that agree, averaged over the frame pairs compared.

    Args:
        depth: (N, H, W) depth in mm, 0 where unknown.
        K: (3, 3) intrinsics of the depth's pixels.
        T: (N, 4, 4) camera-to-world poses in mm.

    Raises:
        ValueError: no pair has `MIN_PIXELS` carried pixels on a depth, so the clip cannot be checked.
    """
    n, (H, W) = len(depth), depth.shape[1:]
    v, u = np.mgrid[0:H:PIXEL_STEP, 0:W:PIXEL_STEP]
    shares = []
    for t in range(0, n - GAP, PAIR_STEP):
        # Back-project frame t + GAP's depth and move it into the camera of frame t.
        z = depth[t + GAP][v, u]
        m = z > 0
        P = np.stack([(u[m] - K[0, 2]) * z[m] / K[0, 0], (v[m] - K[1, 2]) * z[m] / K[1, 1], z[m]], axis=-1)
        rel = np.linalg.inv(T[t]) @ T[t + GAP]
        Q = P @ rel[:3, :3].T + rel[:3, 3]

        # Project it, and read frame t's depth where it lands.
        ok = Q[:, 2] > 1e-6
        qz = np.where(ok, Q[:, 2], 1)
        ut = np.rint(K[0, 0] * Q[:, 0] / qz + K[0, 2]).astype(int)
        vt = np.rint(K[1, 1] * Q[:, 1] / qz + K[1, 2]).astype(int)
        ok &= (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
        dt = np.zeros(len(Q), np.float32)
        dt[ok] = depth[t][vt[ok], ut[ok]]
        ok &= dt > 0
        if ok.sum() < MIN_PIXELS:
            continue
        shares.append(float(np.mean(np.abs(Q[ok, 2] - dt[ok]) / dt[ok] < AGREE)))
    if not shares:
        raise ValueError(f"no frame pair carries {MIN_PIXELS} pixels onto a depth")
    return float(np.mean(shares))


def check_clip(depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> dict:
    """Return the shares with the kinematics and with a still camera, their difference, and whether to exclude.

    Raises:
        ValueError: `agreeing_share` cannot check the clip, or the depth and the poses differ in length.
    """
    if len(depth) != len(T):
        raise ValueError(f"{len(depth)} depth maps for {len(T)} poses")
    kinematics = agreeing_share(depth, K, T)
    still = agreeing_share(depth, K, np.tile(np.eye(4), (len(T), 1, 1)))
    return dict(kinematics=round(kinematics, 4), still=round(still, 4), difference=round(kinematics - still, 4),
                excluded=kinematics - still < -MARGIN)


def clip_record(job: tuple) -> dict:
    """Return the check of one clip of the clip list, from its stereo depth file and its ground truth poses."""
    root, clip, stereo_depth = job
    with np.load(stereo_depth / f"{clip['name']}.npz") as z:
        depth, K = z["depth_dmm"].astype(np.float32) / 10.0, z["K"].astype(float)
    T = stereomis.gt_pose(root, clip["seq"], stereomis.gt_rows(clip))
    try:
        return dict(clip=clip["name"], seq=clip["seq"], **check_clip(depth, K, T))
    except ValueError as e:
        raise ValueError(f"{clip['name']}: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    ap.add_argument("--stereo-depth", required=True, help="The directory `stereo_depth.py` wrote.")
    ap.add_argument("--out", required=True, help="The JSON file the check is written to.")
    ap.add_argument("--jobs", type=int, default=1, help="Clips checked at once.")
    args = ap.parse_args()

    # Check every usable clip.
    clips = stereomis.usable_clips(stereomis.read_clip_list(Path(args.clips)))
    jobs = [(Path(args.root), c, Path(args.stereo_depth)) for c in clips]
    with ProcessPoolExecutor(args.jobs) as ex:
        records = sorted(ex.map(clip_record, jobs), key=lambda r: r["difference"])

    # Print the lowest differences and the clips excluded, and write every clip's check.
    excluded = [r["clip"] for r in records if r["excluded"]]
    print(f"agreeing share, kinematics - still camera ({len(records)} clips; excluded below -{MARGIN}):")
    for r in records[:len(excluded) + 3]:
        print(f"  {r['clip']:17s} {r['difference']:+.3f}{'  excluded' if r['excluded'] else ''}")
    print("excluded: " + " ".join(excluded))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    below = [r["clip"] for r in records if r["difference"] < 0]
    out.write_text(json.dumps(dict(gap_frames=GAP, agree_within=AGREE, margin=MARGIN, excluded=excluded,
                                   below_still_camera=below, clips=records), indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
