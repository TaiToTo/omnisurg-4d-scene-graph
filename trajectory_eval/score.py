"""Score one clip's estimated camera trajectory against its true one, and read a depth stage's trajectory.

`score` returns a clip's row: `ate_rel`, which decides the result, `rpe` and
`scale_consistency`, rounded as the workbench wrote them, with
`trajectory_code_sha`. `method_trajectory` reads the camera centres and
rotations from a depth stage's bundle, whose extrinsics map world to camera.
"""

from pathlib import Path

import numpy as np

from trajectory_eval.code_sha import trajectory_code_sha
from trajectory_eval.pose_metrics import (ate, centers_from_cam_to_world, centers_from_world_to_cam, rpe,
                                          rot_from_world_to_cam, scale_consistency)

# The bundle each depth stage writes into a clip's `exports/mini_npz/`.
METHOD_BUNDLES = {"da3": "results.npz", "pi3x": "results__pi3x.npz"}


def score(clip: dict, cond: str, gt_T: np.ndarray, est_c: np.ndarray, est_R: np.ndarray) -> dict:
    """Return the row of one clip and one condition.

    Args:
        clip: the clip, with `name`, `seq`, `stratum`, `span_mm` and `times` (one per frame, in s).
        cond: the condition's name.
        gt_T: (N, 4, 4) true camera-to-world poses, translations in mm.
        est_c: (N, 3) estimated camera centres.
        est_R: (N, 3, 3) estimated camera-to-world rotations.

    Raises:
        ValueError: `pose_metrics` refuses the trajectories: unequal lengths, times out of order, a value that is
            not finite, or a true camera that does not move.
    """
    gt_c, gt_R = centers_from_cam_to_world(gt_T), gt_T[:, :3, :3]
    a = ate(est_c, gt_c)
    r = rpe(est_c, est_R, gt_c, gt_R, np.asarray(clip["times"], dtype=float), a["scale"])
    sc = scale_consistency(est_c, gt_c)
    return dict(clip=clip["name"], seq=clip["seq"], stratum=clip["stratum"], span_mm=clip["span_mm"],
                ate_rel=round(a["ate_rel"], 4), ate_mm=round(a["ate_mm"], 3), scale=a["scale"],
                rpe_trans_rel=round(r["rpe_trans_rel"], 4), rpe_rot_deg=round(r["rpe_rot_deg"], 3),
                n_rpe=r["n_pairs"],
                scale_ratio=round(sc["scale_ratio"], 3) if np.isfinite(sc["scale_ratio"]) else None,
                cond=cond, trajectory_code_sha=trajectory_code_sha())


def method_trajectory(clip_dir: Path, method: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a depth stage's (N, 3) camera centres and (N, 3, 3) camera-to-world rotations for a clip.

    Raises:
        KeyError: `method` is not in `METHOD_BUNDLES`.
        FileNotFoundError: the clip holds no bundle of the method.
        ValueError: the bundle's extrinsics are not one (3, 4) or (4, 4) matrix for each of `n_frames` frames.
    """
    path = Path(clip_dir) / "exports" / "mini_npz" / METHOD_BUNDLES[method]
    with np.load(path) as z:
        E = z["extrinsics"]
    if E.ndim != 3 or E.shape[0] != n_frames or E.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"{path}: extrinsics of shape {E.shape}, where {n_frames} frames of (3, 4) are expected")
    T = np.tile(np.eye(4), (n_frames, 1, 1))
    T[:, :3, :4] = E[:, :3, :4]
    return centers_from_world_to_cam(T), rot_from_world_to_cam(T)
