"""Measure an estimated camera trajectory against the true one: ATE after a similarity fit, RPE, and scale drift.

A monocular depth model knows no scale, so the estimated camera centres are
first fitted to the true ones by one similarity transform (Umeyama: rotation,
translation and one scale), and the error is the RMSE of the fitted centres.
`ate` divides it by the true trajectory's RMS radius about its centroid, so
a camera that never moves scores exactly 1.0: below 1.0, the poses carry
information. Only `ate_rel` decides the result; `rpe` and
`scale_consistency` help explain it. The model's extrinsics are
world-to-camera and the camera centre is `-R^T t`; a camera-to-world pose
holds the centre as its translation.
"""

import numpy as np


def centers_from_world_to_cam(E: np.ndarray) -> np.ndarray:
    """Return the (N, 3) camera centres of (N, 3, 4) or (N, 4, 4) world-to-camera extrinsics."""
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def centers_from_cam_to_world(T: np.ndarray) -> np.ndarray:
    """Return the (N, 3) camera centres of (N, 4, 4) camera-to-world poses."""
    return T[:, :3, 3].copy()


def rot_from_world_to_cam(E: np.ndarray) -> np.ndarray:
    """Return the camera-to-world rotations of world-to-camera extrinsics."""
    return np.transpose(E[:, :3, :3], (0, 2, 1))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve `dst ≈ s R src + t` by least squares (Umeyama 1991).

    Returns:
        (scale, (3, 3) rotation, (3,) translation). Source points that all coincide give scale 0 and the
        identity, which is what a camera that never moves gets.

    Raises:
        ValueError: the inputs are not two (N, 3) arrays of one shape, or hold fewer than 3 points.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"two (N, 3) arrays are needed, not {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"{len(src)} points; a similarity fit needs 3")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    var_s = float((a ** 2).sum(1).mean())
    if var_s < 1e-18:
        return 0.0, np.eye(3), mu_d.copy()
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    # A reflection would fit a mirrored trajectory perfectly; the sign keeps the fit a rotation.
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    return s, R, mu_d - s * R @ mu_s


def gt_span(gt_c: np.ndarray) -> float:
    """Return the true trajectory's RMS radius about its centroid, `ate`'s denominator."""
    return float(np.sqrt(((gt_c - gt_c.mean(0)) ** 2).sum(1).mean()))


def ate(est_c: np.ndarray, gt_c: np.ndarray) -> dict:
    """Fit the estimate to the truth by a similarity, and return the RMSE of the centres over the truth's span.

    Args:
        est_c: (N, 3) estimated camera centres.
        gt_c: (N, 3) true camera centres, in mm.

    Returns:
        `ate_rel` (the measure), `ate_mm`, `span_mm`, `scale`, `err_mm` (per frame) and `n`.

    Raises:
        ValueError: the two have different numbers of frames, or the truth does not move.
    """
    if len(est_c) != len(gt_c):
        raise ValueError(f"{len(est_c)} estimated frames against {len(gt_c)} true ones")
    s, R, t = umeyama(est_c, gt_c)
    err = np.linalg.norm((est_c @ R.T) * s + t - gt_c, axis=1)
    span = gt_span(gt_c)
    if span < 1e-9:
        raise ValueError("the true camera does not move, so the error has no scale to be divided by")
    rmse = float(np.sqrt((err ** 2).mean()))
    return dict(ate_rel=rmse / span, ate_mm=rmse, span_mm=span, scale=s, err_mm=err, n=len(gt_c))


def _angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def rpe(est_c: np.ndarray, est_R: np.ndarray, gt_c: np.ndarray, gt_R: np.ndarray, times: np.ndarray,
        scale: float, dt: float = 1.0) -> dict:
    """Return the error of the relative poses `dt` seconds apart, a local measure of the trajectory.

    ATE follows one large failure; RPE does not. The translations take the one scale `ate`'s fit found.

    Args:
        est_c, est_R: the estimated camera centres and camera-to-world rotations.
        gt_c, gt_R: the true ones.
        times: each frame's time in seconds, ascending.
        scale: the scale of `ate`'s fit.
        dt: the time between the two frames of a pair.

    Returns:
        `rpe_trans_rel` (the RMSE of the relative translations over their RMS), `rpe_rot_deg` (the median
        rotation error) and `n_pairs`. With fewer than 3 pairs, both errors are NaN.
    """
    j = np.searchsorted(times, times + dt)
    i = np.arange(len(times))
    ok = j < len(times)
    i, j = i[ok], j[ok]
    if len(i) < 3:
        return dict(rpe_trans_rel=float("nan"), rpe_rot_deg=float("nan"), n_pairs=len(i))
    # Frame j's position in frame i's camera axes: the world difference rotated by the transpose of i's rotation.
    d_gt = np.einsum("nji,nj->ni", gt_R[i], gt_c[j] - gt_c[i])
    d_est = np.einsum("nji,nj->ni", est_R[i], est_c[j] - est_c[i]) * scale
    e = np.linalg.norm(d_est - d_gt, axis=1)
    denom = float(np.sqrt((np.linalg.norm(d_gt, axis=1) ** 2).mean()))
    dR = np.einsum("nji,njk->nik", gt_R[i], gt_R[j])
    dRe = np.einsum("nji,njk->nik", est_R[i], est_R[j])
    rot = _angle_deg(np.einsum("nji,njk->nik", dR, dRe))
    return dict(rpe_trans_rel=float(np.sqrt((e ** 2).mean()) / denom) if denom > 1e-9 else float("nan"),
                rpe_rot_deg=float(np.median(rot)), n_pairs=int(len(i)))


def scale_consistency(est_c: np.ndarray, gt_c: np.ndarray, n_parts: int = 4) -> dict:
    """Fit each of `n_parts` equal stretches on its own, and return how far their scales spread.

    A `scale_ratio` near 1 says that the trajectory has one scale throughout. A model that solves a sequence
    chunk by chunk fails this measure first.

    Returns:
        `scale_ratio` (largest over smallest) and `scales`. NaN when the sequence is shorter than 4 frames a
        stretch, or a stretch's fitted scale is 0.
    """
    n = len(gt_c)
    if n < n_parts * 4:
        return dict(scale_ratio=float("nan"), scales=[])
    bounds = np.linspace(0, n, n_parts + 1).astype(int)
    ss = np.array([umeyama(est_c[a:b], gt_c[a:b])[0] for a, b in zip(bounds[:-1], bounds[1:])])
    ratio = float(ss.max() / ss.min()) if ss.min() > 1e-12 else float("nan")
    return dict(scale_ratio=ratio, scales=[round(float(x), 6) for x in ss])
