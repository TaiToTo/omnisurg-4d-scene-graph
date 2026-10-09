"""Measure an estimated camera trajectory against the true one: ATE after a similarity fit, RPE, and scale drift.

The estimated camera centres are fitted to the true ones by one similarity
transform (rotation, translation and one scale), and `ate` is the RMSE of
the fitted centres over the true trajectory's RMS radius about its
centroid. `rpe` is the error of the relative poses a fixed time apart, and
`scale_consistency` the spread of the scales fitted on four equal
stretches. Every measure refuses trajectories of unequal length, times out
of order, and values that are not finite.
"""

import numpy as np


def centers_from_world_to_cam(E: np.ndarray) -> np.ndarray:
    """Return the (N, 3) camera centres of (N, 3, 4) or (N, 4, 4) world-to-camera extrinsics.

    A world-to-camera extrinsic maps `p_cam = R p_world + t`, so the camera centre is `-R^T t`; the depth
    models write their extrinsics this way.
    """
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def centers_from_cam_to_world(T: np.ndarray) -> np.ndarray:
    """Return the (N, 3) camera centres of (N, 4, 4) camera-to-world poses, which hold the centre as `t`."""
    return T[:, :3, 3].copy()


def rot_from_world_to_cam(E: np.ndarray) -> np.ndarray:
    """Return the camera-to-world rotations of world-to-camera extrinsics."""
    return np.transpose(E[:, :3, :3], (0, 2, 1))


def _finite(name: str, *arrays: np.ndarray) -> None:
    """Refuse an array with a value that is not finite; a missing pose would otherwise become a NaN score."""
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} holds {int((~np.isfinite(arr)).sum())} values that are not finite")


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve `dst ≈ s R src + t` by least squares (Umeyama 1991).

    A monocular depth model knows no scale, so its trajectory is compared only after this fit. `R` is a
    rotation, never a reflection. A trajectory that lies in one plane is the exception the fit cannot tell:
    its mirror image is reached by a rotation too, and is fitted at zero error.

    Returns:
        (scale, (3, 3) rotation, (3,) translation). Source points that all coincide give scale 0 and the
        identity, which is what a camera that never moves gets.

    Raises:
        ValueError: the inputs are not two (N, 3) arrays of one shape, hold fewer than 3 points, or hold a
            value that is not finite.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"two (N, 3) arrays are needed, not {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"{len(src)} points; a similarity fit needs 3")
    _finite("the centres", src, dst)
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

    The span is the denominator because of what it makes of a camera that never moves: the fit puts it at
    the truth's centroid, its RMSE is the span itself, and it scores exactly 1.0. So a trajectory below 1.0
    carries information, and no threshold has to be chosen.

    Args:
        est_c: (N, 3) estimated camera centres.
        gt_c: (N, 3) true camera centres, in mm.

    Returns:
        `ate_rel` (the measure), `ate_mm`, `span_mm`, `scale`, `err_mm` (per frame) and `n`.

    Raises:
        ValueError: the two have different numbers of frames, `umeyama` refuses them, or the truth does not
            move.
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
    Frame `i` is paired with the first frame at least `dt` seconds after it.

    Args:
        est_c: (N, 3) estimated camera centres.
        est_R: (N, 3, 3) estimated camera-to-world rotations.
        gt_c: (N, 3) true camera centres.
        gt_R: (N, 3, 3) true camera-to-world rotations.
        times: (N,) each frame's time in seconds, never decreasing.
        scale: the scale of `ate`'s fit.
        dt: the time between the two frames of a pair.

    Returns:
        `rpe_trans_rel` (the RMSE of the relative translations over their RMS), `rpe_rot_deg` (the median
        rotation error) and `n_pairs`. With fewer than 3 pairs, both errors are NaN.

    Raises:
        ValueError: the five arrays do not describe one number of frames, `times` decreases somewhere, or a
            value is not finite.
    """
    n = len(times)
    shapes = {"est_c": (n, 3), "est_R": (n, 3, 3), "gt_c": (n, 3), "gt_R": (n, 3, 3), "times": (n,)}
    for name, arr in zip(shapes, (est_c, est_R, gt_c, gt_R, times)):
        if arr.shape != shapes[name]:
            raise ValueError(f"{name} has shape {arr.shape}; {shapes[name]} is needed for {n} frames")
        _finite(name, arr)
    if np.any(np.diff(times) < 0):
        raise ValueError("times decrease somewhere; the pairs are found by searching them in order")
    j = np.searchsorted(times, times + dt)
    i = np.arange(n)
    ok = j < n
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

    Raises:
        ValueError: `umeyama` refuses a stretch.
    """
    n = len(gt_c)
    if n < n_parts * 4:
        return dict(scale_ratio=float("nan"), scales=[])
    bounds = np.linspace(0, n, n_parts + 1).astype(int)
    ss = np.array([umeyama(est_c[a:b], gt_c[a:b])[0] for a, b in zip(bounds[:-1], bounds[1:])])
    ratio = float(ss.max() / ss.min()) if ss.min() > 1e-12 else float("nan")
    return dict(scale_ratio=ratio, scales=[round(float(x), 6) for x in ss])
