"""Back-projection, projection, label transfer and label warping between frames.

Extrinsics are world-to-camera (w2c) throughout. numpy only.
"""

import numpy as np


def backproject(depth, K, ext_w2c):
    """Depth to world points.

    For camera-to-world extrinsics the formula would be `Pw = (R @ Pc.T + t).T`.

    Args:
        depth: (H, W) depth. 0 and NaN are invalid.
        K: (3, 3) intrinsics.
        ext_w2c: (3, 4) world-to-camera extrinsics.

    Returns:
        `(Pw, m, ys, xs)`: the (N, 3) world points of the valid pixels, the
        (H, W) valid mask, and the row and column of each point.
    """
    # The shapes are not validated at the entry. A (4, 4) extrinsics matrix
    # slices without error into the wrong submatrix and the point cloud comes
    # out silently wrong; the projection and warp below share the slice. A
    # check here would close that for all of them; it is a change in
    # behaviour for callers that pass (4, 4) today, so it is made on purpose,
    # not in a port.
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 1e-6)
    z = depth[m]
    X = (xs[m] - cx) * z / fx
    Y = (ys[m] - cy) * z / fy
    Pc = np.stack([X, Y, z], 1)
    R, t = ext_w2c[:, :3], ext_w2c[:, 3]
    with np.errstate(all="ignore"):
        Pw = (R.T @ (Pc.T - t[:, None])).T
    return Pw, m, ys[m], xs[m]


def project_world_to_frame(Pw, K, ext_w2c):
    """World points to pixel coordinates `(u, v)` and camera depth `Z`."""
    R, t = ext_w2c[:, :3], ext_w2c[:, 3]
    with np.errstate(all="ignore"):
        Pc = (R @ Pw.T + t[:, None]).T
    Z = Pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = Pc[:, 0] * fx / Z + cx
    v = Pc[:, 1] * fy / Z + cy
    return u, v, Z


def project_labels(fused_tree, fused_group, depth, K, ext, transfer_thresh):
    """Fused group id per pixel by nearest-neighbour lookup of each point.

    Fast, but the boundaries come out speckled; `project_labels_region` is
    the one to use.

    Args:
        fused_tree: a KD-tree over the fused points, with a `query(points,
            k=1)` method returning `(distances, indices)`, such as
            `scipy.spatial.cKDTree`.
        fused_group: (M,) group id of each fused point.
        depth, K, ext: the frame's depth, intrinsics and w2c extrinsics.
        transfer_thresh: a pixel farther than this from its nearest fused
            point gets -1.

    Returns:
        (H, W) int labels, -1 where none.
    """
    H, W = depth.shape
    Pw, m, ys, xs = backproject(depth, K, ext)
    dist, idx = fused_tree.query(Pw, k=1)
    grp = fused_group[idx].astype(int)
    grp[dist > transfer_thresh] = -1
    lab = np.full((H, W), -1, dtype=int)
    lab[ys, xs] = grp
    return lab


def project_labels_region(fused_tree, fused_group, depth, K, ext, transfer_thresh, L2d):
    """Region-wise relabelling: the per-pixel projection, then a majority vote
    inside each clean 2D region of `L2d`, so that the boundaries are the 2D
    segmentation's and the ids are the fused ones.

    Args:
        L2d: (H, W) int regions of the frame's 2D segmentation, -1 for none.

    Returns:
        (H, W) int labels, -1 where a region has no projected id.
    """
    proj = project_labels(fused_tree, fused_group, depth, K, ext, transfer_thresh)
    out = np.full(proj.shape, -1, dtype=int)
    for r in np.unique(L2d):
        if r < 0:
            continue
        m = L2d == r
        v = proj[m]
        v = v[v >= 0]
        if v.size:
            out[m] = np.bincount(v).argmax()
    return out


def warp_labels(L_src, depth_src, K_src, ext_src, K_dst, ext_dst):
    """Warp a label map from one frame's grid to another's, with a z-buffer.

    Returns:
        (H, W) int labels on the destination grid, -1 where nothing lands.
    """
    H, W = L_src.shape
    Pw, m, ys, xs = backproject(depth_src, K_src, ext_src)
    lab = L_src[ys, xs]
    u, v, Z = project_world_to_frame(Pw, K_dst, ext_dst)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (Z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z, lab = ui[ok], vi[ok], Z[ok], lab[ok]
    order = np.argsort(-Z)                            # far first, so near overwrites
    out = np.full((H, W), -1, dtype=int)
    out[vi[order], ui[order]] = lab[order]
    return out
