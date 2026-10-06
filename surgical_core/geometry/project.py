"""Back-projection, projection, label transfer and label warping between frames.

Extrinsics are world-to-camera (w2c) throughout. numpy only.
"""

import numpy as np

from surgical_core.geometry.camera import backproject_depth, cam_to_world
from surgical_core.geometry.valid import DEPTH_MIN, valid_depth_mask


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

    Raises:
        ValueError: `K` is not (3, 3), or `ext_w2c` not (3, 4): `cam_to_world`
            refuses the (4, 3) rotation a (4, 4) one slices to.
    """
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    m = valid_depth_mask(depth)
    R, t = ext_w2c[:, :3], ext_w2c[:, 3]
    with np.errstate(all="ignore"):
        Pw = cam_to_world(backproject_depth(depth, K)[m.reshape(-1)], R, t)
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
    ok = (Z > DEPTH_MIN) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z, lab = ui[ok], vi[ok], Z[ok], lab[ok]
    # One explicit winner per pixel: the nearest point. Sorting by pixel and
    # then by depth puts it first in its pixel's run. Writing every point
    # through repeated indices and trusting the last write to win is not
    # something NumPy promises, so it is not relied on. Two points at exactly
    # the same depth on the same pixel are settled by source order (lexsort is
    # stable), where the old far-first sort left the choice to the sort.
    flat = vi * W + ui
    order = np.lexsort((Z, flat))
    flat, lab = flat[order], lab[order]
    first = np.ones(flat.size, dtype=bool)
    first[1:] = flat[1:] != flat[:-1]
    out = np.full((H, W), -1, dtype=int)
    out.flat[flat[first]] = lab[first]
    return out
