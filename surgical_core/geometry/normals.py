"""Camera-space normals from depth, and the geometric edge maps built on them.

numpy and OpenCV only: the evaluation toolkit imports this, and it has to run
on a machine with no data and no model weights.
"""

import cv2
import numpy as np

from surgical_core.geometry.valid import valid_depth_mask


def camera_normals(depth, K):
    """Unit normals in camera space, and the mask of pixels that have one.

    Each pixel is back-projected into camera space and the normal is the
    cross product of the horizontal and vertical tangent vectors, taken by
    central differences. `normal_map` and `geom_edge_map` are computed from
    these normals.

    Args:
        depth: (H, W) depth. 0 and NaN are invalid.
        K: (3, 3) intrinsics at the depth's resolution.

    Returns:
        `(n, m)`: `n` is (H, W, 3) unit normals (NaN where invalid), `m` is the
        (H, W) bool mask of valid pixels.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.mgrid[0:H, 0:W]
    m = valid_depth_mask(depth)
    z = np.where(m, depth, np.nan)
    P = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], -1)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return n, m


def normal_map(depth, K):
    """The normals as an RGB image, (H, W, 3) uint8, black where invalid.

    Bulges and creases show as colour changes, so a segmenter prompted with
    this picks up geometric boundaries that colormapped depth hides.

    Args:
        depth: (H, W) depth. 0 and NaN are invalid.
        K: (3, 3) intrinsics at the depth's resolution.
    """
    n, m = camera_normals(depth, K)
    img = (n * 0.5 + 0.5) * 255
    img[~m] = 0
    return np.nan_to_num(img).astype(np.uint8)


# The contour along the image border and around invalid depth is not geometry.
# The normals are central differences, so they are 0 or NaN on the outer 2
# pixels of the image and the 2 pixels next to an invalid pixel; there the
# 1 - cos to the neighbour is 1, the strongest possible crease, and the depth
# gradient looks like a step at the rim of a hole. Measured on one CholecSeg8k
# frame, 57.6 % of the pixels with edge > 0.5 were the outer 2 pixels of the
# image. By default that ring is zeroed (`mask_ring=True`), as in every
# condition made since the ring was masked. Labels made before the ring was
# masked came from inputs with the ring in; `mask_ring=False` reproduces
# them. The setting is an argument, not a module flag, so that one process
# cannot run two settings under one record: a stage passes it and records
# the value it passed.
#
# The underlying `normal_map` keeps the same ring, and that was measured and
# left alone on purpose. On the border the tangent is 0, so the normal is 0,
# which `normal_map` draws as grey (127, 127, 127), a false "flat"; next to a
# hole it is NaN, drawn black. Two measurements say it does not matter:
# 1. The ring around holes does not occur. The depth the pipeline uses is
#    dense: all four clips measured had 100 % valid pixels, so the only ring is
#    the 1-pixel image border, 0.8 to 1.1 % of the pixels.
# 2. The segmenter does not pick the border up as a region. Counting regions
#    with more than half their pixels within 2 pixels of the border in the
#    existing labels: 0 of 49 (normal), 0 of 48 (edge) and 0 of 187 (rgb) on
#    ATLAS-120k; 1 of 101, 0 of 109 and 0 of 118 on CholecSeg8k.
# Fixing it would mean regenerating every normal-based and edge-based
# condition. GPU time is not spent on an effect that could not be measured;
# measure these two again first.
EDGE_RING_PX = 2


def edge_reliable_mask(m):
    """Pixels whose crease and step are trustworthy: valid, and `EDGE_RING_PX`
    away from any invalid pixel and from the image border."""
    px = int(EDGE_RING_PX)
    if px < 0:
        raise ValueError(f"EDGE_RING_PX is a width in pixels, got {EDGE_RING_PX!r}")
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    r = cv2.erode(m.astype(np.uint8), k).astype(bool)
    # The border slices are written only for a positive width: `r[-0:]` is
    # the whole array, not an empty slice, so a width of 0 would otherwise
    # mask every pixel instead of none.
    if px:
        r[:px, :] = False
        r[-px:, :] = False
        r[:, :px] = False
        r[:, -px:] = False
    return r


def geom_edge_map(depth, K, normal_thresh=0.3, depth_thresh=0.04, parts="both",
                  mask_ring=True, normals=None):
    """Geometric edge strength in [0, 1] from normal discontinuity and depth steps.

    Large at real geometric boundaries (organ creases, occlusion steps), small
    on smooth surfaces. Burnt into a segmenter input as a dark line, it pulls
    the segmentation onto those boundaries.

    Args:
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        normal_thresh: the 1 - cos between neighbouring normals at which the
            edge saturates (0.3 is about 45 degrees).
        depth_thresh: the relative depth step at which the edge saturates
            (4 %, for occlusion boundaries).
        parts: `"both"` (the default), `"normal"` (creases only) or
            `"depth"` (steps only), to tell which cue is doing the work.
        mask_ring: whether to zero the ring along the image border and
            around invalid pixels, True or False.
        normals: `camera_normals(depth, K)`, when the caller has it already.

    Returns:
        (H, W) float edge strength, 0 where invalid.

    Raises:
        ValueError: one of these:
            - `mask_ring` is not a bool. `None` once meant "follow the module
              flag", which no longer exists.
            - `normals` is given and its shapes are not the depth's, so it was
              computed from another depth.

    Note:
        The ring is 2 pixels wide to match the crease term: the normals are
        central differences, so they are 0 or NaN on the outer pixel and the
        pixel next to a hole, and the 1 - cos rings one pixel further. The
        step term uses `np.gradient`, which is one-sided at the border, so it
        has no border artifact and would need only 1 pixel around holes;
        with `parts="depth"` the outer 2 pixels of real steps are lost too.
        That is the conservative side, so the two are kept equal.
    """
    if not isinstance(mask_ring, bool):
        raise ValueError(f"mask_ring is True or False, not {mask_ring!r}")
    if normals is None:
        n, m = camera_normals(depth, K)
    else:
        n, m = normals
        if n.shape != (*depth.shape, 3) or m.shape != depth.shape:
            raise ValueError(f"the normals given are {n.shape} with a mask {m.shape}, not the depth's {depth.shape}; "
                             "they were computed from another depth")
    H, W = depth.shape
    nf = np.nan_to_num(n)
    # Normal discontinuity: 1 - cos to the right and lower neighbour, large at
    # creases and folds.
    en = np.zeros((H, W))
    dot_r = np.sum(nf[:, :-1] * nf[:, 1:], -1)
    en[:, :-1] = np.maximum(en[:, :-1], 1 - dot_r)
    en[:, 1:] = np.maximum(en[:, 1:], 1 - dot_r)
    dot_d = np.sum(nf[:-1, :] * nf[1:, :], -1)
    en[:-1, :] = np.maximum(en[:-1, :], 1 - dot_d)
    en[1:, :] = np.maximum(en[1:, :], 1 - dot_d)
    en = np.clip(en / normal_thresh, 0, 1)
    # Depth step: relative gradient, for occlusion boundaries and the places
    # where the normal breaks down.
    d = np.where(m, depth, np.nan)
    g = np.hypot(np.gradient(np.nan_to_num(d), axis=1), np.gradient(np.nan_to_num(d), axis=0))
    # The 1e-6 keeps the division finite where the depth is 0. It is not a depth test.
    ed = np.clip(np.nan_to_num(g) / (np.nan_to_num(d) * depth_thresh + 1e-6), 0, 1)
    edge = {"normal": en, "depth": ed}.get(parts, np.maximum(en, ed))
    edge = np.array(edge, dtype=float)
    edge[~m] = 0
    if mask_ring:
        edge[~edge_reliable_mask(m)] = 0
    return edge


def normal_edge_map(depth, K, edge_gain=0.85, smooth=True, mask_ring=True):
    """The normal image with the geometric edges burnt in as dark lines.

    Smoothing removes the speckle that depth noise puts into the normals,
    which otherwise splits regions into slivers, and the dark lines pull the
    segmentation onto the real boundaries.

    Args:
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        edge_gain: how dark the edge line is, 0 to 1; 1 makes it nearly black.
        smooth: bilateral-filter the normal image first.
        mask_ring: zero the edges' ring, as `geom_edge_map` does.

    Returns:
        (H, W, 3) uint8, black where invalid.
    """
    n, m = camera_normals(depth, K)
    base = ((np.nan_to_num(n) * 0.5 + 0.5) * 255).astype(np.uint8)
    base[~m] = 0
    if smooth:
        base = cv2.bilateralFilter(base, d=5, sigmaColor=40, sigmaSpace=5)
    img = burn_geom_edge(base, depth, K, edge_gain, normals=(n, m), mask_ring=mask_ring)
    img[~m] = 0
    return img


def burn_geom_edge(base, depth, K, edge_gain=0.85, normals=None, mask_ring=True):
    """Burn the same geometric edges into any 3-channel image.

    The edge comes from `geom_edge_map`, from depth and intrinsics alone, so
    it is identical whatever `base` is, and the difference between an input
    and its `_edge` variant is exactly the dark line. Invalid pixels are left
    to `base` (black in a normal image, untouched in RGB); zeroing them here
    would mix "edges added" with "invalid pixels removed".

    Args:
        base: (H, W, 3) uint8.
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        edge_gain: how dark the edge line is, 0 to 1.
        normals: `camera_normals(depth, K)`, when the caller has it already.
        mask_ring: zero the edges' ring, as `geom_edge_map` does.

    Returns:
        (H, W, 3) uint8.
    """
    edge = geom_edge_map(depth, K, normals=normals, mask_ring=mask_ring)
    img = base.astype(np.float32) * (1.0 - edge_gain * edge)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)
