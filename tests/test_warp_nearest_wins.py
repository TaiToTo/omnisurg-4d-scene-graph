"""`warp_labels` gives every destination pixel the label of its nearest point.

The old code sorted far-first and wrote every point through repeated indices,
trusting the last write to win. NumPy does not promise that, so the winner is
now chosen explicitly. On inputs with distinct depths the two agree exactly;
that is checked here, since the workbench's labels were made with the old one.
"""

import numpy as np
import pytest

from surgical_core.geometry import project


def _cam(h, w):
    return np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]])


def _random_warp(seed, h=24, w=32):
    rng = np.random.default_rng(seed)
    depth = 0.4 + 0.3 * rng.random((h, w))
    labels = rng.integers(0, 6, (h, w))
    labels[rng.random((h, w)) < 0.1] = -1
    ext_src = np.eye(4)[:3]                               # (3, 4) world-to-camera
    ext_dst = np.eye(4)[:3].copy()
    ext_dst[:, 3] = rng.normal(0, 0.03, 3)                # a small camera move
    return labels, depth, _cam(h, w), ext_src, _cam(h, w), ext_dst


def _old_warp(L_src, depth_src, K_src, ext_src, K_dst, ext_dst):
    """The workbench's expression, kept here as the reference it is."""
    H, W = L_src.shape
    Pw, m, ys, xs = project.backproject(depth_src, K_src, ext_src)
    lab = L_src[ys, xs]
    u, v, Z = project.project_world_to_frame(Pw, K_dst, ext_dst)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (Z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z, lab = ui[ok], vi[ok], Z[ok], lab[ok]
    order = np.argsort(-Z)
    out = np.full((H, W), -1, dtype=int)
    out[vi[order], ui[order]] = lab[order]
    return out


@pytest.mark.parametrize("seed", range(20))
def test_agrees_with_the_old_expression_on_distinct_depths(seed):
    args = _random_warp(seed)
    assert np.array_equal(project.warp_labels(*args), _old_warp(*args))


def test_the_nearer_point_wins_on_a_shared_pixel():
    """Two points land on one pixel; the one with the smaller depth names it."""
    h, w = 8, 8
    depth = np.full((h, w), 1.0)
    depth[2, 2] = 0.5                                   # nearer
    labels = np.full((h, w), 1)
    labels[2, 2] = 7
    K = _cam(h, w)
    # Project everything onto a single destination pixel by shrinking the
    # destination focal length to almost nothing.
    K_dst = np.array([[1e-6, 0, 3.0], [0, 1e-6, 3.0], [0, 0, 1]])
    eye = np.eye(4)[:3]
    out = project.warp_labels(labels, depth, K, eye, K_dst, eye)
    assert out[3, 3] == 7
    assert (out == -1).sum() == h * w - 1
