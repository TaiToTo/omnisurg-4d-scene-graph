"""Write a graph's statements from its nodes' states on each frame, by fixed rules and thresholds.

A node's state on a frame is its centroid, as shares of the frame's width and height, and its median depth.
`frame_statements` writes one frame's 2D statements (in view, left of, above) and 3D statements (in front of).
`temporal_statements` writes the statements over the frames: persists, leaves, enters, moves, swaps sides.
`approach_count` counts the nodes whose depth changes over a window, and `camera_statements` writes the camera's
moves, turns and tilts over a window. The thresholds are the ones the paper's supplement lists.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

# A node is in view on a frame when it covers this share of the frame's pixels.
VIS_SHARE = 0.0025
# Two nodes are on different sides when their centroids differ by this share of the frame's width (height).
SIDE_MARGIN = 0.05
# One node is in front of another when its median depth is smaller by this share.
FRONT_MARGIN = 0.05
# A node moves when its centroid travels this share of the frame between its first and last frame in view.
MOVE_SHARE = 0.10
# A node persists when it is in view on this share of the frames.
PERSIST_SHARE = 0.9
# A node leaves (enters) when it is out of (in) view on this many frames in a row after being in (out of) view.
DEBOUNCE = 2
# A window spans this many seconds: StereoMIS's 25 frames 0.2 s apart, where these statements were checked.
WINDOW_S = 4.8
# A node comes closer (moves away) when its median depth falls (rises) by this share over a window.
APPROACH_SHARE = 0.10
# The camera moves along an axis when its centre travels this share of the scene's median depth over a window.
CAM_SHIFT = 0.05
# The camera turns (tilts) when its viewing direction swings by this many degrees over a window.
CAM_TURN_DEG = 3.0

KINDS_2D = ("vis", "left", "above")
KINDS_TEMPORAL = ("persist", "leave", "enter", "move", "swap")


def node_state(mask: np.ndarray, depth: np.ndarray) -> tuple[float, float, float] | None:
    """Return a node's centroid x and y, as shares of the frame, and its median depth; None when out of view."""
    if int(mask.sum()) < VIS_SHARE * depth.size:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return float(xs.mean() / w), float(ys.mean() / h), float(np.median(depth[mask]))


def states_of(masks_per_frame: list[dict], depth: list[np.ndarray]) -> list[dict]:
    """Return each node's state on each frame, from its (H, W) bool mask and the frame's depth."""
    return [{c: node_state(m, d) for c, m in masks.items()} for masks, d in zip(masks_per_frame, depth)]


def side(a: tuple | None, b: tuple | None, axis: int) -> int:
    """Return -1 when node a is left of (above) node b by the margin, +1 when right of (below), 0 otherwise."""
    if a is None or b is None:
        return 0
    d = a[axis] - b[axis]
    return -1 if d <= -SIDE_MARGIN else (1 if d >= SIDE_MARGIN else 0)


def frame_statements(state: dict, names: list[int], k: int) -> set[tuple]:
    """Return the 2D and 3D statements of frame `k` from its nodes' states.

    A statement of two nodes names the left (upper, front) node first.
    """
    out: set[tuple] = set()
    for c in names:
        if state.get(c) is not None:
            out.add(("vis", c, k))
    for a, b in combinations(sorted(names), 2):
        for axis, kind in ((0, "left"), (1, "above")):
            r = side(state.get(a), state.get(b), axis)
            if r:
                out.add((kind,) + ((a, b) if r < 0 else (b, a)) + (k,))
        sa, sb = state.get(a), state.get(b)
        if sa is not None and sb is not None:
            if sa[2] < sb[2] * (1 - FRONT_MARGIN):
                out.add(("front", a, b, k))
            elif sb[2] < sa[2] * (1 - FRONT_MARGIN):
                out.add(("front", b, a, k))
    return out


def temporal_statements(states: list[dict], names: list[int]) -> set[tuple]:
    """Return the statements over the frames: persists, leaves and enters at a frame, moves, swaps sides at a frame."""
    out: set[tuple] = set()
    n = len(states)
    for c in names:
        vis = [s.get(c) is not None for s in states]
        if sum(vis) >= PERSIST_SHARE * n:
            out.add(("persist", c))
        for k in range(1, n):
            after = vis[k:k + DEBOUNCE]
            if vis[k - 1] and not any(after):
                out.add(("leave", c, k))
            if not vis[k - 1] and all(after) and len(after) == DEBOUNCE:
                out.add(("enter", c, k))
        seen = [s[c] for s in states if s.get(c) is not None]
        if len(seen) >= 2:
            dx, dy = seen[-1][0] - seen[0][0], seen[-1][1] - seen[0][1]
            if abs(dx) >= MOVE_SHARE:
                out.add(("move", c, "left" if dx < 0 else "right"))
            if abs(dy) >= MOVE_SHARE:
                out.add(("move", c, "up" if dy < 0 else "down"))
    for a, b in combinations(sorted(names), 2):
        for axis, kind in ((0, "lr"), (1, "ud")):
            last = 0
            for k, s in enumerate(states):
                r = side(s.get(a), s.get(b), axis)
                if r and last and r != last:
                    out.add(("swap", a, b, kind, k))
                if r:
                    last = r
    return out


def windows(times: np.ndarray) -> list[tuple[int, int]]:
    """Return non-overlapping (start, end) frame positions whose times are at least `WINDOW_S` apart."""
    out, s = [], 0
    while s < len(times):
        later = np.nonzero(times[s:] - times[s] >= WINDOW_S - 1e-6)[0]
        if not len(later):
            break
        e = s + int(later[0])
        out.append((s, e))
        s = e + 1
    return out


def approach_count(states: list[dict], names: list[int], spans: list[tuple[int, int]]) -> int:
    """Count the (window, node) pairs whose node is in view at both ends and comes closer or moves away."""
    n = 0
    for s, e in spans:
        for c in names:
            a, b = states[s].get(c), states[e].get(c)
            if a is not None and b is not None and (b[2] < a[2] * (1 - APPROACH_SHARE) or b[2] > a[2] * (1 + APPROACH_SHARE)):
                n += 1
    return n


def camera_statements(E: np.ndarray, scene: np.ndarray, spans: list[tuple[int, int]]) -> list[tuple]:
    """Return the camera's moves, turns and tilts per window, read in the camera's frame at the window's start.

    Args:
        E: (N, 3, 4) world-to-camera extrinsics, one per frame.
        scene: (N,) median depth of each frame, the unit a move is measured in.
        spans: The windows, as (start, end) frame positions.
    """
    R = np.transpose(E[:, :3, :3], (0, 2, 1))
    c = np.einsum("nij,nj->ni", R, -E[:, :3, 3])
    out = []
    for k, (s, e) in enumerate(spans):
        shift = R[s].T @ (c[e] - c[s]) / scene[s]
        for axis, (neg, pos) in enumerate((("left", "right"), ("up", "down"), ("back", "forward"))):
            if abs(shift[axis]) >= CAM_SHIFT:
                out.append(("cam", pos if shift[axis] > 0 else neg, k))
        f = R[s].T @ R[e] @ np.array([0.0, 0.0, 1.0])
        yaw, pitch = np.degrees(np.arctan2(f[0], f[2])), np.degrees(np.arctan2(-f[1], f[2]))
        if abs(yaw) >= CAM_TURN_DEG:
            out.append(("cam", "turn right" if yaw > 0 else "turn left", k))
        if abs(pitch) >= CAM_TURN_DEG:
            out.append(("cam", "tilt up" if pitch > 0 else "tilt down", k))
    return out
