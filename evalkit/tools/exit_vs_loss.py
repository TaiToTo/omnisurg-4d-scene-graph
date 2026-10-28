"""Sort a tracked node's exit and enter events into left-view and lost-in-view, by its 3D position and the camera poses.

A node is one seed region the tracker carries. Where its mask ends the
tracker says it exited, and where a mask begins again, that it entered. The
tool takes the node's last known 3D position, the mean of its centroids on
the frames before the event, and projects it through the poses that follow:
still inside the frustum is a loss, not an exit. The same test sorts the GT
events where a dataset has GT on consecutive frames, and predicted events
are matched to GT events within one sample frame. A still camera and
shuffled poses repeat the test as controls. The tool reads saved outputs
only and refuses a clip whose regions are not the seed's.

Usage:
    python -m evalkit.tools.exit_vs_loss --dataset cholecseg8k --clips LIST --data-root DIR --tracks-root DIR \\
        --tag track_rgb_rgb_gt_forward --out DIR [--geometry da3] [--gt-events] [--k 3] [--h 3] \\
        [--margins 0,0.05,0.1] [--min-present-px 100] [--shuffles 5] [--seed 0] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evalkit.classes import ClassTable, load_table
from evalkit.evaluate import read_population
from evalkit.inputs import ClipInputs, read_clip
from evalkit.objects import gt_objects
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth
from evalkit.tools.component_scores import tool_sha
from evalkit.tools.paired_stats import boot_ci, video_of
from surgical_core.geometry.camera import backproject_depth, cam_to_world
from surgical_core.geometry.project import project_world_to_frame

VIEW = "all"
KINDS = ("exit", "enter")
# The labels of each kind, in the order: still in view (a loss), out of view (a true event), out within the window.
LABELS = {"exit": ("lost_in_view", "left_view", "leaving"), "enter": ("reacquired_in_view", "entered_view", "entering")}
# The depth bundle of each geometry source, under `exports/mini_npz/`.
BUNDLES = {"da3": "results.npz", "pi3x": "results__pi3x.npz"}
# The share of consecutive-frame projections that must land in the next mask's box for a pose reading to pass.
CONVENTION_MIN = 0.8
WITH_POSES = "with_poses"
STATIC = "static_camera"


@dataclass(frozen=True)
class Event:
    """One event of one node: `kind` at frame `t`, the first frame absent (exit) or present again (enter)."""

    kind: str
    node: int
    t: int


def events_of(present: set[int], frames: Sequence[int]) -> list[Event]:
    """Return the exits and enters of one node over `frames`, in time order.

    An exit is the first frame absent after a frame present; an enter the first frame present after a frame
    absent, once the node has been present. A node absent on every frame has no event.
    """
    out, was = [], None
    for t in frames:
        now = t in present
        if was is not None and was != now:
            out.append(Event("exit" if was else "enter", -1, t))
        was = now
    return out


def known_position(centroids: Mapping[int, np.ndarray], t: int, k: int, kind: str) -> np.ndarray:
    """Return the mean centroid over the `k` present frames before an exit at `t`, or from an enter at `t` on.

    Raises:
        ValueError: no present frame lies on that side of `t`.
    """
    frames = sorted(f for f in centroids if (f < t if kind == "exit" else f >= t))
    picked = frames[-k:] if kind == "exit" else frames[:k]
    if not picked:
        raise ValueError(f"no present frame {'before' if kind == 'exit' else 'from'} frame {t}")
    return np.mean([centroids[f] for f in picked], axis=0)


def inside(p: np.ndarray, K: np.ndarray, ext_w2c: np.ndarray, size: tuple[int, int], margin: float) -> bool:
    """Return whether world point `p` lies in front of the camera and inside the frame less `margin` of each side.

    Args:
        size: (W, H) of the frame.
    """
    u, v, z = project_world_to_frame(np.asarray(p, dtype=np.float64)[None], K, ext_w2c)
    W, H = size
    return bool(z[0] > 0 and margin * W <= u[0] <= (1 - margin) * W and margin * H <= v[0] <= (1 - margin) * H)


def classify(kind: str, p: np.ndarray, t: int, frames: Sequence[int], K: np.ndarray, E: np.ndarray,
             size: tuple[int, int], margin: float, h: int) -> tuple[str, bool]:
    """Sort one event by where `p` is on the frames of its window.

    An exit at `t` is a loss when `p` is inside on every frame of `[t, t + h]`, a true exit when it is outside on
    `t`, and `leaving` otherwise. An enter at `t` reads `[t - h, t - 1]` the same way, with `t - 1` deciding.
    Frames past the clip's ends are left out and the event is marked truncated.

    Returns:
        The label, one of `LABELS[kind]`, and whether the window was truncated.
    """
    window = range(t, t + h + 1) if kind == "exit" else range(t - h, t)
    decisive = t if kind == "exit" else t - 1
    have = [f for f in window if f in frames]
    truncated = len(have) < len(window)
    if decisive not in have:
        raise ValueError(f"frame {decisive} that decides the {kind} at {t} is not a frame of the clip")
    still, loss, left = LABELS[kind]
    flags = {f: inside(p, K[f], E[f], size, margin) for f in have}
    if all(flags.values()):
        return still, truncated
    if not flags[decisive]:
        return loss, truncated
    return left, truncated


def static_poses(E: np.ndarray, seed: int) -> np.ndarray:
    """Every frame's pose replaced by the seed frame's: the camera never moves."""
    return np.repeat(E[seed][None], len(E), axis=0)


def shuffled_poses(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The poses given to the frames in a random order."""
    return E[rng.permutation(len(E))]


def inverted(E: np.ndarray) -> np.ndarray:
    """Each (3, 4) pose read the other way round: `[R^T | -R^T t]`."""
    R, t = E[:, :, :3], E[:, :, 3]
    Rt = np.transpose(R, (0, 2, 1))
    return np.concatenate([Rt, -np.einsum("nij,nj->ni", Rt, t)[:, :, None]], axis=2)


def match_events(pred: Sequence[Event], gt: Sequence[Event], tol: int = 1) -> dict[int, int]:
    """Pair predicted events with GT events of the same node and kind within `tol` frames, closest first, each once.

    Returns:
        Each matched prediction's index to its GT event's index.
    """
    pairs = sorted((abs(p.t - g.t), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)
                   if p.kind == g.kind and p.node == g.node and abs(p.t - g.t) <= tol)
    out, taken = {}, set()
    for _, i, j in pairs:
        if i not in out and j not in taken:
            out[i] = j
            taken.add(j)
    return out


@dataclass(frozen=True)
class NodeFrames:
    """One node's masks over the frames, reduced: area, camera-space centroid and box per frame it has pixels on."""

    area: Mapping[int, int]
    centroid_cam: Mapping[int, np.ndarray]
    box: Mapping[int, tuple[int, int, int, int]]


def reduce_masks(masks: Mapping[int, np.ndarray], depth: np.ndarray, K: np.ndarray) -> NodeFrames:
    """Reduce one node's (H, W) bool masks, by frame, to what the test reads.

    The centroid is the mean of the mask's pixels back-projected into the frame's camera space. The world
    centroid of any pose set is that point moved rigidly, since the mean commutes with a rigid transform.
    """
    area, cen, box = {}, {}, {}
    for t, m in masks.items():
        n = int(m.sum())
        if n == 0:
            continue
        pts = backproject_depth(depth[t], K[t])[m.reshape(-1)]
        ys, xs = np.nonzero(m)
        area[t], cen[t] = n, pts.mean(axis=0)
        box[t] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return NodeFrames(area=area, centroid_cam=cen, box=box)


def world_centroids(node: NodeFrames, E: np.ndarray, present: set[int]) -> dict[int, np.ndarray]:
    """The node's world centroid on each present frame, under the pose set `E`."""
    return {t: cam_to_world(node.centroid_cam[t], E[t][:, :3], E[t][:, 3]) for t in present}


def convention_share(nodes: Sequence[NodeFrames], presents: Sequence[set[int]], K: np.ndarray,
                     E: np.ndarray) -> tuple[int, int]:
    """Count the projections of a node's centroid from one present frame into the next that land in its box there.

    Returns:
        The hits and the pairs of consecutive present frames.
    """
    hits = pairs = 0
    for node, present in zip(nodes, presents):
        cen = world_centroids(node, E, present)
        for t in sorted(present):
            if t + 1 not in present:
                continue
            u, v, z = project_world_to_frame(cen[t][None], K[t + 1], E[t + 1])
            x0, y0, x1, y1 = node.box[t + 1]
            pairs += 1
            hits += bool(z[0] > 0 and x0 <= u[0] <= x1 and y0 <= v[0] <= y1)
    return hits, pairs


@dataclass(frozen=True)
class Settings:
    """The test's parameters, each an argument of the command."""

    k: int = 3
    h: int = 3
    margins: tuple[float, ...] = (0.0, 0.05, 0.1)
    min_present_px: int = 100
    shuffles: int = 5
    seed: int = 0
    tol: int = 1


def read_bundle(clip_dir: Path, geometry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The geometry source's depth, intrinsics and world-to-camera extrinsics, one of each per frame.

    Raises:
        ValueError: the three arrays do not hold one entry per frame.
    """
    with np.load(clip_dir / "exports" / "mini_npz" / BUNDLES[geometry]) as z:
        depth, K, E = (np.asarray(z[k], dtype=np.float64) for k in ("depth", "intrinsics", "extrinsics"))
    if not (len(depth) == len(K) == len(E)) or E.shape[1:] != (3, 4) or K.shape[1:] != (3, 3):
        raise ValueError(f"{clip_dir}: depth {depth.shape}, intrinsics {K.shape}, extrinsics {E.shape}")
    return depth, K, E


def _resized(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The bool mask at `shape`, nearest neighbour: the geometry source's depth may be of another size than DA3's."""
    if m.shape == shape:
        return m
    import cv2

    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _resized_ids(ids: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """An integer id map at `shape`, nearest neighbour."""
    if ids.shape == shape:
        return ids
    import cv2

    return cv2.resize(ids.astype(np.int32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def seed_classes(inputs: ClipInputs, table: ClassTable, seed: int) -> list[int]:
    """The class of each seed region, in the seed's order: one GT object per class on the seed frame, ascending."""
    if seed not in inputs.gt or frame_is_excluded(inputs.gt[seed], table):
        raise ValueError(f"{inputs.clip}: seed frame {seed} has no scored GT, so its regions are not GT objects")
    s = scored_pixels(inputs.gt[seed], table, VIEW, valid_depth(inputs.depth[seed]))
    return list(gt_objects(s.classes, s.mask).ids)


def gt_presence(inputs: ClipInputs, table: ClassTable, classes: Sequence[int], shape: tuple[int, int],
                min_px: int) -> tuple[list[int], list[dict[int, np.ndarray]]]:
    """The scored GT frames, and per node its GT mask on each GT frame where it has `min_px` pixels or more."""
    frames = [t for t in sorted(inputs.gt) if not frame_is_excluded(inputs.gt[t], table)]
    masks: list[dict[int, np.ndarray]] = [{} for _ in classes]
    for t in frames:
        s = scored_pixels(inputs.gt[t], table, VIEW, valid_depth(inputs.depth[t]))
        for n, c in enumerate(classes):
            m = _resized(s.mask & (s.classes == c), shape)
            if int(m.sum()) >= min_px:
                masks[n][t] = m
    return frames, masks


def gt_events_of(present: set[int], gt_frames: Sequence[int], seed: int) -> list[Event]:
    """The GT events: a change of presence between two GT frames that are consecutive sample frames, after the seed."""
    out = []
    for a, b in zip(gt_frames, gt_frames[1:]):
        if b != a + 1 or a < seed:
            continue
        if a in present and b not in present:
            out.append(Event("exit", -1, b))
        elif a not in present and b in present:
            out.append(Event("enter", -1, b))
    return out


def analyse_clip(dataset: str, clip: str, data_root: str, tracks_root: str, tag: str, geometry: str,
                 settings: Settings, gt_events: bool) -> dict:
    """Run every part of the test on one clip, and return its records.

    Raises:
        ValueError: the condition's regions are not the seed's (a region id outside 1..n or the count differs),
            the seed frame is not a GT frame, or the extrinsics read as given fail the convention check while
            read the other way they pass.
    """
    table = load_table(dataset)
    inputs = read_clip(data_root, tracks_root, tag, clip, table)
    info = json.loads((Path(tracks_root) / clip / tag / "seed_info.json").read_text())
    seed, n_nodes = int(info["seed_frame"]), int(info["n_seed_regions"])
    depth, K, E = read_bundle(Path(data_root) / clip, geometry)
    shape = depth.shape[1:]
    size = (shape[1], shape[0])
    frames = list(range(len(depth)))
    if inputs.order != tuple(frames):
        raise ValueError(f"{clip}: the frames are not in time order by index: {inputs.order[:8]}")
    # A node is a GT object of the seed frame; its region id is its place plus one. The tracking stage drops a
    # seed region under its minimum area and keeps the others' ids, so such a node has no region and no event.
    classes = seed_classes(inputs, table, seed)
    n_tracked, n_nodes = n_nodes, len(classes)
    if n_tracked > n_nodes:
        raise ValueError(f"{clip}: seed_info says {n_tracked} regions, the seed frame has {n_nodes} GT objects")
    ids = {int(v) for t in inputs.regions for v in np.unique(inputs.regions[t]) if v >= 0}
    if not ids <= set(range(1, n_nodes + 1)):
        raise ValueError(f"{clip}: region ids {sorted(ids - set(range(1, n_nodes + 1)))} are not seed regions 1..{n_nodes}")

    # Each node's masks reduced, its present frames and its events.
    nodes, presents, events = [], [], []
    for n in range(n_nodes):
        masks = {t: _resized(inputs.regions[t] == n + 1, shape) for t in inputs.regions}
        node = reduce_masks(masks, depth, K)
        present = {t for t, a in node.area.items() if a >= settings.min_present_px}
        nodes.append(node)
        presents.append(present)
        events.extend(Event(e.kind, n, e.t) for e in events_of(present, frames))

    # The pose convention: as stored, and read the other way.
    hits, pairs = convention_share(nodes, presents, K, E)
    hits_inv, _ = convention_share(nodes, presents, K, inverted(E))
    share = hits / pairs if pairs else None
    share_inv = hits_inv / pairs if pairs else None
    if share is not None and share < CONVENTION_MIN <= share_inv:
        raise ValueError(f"{clip}: the extrinsics read as world-to-camera land {share:.2f} of projections in the next "
                         f"box, read the other way {share_inv:.2f}; the bundle's convention is not the one assumed")

    # The pose sets: the poses, the still camera, and the shuffles.
    rng = np.random.default_rng([settings.seed, int.from_bytes(hashlib.blake2b(clip.encode(), digest_size=4).digest(), "big")])
    pose_sets = {WITH_POSES: E, STATIC: static_poses(E, seed)}
    for i in range(settings.shuffles):
        pose_sets[f"shuffle_{i}"] = shuffled_poses(E, rng)

    # The GT events of each node, and the predicted events' matches.
    gt_frames: list[int] = []
    gt_masks: list[dict[int, np.ndarray]] = [{} for _ in classes]
    gt_evts: list[Event] = []
    matches: dict[int, int] = {}
    if gt_events:
        gt_frames, gt_masks = gt_presence(inputs, table, classes, shape, settings.min_present_px)
        for n in range(n_nodes):
            gt_evts.extend(Event(e.kind, n, e.t) for e in gt_events_of(set(gt_masks[n]), gt_frames, seed))
        matches = match_events(events, gt_evts, settings.tol)
    gt_set = set(gt_frames)
    # The region and GT id maps at the geometry's size, read under the projected point.
    regions_at = {t: _resized_ids(inputs.regions[t], shape) for t in inputs.regions}
    gt_at = {t: _resized_ids(inputs.gt[t], shape) for t in gt_set}

    def verifiable(t: int) -> bool:
        return t in gt_set and t - 1 in gt_set

    # Every predicted event, under every pose set and margin.
    rows = []
    for i, e in enumerate(events):
        node = nodes[e.node]
        for name, P in pose_sets.items():
            cen = world_centroids(node, P, presents[e.node])
            p = known_position(cen, e.t, settings.k, e.kind)
            decisive = e.t if e.kind == "exit" else e.t - 1
            u, v, z = project_world_to_frame(p[None], K[decisive], P[decisive])
            under = int(regions_at[decisive].reshape(-1)[_pixel(u[0], v[0], shape)]) if np.isfinite(u[0] + v[0]) else -1
            gt_under = (int(gt_at[decisive].reshape(-1)[_pixel(u[0], v[0], shape)])
                        if decisive in gt_set and np.isfinite(u[0] + v[0]) else None)
            for m in settings.margins:
                label, truncated = classify(e.kind, p, e.t, frames, K, P, size, m, settings.h)
                rows.append({
                    "clip": clip, "video": video_of(clip), "kind": e.kind, "node": e.node,
                    "class_id": classes[e.node], "t": e.t, "pose_set": name, "margin": m, "label": label,
                    "truncated": truncated, "verifiable": verifiable(e.t),
                    "gt_t": gt_evts[matches[i]].t if i in matches else None,
                    "u": round(float(u[0]), 1), "v": round(float(v[0]), 1), "z": round(float(z[0]), 4),
                    "region_under": under, "gt_class_under": gt_under,
                    "area_last": node.area[max(f for f in presents[e.node] if f < e.t)] if e.kind == "exit"
                    else node.area[e.t],
                })

    # Every GT event, sorted by the same test on the GT masks' centroids, under the poses.
    gt_rows = []
    matched_gt = set(matches.values())
    for j, g in enumerate(gt_evts):
        node = reduce_masks(gt_masks[g.node], depth, K)
        cen = world_centroids(node, E, set(node.area))
        p = known_position(cen, g.t, settings.k, g.kind)
        for m in settings.margins:
            label, truncated = classify(g.kind, p, g.t, frames, K, E, size, m, settings.h)
            gt_rows.append({"clip": clip, "video": video_of(clip), "kind": g.kind, "node": g.node,
                            "class_id": classes[g.node], "t": g.t, "margin": m, "label": label,
                            "truncated": truncated, "matched": j in matched_gt})

    return {
        "clip": clip, "n_nodes": n_nodes, "n_tracked": n_tracked, "n_frames": len(frames), "seed": seed,
        "classes": classes,
        "n_gt_frames": len(gt_frames), "convention": {"w2c": share, "c2w": share_inv, "pairs": pairs},
        "events": rows, "gt_events": gt_rows,
    }


def _pixel(u: float, v: float, shape: tuple[int, int]) -> int:
    """The flat index of the pixel nearest to (u, v), clamped to the frame."""
    H, W = shape
    return int(np.clip(round(v), 0, H - 1)) * W + int(np.clip(round(u), 0, W - 1))


def _ci(values: Sequence[float], groups: Sequence[str]) -> list[float] | None:
    """The video-level bootstrap interval of a mean, None below two videos or without values."""
    if not values or len(set(groups)) < 2:
        return None
    lo, hi = boot_ci(np.asarray(values, dtype=np.float64), np.asarray(groups))
    return [lo, hi]


def shares(rows: Sequence[dict], kind: str, pose_set: str, margin: float) -> dict:
    """The share of each label among the events of one kind under one pose set and margin, with intervals."""
    sel = [r for r in rows if r["kind"] == kind and r["pose_set"] == pose_set and r["margin"] == margin]
    out = {"n": len(sel), "n_truncated": sum(r["truncated"] for r in sel)}
    groups = [r["video"] for r in sel]
    for label in LABELS[kind]:
        d = [float(r["label"] == label) for r in sel]
        out[label] = {"share": float(np.mean(d)) if d else None, "ci95_video": _ci(d, groups)}
    return out


def precision_recall(rows: Sequence[dict], gt_rows: Sequence[dict], kind: str, margin: float,
                     pose_set: str = WITH_POSES) -> dict:
    """The event precision and recall before and after the test, on the verifiable events under one pose set.

    `before` takes every predicted event, `after` the true events only (`left_view` or `entered_view`), and
    `after_plus_window` those and the window's events as well. `tp_removed` counts the matched events the test
    called losses. `gt_true_only` restricts the GT side to the GT events the same test calls true under the
    poses. A control pose set changes which predicted events `after` keeps, and nothing else.
    """
    still, true_, window = LABELS[kind]
    pred = [r for r in rows if r["kind"] == kind and r["pose_set"] == pose_set and r["margin"] == margin
            and r["verifiable"]]
    gt = [r for r in gt_rows if r["kind"] == kind and r["margin"] == margin]
    gt_true = {(r["clip"], r["node"], r["t"]) for r in gt if r["label"] == true_}

    def table(kept: Sequence[dict], gt_keys: set | None) -> dict:
        tp = [r for r in kept if r["gt_t"] is not None and (gt_keys is None or (r["clip"], r["node"], r["gt_t"]) in gt_keys)]
        gt_n = len(gt) if gt_keys is None else len(gt_keys)
        hit = [float(r["gt_t"] is not None and (gt_keys is None or (r["clip"], r["node"], r["gt_t"]) in gt_keys))
               for r in kept]
        gt_sel = gt if gt_keys is None else [r for r in gt if (r["clip"], r["node"], r["t"]) in gt_keys]
        kept_keys = {(r["clip"], r["node"], r["gt_t"]) for r in tp}
        rec = [float((r["clip"], r["node"], r["t"]) in kept_keys) for r in gt_sel]
        return {"n_pred": len(kept), "n_gt": gt_n, "tp": len(tp),
                "precision": len(tp) / len(kept) if kept else None,
                "precision_ci95_video": _ci(hit, [r["video"] for r in kept]),
                "recall": len(tp) / gt_n if gt_n else None,
                "recall_ci95_video": _ci(rec, [r["video"] for r in gt_sel])}

    by = {label: [r for r in pred if r["label"] == label] for label in LABELS[kind]}
    return {
        "before": table(pred, None),
        "after": table(by[true_], None),
        "after_plus_window": table(by[true_] + by[window], None),
        "tp_removed": sum(r["gt_t"] is not None for r in by[still]),
        "gt_labels": {label: sum(r["label"] == label for r in gt) for label in LABELS[kind]},
        "gt_true_only": {"before": table(pred, gt_true), "after": table(by[true_], gt_true),
                         "after_plus_window": table(by[true_] + by[window], gt_true)},
    }


def summarise(results: Sequence[dict], settings: Settings) -> dict:
    """Aggregate every clip's records: the shares under each pose set and margin, the precision tables, the checks."""
    rows = [r for res in results for r in res["events"]]
    gt_rows = [r for res in results for r in res["gt_events"]]
    pose_sets = sorted({r["pose_set"] for r in rows}, key=lambda s: (s not in (WITH_POSES, STATIC), s))
    pairs = sum(res["convention"]["pairs"] for res in results)
    hits = {key: sum((res["convention"][key] or 0) * res["convention"]["pairs"] for res in results) / pairs if pairs else None
            for key in ("w2c", "c2w")}
    out = {
        "settings": asdict(settings),
        "clips": len(results), "videos": len({video_of(res["clip"]) for res in results}),
        "nodes": sum(res["n_nodes"] for res in results), "nodes_tracked": sum(res["n_tracked"] for res in results),
        "events": {kind: sum(r["kind"] == kind for r in rows if r["pose_set"] == WITH_POSES and r["margin"] == settings.margins[0])
                   for kind in KINDS},
        "convention_check": {"share_in_next_box": hits, "pairs": pairs, "min": CONVENTION_MIN, "read_as": "w2c"},
        "shares": {kind: {name: {str(m): shares(rows, kind, name, m) for m in settings.margins} for name in pose_sets}
                   for kind in KINDS},
    }
    if gt_rows or any(res["n_gt_frames"] for res in results):
        out["gt_events"] = {kind: sum(r["kind"] == kind for r in gt_rows if r["margin"] == settings.margins[0])
                            for kind in KINDS}
        out["precision_recall"] = {kind: {str(m): precision_recall(rows, gt_rows, kind, m) for m in settings.margins}
                                   for kind in KINDS}
        out["precision_recall_controls"] = {
            name: {kind: {str(m): precision_recall(rows, gt_rows, kind, m, name) for m in settings.margins}
                   for kind in KINDS} for name in pose_sets if name != WITH_POSES}
    return out


def per_clip_rows(results: Sequence[dict]) -> list[dict]:
    """One row per clip, pose set, margin and kind: the events and how many took each label."""
    out = []
    for res in results:
        keys = sorted({(r["pose_set"], r["margin"], r["kind"]) for r in res["events"]})
        for name, m, kind in keys:
            sel = [r for r in res["events"] if (r["pose_set"], r["margin"], r["kind"]) == (name, m, kind)]
            row = {"clip": res["clip"], "video": video_of(res["clip"]), "pose_set": name, "margin": m, "kind": kind,
                   "n": len(sel), "n_truncated": sum(r["truncated"] for r in sel)}
            # The three labels of a kind, under one name each so that exits and enters share the columns.
            for column, label in zip(("n_still_in_view", "n_out_of_view", "n_out_in_window"), LABELS[kind]):
                row[column] = sum(r["label"] == label for r in sel)
            out.append(row)
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _job(args: tuple) -> dict:
    return analyse_clip(*args)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tracks-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", choices=sorted(BUNDLES), default="da3")
    ap.add_argument("--gt-events", action="store_true", help="match the events to GT events on consecutive GT frames")
    ap.add_argument("--k", type=int, default=Settings.k)
    ap.add_argument("--h", type=int, default=Settings.h)
    ap.add_argument("--margins", default=",".join(str(m) for m in Settings.margins))
    ap.add_argument("--min-present-px", type=int, default=Settings.min_present_px)
    ap.add_argument("--shuffles", type=int, default=Settings.shuffles)
    ap.add_argument("--seed", type=int, default=Settings.seed)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summarise-only", action="store_true",
                    help="read `clip_results.json` under --out instead of the clips, and write the summary again")
    a = ap.parse_args(argv)
    settings = Settings(k=a.k, h=a.h, margins=tuple(float(m) for m in a.margins.split(",")),
                        min_present_px=a.min_present_px, shuffles=a.shuffles, seed=a.seed)
    out = Path(a.out)
    if a.summarise_only:
        results = json.loads((out / "clip_results.json").read_text())
    else:
        clips = read_population(a.clips)
        jobs = [(a.dataset, c, a.data_root, a.tracks_root, a.tag, a.geometry, settings, a.gt_events) for c in clips]
        with Pool(a.workers) as pool:
            results = pool.map(_job, jobs, chunksize=1)
        out.mkdir(parents=True, exist_ok=True)
        (out / "clip_results.json").write_text(json.dumps(results))
    summary = summarise(results, settings)
    summary.update({"dataset": a.dataset, "tag": a.tag, "geometry": a.geometry, "tool_sha": tool_sha(__file__),
                    "clips_skipped": []})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    _write_csv(out / "per_clip.csv", per_clip_rows(results))
    _write_csv(out / "events.csv", [r for res in results for r in res["events"]])
    _write_csv(out / "gt_events.csv", [r for res in results for r in res["gt_events"]])
    print(json.dumps({k: summary[k] for k in ("clips", "videos", "nodes", "events", "convention_check")}, indent=1))


if __name__ == "__main__":
    main()
