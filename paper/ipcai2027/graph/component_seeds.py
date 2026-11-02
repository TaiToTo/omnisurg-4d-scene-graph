"""Write each clip's seed regions, one node per reference class or per connected component, and the components' tracks.

A class node is one region per reference class on the seed frame. A component node is one 8-connected component of
a class, of `--min-area` pixels or more, since the tracking stage drops a smaller seed region. A component node's
reference track follows, frame to frame, the component of its class that overlaps its last mask most; a component
that two nodes claim goes to the larger overlap. Each map is `label_<frame:04d>.npy` of node indices, -1 outside
every node, at the depth's shape, with the nodes' classes in `nodes.json`. `pipeline.track --seed-labels` reads the
seeds, and `evalkit.tools.exit_vs_loss --gt-tracks` reads the tracks.

Usage:
    python paper/ipcai2027/graph/component_seeds.py --dataset cholecseg8k --clips LIST --data-root DIR --out DIR \\
        [--min-area 400]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from evalkit.classes import ClassTable, load_table
from evalkit.evaluate import read_population
from evalkit.fill import fill_from_nearest
from evalkit.inputs import DEPTH_FILE, GT_FLAG, MANIFEST, MASK_DIR, MASK_SUFFIX
from evalkit.objects import gt_objects
from evalkit.scored import Scored, frame_is_excluded, scored_pixels, valid_depth

VIEW = "all"
SEED_FRAME = 0
EIGHT = np.ones((3, 3), dtype=int)
SEEDS_CLASS, SEEDS_COMPONENT, TRACKS_COMPONENT = "seeds_class", "seeds_component", "gt_tracks_component"


def reference_ids(path: Path, table: ClassTable, shape: tuple[int, int]) -> np.ndarray:
    """Return a reference mask's ids at `shape`, its `filled` classes filled first, as `evalkit.inputs.read_clip` reads it."""
    with Image.open(path) as image:
        ids, _ = fill_from_nearest(table.mask_ids(image), table.filled_mask_ids)
    if ids.shape == shape:
        return ids
    return cv2.resize(ids, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def read_reference(clip_dir: Path, table: ClassTable) -> tuple[tuple[int, int], dict[int, Scored]]:
    """Return the depth's shape and, per reference frame that is not excluded, its scored pixels and classes."""
    manifest = json.loads((clip_dir / MANIFEST).read_text())
    with np.load(clip_dir / DEPTH_FILE) as z:
        depth = np.asarray(z["depth"], dtype=np.float32)
    shape = depth.shape[1:]
    out = {}
    for i, f in enumerate(manifest["frames"]):
        p = clip_dir / MASK_DIR / f"{i:06d}{MASK_SUFFIX[table.dataset]}"
        if not f.get(GT_FLAG[table.dataset]) or not p.is_file():
            continue
        ids = reference_ids(p, table, shape)
        if not frame_is_excluded(ids, table):
            out[i] = scored_pixels(ids, table, VIEW, valid_depth(depth[i]))
    return shape, out


def components(s: Scored, class_id: int, min_area: int) -> list[np.ndarray]:
    """Return the 8-connected components of one class's scored pixels of `min_area` or more, in label order."""
    lab, n = ndimage.label(s.mask & (s.classes == class_id), structure=EIGHT)
    return [c for c in (lab == k for k in range(1, n + 1)) if int(c.sum()) >= min_area]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return the intersection over union of two bool masks."""
    inter = int((a & b).sum())
    return inter / int((a | b).sum()) if inter else 0.0


def class_seed(s0: Scored, shape: tuple[int, int]) -> tuple[list[int], np.ndarray]:
    """Return the class of each class node and the seed frame's map: one node per class, ascending."""
    classes = list(gt_objects(s0.classes, s0.mask).ids)
    m = np.full(shape, -1, dtype=np.int32)
    for n, c in enumerate(classes):
        m[s0.mask & (s0.classes == c)] = n
    return classes, m


def component_tracks(frames: dict[int, Scored], shape: tuple[int, int],
                     min_area: int) -> tuple[list[int], list[int], dict[int, np.ndarray]]:
    """Return the class and seed area of each component node, and its map on every reference frame."""
    # The nodes: the seed frame's components, class by class ascending.
    s0 = frames[SEED_FRAME]
    classes, areas, last = [], [], []
    for c in sorted(int(v) for v in np.unique(s0.classes[s0.mask])):
        for comp in components(s0, c, min_area):
            classes.append(c)
            areas.append(int(comp.sum()))
            last.append(comp)
    tracks = {}
    for t in sorted(frames):
        m = np.full(shape, -1, dtype=np.int32)
        if t == SEED_FRAME:
            for n, comp in enumerate(last):
                m[comp] = n
            tracks[t] = m
            continue
        # Each node claims the component of its class it overlaps most; a contested component goes to the larger
        # overlap, and a node that loses it, or overlaps nothing, is absent on this frame.
        comps = {c: components(frames[t], c, 1) for c in set(classes)}
        claims: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for n, c in enumerate(classes):
            best = max(((iou(last[n], comp), k) for k, comp in enumerate(comps[c])), default=(0.0, -1))
            if best[0] > 0:
                claims.setdefault((c, best[1]), []).append((best[0], n))
        for (c, k), claimants in claims.items():
            _, n = max(claimants)
            m[comps[c][k]] = n
            last[n] = comps[c][k]
        tracks[t] = m
    return classes, areas, tracks


def write(dst: Path, maps: dict[int, np.ndarray], nodes: dict) -> None:
    """Write one clip's maps and its `nodes.json`."""
    dst.mkdir(parents=True, exist_ok=True)
    for t, m in maps.items():
        np.save(dst / f"label_{t:04d}.npy", m)
    (dst / "nodes.json").write_text(json.dumps(nodes, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, type=Path,
                    help=f"writes {SEEDS_CLASS}/, {SEEDS_COMPONENT}/ and {TRACKS_COMPONENT}/ under it")
    ap.add_argument("--min-area", type=int, default=400)
    a = ap.parse_args()
    table = load_table(a.dataset)
    for clip in read_population(a.clips):
        shape, frames = read_reference(Path(a.data_root) / clip, table)
        if SEED_FRAME not in frames:
            raise ValueError(f"{clip}: frame {SEED_FRAME} has no scored reference mask")
        classes, seed = class_seed(frames[SEED_FRAME], shape)
        write(a.out / SEEDS_CLASS / clip, {SEED_FRAME: seed}, {"seed_frame": SEED_FRAME, "class_ids": classes})
        comp_classes, areas, tracks = component_tracks(frames, shape, a.min_area)
        nodes = {"seed_frame": SEED_FRAME, "class_ids": comp_classes, "areas": areas, "min_area": a.min_area}
        write(a.out / SEEDS_COMPONENT / clip, {SEED_FRAME: tracks[SEED_FRAME]}, nodes)
        write(a.out / TRACKS_COMPONENT / clip, tracks, nodes)
        print(f"{clip}: {len(classes)} classes, {len(comp_classes)} components, {len(frames)} reference frames",
              flush=True)


if __name__ == "__main__":
    main()
