"""Merge each frame of a condition's predictions down to K regions, and write them as a condition of their own.

Conditions that cut a frame into different numbers of regions are not
compared on instance metrics as they are: more regions lose on them
whatever their boundaries. Merged to one K, they are. The rule was fixed
before any result: the region of least area is absorbed into the
neighbour it shares the longest boundary with, until K are left. It reads
areas and adjacency only, so it favours no input. The merged predictions
are scored by the evaluator like any other condition, and compared with
`compare_eval`.

Usage:
    python -m evalkit.tools.kmerge --dataset atlas120k --clips atlas120k_meta/clips.txt \\
        --data-root /path/to/clips --tracks-root /path/to/tracks --tag <tag> [--k 10] [--out-tag <tag>_k10]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from evalkit.classes import load_table
from evalkit.evaluate import SEED_INFO, read_population
from evalkit.inputs import read_clip

RECORD = "kmerge.json"

# The K the paper's granularity result merges to.
DEFAULT_K = 10


def fill_seams(labels: np.ndarray) -> np.ndarray:
    """Give each unlabelled pixel the label of the nearest labelled one, for measuring adjacency only.

    Tracked regions are often parted by a seam of unlabelled pixels one wide, and across it two regions that
    touch are not 4-neighbours. Without the fill a region can look isolated and stop the merge. The merged map
    keeps its unlabelled pixels.
    """
    known = labels >= 0
    if not known.any() or known.all():
        return labels
    _, idx = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    out = labels.copy()
    out[~known] = labels[ys[idx[~known] - 1], xs[idx[~known] - 1]]
    return out


def region_adjacency(labels: np.ndarray) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    """Return each region's area, and the length of the boundary it shares with each neighbour.

    Areas count the labelled pixels; adjacency is measured on `fill_seams`' map.
    """
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    areas = {int(i): int(c) for i, c in zip(ids, counts)}
    adj: dict[int, dict[int, int]] = {int(i): {} for i in ids}
    filled = fill_seams(labels.astype(np.int64))
    for a, b in ((filled[:, :-1], filled[:, 1:]), (filled[:-1, :], filled[1:, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        if not m.any():
            continue
        u, v = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        keys, counts = np.unique((u << 32) | v, return_counts=True)
        for key, c in zip(keys.tolist(), counts.tolist()):
            x, y = key >> 32, key & 0xFFFFFFFF
            adj[x][y] = adj[x].get(y, 0) + c
            adj[y][x] = adj[y].get(x, 0) + c
    return areas, adj


def kmerge(labels: np.ndarray, k: int) -> np.ndarray:
    """Merge the regions of one frame down to `k`, and return the merged map.

    The region of least area is absorbed into the neighbour it shares the longest boundary with; ties go to the
    lower id, then to the smaller neighbour. A merged region keeps the id of the region that absorbed it. A map
    of `k` regions or fewer is returned as it is.

    Args:
        labels: An (H, W) integer map, -1 for no region.
        k: The number of regions to merge down to, at least 1.

    Raises:
        ValueError: `k` is below 1.
        RuntimeError: a region has no neighbour. Once the seams are filled every pixel has a region, so each
            region touches another; one that does not means the adjacency was measured wrong.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    areas, adj = region_adjacency(labels)
    active = set(areas)
    parent = {r: r for r in active}
    while len(active) > k:
        r = min(active, key=lambda x: (areas[x], x))
        nb = {n: w for n, w in adj[r].items() if n in active}
        if not nb:
            raise RuntimeError(f"region {r} has no neighbour among {len(active)} regions")
        t = max(nb, key=lambda n: (nb[n], -areas[n]))
        areas[t] += areas[r]
        for n, w in adj[r].items():
            if n != t and n in active:
                adj[t][n] = adj[t].get(n, 0) + w
                adj[n][t] = adj[n].get(t, 0) + w
                adj[n].pop(r, None)
        adj[t].pop(r, None)
        active.discard(r)
        parent[r] = t

    def root(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    out = labels.copy()
    for r in areas:
        if root(r) != r:
            out[labels == r] = root(r)
    return out


def merge_condition(dataset: str, clips: Sequence[str], data_root: str | Path, tracks_root: str | Path, tag: str,
                    out_tag: str, k: int = DEFAULT_K, overwrite: bool = False) -> None:
    """Merge every frame of a condition and write it under `out_tag`.

    Each prediction is read as the evaluator reads it, at the shape of the clip's depth, and the merged map is
    written at that shape, under the source file's name. The clip's `seed_info.json`, if any, is copied with
    `kmerge` added, so that the merged condition keeps its propagation rule; `kmerge.json` records the source
    and `k` in any case.

    Raises:
        ValueError: `out_tag` is `tag`, or a clip already has `out_tag` and `overwrite` is false; or
            `read_clip` refuses a clip.
        FileNotFoundError: `read_clip` finds an input missing.
    """
    if out_tag == tag:
        raise ValueError("the merged condition needs a tag of its own")
    tracks_root = Path(tracks_root)
    table = load_table(dataset)

    # Refuse merged predictions an earlier run left before anything is written.
    held = [c for c in clips if (tracks_root / c / out_tag).exists()]
    if held and not overwrite:
        raise ValueError(f"{out_tag} already exists for {held}; pass --overwrite to replace it")

    for clip in clips:
        # Read the clip as the evaluator does, and find each prediction's file name.
        inputs = read_clip(data_root, tracks_root, tag, clip, table)
        src = tracks_root / clip / tag
        names = {int(p.stem.split("_")[1]): p.name for p in src.glob("label_*.npy")}

        # Merge each frame and write it, with the seed's record and this tool's.
        dst = tracks_root / clip / out_tag
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for i, regions in sorted(inputs.regions.items()):
            np.save(dst / names[i], kmerge(np.asarray(regions), k))
        if (src / SEED_INFO).is_file():
            info = json.loads((src / SEED_INFO).read_text(encoding="utf-8"))
            info["kmerge"] = {"from_tag": tag, "k": k}
            (dst / SEED_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")
        (dst / RECORD).write_text(json.dumps({"from_tag": tag, "k": k}, indent=2), encoding="utf-8")
        print(f"{clip}: {len(inputs.regions)} frames merged to {k} regions")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="cholecseg8k or atlas120k; its class table reads the GT.")
    ap.add_argument("--clips", required=True, help="The population file, one clip per line.")
    ap.add_argument("--data-root", required=True, help="The directory holding one directory per clip.")
    ap.add_argument("--tracks-root", required=True, help="The directory holding each clip's conditions.")
    ap.add_argument("--tag", required=True, help="The condition to merge.")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="The number of regions to merge down to.")
    ap.add_argument("--out-tag", help="The merged condition's tag. Default: <tag>_k<k>.")
    ap.add_argument("--overwrite", action="store_true", help="Replace merged predictions an earlier run left.")
    args = ap.parse_args(argv)
    try:
        merge_condition(args.dataset, read_population(args.clips), args.data_root, args.tracks_root, args.tag,
                        args.out_tag or f"{args.tag}_k{args.k}", args.k, args.overwrite)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
