"""Count a condition's regions per frame without reading the GT: the axis along which inputs are compared.

A frame's region count is the number of region ids its label map holds,
every pixel read and every region counted whatever its size. A clip's count
is the mean over its frames with a prediction, and a condition's the mean
over its clips. No view, GT or depth enters it, so it is the same for every
key it is drawn against. The JSON records each clip's count and the hash of
this file.

Usage:
    python -m evalkit.tools.region_counts --clips atlas120k_meta/clips.txt --tracks-root <dir of predictions> \\
        --tag <label directory> --out <condition>.regions.json
"""
from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evalkit.evaluate import read_population, write_scores
from evalkit.tools.component_scores import tool_sha

# A prediction's file name, as the stages write it and `evalkit.inputs` reads it.
LABEL = re.compile(r"label_(\d+)\.npy")


def frame_count(labels: np.ndarray) -> int:
    """Return the number of region ids a label map holds; -1 is no region.

    Raises:
        ValueError: The map is not an (H, W) integer array, or holds an id below -1.
    """
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"a label map is an (H, W) integer array, got {labels.dtype} of shape {labels.shape}")
    if labels.min(initial=0) < -1:
        raise ValueError("a label map holds region ids >= 0 and -1 for no region")
    return int(np.unique(labels[labels >= 0]).size)


def count_clip(tracks_root: str | Path, tag: str, clip: str) -> dict:
    """Return one clip's mean region count over its frames with a prediction, and the number of those frames.

    Raises:
        FileNotFoundError: The clip has no directory under `tag`, or no prediction in it.
        ValueError: A file is not named `label_<index>.npy`, or a map is refused (`frame_count`).
    """
    paths = sorted((Path(tracks_root) / clip / tag).glob("label_*.npy"))
    if not paths:
        raise FileNotFoundError(f"{clip}: no prediction under {Path(tracks_root) / clip / tag}")
    bad = [p.name for p in paths if LABEL.fullmatch(p.name) is None]
    if bad:
        raise ValueError(f"{clip}: {bad} are not label_<index>.npy")
    counts = [frame_count(np.load(p)) for p in paths]
    return {"clip": clip, "regions_per_frame": float(np.mean(counts)), "n_frames": len(counts)}


def count_condition(clips: Sequence[str], tracks_root: str | Path, tag: str) -> dict:
    """Count every clip of a condition, and return its JSON as a dict."""
    # The hash before any clip is counted, so that a source file edited during the run cannot relabel the values.
    tool = tool_sha(__file__)
    rows = [count_clip(tracks_root, tag, clip) for clip in clips]
    return {"tool_sha": tool, "track_dir_name": tag, "clips": list(clips),
            "regions_per_frame": float(np.mean([r["regions_per_frame"] for r in rows])), "per_clip": rows}


def main(argv: Sequence[str] | None = None) -> None:
    """Count the condition the arguments name, and write its JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", required=True, help="the population: one clip name per line")
    ap.add_argument("--tracks-root", required=True, help="one directory per clip, one per condition inside")
    ap.add_argument("--tag", required=True, help="the condition's label directory under each clip")
    ap.add_argument("--out", required=True, help="the JSON to write")
    args = ap.parse_args(argv)
    summary = count_condition(read_population(args.clips), args.tracks_root, args.tag)
    write_scores(summary, args.out)
    print(f"{args.tag}: {summary['regions_per_frame']:.2f} regions per frame over {len(summary['clips'])} clips")


if __name__ == "__main__":
    main()
