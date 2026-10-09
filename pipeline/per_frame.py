"""Run the per-frame segmentation stage: cut every frame of each clip into regions, with nothing carried between frames.

The stage makes the tracking stage's seed on every frame. SAM's automatic
mask generator cuts each frame, prompted with one of the segmenter inputs,
and the regions under `--seed-min-area` pixels are dropped. Region `r` is
written as `r + 1`, the id the tracker would give it; a pixel in no region
is written as -1. Ids do not follow an object from frame to frame, so a
measure over time means nothing here. The stage writes a clip's labels,
`label_NNNN.npy`, and `seed_info.json` to
`<tracks-root>/<clip>/track_rgb_<tag>/`, a directory named as the tracking
stage's are.

Usage:
    python -m pipeline.per_frame --input-dir /path/to/clips --tracks-root /path/to/tracks --tag <tag> \\
        --sam-input normal_edge --sam-ckpt sam_vit_h_4b8939.pth [--clips <clip> ...] [--points-per-side 24] \\
        [--edge-gain 1.0] [--depth-source pi3] [--keep-edge-ring] [--device auto] [--gpu N] [--overwrite]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from pipeline.track import (BUNDLE, DEPTH_SOURCES, SEED_SAM_KWARGS, SeedSegmenter, read_clip, read_rgb,
                            seed_masks)
from surgical_core.geometry.render import SAM_INPUT_MODES, global_depth01, sam_input_image, uses_geom_edge

# The per-frame conditions' normal images are not smoothed, unlike the tracker's seed.
SMOOTH = False


def run_per_frame(clip_dir: Path, tracks_root: Path, tag: str, segmenter, sam_input: str,
                  depth_source: str = "da3", points_per_side: int = 24, seed_min_area: int = 400,
                  edge_gain: float = 1.0, mask_ring: bool = True, overwrite: bool = False) -> Path:
    """Cut every frame of one clip and return the directory of its labels.

    Args:
        clip_dir: the clip.
        tracks_root: where each clip's conditions go.
        tag: the condition's name; its directory is `track_rgb_<tag>`.
        segmenter: `pipeline.track.SeedSegmenter`, or a stand-in with its `label_map`.
        sam_input: the input mode, of `SAM_INPUT_MODES`.
        depth_source: one of `DEPTH_SOURCES`.
        points_per_side: the mask generator's grid.
        seed_min_area: the fewest pixels a region keeps.
        edge_gain: how dark the burnt-in edges are.
        mask_ring: zero the ring of the burnt-in edges.
        overwrite: replace the condition's labels when the clip already has them. They are replaced only once
            the run has written every file, so a run that fails leaves them as they were.

    Raises:
        FileNotFoundError: a bundle or an image is missing.
        ValueError: one of these:
            - the mode or the depth source is unknown
            - the labels exist and `overwrite` is false
            - the labels exist and have no `seed_info.json` that records `seed_source` `per_frame`
            - Pi3X's depth has another number of frames than DA3's
    """
    # Check the settings, and refuse labels an earlier run left.
    uses_geom_edge(sam_input)
    if depth_source not in DEPTH_SOURCES:
        raise ValueError(f"unknown depth source {depth_source!r}; one of {DEPTH_SOURCES}")
    lab_dir = tracks_root / clip_dir.name / f"track_rgb_{tag}"
    if lab_dir.exists():
        if not overwrite:
            raise ValueError(f"{clip_dir.name} already has {lab_dir.name}; pass --overwrite to replace it")
        # Refuse to replace labels this stage did not write: a tag the tracking stage used would lose its condition.
        info = lab_dir / "seed_info.json"
        if not info.is_file() or json.loads(info.read_text(encoding="utf-8")).get("seed_source") != "per_frame":
            raise ValueError(f"{clip_dir.name}'s {lab_dir.name} has no seed_info.json with seed_source per_frame, "
                             "so this stage did not write it; it is not replaced")

    # Read the depth.
    depth, K = read_clip(clip_dir, depth_source)
    n, h, w = depth.shape
    depth01 = global_depth01(depth)

    # Cut each frame, keep the regions large enough, and write them under the tracker's ids, with the record, to a
    # partial directory. An earlier run's labels are replaced only after every file is written, so a run that fails
    # keeps them.
    partial = tracks_root / clip_dir.name / f".{lab_dir.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        for i in range(n):
            image = sam_input_image(sam_input, depth[i], K[i], depth01[i], read_rgb(clip_dir, i, (h, w)),
                                    edge_gain=edge_gain, smooth=SMOOTH, mask_ring=mask_ring)
            masks, obj_ids = seed_masks(segmenter.label_map(image, points_per_side), seed_min_area)
            labels = np.full((h, w), -1, dtype=np.int16)
            for m, oid in zip(masks, obj_ids):
                labels[m] = oid
            np.save(partial / f"label_{i:04d}.npy", labels)
        # The keys and their order are the workbench's, so that a record compares with one written there;
        # `seed_topk` and `point_grids` name options the workbench had and this stage leaves out.
        seed_input = dict(produced_by="sam", cache_path=None, points_per_side=points_per_side,
                          seed_sam_kwargs=SEED_SAM_KWARGS, seed_edge_gain=edge_gain, seed_smooth=SMOOTH,
                          edge_ring_masked=mask_ring if uses_geom_edge(sam_input) else None)
        with open(partial / "seed_info.json", "w", encoding="utf-8") as f:
            json.dump(dict(clip=clip_dir.name, tag=tag, seed_source="per_frame", track_base="rgb",
                           sam_input=sam_input, frames="all", depth_source=depth_source, seed_min_area=seed_min_area,
                           seed_topk=0, point_grids=None, seed_input=seed_input), f, indent=1, ensure_ascii=False)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    # Replace the earlier run's labels with the new ones.
    if lab_dir.exists():
        shutil.rmtree(lab_dir)
    partial.rename(lab_dir)
    print(f"  {n} label maps to {lab_dir}")
    return lab_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip with depth.")
    ap.add_argument("--tracks-root", required=True, help="Where each clip's conditions go.")
    ap.add_argument("--tag", required=True, help="The condition's name.")
    ap.add_argument("--sam-input", required=True, choices=SAM_INPUT_MODES, help="The input mode.")
    ap.add_argument("--sam-ckpt", required=True, help="The SAM ViT-H weights, sam_vit_h_4b8939.pth.")
    ap.add_argument("--depth-source", default="da3", choices=DEPTH_SOURCES,
                    help="The depth the normals and edges come from.")
    ap.add_argument("--points-per-side", type=int, default=24, help="The mask generator's grid.")
    ap.add_argument("--seed-min-area", type=int, default=400, help="The fewest pixels a region keeps.")
    ap.add_argument("--edge-gain", type=float, default=1.0, help="How dark the burnt-in edges are.")
    ap.add_argument("--keep-edge-ring", action="store_true",
                    help="Keep the edges' ring along the image border and around invalid depth.")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu. auto takes CUDA when there is one.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="The index of the GPU to use. Default: what CUDA_VISIBLE_DEVICES says, or the first GPU.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a condition's labels an earlier run left.")
    args = ap.parse_args()

    # Find the clips before the model loads: its weights take 2.5 GB, and a mistyped clip name should fail first.
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    clips = ([root / c for c in args.clips] if args.clips
             else sorted(d for d in root.iterdir() if (d / BUNDLE).is_file()))
    missing = [c.name for c in clips if not (c / BUNDLE).is_file()]
    if missing:
        raise SystemExit(f"no clip with depth ({BUNDLE}) at: {', '.join(missing)}")
    if not clips:
        raise SystemExit(f"no clip with depth under {root}")

    # Select the GPU before the segmenter imports torch: CUDA_VISIBLE_DEVICES takes effect only before torch's first
    # import.
    if args.gpu is not None:
        if "torch" in sys.modules:
            raise SystemExit("torch is already imported, so CUDA_VISIBLE_DEVICES cannot select the GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    segmenter = SeedSegmenter(args.sam_ckpt, args.device)
    print(f"SAM {args.sam_input} on {segmenter.device}, {len(clips)} clip(s)")

    # Run every clip, even after one fails, and list the failures at the end.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            run_per_frame(clip_dir, Path(args.tracks_root), args.tag, segmenter, args.sam_input,
                          depth_source=args.depth_source, points_per_side=args.points_per_side,
                          seed_min_area=args.seed_min_area, edge_gain=args.edge_gain,
                          mask_ring=not args.keep_edge_ring, overwrite=args.overwrite)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
