"""Run the tracking stage: cut one frame of each clip into regions, and carry them to every other frame with SAM 3.

SAM's automatic mask generator cuts the seed frame, prompted with one of the
segmenter inputs (`--sam-input`); `--seed-labels` gives the regions instead.
Regions under `--seed-min-area` pixels are dropped. SAM 3's video tracker
carries each region, as one object, through the clip, which it sees as
`--track-base` images. Each frame's objects are painted into one label map,
the higher presence score on top. `both_ways_from_centre` seeds the middle
frame and carries both ways; `forward_from_first` seeds frame 0 and carries
forwards. The stage writes a clip's labels, `label_NNNN.npy`, and
`seed_info.json` to `<tracks-root>/<clip>/track_<track-base>_<tag>/`. It
writes a montage of the labels to `<tracks-root>/<clip>/viz/`.

Usage:
    python -m pipeline.track --input-dir /path/to/clips --tracks-root /path/to/tracks --tag <tag> \\
        --rule both_ways_from_centre --sam-input normal_edge --track-base rgb --sam-ckpt sam_vit_h_4b8939.pth \\
        [--clips <clip> ...] [--seed-labels DIR] [--depth-source pi3] [--keep-edge-ring] [--device auto] [--gpu N]
        [--overwrite]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam3_wrapper import DEFAULT_MODEL_ID, FrameResult, Sam3VideoInstanceSession
from surgical_core.geometry.render import SAM_INPUT_MODES, global_depth01, sam_input_image, uses_geom_edge

RULES = ("both_ways_from_centre", "forward_from_first")
DEPTH_SOURCES = ("da3", "pi3")
BUNDLE = Path("exports", "mini_npz", "results.npz")
PI3X_BUNDLE = Path("exports", "mini_npz", "results__pi3x.npz")

# The automatic mask generator's settings for the seed, other than `points_per_side`.
SEED_SAM_MODEL_TYPE = "vit_h"
SEED_SAM_KWARGS = dict(pred_iou_thresh=0.8, stability_score_thresh=0.8, min_mask_region_area=100)

# How dark the tracker's edges are, and whether its normals are smoothed: what every condition ran.
TRACK_EDGE_GAIN = 0.85
TRACK_SMOOTH = True

# The montage's tiles: a header bar above each frame, three frames to a row.
MONTAGE_BAR = 26
MONTAGE_COLS = 3


class SeedSegmenter:
    """Load SAM's automatic mask generator once.

    Args:
        checkpoint: the SAM ViT-H weights, `sam_vit_h_4b8939.pth`.
        device: "cpu" or "cuda", the tracker's device.
    """

    def __init__(self, checkpoint: str, device: str) -> None:
        # Imported here: the `track` extra provides it, and the stage's other parts run without it.
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self._generator = SamAutomaticMaskGenerator
        self._sam = sam_model_registry[SEED_SAM_MODEL_TYPE](checkpoint=checkpoint).to(device)

    def label_map(self, image: np.ndarray, points_per_side: int) -> np.ndarray:
        """Return the regions of an (H, W, 3) uint8 image as an (H, W) map of 0..K-1, -1 for none."""
        masks = self._generator(self._sam, points_per_side=points_per_side, **SEED_SAM_KWARGS).generate(image)
        return paint_masks([m["segmentation"] for m in masks], image.shape[:2])


def paint_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Paint the generator's masks into one map, the first mask on top, and number the regions 0..K-1.

    The generator lists its masks roughly by predicted IoU, highest first, so where masks overlap the more
    confident one wins, whatever the sizes. The last mask paints id 0 and the first the highest id; the ids
    that survive are then numbered 0..K-1 in their order, so the regions are numbered from the last mask up.
    """
    labels = np.full(shape, -1, dtype=int)
    for k, i in enumerate(reversed(range(len(masks)))):
        labels[masks[i]] = k
    ids = np.unique(labels[labels >= 0])
    if ids.size == 0:
        return labels
    lut = np.full(int(ids.max()) + 2, -1)
    lut[ids + 1] = np.arange(ids.size)
    return lut[labels + 1]


def read_seed_labels(seed_labels: str, clip: str, frame: int, shape: tuple[int, int]) -> np.ndarray:
    """Return the seed regions made outside the stage, numbered 0..K-1 in the order of their ids.

    The file is `label_<frame:04d>.npy` in `seed_labels`, with `{clip}` replaced by the clip's name; such a
    directory can be a condition that `evalkit.tools.kmerge` wrote. Without `{clip}`, the file is in
    `<seed_labels>/<clip>/`. The regions are numbered again, so that the tracker's ids are 1..K whatever ids the
    file holds. The workbench's seed files were numbered 0..K-1 when written, so they gave the same ids.

    Raises:
        FileNotFoundError: the file is missing, as when the seeds were made for another seed frame.
        ValueError: the map is not the depth's shape, or holds no region.
    """
    d = Path(seed_labels.replace("{clip}", clip)) if "{clip}" in seed_labels else Path(seed_labels) / clip
    p = d / f"label_{frame:04d}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"no seed labels {p}; were they made for seed frame {frame}?")
    labels = np.load(p).astype(int)
    if labels.shape != shape:
        raise ValueError(f"{p} is {labels.shape}, not the depth's {shape}")
    ids = np.unique(labels[labels >= 0])
    if ids.size == 0:
        raise ValueError(f"{p} holds no region")
    out = np.full(shape, -1, dtype=int)
    for i, r in enumerate(ids):
        out[labels == r] = i
    return out


def seed_masks(labels: np.ndarray, min_area: int) -> tuple[list[np.ndarray], list[int]]:
    """Return a mask and an object id (region + 1) for each region of at least `min_area` pixels."""
    masks, ids = [], []
    for r in np.unique(labels):
        m = labels == r
        if r >= 0 and int(m.sum()) >= min_area:
            masks.append(m)
            ids.append(int(r) + 1)
    return masks, ids


def collapse(result: FrameResult, shape: tuple[int, int]) -> np.ndarray:
    """Return one frame's objects as an (H, W) int16 label map, the higher presence score painted last.

    A missing score counts as the lowest, so such objects are painted under the others. On the prompted
    frame every score is missing, so the objects tie. `np.argsort` keeps the tracker's order among ties only
    up to 16 objects; above that, the order of ties depends on numpy's build and the CPU, as it did in the
    workbench, whose labels were made with this sort. Where those objects' masks overlap, the label is
    therefore the same only on the same numpy and CPU.
    """
    labels = np.full(shape, -1, dtype=np.int16)
    scores = np.nan_to_num(np.asarray(result.scores, dtype=float), nan=-1e9)
    for k in np.argsort(scores):
        labels[result.masks[k] > 0] = int(result.object_ids[k])
    return labels


def read_clip(clip_dir: Path, depth_source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the clip's (N, H, W) float32 depth and (N, 3, 3) float64 intrinsics, from the chosen depth stage.

    Pi3X's depth is resized to DA3's grid with the nearest neighbour, and its intrinsics are scaled alike. Its
    depth is then scaled to DA3's median: the normals' normalisation has an absolute floor, so depth of another
    magnitude would flatten them.

    Raises:
        FileNotFoundError: a bundle is missing.
        ValueError: Pi3X's bundle has another number of frames than DA3's.
    """
    if not (clip_dir / BUNDLE).is_file():
        raise FileNotFoundError(f"{clip_dir.name} has no {BUNDLE}; run the depth stage first")
    with np.load(clip_dir / BUNDLE) as z:
        depth, K = z["depth"].astype(np.float32), z["intrinsics"].astype(np.float64)
    if depth_source == "da3":
        return depth, K
    if not (clip_dir / PI3X_BUNDLE).is_file():
        raise FileNotFoundError(f"{clip_dir.name} has no {PI3X_BUNDLE}; run the Pi3X stage first")
    with np.load(clip_dir / PI3X_BUNDLE) as z:
        d, K = z["depth"].astype(np.float32), z["intrinsics"].astype(np.float64)
    if d.shape[0] != depth.shape[0]:
        raise ValueError(f"{clip_dir.name}: Pi3X has {d.shape[0]} frames, but DA3 has {depth.shape[0]}")
    if d.shape[1:] != depth.shape[1:]:
        h, w = depth.shape[1:]
        sy, sx = h / d.shape[1], w / d.shape[2]
        d = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in d])
        K = K.copy()
        K[:, 0, 0] *= sx
        K[:, 0, 2] *= sx
        K[:, 1, 1] *= sy
        K[:, 1, 2] *= sy
    d = d * (float(np.nanmedian(depth)) / max(float(np.nanmedian(d)), 1e-12))
    return np.nan_to_num(d, nan=float(np.nanmedian(d))), K


def read_rgb(clip_dir: Path, i: int, shape: tuple[int, int]) -> np.ndarray:
    """Return frame `i` as (H, W, 3) uint8 RGB at the depth's shape, resized with area averaging.

    Raises:
        FileNotFoundError: the image is missing. A black frame in its place would be segmented and tracked as if
            it were the scene.
    """
    p = clip_dir / "input_images" / f"{i:06d}.png"
    if not p.is_file():
        raise FileNotFoundError(f"no image {p}")
    im = np.array(Image.open(p).convert("RGB"))
    if im.shape[:2] != shape:
        im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return im


def window(rule: str, n: int) -> tuple[list[int], int, bool]:
    """Return the frames, the seed frame and whether to carry backwards too, under a rule.

    Raises:
        ValueError: the rule is not one of `RULES`.
    """
    frames = list(range(n))
    if rule == "both_ways_from_centre":
        return frames, frames[len(frames) // 2], True
    if rule == "forward_from_first":
        return frames, 0, False
    raise ValueError(f"unknown rule {rule!r}; one of {RULES}")


def write_montage(path: Path, labels: dict[int, np.ndarray], frames: list[int], shape: tuple[int, int]) -> None:
    """Draw every frame's labels, one colour per object across the clip, with the frame's number above it."""
    ids = sorted({int(v) for lab in labels.values() for v in np.unique(lab) if v >= 0})
    rng = np.random.default_rng(2)
    lut = {g: rng.integers(40, 235, 3).astype(np.uint8) for g in ids}
    h, w = shape
    tiles = []
    for k, lab in sorted(labels.items()):
        tile = np.zeros((h + MONTAGE_BAR, w, 3), np.uint8)
        for g in np.unique(lab):
            if g >= 0:
                tile[MONTAGE_BAR:][lab == g] = lut[int(g)]
        cv2.putText(tile, f"abs {frames[k]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    rows = (len(tiles) + MONTAGE_COLS - 1) // MONTAGE_COLS
    grid = np.zeros((rows * (h + MONTAGE_BAR), MONTAGE_COLS * w, 3), np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, MONTAGE_COLS)
        grid[r * (h + MONTAGE_BAR):(r + 1) * (h + MONTAGE_BAR), c * w:(c + 1) * w] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def run_track(clip_dir: Path, tracks_root: Path, tag: str, rule: str, segmenter, tracker, sam_input: str,
              track_base: str, depth_source: str = "da3", points_per_side: int = 24, seed_min_area: int = 400,
              seed_edge_gain: float = 0.85, seed_smooth: bool = True, mask_ring: bool = True,
              seed_labels: str | None = None,
              overwrite: bool = False) -> Path:
    """Run the stage on one clip and return the directory of its labels.

    Args:
        clip_dir: the clip.
        tracks_root: where each clip's conditions go.
        tag: the condition's name; its directory is `track_<track_base>_<tag>`.
        rule: one of `RULES`.
        segmenter: cuts the seed frame; `SeedSegmenter` or a stand-in with its `label_map`. Unused with
            `seed_labels`.
        tracker: carries the regions; `Sam3VideoInstanceSession` or a stand-in with its methods.
        sam_input, track_base: the seed's input mode and the tracker's, of `SAM_INPUT_MODES`.
        depth_source: one of `DEPTH_SOURCES`, the depth the normals and edges come from.
        points_per_side: the mask generator's grid.
        seed_min_area: the fewest pixels a seed region keeps.
        seed_edge_gain, seed_smooth: the seed input's edge darkness and normal smoothing. The tracker's are
            `TRACK_EDGE_GAIN` and `TRACK_SMOOTH`.
        mask_ring: zero the ring of the burnt-in edges, for the seed and the tracker alike.
        seed_labels: a directory of seed regions made outside the stage (`read_seed_labels`), in place of the
            segmenter's. `sam_input` is then recorded but not used.
        overwrite: replace the condition's labels and montage when the clip already has them. They are
            replaced only once the run has written every file, so a run that fails leaves them as they were.

    Raises:
        FileNotFoundError: a bundle, an image or the seed labels are missing.
        ValueError: one of these:
            - a mode, the rule or the depth source is unknown
            - the condition's labels exist and `overwrite` is false
            - Pi3X's depth has another number of frames than DA3's
            - the seed labels are of another shape or hold no region
            - no seed region is left
            - the tracker carries no frame
    """
    # Check the settings, and refuse labels an earlier run left.
    for mode in (sam_input, track_base):
        uses_geom_edge(mode)
    if depth_source not in DEPTH_SOURCES:
        raise ValueError(f"unknown depth source {depth_source!r}; one of {DEPTH_SOURCES}")
    subdir = f"track_{track_base}_{tag}"
    lab_dir = tracks_root / clip_dir.name / subdir
    montage = tracks_root / clip_dir.name / "viz" / f"montage_{subdir}.png"
    if (lab_dir.exists() or montage.exists()) and not overwrite:
        raise ValueError(f"{clip_dir.name} already has {subdir}; pass --overwrite to replace it")

    # Read the depth, and place the seed under the rule.
    depth, K = read_clip(clip_dir, depth_source)
    n, h, w = depth.shape
    depth01 = global_depth01(depth)
    frames, seed_frame, both_ways = window(rule, n)
    seed_idx = frames.index(seed_frame)
    print(f"  {n} frames, {rule}: seed frame {seed_frame}")

    # Cut the seed frame into regions, or read regions made outside, and keep those large enough.
    if seed_labels is None:
        image = sam_input_image(sam_input, depth[seed_frame], K[seed_frame], depth01[seed_frame],
                                read_rgb(clip_dir, seed_frame, (h, w)), edge_gain=seed_edge_gain,
                                smooth=seed_smooth, mask_ring=mask_ring)
        regions = segmenter.label_map(image, points_per_side)
        seed_input = dict(produced_by="sam", cache_path=None, points_per_side=points_per_side,
                          seed_sam_kwargs=SEED_SAM_KWARGS, seed_edge_gain=seed_edge_gain, seed_smooth=seed_smooth,
                          edge_ring_masked=mask_ring if uses_geom_edge(sam_input) else None)
    else:
        regions = read_seed_labels(seed_labels, clip_dir.name, seed_frame, (h, w))
        # Settings that did not make the seed are recorded as unknown, not as this run's.
        seed_input = dict(produced_by="external", cache_path=seed_labels, points_per_side=None,
                          seed_sam_kwargs=None, seed_edge_gain=None, seed_smooth=None, edge_ring_masked=None)
    masks, obj_ids = seed_masks(regions, seed_min_area)
    if not masks:
        raise ValueError(f"{clip_dir.name}: no seed region of {seed_min_area} pixels or more")
    print(f"  seed: {len(masks)} regions")

    # Carry the regions through the tracker's frames, forwards, and backwards too under both_ways.
    tracker.init_video(np.stack([
        sam_input_image(track_base, depth[i], K[i], depth01[i], read_rgb(clip_dir, i, (h, w)),
                        edge_gain=TRACK_EDGE_GAIN, smooth=TRACK_SMOOTH, mask_ring=mask_ring) for i in frames]))
    tracker.add_masks(seed_idx, masks, obj_ids)
    labels = {r.frame_idx: collapse(r, (h, w)) for r in tracker.propagate(seed_idx)}
    if both_ways and seed_idx > 0:
        for r in tracker.propagate(seed_idx, reverse=True):
            labels.setdefault(r.frame_idx, collapse(r, (h, w)))
    if not labels:
        raise ValueError(f"{clip_dir.name}: the tracker carried no frame")

    # Write the labels, the seed's record and the montage to a partial directory.
    # An earlier run's labels are replaced only after every file is written, so a run that fails keeps them.
    partial = tracks_root / clip_dir.name / f".{subdir}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        for k, lab in sorted(labels.items()):
            np.save(partial / f"label_{frames[k]:04d}.npy", lab)
        # The keys and their order are the workbench's, so that a record compares with one written there;
        # `instrument_seed` names an option the workbench had and this stage leaves out.
        with open(partial / "seed_info.json", "w", encoding="utf-8") as f:
            json.dump(dict(clip=clip_dir.name, seed_frame=int(seed_frame),
                           seed_source="sam" if seed_labels is None else "external",
                           seed_labels=seed_labels or "", sam_input=sam_input, depth_source=depth_source,
                           track_base=track_base, bidir=both_ways, stride=1, seed_min_area=seed_min_area,
                           seed_input=seed_input, n_seed_regions=len(masks), instrument_seed="off",
                           frames=[int(frames[k]) for k in sorted(labels)]),
                      f, indent=1, ensure_ascii=False)
        write_montage(partial / montage.name, labels, frames, (h, w))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    # Replace the earlier run's labels and montage with the new ones.
    if lab_dir.exists():
        shutil.rmtree(lab_dir)
    montage.parent.mkdir(parents=True, exist_ok=True)
    (partial / montage.name).replace(montage)
    partial.rename(lab_dir)
    print(f"  {len(labels)} label maps to {lab_dir}")
    return lab_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip with depth.")
    ap.add_argument("--tracks-root", required=True, help="Where each clip's conditions go.")
    ap.add_argument("--tag", required=True, help="The condition's name.")
    ap.add_argument("--rule", required=True, choices=RULES, help="Where the seed goes and which way it is carried.")
    ap.add_argument("--sam-input", required=True, choices=SAM_INPUT_MODES, help="The seed frame's input mode.")
    ap.add_argument("--track-base", required=True, choices=SAM_INPUT_MODES, help="The tracker's input mode.")
    ap.add_argument("--sam-ckpt", help="The SAM ViT-H weights, sam_vit_h_4b8939.pth. Needed without --seed-labels.")
    ap.add_argument("--seed-labels",
                    help="Seed regions made outside the stage: a directory path with {clip} in it, replaced by the "
                         "clip's name, such as '<tracks-root>/{clip}/<merged tag>', or a directory holding "
                         "<clip>/label_<frame>.npy.")
    ap.add_argument("--depth-source", default="da3", choices=DEPTH_SOURCES,
                    help="The depth the normals and edges come from.")
    ap.add_argument("--points-per-side", type=int, default=24, help="The mask generator's grid.")
    ap.add_argument("--seed-min-area", type=int, default=400, help="The fewest pixels a seed region keeps.")
    ap.add_argument("--seed-edge-gain", type=float, default=0.85, help="How dark the seed input's edges are.")
    ap.add_argument("--seed-no-smooth", action="store_true", help="Do not smooth the seed input's normals.")
    ap.add_argument("--keep-edge-ring", action="store_true",
                    help="Keep the edges' ring along the image border and around invalid depth, as in the "
                         "labels made before the ring was masked.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="SAM 3's Hugging Face model id.")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu. auto takes CUDA when there is one.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="The index of the GPU to use. Default: what CUDA_VISIBLE_DEVICES says, or the first GPU.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a condition's labels an earlier run left.")
    args = ap.parse_args()

    # Find the clips before the models load: they take 3 GB, and a mistyped clip name should fail first.
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
    if not args.seed_labels and not args.sam_ckpt:
        raise SystemExit("--sam-ckpt is needed to cut the seed frames, unless --seed-labels gives them")
    tracker = Sam3VideoInstanceSession(args.model_id, device=args.device, gpu=args.gpu)
    segmenter = None if args.seed_labels else SeedSegmenter(args.sam_ckpt, str(tracker.device))
    print(f"{args.model_id} on {tracker.device}, {len(clips)} clip(s)")

    # Run every clip, even after one fails, and list the failures at the end.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            run_track(clip_dir, Path(args.tracks_root), args.tag, args.rule, segmenter, tracker, args.sam_input,
                      args.track_base, args.depth_source, args.points_per_side, args.seed_min_area,
                      args.seed_edge_gain, not args.seed_no_smooth, not args.keep_edge_ring, args.seed_labels,
                      args.overwrite)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
