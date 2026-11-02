"""Write each clip's first-frame reference masks as the seed regions that `pipeline.track --seed-labels` reads.

A seed region is one class present on frame 0 in the scored pixels of the all-class view, the object that
`evalkit.objects.gt_objects` builds for the score. The mask is read as `evalkit.inputs.read_clip` reads
it: through the class table, filled, and resized to the depth map's shape. The map is written as
`<out>/<clip>/label_0000.npy`, -1 outside every region, with `seed_regions.json` beside it listing each
region's class and area. The tracking stage drops the regions under its `--seed-min-area`. A clip whose
frame 0 is not a scored reference frame is refused.

Usage:
    python paper/ipcai2027/nodes/gt_seeds.py --dataset atlas120k --data-root DIR --clips LIST --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evalkit.classes import ClassTable, load_table
from evalkit.fill import fill_from_nearest
from evalkit.inputs import ANCHORED_DATASETS, DEPTH_FILE, GT_FLAG, MANIFEST, MASK_DIR, MASK_SUFFIX, gt_frames
from evalkit.objects import gt_objects
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth

# The frame `forward_from_first` seeds, and the view whose objects the score pairs on it.
SEED_FRAME = 0
VIEW = "all"


def first_frame_reference(clip_dir: Path, table: ClassTable) -> tuple[np.ndarray, np.ndarray]:
    """Return frame 0's reference mask ids and depth map, both at the depth map's shape.

    Raises:
        ValueError: Frame 0 is not a reference frame by the manifest's flags.
    """
    clip = clip_dir.name

    # Frame 0 must be a reference frame, as the evaluator tells one from the manifest's flags.
    frames = json.loads((clip_dir / MANIFEST).read_text(encoding="utf-8"))["frames"]
    suffix = MASK_SUFFIX[table.dataset]
    with_mask = {i for i in range(len(frames)) if (clip_dir / MASK_DIR / f"{i:06d}{suffix}").is_file()}
    reference = gt_frames(frames, GT_FLAG[table.dataset], with_mask, clip,
                          anchored=table.dataset in ANCHORED_DATASETS)
    if SEED_FRAME not in reference:
        raise ValueError(f"{clip}: frame {SEED_FRAME} is not a reference frame")

    # Frame 0's depth map, which sets the shape.
    with np.load(clip_dir / DEPTH_FILE) as z:
        depth = np.asarray(z["depth"][SEED_FRAME], dtype=np.float32)
    h, w = depth.shape

    # The mask, read through the table, filled before it is resized, as the evaluator's reader does.
    with Image.open(clip_dir / MASK_DIR / f"{SEED_FRAME:06d}{suffix}") as image:
        ids = table.mask_ids(image)
    ids, _ = fill_from_nearest(ids, table.filled_mask_ids)
    if ids.shape != (h, w):
        ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
    return ids, depth


def seed_regions(ids: np.ndarray, depth: np.ndarray, table: ClassTable) -> tuple[np.ndarray, dict]:
    """Return the seed map, one region per class of the view's scored pixels, and its record.

    Raises:
        ValueError: The frame is excluded, or holds no class in the view's scored pixels.
    """
    if frame_is_excluded(ids, table):
        raise ValueError("frame 0 is excluded from the score")
    scored = scored_pixels(ids, table, VIEW, valid_depth(depth))
    objects = gt_objects(scored.classes, scored.mask)
    if not objects.ids:
        raise ValueError("frame 0 holds no class in the scored pixels")
    labels = objects.index.astype(np.int32) - 1
    record = {"seed_frame": SEED_FRAME, "rule": "forward", "view": VIEW, "class_ids": list(objects.ids),
              "areas": list(objects.areas)}
    return labels, record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=("atlas120k", "cholecseg8k"))
    ap.add_argument("--data-root", required=True, type=Path, help="The extracted clips, one directory each.")
    ap.add_argument("--clips", required=True, type=Path, help="The clip names, one per line.")
    ap.add_argument("--out", required=True, type=Path, help="The seed directory that `--seed-labels` names.")
    a = ap.parse_args()
    table = load_table(a.dataset)
    clips = a.clips.read_text(encoding="utf-8").split()
    n_regions = 0
    for clip in clips:
        ids, depth = first_frame_reference(a.data_root / clip, table)
        try:
            labels, record = seed_regions(ids, depth, table)
        except ValueError as e:
            raise ValueError(f"{clip}: {e}") from None
        dst = a.out / clip
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / f"label_{SEED_FRAME:04d}.npy", labels)
        (dst / "seed_regions.json").write_text(json.dumps(record, indent=1))
        n_regions += len(record["class_ids"])
        print(f"{clip}: {len(record['class_ids'])} seed regions", flush=True)
    print(f"{len(clips)} clips, {n_regions} seed regions, written to {a.out}")


if __name__ == "__main__":
    main()
