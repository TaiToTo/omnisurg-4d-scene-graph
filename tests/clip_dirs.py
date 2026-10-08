"""Clip directories on disk, laid out as the pipeline writes them, for the entry point's tests.

`write_clip` writes one CholecSeg8k clip from class-id maps: its manifest,
its depth and its colour masks, the masks `scale` times larger than the
depth so that the resize is exercised. `write_labels` writes one condition's
predictions for it.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evalkit.classes import load_table
from evalkit.inputs import DEPTH_FILE, MANIFEST, MASK_DIR, MASK_SUFFIX

TABLE = load_table("cholecseg8k")
LIVER, GALLBLADDER, FAT, GRASPER = 2, 10, 4, 5


def two_organs(h: int = 20, w: int = 30, cut: int = 15) -> np.ndarray:
    """Liver left of column `cut`, gallbladder right of it."""
    ids = np.full((h, w), GALLBLADDER, dtype=np.int32)
    ids[:, :cut] = LIVER
    return ids


def colour_mask(ids: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for cid in np.unique(ids).tolist():
        rgb[ids == cid] = TABLE.entries[cid].colour
    return rgb


def write_clip(data_root: Path, clip: str, gt: list[np.ndarray], *, gt_frames=None, sam_frames=(), times=None,
               scale: int = 2, manifest_extra: dict | None = None) -> Path:
    """One clip of `len(gt)` frames; `gt_frames` (default all) have a GT mask, `times` (default 0.6 s apart) their timestamps.

    `sam_frames` have a mask the viewer's step wrote under the GT's name, marked by `seg_provenance`.
    """
    n, (h, w) = len(gt), gt[0].shape
    gt_frames = set(range(n)) if gt_frames is None else set(gt_frames)
    sam_frames = set(sam_frames)
    times = [0.6 * i for i in range(n)] if times is None else times
    clip_dir = Path(data_root) / clip
    (clip_dir / MASK_DIR).mkdir(parents=True)
    (clip_dir / DEPTH_FILE).parent.mkdir(parents=True)
    frames = [{"seq_idx": i, "native_frame": 100 + 15 * i, "timestamp_sec": times[i],
               "has_seg_mask": i in gt_frames | sam_frames, "is_anchor": i in gt_frames} for i in range(n)]
    for i in sam_frames:
        frames[i]["seg_provenance"] = "sam3_gt_propagated"
    manifest = {"frames": frames, "crop_info": {"x0": 0, "x1": w, "y0": 0, "y1": h}, **(manifest_extra or {})}
    (clip_dir / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(clip_dir / DEPTH_FILE, depth=np.ones((n, h, w), dtype=np.float32))
    for i in sorted(gt_frames | sam_frames):
        big = np.kron(gt[i], np.ones((scale, scale), dtype=np.int32))
        Image.fromarray(colour_mask(big)).save(clip_dir / MASK_DIR / f"{i:06d}{MASK_SUFFIX['cholecseg8k']}")
    return clip_dir


def write_labels(tracks_root: Path, clip: str, tag: str, regions: dict[int, np.ndarray], scale: int = 1,
                 seed_info: dict | None = None) -> Path:
    """One condition's predictions for one clip, with the tracker's `seed_info.json` if one is given."""
    out = Path(tracks_root) / clip / tag
    out.mkdir(parents=True)
    if seed_info is not None:
        (out / "seed_info.json").write_text(json.dumps(seed_info), encoding="utf-8")
    for i, r in regions.items():
        np.save(out / f"label_{i:04d}.npy", np.kron(r, np.ones((scale, scale), dtype=r.dtype)).astype(np.int16))
    return out
