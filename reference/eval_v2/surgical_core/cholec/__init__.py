"""Canonical CholecSeg8k palette and id-map helpers.

Single source of truth for the CholecSeg8k colour palette. Imported as
`from surgical_core.cholec import COLOR_TO_CLASS, mask_to_id_map, ...` by
the repo-level cholec scripts.

Two views are exposed and both are derived from the same `_CLASSES` table:

- `COLOR_TO_CLASS`: RGB triplet -> (class_id, "Display Name").
  Multiple RGB triplets can map to the same class (the background entry
  carries (0,0,0), (50,50,50), (127,127,127) — defensive fallbacks for
  padding/edge artifacts found in CholecSeg8k masks).
- `CHOLECSEG8K_CLASS_TO_RGB`: snake_case class name -> canonical RGB.
  The "canonical" RGB is the first triplet listed for each class in
  `_CLASSES`.

Note: sam3_wrapper/scripts/cholec_utils.py keeps its own copy of
CHOLECSEG8K_CLASS_TO_RGB so the wrapper stays self-contained per the
convention in docs/repo_structure.md. If you edit the palette here, update
that copy too — tests/test_cholec_palette.py asserts both stay byte-for-byte
equivalent and will fail on drift.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# (class_id, snake_case_name, "Display Name", (canonical_rgb, *aliases))
_CLASSES: list[tuple[int, str, str, tuple[tuple[int, int, int], ...]]] = [
    (0,  "background",        "Background",             ((127, 127, 127), (0, 0, 0), (50, 50, 50))),
    (1,  "abdominal_wall",    "Abdominal Wall",         ((210, 140, 140),)),
    (2,  "liver",             "Liver",                  ((255, 114, 114),)),
    (3,  "gastrointestinal",  "Gastrointestinal Tract", ((231,  70, 156),)),
    (4,  "fat",               "Fat",                    ((186, 183,  75),)),
    (5,  "grasper",           "Grasper",                ((170, 255,   0),)),
    (6,  "connective_tissue", "Connective Tissue",      ((255,  85,   0),)),
    (7,  "blood",             "Blood",                  ((255,   0,   0),)),
    (8,  "cystic_duct",       "Cystic Duct",            ((255, 255,   0),)),
    (9,  "l_hook",            "L-hook Electrocautery",  ((169, 255, 184),)),
    (10, "gallbladder",       "Gallbladder",            ((255, 160, 165),)),
    (11, "hepatic_vein",      "Hepatic Vein",           ((  0, 255,   0),)),
    (12, "liver_ligament",    "Liver Ligament",         ((111,  74,   0),)),
]


COLOR_TO_CLASS: dict[tuple[int, int, int], tuple[int, str]] = {
    rgb: (cid, display)
    for cid, _, display, rgbs in _CLASSES
    for rgb in rgbs
}

CHOLECSEG8K_CLASS_TO_RGB: dict[str, tuple[int, int, int]] = {
    snake: rgbs[0] for _, snake, _, rgbs in _CLASSES
}


def color_to_id(rgb: tuple[int, int, int]) -> int:
    """Exact-RGB lookup; unknown colours fall through to background (id 0)."""
    entry = COLOR_TO_CLASS.get(rgb)
    return entry[0] if entry is not None else 0


def mask_to_id_map(mask_path: Path, target_H: int, target_W: int) -> np.ndarray | None:
    """Load color mask PNG and return (H, W) int32 class-ID array.

    Returns None if the file is unreadable (corrupt PNG, permission issues,
    etc.) — cv2.imread() returns None instead of raising, and passing None to
    cvtColor() trips an OpenCV assertion that would otherwise abort the whole
    video. The caller treats None the same as a missing mask.
    """
    bgr_image = cv2.imread(str(mask_path))
    if bgr_image is None:
        return None
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # INTER_NEAREST preserves exact palette colours (no interpolated blends).
    rgb_image = cv2.resize(rgb_image, (target_W, target_H), interpolation=cv2.INTER_NEAREST)
    pixels_rgb = rgb_image.reshape(-1, 3)
    unique_palette, pixel_to_palette_idx = np.unique(pixels_rgb, axis=0, return_inverse=True)
    palette_class_ids = np.array(
        [color_to_id(tuple(int(c) for c in rgb)) for rgb in unique_palette],
        dtype=np.int32,
    )
    return palette_class_ids[pixel_to_palette_idx].reshape(target_H, target_W)


def nearest_anchor(idx: int, anchors: list[int]) -> int | None:
    """Return the anchor frame index closest to `idx` (None if no anchors).

    Args:
        idx: Frame index whose nearest anchor is sought.
        anchors: Anchor frame indices (annotation frames); may be empty.
    """
    return min(anchors, key=lambda a: abs(a - idx)) if anchors else None
