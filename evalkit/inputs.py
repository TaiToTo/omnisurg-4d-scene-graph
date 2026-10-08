"""Read one clip's inputs: its frames, GT masks, depth and predictions.

The inputs sit in three places:
- `<data_root>/<clip>/frame_manifest.json` lists the frames in the order
  the pipeline processed them, and marks the GT frames (`docs/evaluation.md`,
  "Which frames").
- `<data_root>/<clip>/exports/mini_npz/results.npz` holds the depth, frame
  `i` as `depth[i]`. `seg_masks/<i:06d><suffix>` beside it holds a GT
  frame's mask, already cut to the endoscope's rectangle.
- `<tracks_root>/<clip>/<tag>/label_<i>.npy` is the prediction for frame `i`.
A mask or prediction of another shape is resized to the depth map's, with
nearest neighbour. Every input is fingerprinted, so a score says what it read.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
from PIL import Image

from evalkit.classes import ClassTable

MANIFEST = "frame_manifest.json"
DEPTH_FILE = Path("exports", "mini_npz", "results.npz")
MASK_DIR = "seg_masks"

# The file a GT mask is stored in, after the frame's index. CholecSeg8k's
# colour masks; ATLAS-120k's id masks, as the extraction wrote them.
MASK_SUFFIX: Mapping[str, str] = MappingProxyType({
    "cholecseg8k": "_color_mask.png",
    "atlas120k": "_class.png",
})

# The manifest's flag for a frame with a mask under `seg_masks/`, per dataset.
GT_FLAG: Mapping[str, str] = MappingProxyType({
    "cholecseg8k": "has_seg_mask",
    "atlas120k": "has_gt",
})
# The manifest's crop rectangle, per dataset.
_CROP_KEYS = ("crop_info", "crop")
_LABEL = re.compile(r"label_(\d+)\.npy")


@dataclass(frozen=True)
class ClipInputs:
    """Everything one clip is scored from, at the depth map's shape.

    Attributes:
        clip: The clip's name.
        order: The indexes of the clip's frames, in time order.
        numbers: Each index to the frame's number in the source video.
        depth: The (N, H, W) float32 depth of every frame, by index.
        gt: Each index with a GT mask to its (H, W) int32 mask ids.
            Read-only.
        regions: Each index with a prediction to its (H, W) int32 region
            map, -1 for no region, in the order of the files' names.
            Read-only.
        shas: Each input's sha256: `gt_masks`, `depth`, `crop`, `frames`,
            and `predictions`, the one that is the condition's own.
    """

    clip: str
    order: tuple[int, ...]
    numbers: Mapping[int, int]
    depth: np.ndarray
    gt: Mapping[int, np.ndarray]
    regions: Mapping[int, np.ndarray]
    shas: Mapping[str, str]


def sha_of_files(paths: list[Path]) -> str:
    """Return the sha256 of the files' names and contents, in the order given."""
    # Each part is hashed after its length, so that a renamed file and a moved
    # boundary between two files both change the hash.
    h = hashlib.sha256()
    for p in paths:
        for part in (p.name.encode("utf-8"), p.read_bytes()):
            h.update(struct.pack(">Q", len(part)))
            h.update(part)
    return h.hexdigest()


def _sha_of_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def depth_sha(depth: np.ndarray) -> str:
    """Return the depth's fingerprint: the sha256 of its float32 values, as `atlas120k_meta/depth_manifest.json` records it."""
    return hashlib.sha256(np.ascontiguousarray(depth, dtype=np.float32).tobytes()).hexdigest()


def _time_order(frames: list[dict], clip: str) -> list[int]:
    """Return the frames' indexes in time order: by `timestamp_sec`, or by frame number where the dataset gives no time.

    Raises:
        ValueError: Some frames have a timestamp and some do not, or two
            frames share a time.
    """
    with_time = [f for f in frames if "timestamp_sec" in f]
    if with_time and len(with_time) != len(frames):
        raise ValueError(f"{clip}: {len(with_time)} of {len(frames)} frames have a timestamp; all or none must")
    key = "timestamp_sec" if with_time else "native_frame"
    times = [f[key] for f in frames]
    if len(set(times)) != len(times):
        raise ValueError(f"{clip}: two frames share one {key}, so their order in time is unknown")
    return sorted(range(len(frames)), key=lambda i: times[i])


def gt_frames(frames: list[dict], flag: str, with_mask: set[int], clip: str) -> list[int]:
    """Return the GT frames: those with the dataset's GT flag, `is_anchor` true and no `seg_provenance`.

    The mask files cannot decide: the pipeline also writes the viewer's SAM 3 masks into `seg_masks/` under the
    GT's names, and marks those frames with `seg_provenance`.

    Args:
        frames: The manifest's frames, by index.
        flag: The dataset's GT flag, from `GT_FLAG`.
        with_mask: The indexes that have a mask file.
        clip: The clip's name, for the messages.

    Raises:
        ValueError: A frame has no GT flag; a mask file has no GT flag; a GT flag has no mask file; or a
            flagged frame is neither a GT frame nor marked by `seg_provenance`.
    """
    out = []
    for i, f in enumerate(frames):
        if flag not in f:
            raise ValueError(f"{clip}: frame {i} has no {flag} flag")
        flagged = bool(f[flag])
        if flagged != (i in with_mask):
            raise ValueError(f"{clip}: frame {i} is flagged {flag}={flagged} but its mask is "
                             f"{'there' if i in with_mask else 'missing'}")
        if not flagged or f.get("seg_provenance"):
            continue
        if not f.get("is_anchor"):
            raise ValueError(f"{clip}: frame {i} has a mask but is neither an anchor nor marked by "
                             "seg_provenance, so whether it is GT is unknown")
        out.append(i)
    return out


def _resized(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _labels(label_dir: Path, n: int, clip: str) -> dict[int, Path]:
    """Map each frame index with a prediction to its file, in the order of the files' names.

    Raises:
        FileNotFoundError: The condition has no directory for this clip.
        ValueError: A file name gives an index twice, or one outside the clip.
    """
    if not label_dir.is_dir():
        raise FileNotFoundError(f"{clip}: no prediction directory {label_dir}")
    out: dict[int, Path] = {}
    for p in sorted(label_dir.glob("label_*.npy")):
        m = _LABEL.fullmatch(p.name)
        if m is None:
            raise ValueError(f"{clip}: {p.name} is not label_<index>.npy")
        i = int(m.group(1))
        if i in out:
            raise ValueError(f"{clip}: frame {i} has two predictions, {out[i].name} and {p.name}")
        if not 0 <= i < n:
            raise ValueError(f"{clip}: {p.name} names frame {i}, but the clip has frames 0 to {n - 1}")
        out[i] = p
    return out


def read_clip(data_root: str | Path, tracks_root: str | Path, tag: str, clip: str,
              table: ClassTable) -> ClipInputs:
    """Read one clip's inputs for one condition.

    Args:
        data_root: The directory holding one directory per clip.
        tracks_root: The directory holding one directory per clip, each with
            one directory per condition.
        tag: The condition's directory name under each clip.
        clip: The clip's name.
        table: The dataset's class table, which reads the GT masks.

    Raises:
        FileNotFoundError: The manifest, the depth or the prediction
            directory is missing.
        ValueError: The manifest disagrees with the depth or the masks on
            disk, or leaves a frame's GT unknown (`gt_frames`); the frames'
            order in time is unknown; the clip has no GT frame; a GT frame
            has no prediction; or a prediction is not an (H, W) integer map.
        KeyError: A GT mask holds an id or a colour the table does not have.
    """
    clip_dir = Path(data_root) / clip

    # The frames, as the manifest lists them, and their order in time.
    manifest = json.loads((clip_dir / MANIFEST).read_text(encoding="utf-8"))
    frames = manifest["frames"]
    for i, f in enumerate(frames):
        if f["seq_idx"] != i:
            raise ValueError(f"{clip}: the manifest's frame {i} says seq_idx {f['seq_idx']}")
    order = _time_order(frames, clip)
    numbers = {i: int(f["native_frame"]) for i, f in enumerate(frames)}
    crops = [k for k in _CROP_KEYS if k in manifest]
    if len(crops) != 1:
        raise ValueError(f"{clip}: the manifest holds {crops or 'no'} crop rectangle; one of {_CROP_KEYS} is needed")

    # The depth of every frame, which sets the shape everything is scored at.
    with np.load(clip_dir / DEPTH_FILE) as z:
        depth = np.asarray(z["depth"], dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] != len(frames):
        raise ValueError(f"{clip}: depth has shape {depth.shape}, the manifest lists {len(frames)} frames")
    shape = depth.shape[1:]

    # The GT frames, as the manifest's flags say, and their masks, read through the table and resized.
    suffix = MASK_SUFFIX[table.dataset]
    with_mask = {i for i in range(len(frames)) if (clip_dir / MASK_DIR / f"{i:06d}{suffix}").is_file()}
    mask_paths = {i: clip_dir / MASK_DIR / f"{i:06d}{suffix}"
                  for i in gt_frames(frames, GT_FLAG[table.dataset], with_mask, clip)}
    if not mask_paths:
        raise ValueError(f"{clip}: no frame is a GT frame")
    gt = {}
    for i, p in mask_paths.items():
        with Image.open(p) as image:
            gt[i] = _resized(table.mask_ids(image), shape)

    # The predictions, every GT frame among them, resized.
    label_paths = _labels(Path(tracks_root) / clip / tag, len(frames), clip)
    missing = sorted(set(mask_paths) - set(label_paths))
    if missing:
        raise ValueError(f"{clip}: GT frames {missing} have no prediction under {tag}; none is skipped")
    regions = {}
    for i, p in label_paths.items():
        r = np.load(p)
        if r.ndim != 2 or not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"{clip}: {p.name} is not an (H, W) integer map, got {r.dtype} of shape {r.shape}")
        regions[i] = _resized(r.astype(np.int32), shape)

    shas = {
        "gt_masks": sha_of_files([mask_paths[i] for i in sorted(mask_paths)]),
        "depth": depth_sha(depth),
        "crop": _sha_of_json(manifest[crops[0]]),
        "frames": _sha_of_json([[i, numbers[i]] for i in order]),
        "predictions": sha_of_files([label_paths[i] for i in sorted(label_paths)]),
    }
    return ClipInputs(
        clip=clip, order=tuple(order), numbers=MappingProxyType(numbers), depth=depth,
        gt=MappingProxyType(gt), regions=MappingProxyType(regions), shas=MappingProxyType(shas),
    )
