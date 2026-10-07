"""Pilot mode: one clip read, scored and averaged by the pilot evaluator's rules.

Pilot mode exists for the check against the pilot evaluator and is removed
before the freeze. It reads what the pilot read: every `label_<i>.npy` in
file-name order, the mask of each GT frame the manifest's flags give, a
colour mask through the pilot's own colours, and the pixels with valid
depth. The pilot read a mask wherever its file was, but its data root held
the annotated masks only, so the flags give the frames it scored. Each
frame is scored with the evaluator's modules under the pilot's objects and
domains (`evalkit.pilot`), with the pilot's arithmetic where it differs.
The clip is averaged and rounded as the pilot did. The row holds the keys
`evalkit.tools.pilot_check` compares, spelled the evaluator's way.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
from PIL import Image

from evalkit.boundary import BoundaryScore, boundary_pixels, boundary_score
from evalkit.classmap import class_map
from evalkit.inputs import DEPTH_FILE, GT_FLAG, MANIFEST, MASK_DIR, MASK_SUFFIX, depth_sha, gt_frames, sha_of_files
from evalkit.inst_bf import instance_boundary_f
from evalkit.keys import metric_key
from evalkit.objects import instance_scores
from evalkit.pilot import (
    PILOT_BACKGROUND,
    PILOT_DOMAINS,
    PILOT_INSTRUMENT_IDS,
    pilot_domains,
    pilot_gt_objects,
    pilot_predicted_objects,
)
from evalkit.scored import valid_depth
from evalkit.vi import variation_of_information

# The pilot evaluator's CholecSeg8k colours. It read Hepatic Vein as
# (0, 255, 0), took three colours for background, and read any other colour,
# the region line's white among them, as background.
PILOT_CHOLEC_COLOURS: Mapping[tuple[int, int, int], int] = MappingProxyType({
    (127, 127, 127): 0, (0, 0, 0): 0, (50, 50, 50): 0,
    (210, 140, 140): 1, (255, 114, 114): 2, (231, 70, 156): 3, (186, 183, 75): 4,
    (170, 255, 0): 5, (255, 85, 0): 6, (255, 0, 0): 7, (255, 255, 0): 8,
    (169, 255, 184): 9, (255, 160, 165): 10, (0, 255, 0): 11, (111, 74, 0): 12,
})

# The decimals the pilot evaluator rounded every clip value to.
PILOT_DECIMALS = 4

# The pilot's F had this in its denominator.
_F_EPSILON = 1e-9


def _pilot_gt(mask_dir: Path, i: int, shape: tuple[int, int]) -> tuple[np.ndarray, Path] | None:
    """Frame `i`'s GT ids at `shape`, as the pilot read them, and the file; None without a mask."""
    colour = mask_dir / f"{i:06d}_color_mask.png"
    if colour.is_file():
        bgr = cv2.imread(str(colour))
        if bgr is None:
            raise ValueError(f"{colour}: not a readable image")
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (shape[1], shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        palette, index = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
        ids = np.array([PILOT_CHOLEC_COLOURS.get(tuple(int(c) for c in p), PILOT_BACKGROUND) for p in palette],
                       dtype=np.int32)
        return ids[index.ravel()].reshape(shape), colour
    index_mask = mask_dir / f"{i:06d}_class.png"
    if index_mask.is_file():
        with Image.open(index_mask) as image:
            if image.mode not in ("P", "L"):
                raise ValueError(f"{index_mask}: mode {image.mode} holds no class ids")
            ids = np.asarray(image, dtype=np.int32)
        if ids.shape != shape:
            ids = cv2.resize(ids, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return ids, index_mask
    return None


def _pilot_f(score: BoundaryScore | None) -> float:
    # The pilot scored 0 where either boundary was empty.
    if score is None:
        return 0.0
    return 2 * score.precision * score.recall / (score.precision + score.recall + _F_EPSILON)


def pilot_frame(gt: np.ndarray, regions: np.ndarray, valid: np.ndarray, instrument_ids: frozenset[int]) -> dict:
    """One frame's values by the pilot's rules, unrounded.

    Returns:
        `miou`, `boundary_f` and `boundary_r_raw` on the `full` domain;
        `inst`, each domain to its `f1_50`, `sq` and `inst_bf`, or to None
        when the domain holds no GT object; and `vi_split` and `vi_merge` on
        the `labeled` domain, None when it holds no pixel.
    """
    # The class map: each region named by the vote over its valid pixels,
    # background voting, and the name painted on every pixel of the region.
    names = class_map(gt, regions, valid).names
    pred = np.zeros_like(gt)
    for r, c in names.items():
        pred[regions == r] = c

    # mIoU over the classes in the GT or the class map, background left out,
    # averaged in the order the pilot's set gave.
    classes = set(np.unique(gt[valid]).tolist()) | set(np.unique(pred[valid]).tolist())
    classes.discard(PILOT_BACKGROUND)
    ious = []
    for c in classes:
        p, g = (pred == c) & valid, (gt == c) & valid
        union = int((p | g).sum())
        ious.append(int((p & g).sum()) / union if union else 0.0)

    # The boundaries, marked on the whole map and kept on the valid pixels.
    gt_boundary = boundary_pixels(gt) & valid
    raw = boundary_score(boundary_pixels(regions) & valid, gt_boundary)

    # The objects of each domain, paired as the evaluator pairs them.
    domains = pilot_domains(gt, valid, instrument_ids)
    inst = {}
    for name, domain in domains.items():
        g, p = pilot_gt_objects(gt, domain), pilot_predicted_objects(regions, domain)
        if not len(g):
            inst[name] = None
            continue
        scores = instance_scores(g, p)
        per_hit = [_pilot_f(s) for s in instance_boundary_f(g, p, scores, pilot=True).scores]
        inst[name] = {"f1_50": scores.f1_50, "sq": scores.sq,
                      "inst_bf": float(np.mean(per_hit)) if per_hit else None}

    vi = variation_of_information(gt, regions, domains["labeled"])
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "boundary_f": _pilot_f(boundary_score(boundary_pixels(pred) & valid, gt_boundary)),
        "boundary_r_raw": 0.0 if raw is None else raw.recall,
        "inst": inst,
        "vi_split": None if vi is None else vi.split,
        "vi_merge": None if vi is None else vi.merge,
    }


def pilot_time_iou(regions: Sequence[np.ndarray], valid: Sequence[np.ndarray]) -> float:
    """`time_IoU` as the pilot computed it: ids in its set order, and 0 where nothing is pooled."""
    values = []
    for t in range(len(regions) - 1):
        a, b = regions[t], regions[t + 1]
        both = valid[t] & valid[t + 1]
        for r in set(np.unique(a).tolist()) & set(np.unique(b).tolist()):
            if r < 0:
                continue
            in_a, in_b = (a == r) & both, (b == r) & both
            union = (in_a | in_b).sum()
            if union:
                values.append((in_a & in_b).sum() / union)
    return float(np.mean(values)) if values else 0.0


def _mean_or_none(values: list) -> tuple[float | None, int]:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, PILOT_DECIMALS)


def pilot_row(clip: str, frames: list[tuple[int, dict]], time_iou: float) -> dict:
    """The clip's row: each key averaged over its frames as the pilot did, rounded, with the counts.

    Args:
        clip: The clip's name.
        frames: Each GT frame's number and `pilot_frame` values, in file order.
        time_iou: The clip's `pilot_time_iou`.
    """
    values = [v for _, v in frames]
    row: dict = {"clip": clip}
    n_frames: dict[str, int] = {}

    # The class map and the boundaries, a mean over every GT frame.
    for metric, field in (("mIoU", "miou"), ("boundary_F", "boundary_f"), ("boundary_R_raw", "boundary_r_raw")):
        key = metric_key(metric, "full")
        row[key] = round(float(np.mean([v[field] for v in values])), PILOT_DECIMALS)
        n_frames[key] = len(values)

    # The instance metrics, over the frames with a GT object in the domain.
    # Where there is none the pilot wrote 0 for F1_50 in `full`, None elsewhere.
    for domain in PILOT_DOMAINS:
        inst = [v["inst"][domain] for v in values if v["inst"][domain] is not None]
        f1_key, sq_key, bf_key = (metric_key(m, domain) for m in ("F1_50", "SQ", "inst_BF"))
        if inst:
            row[f1_key] = round(float(np.mean([x["f1_50"] for x in inst])), PILOT_DECIMALS)
        else:
            row[f1_key] = 0.0 if domain == "full" else None
        sq, n_sq = _mean_or_none([x["sq"] for x in inst])
        bf, n_bf = _mean_or_none([x["inst_bf"] for x in inst])
        row[sq_key], row[bf_key] = _rounded(sq), _rounded(bf)
        n_frames.update({f1_key: len(inst), sq_key: n_sq, bf_key: n_bf})

    # VI over the frames whose `labeled` domain holds a pixel.
    for metric, field in (("VI_split", "vi_split"), ("VI_merge", "vi_merge")):
        key = metric_key(metric, "labeled")
        mean, n = _mean_or_none([v[field] for v in values])
        row[key] = _rounded(mean)
        n_frames[key] = n

    row["time_IoU"] = round(time_iou, PILOT_DECIMALS)
    row["n_frames"] = n_frames
    row["frames"] = [{"frame": number, **_frame_values(v)} for number, v in frames]
    return row


def _frame_values(v: dict) -> dict:
    out = {metric_key("mIoU", "full"): v["miou"], metric_key("boundary_F", "full"): v["boundary_f"],
           metric_key("boundary_R_raw", "full"): v["boundary_r_raw"],
           metric_key("VI_split", "labeled"): v["vi_split"], metric_key("VI_merge", "labeled"): v["vi_merge"]}
    for domain, inst in v["inst"].items():
        for metric, field in (("F1_50", "f1_50"), ("SQ", "sq"), ("inst_BF", "inst_bf")):
            out[metric_key(metric, domain)] = None if inst is None else inst[field]
    return out


def score_pilot_clip(data_root: str | Path, tracks_root: str | Path, tag: str, clip: str,
                     dataset: str) -> tuple[dict, dict[str, str]]:
    """Score one clip in pilot mode.

    Returns:
        The clip's row and the sha of each input read.

    Raises:
        FileNotFoundError: The manifest, the depth or the prediction directory is missing.
        ValueError: The manifest leaves a frame's GT unknown (`evalkit.inputs.gt_frames`); the condition has
            no prediction for the clip; a label file names a frame outside the depth; no predicted frame is a
            GT frame; or a mask is not readable.
    """
    clip_dir = Path(data_root) / clip
    frames_listed = json.loads((clip_dir / MANIFEST).read_text(encoding="utf-8"))["frames"]
    with_mask = {i for i in range(len(frames_listed))
                 if (clip_dir / MASK_DIR / f"{i:06d}{MASK_SUFFIX[dataset]}").is_file()}
    gt_set = set(gt_frames(frames_listed, GT_FLAG[dataset], with_mask, clip))
    with np.load(clip_dir / DEPTH_FILE) as z:
        depth = np.asarray(z["depth"], dtype=np.float32)
    shape = depth.shape[1:]
    label_dir = Path(tracks_root) / clip / tag
    if not label_dir.is_dir():
        raise FileNotFoundError(f"{clip}: no prediction directory {label_dir}")
    label_paths = sorted(label_dir.glob("label_*.npy"), key=lambda p: p.name)
    if not label_paths:
        raise ValueError(f"{clip}: no label_*.npy in {label_dir}")

    # Every predicted frame in file order; the ones with a mask are scored.
    regions, valids, frames, mask_paths = [], [], [], []
    for p in label_paths:
        i = int(p.name[len("label_"):-len(".npy")])
        if not 0 <= i < depth.shape[0]:
            raise ValueError(f"{clip}: {p.name} names frame {i}, outside the depth's {depth.shape[0]} frames")
        r = np.load(p).astype(np.int32)
        if r.shape != shape:
            r = cv2.resize(r, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = valid_depth(depth[i], pilot=True)
        regions.append(r)
        valids.append(valid)
        gt = _pilot_gt(clip_dir / MASK_DIR, i, shape) if i in gt_set else None
        if gt is not None:
            frames.append((i, pilot_frame(gt[0], r, valid, PILOT_INSTRUMENT_IDS[dataset])))
            mask_paths.append(gt[1])
    if not frames:
        raise ValueError(f"{clip}: no predicted frame is a GT frame")

    shas = {"gt_masks": sha_of_files(mask_paths), "depth": depth_sha(depth),
            "predictions": sha_of_files(label_paths)}
    return pilot_row(clip, frames, pilot_time_iou(regions, valids)), shas
