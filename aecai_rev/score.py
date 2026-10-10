"""Score saved tracks frame by frame, under every area cut and merge variant.

For each window, condition (modality, points_per_side) and annotated frame,
the script computes instance F1 at each IoU threshold, the region count
`n_reg`, the regions and GT instances that enter F1 (`n_pred`, `n_gt`) and
oracle mIoU. It does so for every area cut (the AE-CAI 300 px and each
`min_area_frac`) and for two variants: the regions as tracked (`raw`) and
the regions merged per GT instance (`merged`). Matching and the oracle class
map call the tagged evaluator; only the area cut takes a parameter here,
and `test_score.py` checks it equals the tagged cut at 300 px.

Usage:
    python3 -m aecai_rev.score --workers 32
"""
import argparse
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.tagged import use
from aecai_rev.windows_io import window_clips

LEGACY = "px300"


def present(values):
    """Return the sorted distinct non-negative ints of an array, as `np.unique` would."""
    values = values.ravel()
    values = values[values >= 0]
    if values.size == 0:
        return []
    return np.flatnonzero(np.bincount(values)).tolist()


def filter_names(cfg):
    """Return the area cuts scored: the AE-CAI one and each fraction."""
    return [LEGACY] + [f"frac{f:g}" for f in cfg["score"]["min_area_frac_sweep"]]


def min_area(name, valid):
    """Return the area cut in pixels for one frame.

    Args:
        name: `px<N>` for an absolute cut, `frac<f>` for a fraction of the
            frame's valid area.
        valid: (H, W) bool, the pixels with valid depth.

    Returns:
        The smallest area, in pixels, a component or region must have.
    """
    if name.startswith("px"):
        return float(name[2:])
    if name.startswith("frac"):
        return float(name[4:]) * int(valid.sum())
    raise ValueError(f"unknown area cut {name!r}")


def gt_instances(gt, valid, cut, ignore):
    """Split GT into instances: connected components of each class kept.

    Same as the tagged `_gt_instances`, with the area cut as an argument.

    Args:
        gt: (H, W) class id map.
        valid: (H, W) bool evaluation domain.
        cut: Smallest area kept, in pixels.
        ignore: Class ids that yield no instance.

    Returns:
        A list of (mask, class id) pairs.
    """
    out = []
    for c in present(gt[valid]):
        if int(c) in ignore:
            continue
        n, lbl = cv2.connectedComponents(((gt == c) & valid).astype(np.uint8))
        for i in range(1, n):
            m = lbl == i
            if int(m.sum()) >= cut:
                out.append((m, int(c)))
    return out


def pred_regions(lab, valid, cut):
    """Return each predicted region inside the domain, kept by the area cut.

    Same as the tagged `_pred_regions`, with the area cut as an argument.

    Returns:
        A list of (mask, region id) pairs.
    """
    out = []
    for r in present(lab[valid]):
        m = (lab == r) & valid
        if int(m.sum()) >= cut:
            out.append((m, int(r)))
    return out


def match_pairs(gts, preds):
    """Match GT instances to regions greedily by IoU, one to one.

    The order is the tagged `_match_ious`: pairs sorted by (IoU, GT index,
    region index) descending; it also returns the indices.

    Args:
        gts: List of (H, W) bool GT masks, disjoint.
        preds: List of (H, W) bool region masks, disjoint.

    Returns:
        A list of (iou, gt index, region index), best first.
    """
    return greedy(iou_pairs(gts, preds))


def region_to_class(lab, gt, valid):
    """Give each region the GT class it overlaps most, as `_region_to_class` does.

    Ties go to the smaller class id, as `np.bincount(...).argmax()` does.
    """
    m = valid & (lab >= 0)
    assign = {}
    if not m.any():
        return assign
    n_cls = max(13, int(gt.max()) + 1)
    counts = np.bincount(lab[m].astype(np.int64) * n_cls + gt[m],
                         minlength=(int(lab.max()) + 1) * n_cls).reshape(-1, n_cls)
    for r in np.flatnonzero(counts.sum(1)).tolist():
        assign[int(r)] = int(counts[r].argmax())
    return assign


def oracle_miou(et, lab, gt, valid):
    """Return oracle mIoU as the tagged `eval_frame` computes it."""
    assign = region_to_class(lab, gt, valid)
    pred = np.zeros_like(gt)
    if assign:
        lut_ids = np.array(sorted(assign))
        lut_cls = np.array([assign[r] for r in lut_ids])
        hit = np.isin(lab, lut_ids)
        pred[hit] = lut_cls[np.searchsorted(lut_ids, lab[hit])]
    classes = set(present(gt[valid])) | set(present(pred[valid]))
    classes.discard(et.BACKGROUND)
    ious = []
    for c in classes:
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = int((p | g).sum())
        ious.append(int((p & g).sum()) / union if union else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def merge_by_instance(lab, valid, gts):
    """Merge the regions that overlap one GT instance most into one region.

    Every region with pixels in the domain goes to the GT instance it shares
    the most pixels with (the first such instance on a tie); the regions of
    one instance take the smallest id among them. A region that touches no
    instance keeps its id.

    Args:
        lab: (H, W) int labels, -1 for no region.
        valid: (H, W) bool evaluation domain.
        gts: List of (H, W) bool GT instance masks (disjoint).

    Returns:
        The merged (H, W) labels.
    """
    inst = np.full(lab.shape, -1, np.int32)
    for j, g in enumerate(gts):
        inst[g] = j
    v = valid & (lab >= 0) & (inst >= 0)
    owner = {}
    if v.any():
        k = len(gts)
        counts = np.bincount(lab[v].astype(np.int64) * k + inst[v],
                             minlength=(int(lab.max()) + 1) * k).reshape(-1, k)
        for r in np.flatnonzero(counts.sum(1)).tolist():
            owner[r] = int(counts[r].argmax())
    groups = {}
    for r, j in owner.items():
        groups.setdefault(j, []).append(r)
    out = lab.copy()
    for rs in groups.values():
        target = min(rs)
        for r in rs:
            if r != target:
                out[lab == r] = target
    return out


def iou_pairs(gts, preds):
    """Return every (iou, gt index, region index) with IoU > 0, from counts.

    The IoU of each pair is the same integer ratio the tagged `_match_ious`
    computes from masks, so `greedy` on these pairs gives its matches.

    Args:
        gts: List of (H, W) bool GT masks, disjoint.
        preds: List of (H, W) bool region masks, disjoint.
    """
    if not gts or not preds:
        return []
    shape = gts[0].shape
    gi = np.full(shape, -1, np.int64)
    for i, g in enumerate(gts):
        gi[g] = i
    pj = np.full(shape, -1, np.int64)
    for j, p in enumerate(preds):
        pj[p] = j
    both = (gi >= 0) & (pj >= 0)
    counts = np.bincount(gi[both] * len(preds) + pj[both], minlength=len(gts) * len(preds))
    ga = [int(g.sum()) for g in gts]
    pa = [int(p.sum()) for p in preds]
    out = []
    for c in np.flatnonzero(counts).tolist():
        i, j = divmod(c, len(preds))
        n = int(counts[c])
        out.append((n / (ga[i] + pa[j] - n), i, j))
    return out


def greedy(pairs):
    """Match greedily, one to one, in the tagged order: (IoU, GT, region) descending."""
    gused, pused, out = set(), set(), []
    for iou, gi, pj in sorted(pairs, reverse=True):
        if gi in gused or pj in pused:
            continue
        gused.add(gi)
        pused.add(pj)
        out.append((iou, gi, pj))
    return out


def frame_scores(et, lab, gt, valid, cut, thresholds, ignore, gts=None):
    """Score one frame: instance F1, region and instance counts, oracle mIoU.

    Args:
        gts: The frame's GT instance masks at this cut, when already split.

    Returns:
        A dict; the F1 entries are None when the frame has no GT instance.
    """
    if gts is None:
        gts = [m for m, _ in gt_instances(gt, valid, cut, ignore)]
    preds = [m for m, _ in pred_regions(lab, valid, cut)]
    out = dict(n_reg=len(present(lab)), n_pred=len(preds),
               n_gt=len(gts), miou=oracle_miou(et, lab, gt, valid))
    ious = [p[0] for p in greedy(iou_pairs(gts, preds))]
    ng, npd = len(gts), len(preds)
    for t in thresholds:
        tp = sum(1 for i in ious if i >= t)
        denom = 2 * tp + (npd - tp) + (ng - tp)
        out[f"tp@{t}"] = tp
        out[f"f1@{t}"] = (2 * tp / denom if denom else 0.0) if gts else None
    return out


def load_window(et, cfg, clip):
    """Load a window's valid-depth masks and GT class maps.

    Returns:
        (valids, gts): valids is (N, H, W) bool; gts maps a frame index to
        its (H, W) class id map, for the annotated frames only.
    """
    root = cfg["paths"]["cholec_gt"]
    depth = et._load_depth(root, clip)
    n, h, w = depth.shape
    valids = np.isfinite(depth) & (depth > 1e-6)
    gts = {}
    for af in range(n):
        g = et._gt_idmap(root, clip, af, h, w)
        if g is not None:
            gts[af] = g
    return valids, gts


def load_labels(track_dir, n, shape):
    """Load a condition's labels as the tagged evaluator does, one per frame.

    Raises:
        FileNotFoundError: A frame has no label file.
    """
    out = {}
    for af in range(n):
        p = os.path.join(track_dir, f"label_{af:04d}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        lab = np.load(p).astype(np.int32)
        if lab.shape != shape:
            lab = cv2.resize(lab, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out[af] = lab
    return out


def all_pps(cfg):
    """Return the specified points_per_side values and the added ones, sorted."""
    t = cfg["track"]
    return sorted(set(t["points_per_side"]) | set(t.get("points_per_side_extended", [])))


def conditions(cfg):
    """Return every (modality, points_per_side, track_dir template)."""
    return [(m, pps, os.path.join(cfg["track"]["sources"][pps], "{clip}", f"track_rgb_w1_{m}"))
            for pps in all_pps(cfg) for m in cfg["track"]["modalities"]]


def score_window(args):
    """Score every condition, area cut and variant of one window."""
    cfg, clip = args
    cv2.setNumThreads(1)
    et = use(cfg)
    valids, gts = load_window(et, cfg, clip)
    n, h, w = valids.shape
    thresholds = cfg["score"]["iou_thresholds"]
    ignore = {et.BACKGROUND}
    split = {(af, f): (min_area(f, valids[af]), None) for af in gts for f in filter_names(cfg)}
    for (af, f), (cut, _) in split.items():
        split[(af, f)] = (cut, [m for m, _ in gt_instances(gts[af], valids[af], cut, ignore)])
    rows = []
    for mode, pps, tmpl in conditions(cfg):
        labels = load_labels(tmpl.format(clip=clip), n, (h, w))
        for af, gt in gts.items():
            lab, valid = labels[af], valids[af]
            for fname in filter_names(cfg):
                cut, gtm = split[(af, fname)]
                base = dict(clip=clip, modality=mode, pps=pps, frame=af, filter=fname,
                            valid_px=int(valid.sum()))
                rows.append(dict(base, variant="raw", **frame_scores(
                    et, lab, gt, valid, cut, thresholds, ignore, gtm)))
                merged = merge_by_instance(lab, valid, gtm)
                rows.append(dict(base, variant="merged", **frame_scores(
                    et, merged, gt, valid, cut, thresholds, ignore, gtm)))
    return rows


def window_table(frames):
    """Average frame scores into window scores, as the tagged evaluator does.

    F1 averages the frames with at least one GT instance; the other measures
    average every annotated frame.

    Args:
        frames: The frame-level table.

    Returns:
        A long table: clip, modality, pps, filter, variant, metric, value.
    """
    keys = ["clip", "modality", "pps", "filter", "variant"]
    f1 = [c for c in frames.columns if c.startswith("f1@")]
    rest = ["n_reg", "n_pred", "n_gt", "miou"]
    a = frames.groupby(keys)[f1].mean()          # pandas skips the None (NaN) frames
    b = frames.groupby(keys)[rest].mean()
    n = frames.groupby(keys).size().rename("n_gt_frames")
    wide = pd.concat([a, b, n], axis=1).reset_index()
    return wide.melt(id_vars=keys, var_name="metric", value_name="value")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # Score: one process per window, every condition inside it.
    cfg = load()
    clips = window_clips(cfg)
    with Pool(args.workers) as pool:
        parts = pool.map(score_window, [(cfg, c) for c in clips])
    frames = pd.DataFrame([r for part in parts for r in part])

    # Save: the frame table, then the window table built from it.
    out = os.path.join(cfg["paths"]["work"], "scores")
    os.makedirs(out, exist_ok=True)
    frames.to_csv(os.path.join(out, "frames.csv.gz"), index=False)
    window_table(frames).to_csv(os.path.join(out, "windows.csv"), index=False)
    write_meta(out, cfg, what="frame and window scores of every condition",
               clips=clips, filters=filter_names(cfg),
               sources={int(k): v for k, v in cfg["track"]["sources"].items()})
    print(f"{len(frames)} frame rows from {len(clips)} windows -> {out}")


if __name__ == "__main__":
    main()
