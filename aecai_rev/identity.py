"""Item 4: identity consistency of the tracked regions over time.

On each annotated frame, regions are matched to the frame's true GT
instances by IoU >= 0.5, greedily and one to one, with the evaluator's
matching. GT instances are linked into GT tracks between adjacent sampled
frames that both carry annotation (same class, largest IoU first). From the
matches the script measures, per window: the ID switch rate, the
fragmentation of GT tracks, the survival of each tracked region forward and
backward from the seed, and the rate at which a region reappears after
vanishing (and whether it comes back on the same GT track). It writes the
long, summary and test tables, and images of ID switches. Definitions:
`04_identity/README.md`.

Usage:
    python3 -m aecai_rev.identity --workers 27
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.score import gt_instances, load_labels, load_window, match_pairs, min_area
from aecai_rev.stats import versions
from aecai_rev.sweep import default_filter
from aecai_rev.tables import window_ids, write_tables
from aecai_rev.tagged import use
from aecai_rev.windows_io import window_clips


def consecutive(frames, max_gap):
    """Return the pairs of consecutive annotated frames at most `max_gap` apart.

    Args:
        frames: Sorted annotated frame indices.
        max_gap: The largest step in sampled frames; None for any step.
    """
    return [(a, b) for a, b in zip(frames, frames[1:])
            if max_gap is None or b - a <= max_gap]


def link_gt(insts, frames, min_iou, max_gap):
    """Give every GT instance a GT track id.

    Args:
        insts: {frame: [(mask, class), ...]} for the annotated frames.
        frames: Sorted annotated frame indices.
        min_iou: A link needs an IoU above this.
        max_gap: Link only consecutive annotated frames at most this many
            sampled frames apart; None links across any gap.

    Returns:
        {frame: [track id per instance]}.
    """
    linked = dict((b, a) for a, b in consecutive(frames, max_gap))
    tracks, nxt = {}, 0
    for f in frames:
        ids = [None] * len(insts[f])
        prev = linked.get(f)
        if prev is not None:
            pairs = []
            for i, (mi, ci) in enumerate(insts[f]):
                for j, (mj, cj) in enumerate(insts[prev]):
                    if ci != cj:
                        continue
                    u = int((mi | mj).sum())
                    iou = int((mi & mj).sum()) / u if u else 0.0
                    if iou > min_iou:
                        pairs.append((iou, i, j))
            used_i, used_j = set(), set()
            for iou, i, j in sorted(pairs, reverse=True):
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                ids[i] = tracks[prev][j]
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = nxt
                nxt += 1
        tracks[f] = ids
    return tracks


def frame_matches(lab, valid, insts, match_iou, cut):
    """Return {region id: GT instance index} for the matches with IoU >= match_iou."""
    regions = [(m, r) for r in np.unique(lab[valid & (lab >= 0)])
               for m in [(lab == r) & valid] if int(m.sum()) >= cut]
    pairs = match_pairs([m for m, _ in insts], [m for m, _ in regions])
    return {regions[pj][1]: gi for iou, gi, pj in pairs if iou >= match_iou}


def survival_and_reappearance(labels, seed, ids):
    """Measure, per region seeded, survival on each side and reappearance.

    Returns:
        A list of dicts, one per region id, with the frames it is present
        on each side and its reappearance events (vanish frame, return
        frame, direction).
    """
    n = len(labels)
    present = {r: [bool((labels[f] == r).any()) for f in range(n)] for r in ids}
    out = []
    for r in ids:
        p = present[r]
        fwd, bwd = p[seed + 1:], p[:seed][::-1]
        events = []
        for direction, seq, frames in (("forward", fwd, range(seed + 1, n)),
                                       ("backward", bwd, range(seed - 1, -1, -1))):
            frames = list(frames)
            gone = None
            alive_before = p[seed]
            for f, here in zip(frames, seq):
                if not here and alive_before and gone is None:
                    gone = f
                if here and gone is not None:
                    events.append((gone, f, direction))
                    gone = None
                alive_before = alive_before or here
        out.append(dict(id=int(r), fwd=float(np.mean(fwd)) if fwd else None,
                        bwd=float(np.mean(bwd)) if bwd else None, events=events))
    return out


def window_identity(args):
    """Measure every identity metric of one window and condition set."""
    cfg, clip = args
    et = use(cfg)
    valids, gts = load_window(et, cfg, clip)
    n, h, w = valids.shape
    ic = cfg["identity"]
    seed = cfg["windows"]["seed_index"]
    pps = cfg["track"]["default_points_per_side"]
    fname = default_filter(cfg)
    frames = sorted(gts)
    insts, cuts = {}, {}
    for f in frames:
        cuts[f] = min_area(fname, valids[f])
        insts[f] = gt_instances(gts[f], valids[f], cuts[f], {et.BACKGROUND})
    labels_of = {}
    for mode in cfg["track"]["modalities"]:
        tdir = os.path.join(cfg["track"]["sources"][pps], clip, f"track_rgb_w1_{mode}")
        labels_of[mode] = load_labels(tdir, n, (h, w))
    rows, events = [], []
    for link, max_gap in ic["links"].items():
        gtrack = link_gt(insts, frames, ic["gt_link_min_iou"], max_gap)
        pairs_of = consecutive(frames, max_gap)
        for mode in cfg["track"]["modalities"]:
            labels = labels_of[mode]
            match = {f: frame_matches(labels[f], valids[f], insts[f], ic["match_iou"], cuts[f])
                     for f in frames}
            track_of = {f: {r: gtrack[f][gi] for r, gi in match[f].items()} for f in frames}
            class_of = {f: {r: insts[f][gi][1] for r, gi in match[f].items()} for f in frames}

            # ID switches over the pairs of frames the linking rule links.
            pairs = switches = cswitches = 0
            for a, b in pairs_of:
                for r in set(track_of[a]) & set(track_of[b]):
                    pairs += 1
                    if track_of[a][r] != track_of[b][r]:
                        switches += 1
                        events.append(dict(clip=clip, link=link, modality=mode, region=int(r),
                                           frame_a=a, frame_b=b, gt_a=track_of[a][r],
                                           gt_b=track_of[b][r], class_a=class_of[a][r],
                                           class_b=class_of[b][r]))
                    if class_of[a][r] != class_of[b][r]:
                        cswitches += 1

            # Fragmentation: distinct regions matched to one GT track.
            by_gt = {}
            for f in frames:
                for r, t in track_of[f].items():
                    by_gt.setdefault(t, set()).add(r)
            frag = [len(s) for s in by_gt.values()]

            # Survival and reappearance of the regions present on the seed frame.
            ids = [int(r) for r in np.unique(labels[seed]) if r >= 0]
            sr = survival_and_reappearance([labels[f] for f in range(n)], seed, ids)
            same_gt, same_cls = [], []
            for rec in sr:
                r = rec["id"]
                for gone, back, direction in rec["events"]:
                    step = 1 if direction == "forward" else -1
                    end = n if step == 1 else -1
                    before = [f for f in range(seed, gone, step) if r in track_of.get(f, {})]
                    after = [f for f in range(back, end, step) if r in track_of.get(f, {})]
                    if before and after:
                        fb, fa = before[-1], after[0]
                        same_gt.append(track_of[fb][r] == track_of[fa][r])
                        same_cls.append(class_of[fb][r] == class_of[fa][r])
            reappear = [bool(rec["events"]) for rec in sr]
            vals = dict(
                id_switch_rate=switches / pairs if pairs else None,
                class_switch_rate=cswitches / pairs if pairs else None,
                matched_pairs=pairs,
                fragmentation_mean=float(np.mean(frag)) if frag else None,
                fragmentation_ge2=float(np.mean([k >= 2 for k in frag])) if frag else None,
                gt_tracks_matched=len(frag),
                survival_forward=float(np.mean([rec["fwd"] for rec in sr])) if sr else None,
                survival_backward=float(np.mean([rec["bwd"] for rec in sr])) if sr else None,
                reappearance_rate=float(np.mean(reappear)) if sr else None,
                reappear_same_gt=float(np.mean(same_gt)) if same_gt else None,
                reappear_same_class=float(np.mean(same_cls)) if same_cls else None,
                reappear_events_judged=len(same_gt),
                seeded_regions=len(sr))
            for k, v in vals.items():
                rows.append(dict(clip=clip, link=link, modality=mode, pps=pps, metric=k,
                                 value=v))
    return rows, events


def frames_in_order(cfg, clip):
    """Say whether a window's native frame numbers increase throughout."""
    with open(os.path.join(cfg["paths"]["cholec_gt"], clip, "frame_manifest.json")) as f:
        nat = [fr["native_frame"] for fr in json.load(f)["frames"]]
    return all(b > a for a, b in zip(nat, nat[1:]))


def draw_examples(cfg, events, out, k):
    """Draw k ID switches: the region and its matched GT instances, both frames."""
    et = use(cfg)
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        if old.endswith(".png"):
            os.remove(os.path.join(out, old))
    pps = cfg["track"]["default_points_per_side"]
    primary = next(iter(cfg["identity"]["links"]))
    ev = events[events["link"] == primary].copy()
    ev["same_class"] = ev["class_a"] == ev["class_b"]
    picked = ev.sort_values(["same_class", "clip"]).drop_duplicates("clip").head(k)
    for _, e in picked.iterrows():
        valids, gts = load_window(et, cfg, e["clip"])
        n, h, w = valids.shape
        tdir = os.path.join(cfg["track"]["sources"][pps], e["clip"], f"track_rgb_w1_{e['modality']}")
        labels = load_labels(tdir, n, (h, w))
        tiles = []
        for f, cls in ((e["frame_a"], e["class_a"]), (e["frame_b"], e["class_b"])):
            img = cv2.imread(os.path.join(cfg["paths"]["cholec_gt"], e["clip"], "input_images",
                                          f"{f:06d}.png"))
            img = cv2.resize(img, (w, h))
            reg = (labels[f] == e["region"]).astype(np.uint8)
            gt = ((gts[f] == cls) & valids[f]).astype(np.uint8)
            over = img.copy()
            over[reg > 0] = (0.5 * over[reg > 0] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)
            cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cs, -1, (0, 255, 255), 2)
            name = cfg["classes"]["names"][int(cls)]
            cv2.putText(over, f"t={f} region {e['region']} -> GT {name}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(over)
        cv2.imwrite(os.path.join(out, f"{e['clip']}_{e['modality']}_r{e['region']}_"
                                      f"t{e['frame_a']}-{e['frame_b']}.png"), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # Measure every window.
    cfg = load()
    clips = window_clips(cfg)
    with Pool(args.workers) as pool:
        parts = pool.map(window_identity, [(cfg, c) for c in clips])
    long = pd.DataFrame([r for rows, _ in parts for r in rows])
    events = pd.DataFrame([e for _, ev in parts for e in ev])
    ids = window_ids(cfg)
    long["window_id"] = long["clip"].map(ids)

    # Tables: all windows, and the windows whose frames run in order.
    out = os.path.join(cfg["paths"]["results"], "04_identity")
    in_order = {c: frames_in_order(cfg, c) for c in clips}
    long["subset"] = "all"
    sub = long[long["clip"].map(in_order)].copy()
    sub["subset"] = "frames_in_order"
    both = pd.concat([long, sub])
    write_tables(out, both, ["subset", "link", "modality", "metric"], cfg)
    events.to_csv(os.path.join(out, "id_switch_events.csv"), index=False)
    pd.DataFrame(sorted(in_order.items()), columns=["clip", "frames_in_order"]).to_csv(
        os.path.join(out, "frames_in_order.csv"), index=False)
    if len(events):
        draw_examples(cfg, events, os.path.join(out, "examples"), cfg["identity"]["n_examples"])
    write_meta(out, cfg, what="item 4: identity consistency", filter=default_filter(cfg),
               points_per_side=cfg["track"]["default_points_per_side"],
               identity=cfg["identity"], libraries=versions())
    print(f"wrote {out}: {len(events)} ID switches")


if __name__ == "__main__":
    main()
