"""Items 5 and 6: instrument-tissue merges and nameability, from the overlay.

On each annotated frame the script builds the clinical overlay with the
tagged `build_hierarchy_frame` and its default thresholds: a region and a
CholecSeg8k class are linked when one covers at least `CONTAIN_TAU` of the
other, or their IoU is at least `OVERLAP_TAU`. From the links it measures,
per window and modality at the AE-CAI setting:

- the merge rate (item 5): the share of regions linked to an instrument
  class and a tissue class at once, and the share of frames with such a
  region;
- nameability (item 6): the share of regions linked to exactly one class,
  to two or more, and to none.

It writes `05_merge_rate/` and `06_naming/` (long, summary and test tables)
and three images of merged regions.

Usage:
    python3 -m aecai_rev.overlay --workers 27
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.score import load_labels, load_window
from aecai_rev.stats import versions
from aecai_rev.tables import window_ids, write_tables
from aecai_rev.tagged import use
from aecai_rev.windows_io import window_clips


def overlay_api(cfg):
    """Return the tagged overlay builder, its thresholds and the label tables."""
    use(cfg)
    from surgical_core.viewer import hierarchy_frame as hf
    from surgical_core.viewer.labels import LabelTable, cholec_gt_table
    return hf, cholec_gt_table(), LabelTable(entries={}, background_ids={-1})


def links_of(hf, gt_table, region_table, gt, lab, valid):
    """Return {region id: set of linked class ids} and the regions present."""
    frame = hf.build_hierarchy_frame(gt.astype(np.int64), gt_table, "cholecseg8k",
                                     lab.astype(np.int64), region_table, "sam3d", valid=valid)
    regions = [n["id"] for n in frame["nodes"] if n["track"] == "sam3d"]
    links = {r: set() for r in regions}
    for e in frame["edges"]:
        a_track, a_id = e["src"].split(":")
        b_id = int(e["dst"].split(":")[1])
        cls, reg = (int(a_id), b_id) if a_track == "cholecseg8k" else (b_id, int(a_id))
        links[reg].add(cls)
    return links, regions


def window_overlay(args):
    """Measure merge rate and nameability for one window, every modality."""
    cfg, clip = args
    et = use(cfg)
    hf, gt_table, region_table = overlay_api(cfg)
    valids, gts = load_window(et, cfg, clip)
    n, h, w = valids.shape
    pps = cfg["track"]["default_points_per_side"]
    inst = set(cfg["classes"]["instrument"])
    tissue = set(cfg["classes"]["tissue"])
    rows, examples = [], []
    for mode in cfg["track"]["modalities"]:
        tdir = os.path.join(cfg["track"]["sources"][pps], clip, f"track_rgb_w1_{mode}")
        labels = load_labels(tdir, n, (h, w))
        per = []
        for f, gt in sorted(gts.items()):
            links, regions = links_of(hf, gt_table, region_table, gt, labels[f], valids[f])
            if not regions:
                continue
            merged = [r for r in regions if links[r] & inst and links[r] & tissue]
            counts = [len(links[r]) for r in regions]
            per.append(dict(merge_region_rate=len(merged) / len(regions),
                            merge_frame=float(bool(merged)),
                            named_one=np.mean([c == 1 for c in counts]),
                            named_many=np.mean([c >= 2 for c in counts]),
                            named_none=np.mean([c == 0 for c in counts]),
                            regions=len(regions)))
            for r in merged:
                examples.append(dict(clip=clip, modality=mode, frame=f, region=int(r),
                                     classes=" ".join(map(str, sorted(links[r])))))
        for k in per[0] if per else []:
            rows.append(dict(clip=clip, modality=mode, pps=pps, metric=k,
                             value=float(np.mean([p[k] for p in per]))))
    return rows, examples


def draw(cfg, ex, out, k):
    """Draw k merged regions: the region filled, instrument and tissue outlined."""
    et = use(cfg)
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        if old.endswith(".png"):
            os.remove(os.path.join(out, old))
    pps = cfg["track"]["default_points_per_side"]
    picked = ex[ex["modality"] == "normal"].drop_duplicates("clip").head(k)
    for _, e in picked.iterrows():
        valids, gts = load_window(et, cfg, e["clip"])
        n, h, w = valids.shape
        tdir = os.path.join(cfg["track"]["sources"][pps], e["clip"], f"track_rgb_w1_{e['modality']}")
        lab = load_labels(tdir, n, (h, w))[e["frame"]]
        img = cv2.imread(os.path.join(cfg["paths"]["cholec_gt"], e["clip"], "input_images",
                                      f"{e['frame']:06d}.png"))
        img = cv2.resize(img, (w, h))
        reg = lab == e["region"]
        img[reg] = (0.5 * img[reg] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)
        for classes, color in ((cfg["classes"]["instrument"], (0, 255, 255)),
                               (cfg["classes"]["tissue"], (0, 0, 255))):
            m = (np.isin(gts[e["frame"]], classes) & valids[e["frame"]]).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, color, 2)
        names = ", ".join(cfg["classes"]["names"][int(c)] for c in e["classes"].split())
        cv2.putText(img, f"t={e['frame']} region {e['region']}: {names}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out, f"{e['clip']}_{e['modality']}_r{e['region']}_t{e['frame']}.png"),
                    img)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # Measure every window.
    cfg = load()
    clips = window_clips(cfg)
    with Pool(args.workers) as pool:
        parts = pool.map(window_overlay, [(cfg, c) for c in clips])
    long = pd.DataFrame([r for rows, _ in parts for r in rows])
    ex = pd.DataFrame([e for _, es in parts for e in es])
    long["window_id"] = long["clip"].map(window_ids(cfg))
    hf, _, _ = overlay_api(cfg)
    thresholds = dict(contain_tau=hf.CONTAIN_TAU, overlap_tau=hf.OVERLAP_TAU)

    # Item 5: the merge rate, and images of merged regions.
    out5 = os.path.join(cfg["paths"]["results"], "05_merge_rate")
    m5 = long[long["metric"].isin(["merge_region_rate", "merge_frame", "regions"])]
    write_tables(out5, m5, ["modality", "metric"], cfg)
    ex.to_csv(os.path.join(out5, "merged_regions.csv"), index=False)
    draw(cfg, ex, os.path.join(out5, "examples"), 3)
    write_meta(out5, cfg, what="item 5: instrument-tissue merge rate",
               overlay_thresholds=thresholds, classes=cfg["classes"], libraries=versions())

    # Item 6: nameability.
    out6 = os.path.join(cfg["paths"]["results"], "06_naming")
    m6 = long[long["metric"].isin(["named_one", "named_many", "named_none", "regions"])]
    write_tables(out6, m6, ["modality", "metric"], cfg)
    write_meta(out6, cfg, what="item 6: nameability from the overlay",
               overlay_thresholds=thresholds, libraries=versions())
    print(f"wrote {out5} and {out6}")


if __name__ == "__main__":
    main()
