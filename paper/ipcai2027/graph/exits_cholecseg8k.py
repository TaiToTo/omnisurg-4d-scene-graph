"""Print the exit check on CholecSeg8k: the tracking windows, and the dense chunks with class and component nodes.

Reads three finished `evalkit.tools.exit_vs_loss` runs with DA3 poses and reference events: the tracking windows,
the dense chunks with one node per class, and the dense chunks with one node per connected component, the last run
with the reference tracks of `component_seeds.py`. For the components it gives the precision before and after
keeping the out-of-view exits, and their difference drawn over the videos. It sorts what lies under the projected
point of the exits classified within the view. For each reference exit classified within the view, it reads which
class the next reference frame holds under the node's last mask.

Usage:
    python paper/ipcai2027/graph/exits_cholecseg8k.py --windows-run DIR --class-run DIR --component-run DIR \\
        --data-root DIR --gt-tracks DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evalkit.classes import ClassTable, load_table
from evalkit.inputs import MASK_DIR, MASK_SUFFIX
from evalkit.tools.exit_vs_loss import LABELS
from evalkit.tools.paired_stats import mark_of

from component_seeds import reference_ids
from exit_events import POINT_KINDS, counts_of, per_video, point_kind, read_events, read_reference, within_view
from video_bootstrap import interval, resampled_sums

DATASET = "cholecseg8k"
# The tool keys its summaries by the margin as a string; the paper reads margin 0.
M = str(0.0)
REFERENCE_KINDS = ("relabelled", "instrument", "same_class", "background")
N_BOOT = 10_000
SEED = 0


def checkable_exits(run: Path) -> dict:
    """Return a run's exits, and those whose frame and the frame before both have a reference mask."""
    exits = [r for r in read_events(run) if r["kind"] == "exit"]
    return {"exits": len(exits), "checkable": sum(r["verifiable"] == "True" for r in exits)}


def after_minus_before(run: Path) -> dict:
    """Return the precision after keeping the out-of-view exits minus the precision before, drawn over the videos."""
    rows = [r for r in read_events(run) if r["kind"] == "exit" and r["verifiable"] == "True"]
    out_of_view = LABELS["exit"][1]
    videos, pv = per_video(rows, [lambda r: r["label"] == out_of_view and r["gt_t"] != "",
                                  lambda r: r["label"] == out_of_view, lambda r: r["gt_t"] != "", lambda r: True])
    draws = resampled_sums(pv, N_BOOT, SEED)
    t = pv.sum(axis=0)
    ci = interval(draws[:, 0] / draws[:, 1] - draws[:, 2] / draws[:, 3])
    return {"diff": float(t[0] / t[1] - t[2] / t[3]), "ci95_video": ci, "videos": len(videos),
            "mark": mark_of(ci, +1, n_videos=len(videos))}


def reference_kind(table: ClassTable, node_class: int, ids: np.ndarray, before: np.ndarray) -> str:
    """Return what the next reference frame's majority mask id under a vanished node's last mask is."""
    mid, _ = Counter(int(v) for v in ids[before]).most_common(1)[0]
    c = table.mask_id_to_class.get(mid)
    kind = str(table.type_of(c)) if c is not None else "ignored"
    if kind in ("ignored", "background", "excluded"):
        return "background"
    if kind == "tool":
        return "instrument"
    return "same_class" if c == node_class else "relabelled"


def reference_within_view(run: Path, data_root: Path, gt_tracks: Path, table: ClassTable) -> dict:
    """Return the reference exits, those classified within the view, and what replaces each of these."""
    ref = read_reference(run, "exit")
    kinds = []
    for r in within_view(ref, "exit"):
        clip, t, node = r["clip"], int(r["t"]), int(r["node"])
        before = np.load(gt_tracks / clip / f"label_{t - 1:04d}.npy") == node
        ids = reference_ids(data_root / clip / MASK_DIR / f"{t:06d}{MASK_SUFFIX[DATASET]}", table, before.shape)
        kinds.append(reference_kind(table, int(r["class_id"]), ids, before))
    return {"reference_exits": len(ref), "within_view": len(kinds), "kinds": dict(Counter(kinds))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows-run", required=True, type=Path, help="the exit run on the tracking windows")
    ap.add_argument("--class-run", required=True, type=Path, help="the exit run on the chunks, one node per class")
    ap.add_argument("--component-run", required=True, type=Path,
                    help="the exit run on the chunks, one node per connected component")
    ap.add_argument("--data-root", required=True, type=Path, help="the dense chunks with their reference masks")
    ap.add_argument("--gt-tracks", required=True, type=Path, help="the components' reference tracks")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    table = load_table(DATASET)
    s = json.loads((a.component_run / "summary.json").read_text())
    pr = s["precision_recall"]["exit"][M]
    shuffles = [v["exit"][M]["after"]["precision"] for k, v in s["precision_recall_controls"].items()
                if k.startswith("shuffle_")]

    # The windows and the class nodes: how many exits, and how many can be checked.
    out = {"windows": checkable_exits(a.windows_run), "class_nodes": checkable_exits(a.class_run)}
    # The component nodes: the counts, precision and recall before and after, and the shuffled poses.
    lost = within_view([r for r in read_events(a.component_run) if r["kind"] == "exit"], "exit")
    out["component_nodes"] = {
        "clips": s["clips"], "videos": s["videos"], "nodes": s["nodes"], "exits": s["events"]["exit"],
        "reference_exits": s["gt_events"]["exit"],
        "before": {k: pr["before"][k] for k in ("n_pred", "tp", "precision", "precision_ci95_video", "recall")},
        "after": {k: pr["after"][k] for k in ("n_pred", "tp", "precision", "precision_ci95_video", "recall")},
        "after_minus_before": after_minus_before(a.component_run),
        "shuffled_precision_after": {"mean": float(np.mean(shuffles)), "shuffles": len(shuffles)},
        # What lies under the projected point of the exits classified within the view, and what replaces the
        # reference exits classified within the view.
        "within_view_exits": len(lost),
        "class_under": counts_of([point_kind(table, int(r["class_id"]), r["gt_class_under"]) for r in lost],
                                 POINT_KINDS),
        "reference_within_view": reference_within_view(a.component_run, a.data_root, a.gt_tracks, table)}
    a.out.write_text(json.dumps(out, indent=1))

    w, c, cn = out["windows"], out["class_nodes"], out["component_nodes"]
    d, cu, rw = cn["after_minus_before"], cn["class_under"], cn["reference_within_view"]
    print(f"windows: {w['exits']} exits, {w['checkable']} checkable; class nodes: {c['exits']} exits")
    print(f"component nodes: {cn['nodes']} nodes, {cn['clips']} chunks, {cn['videos']} videos; {cn['exits']} exits, "
          f"{cn['reference_exits']} reference exits")
    print(f"precision {cn['before']['precision']:.2f} -> {cn['after']['precision']:.2f} "
          f"[{cn['after']['precision_ci95_video'][0]:.2f}, {cn['after']['precision_ci95_video'][1]:.2f}] "
          f"({d['diff']:+.2f} [{d['ci95_video'][0]:+.2f}, {d['ci95_video'][1]:+.2f}] {d['mark']}, {d['videos']} videos); "
          f"recall {cn['before']['recall']:.2f} -> {cn['after']['recall']:.2f}; "
          f"shuffled {cn['shuffled_precision_after']['mean']:.2f}")
    print(f"{cn['within_view_exits']} exits within view, class under the point: "
          + ", ".join(f"{k} {100 * cu[k]['share']:.0f} %" for k in POINT_KINDS))
    print(f"{rw['within_view']} of {rw['reference_exits']} reference exits within view: "
          + ", ".join(f"{k} {rw['kinds'].get(k, 0)}" for k in REFERENCE_KINDS))


if __name__ == "__main__":
    main()
