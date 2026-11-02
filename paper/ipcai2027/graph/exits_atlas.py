"""Print the exit check on ATLAS-120k: how its exits and entries are classified, and their precision and recall.

Reads a finished `evalkit.tools.exit_vs_loss` run with DA3 poses and reference events, and the image-border rule's
labels that `evalkit.tools.exit_vs_loss_rescore` wrote under it (`rescore_poseless_ref/per_event.csv`). The shares,
precisions and intervals of the poses and the static camera are the tool's. This script draws the interval of the
shuffled poses' precision minus the estimated poses' and of the border rule's precision. It also counts what lies
under the projected point of the exits classified within the view, and that share by class type.

Usage:
    python paper/ipcai2027/graph/exits_atlas.py --run DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evalkit.classes import load_table
from evalkit.tools.exit_vs_loss import LABELS, STATIC, WITH_POSES
from evalkit.tools.exit_vs_loss_rescore import BOUNDARY2D, OUT_DIR
from evalkit.tools.paired_stats import boot_ci, mark_of

from exit_events import (NODE_KINDS, POINT_KINDS, counts_of, node_kind, per_video, point_kind, read_events,
                         read_reference, within_view)
from video_bootstrap import interval, resampled_sums

DATASET = "atlas120k"
# The tool keys its summaries by the margin as a string; the paper reads margin 0.
M = str(0.0)
# The border rule keeps an exit whose last tracked mask lies within this share of the frame's width of the border.
BORDER = "0.02"
# The class types the paper reports the within-view share of, under the names it gives them.
TYPES = {"tool": "instruments", "tissue": "tissue", "expert": "vessels_and_ducts"}
N_BOOT = 10_000
SEED = 0


def shares(summary: dict, kind: str, pose_set: str) -> dict:
    """Return the tool's share of each label, with its interval, for one kind and pose set."""
    s = summary["shares"][kind][pose_set][M]
    return {label: {"share": s[label]["share"], "ci95_video": s[label]["ci95_video"]} for label in LABELS[kind]}


def scores(table: dict) -> dict:
    """Return the precision with its interval and the recall of one of the tool's precision tables."""
    return {k: table[k] for k in ("n_pred", "tp", "n_gt", "precision", "precision_ci95_video", "recall")}


def shuffled(summary: dict, kind: str) -> dict:
    """Return the mean and the range over the shuffles of each label's share and of the precision after."""
    names = sorted(p for p in summary["shares"][kind] if p.startswith("shuffle_"))
    out = {}
    for label in LABELS[kind]:
        v = [summary["shares"][kind][p][M][label]["share"] for p in names]
        out[label] = {"mean": float(np.mean(v)), "range": [min(v), max(v)]}
    p = [summary["precision_recall_controls"][n][kind][M]["after"]["precision"] for n in names]
    out["precision_after"] = {"mean": float(np.mean(p)), "range": [min(p), max(p)], "shuffles": len(names)}
    return out


def shuffled_minus_estimated(run: Path) -> dict:
    """Return the precision after of the pooled shuffles minus that of the estimated poses, drawn over the videos."""
    rows = [r for r in read_events(run, None) if r["kind"] == "exit" and r["verifiable"] == "True"]
    out_of_view = LABELS["exit"][1]

    def kept(shuffles: bool, matched: bool):
        """Count a row kept under the estimated poses or a shuffle, and only a matched one when `matched`."""
        return lambda r: ((r["pose_set"].startswith("shuffle_") if shuffles else r["pose_set"] == WITH_POSES)
                          and r["label"] == out_of_view and (r["gt_t"] != "" or not matched))
    videos, pv = per_video(rows, [kept(False, False), kept(False, True), kept(True, False), kept(True, True)])
    draws = resampled_sums(pv, N_BOOT, SEED)
    t = pv.sum(axis=0)
    ci = interval(draws[:, 3] / draws[:, 2] - draws[:, 1] / draws[:, 0])
    return {"diff": float(t[3] / t[2] - t[1] / t[0]), "ci95_video": ci, "videos": len(videos),
            "mark": mark_of(ci, +1, n_videos=len(videos))}


def border_rule(run: Path, n_gt: int) -> dict:
    """Return the border rule's within-view share, and the precision and recall of the exits it keeps.

    Raises:
        ValueError: the rescoring and the run name different exits.
    """
    ev = {(r["clip"], r["node"], r["t"]): r for r in read_events(run) if r["kind"] == "exit"}
    with (run / OUT_DIR / "per_event.csv").open() as f:
        rule = {(r["clip"], r["node"], r["t"]): r[f"{BOUNDARY2D}_{BORDER}"] for r in csv.DictReader(f)
                if r["kind"] == "exit"}
    if set(ev) != set(rule):
        raise ValueError(f"{run}: the rescoring names {len(rule)} exits, the run {len(ev)}; they must be the same")
    kept = [ev[k] for k in sorted(ev) if rule[k] == LABELS["exit"][1]]
    hit = np.array([r["gt_t"] != "" for r in kept], dtype=float)
    lo, hi = boot_ci(hit, np.array([r["video"] for r in kept]))
    return {"within_view": 1 - len(kept) / len(ev), "out_of_view": len(kept) / len(ev), "n_kept": len(kept),
            "tp": int(hit.sum()), "precision": float(hit.mean()), "precision_ci95_video": [lo, hi],
            "recall": hit.sum() / n_gt}


def within_view_facts(run: Path) -> dict:
    """Return what lies under the point of the exits classified within the view, and that share by class type."""
    table = load_table(DATASET)
    rows = [r for r in read_events(run) if r["kind"] == "exit"]
    lost = within_view(rows, "exit")
    by_type = {}
    for ty, name in TYPES.items():
        sel = [r for r in rows if str(table.type_of(int(r["class_id"]))) == ty]
        by_type[name] = {"n": len(sel), "within_view": len(within_view(sel, "exit"))}
    ref = read_reference(run, "exit")
    return {"n_within_view": len(lost),
            "node_under": counts_of([node_kind(r) for r in lost], NODE_KINDS),
            "class_under": counts_of([point_kind(table, int(r["class_id"]), r["gt_class_under"]) for r in lost],
                                     POINT_KINDS),
            "by_type": by_type,
            "reference_exits": len(ref), "reference_exits_within_view": len(within_view(ref, "exit"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="the exit tool's output directory, rescored")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    s = json.loads((a.run / "summary.json").read_text())
    pr, prc = s["precision_recall"], s["precision_recall_controls"]

    # The counts, the classification under each pose set, and the precision before and after.
    out = {"clips": s["clips"], "videos": s["videos"], "nodes_tracked": s["nodes_tracked"], "events": s["events"],
           "reference_events": s["gt_events"],
           "exits": {"before": scores(pr["exit"][M]["before"]),
                     "estimated": {"shares": shares(s, "exit", WITH_POSES), "after": scores(pr["exit"][M]["after"])},
                     "static_camera": {"shares": shares(s, "exit", STATIC),
                                       "after": scores(prc[STATIC]["exit"][M]["after"])},
                     "shuffled": shuffled(s, "exit"),
                     "shuffled_minus_estimated": shuffled_minus_estimated(a.run),
                     "border_rule": border_rule(a.run, pr["exit"][M]["before"]["n_gt"])},
           "entries": {"before": scores(pr["enter"][M]["before"]),
                       "estimated": {"shares": shares(s, "enter", WITH_POSES),
                                     "after": scores(pr["enter"][M]["after"])}},
           "within_view_exits": within_view_facts(a.run)}
    a.out.write_text(json.dumps(out, indent=1))

    # The table, shares in %.
    ex, en, w = out["exits"], out["entries"], out["within_view_exits"]

    def row(name: str, sh: dict, kind: str) -> str:
        lost, left, leaving = (sh[label] for label in LABELS[kind])
        lo, hi = (100 * x for x in lost["ci95_video"])
        return f"{name:<16} {100 * lost['share']:5.1f} [{lo:.1f}, {hi:.1f}]  {100 * left['share']:5.1f}  {100 * leaving['share']:5.1f}"

    def pr_text(t: dict) -> str:
        lo, hi = t["precision_ci95_video"]
        return f"{t['precision']:.2f} [{lo:.2f}, {hi:.2f}] / {t['recall']:.2f}"
    print(f"{out['nodes_tracked']} nodes, {out['clips']} clips: {out['events']['exit']} exits, {out['events']['enter']} "
          f"entries; reference {out['reference_events']['exit']} exits, {out['reference_events']['enter']} entries")
    print("exits            within view         out    leaving   precision / recall")
    print(f"{'before':<46} {pr_text(ex['before'])}")
    print(row("estimated", ex["estimated"]["shares"], "exit") + f"   {pr_text(ex['estimated']['after'])}")
    print(row("static camera", ex["static_camera"]["shares"], "exit"))
    sh, d = ex["shuffled"], ex["shuffled_minus_estimated"]
    print(f"{'shuffled':<16} {100 * sh['lost_in_view']['mean']:5.1f} ({100 * sh['lost_in_view']['range'][0]:.1f}-"
          f"{100 * sh['lost_in_view']['range'][1]:.1f})  {100 * sh['left_view']['mean']:5.1f}  {100 * sh['leaving']['mean']:5.1f}"
          f"   {sh['precision_after']['mean']:.2f} ({sh['precision_after']['range'][0]:.2f}-{sh['precision_after']['range'][1]:.2f}); "
          f"shuffled - estimated {d['diff']:+.2f} [{d['ci95_video'][0]:+.2f}, {d['ci95_video'][1]:+.2f}] {d['mark']}")
    b = ex["border_rule"]
    print(f"{'border rule':<16} {100 * b['within_view']:5.1f}               {100 * b['out_of_view']:5.1f}           {pr_text(b)}")
    print("entries")
    print(f"{'before':<46} {pr_text(en['before'])}")
    print(row("estimated", en["estimated"]["shares"], "enter") + f"   {pr_text(en['estimated']['after'])}")
    nu, cu = w["node_under"], w["class_under"]
    print(f"within view: {nu['another_node']['n']} of {w['n_within_view']} with another node under the point; class there: "
          + ", ".join(f"{k} {100 * cu[k]['share']:.0f} %" for k in POINT_KINDS))
    print("within view by type: " + ", ".join(f"{k} {v['within_view']} of {v['n']}" for k, v in w["by_type"].items())
          + f"; reference exits within view {w['reference_exits_within_view']} of {w['reference_exits']}")


if __name__ == "__main__":
    main()
