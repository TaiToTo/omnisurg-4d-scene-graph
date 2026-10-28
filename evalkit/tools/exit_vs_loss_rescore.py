"""Score the exit tool's labels against a reference that uses no pose: the GT mask's presence and border contact.

`exit_vs_loss` sorts a predicted exit by projecting the node's last 3D position through the poses, and the first
round sorted the GT exits the same way, so the two sides shared the poses. Here the reference is read from the GT
masks alone. A predicted exit at frame t is `gt_present` when the node's GT mask is there at t (the exit is a loss),
`gt_exit_border` when it is not and the mask's box on the last GT frame that holds it touches the frame's border
within b, `gt_exit_interior` otherwise, and `gt_unverifiable` where t or the frame before has no GT. An enter is
sorted the same way backwards. A pose-free baseline, `boundary2d`, calls an exit left-view when the tracked mask's
box on its last frame touches the border within b. Precision and recall take `gt_exit_border` as the true exits;
intervals are the video bootstrap. The tool reads a finished run (`clip_results.json`) and the clips.

Usage:
    python -m evalkit.tools.exit_vs_loss_rescore --dataset atlas120k --run RESULTS_DIR --clips LIST --data-root DIR \\
        --tracks-root DIR --tag track_rgb_rgb_gt_forward [--gt-tracks DIR] [--borders 0.01,0.02,0.05] \\
        [--random-draws 1000] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evalkit.classes import load_table
from evalkit.evaluate import read_population
from evalkit.inputs import read_clip
from evalkit.tools.component_scores import tool_sha
from evalkit.tools.exit_vs_loss import (
    LABELS, STATIC, WITH_POSES, _resized, gt_events_of, gt_presence, read_bundle, read_gt_tracks, video_of,
)
from evalkit.tools.paired_stats import mark_of

OUT_DIR = "rescore_poseless_ref"
BORDERS = (0.01, 0.02, 0.05)
# The reference labels of each kind: the node is still in the GT, it is gone and was at the border, it is gone and
# was inside, or the GT cannot say.
REFS = {"exit": ("gt_present", "gt_exit_border", "gt_exit_interior", "gt_unverifiable"),
        "enter": ("gt_present", "gt_enter_border", "gt_enter_interior", "gt_unverifiable")}
BOUNDARY2D = "boundary2d"
SHUFFLED = "shuffled"
RANDOM = "random_subset"
BOOT = 10_000
SEED = 0


def box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The mask's box as (y0, y1, x0, x1), inclusive; None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def touches(b: tuple[int, int, int, int], shape: tuple[int, int], border_px: float) -> bool:
    """Whether the box comes within `border_px` of any edge of a frame of `shape`."""
    y0, y1, x0, x1 = b
    h, w = shape
    return y0 <= border_px or x0 <= border_px or y1 >= h - 1 - border_px or x1 >= w - 1 - border_px


def reference(kind: str, t: int, present: Mapping[int, np.ndarray], gt_frames: set[int], shape: tuple[int, int],
              border_px: float) -> tuple[str, int | None]:
    """The reference label of a predicted event at `t` from the node's GT masks, and the GT frame it was read on.

    An exit at `t` is read on `t` (present) or on the last GT frame before `t` that holds the mask; an enter at `t`
    is read on `t - 1` (present) or on the first GT frame from `t` that holds the mask. `t` and `t - 1` must both
    be GT frames, else the event is unverifiable.
    """
    if t not in gt_frames or t - 1 not in gt_frames:
        return "gt_unverifiable", None
    if kind == "exit":
        if t in present:
            return "gt_present", t
        earlier = [f for f in present if f < t]
        if not earlier:
            return "gt_unverifiable", None
        f = max(earlier)
    else:
        if t - 1 in present:
            return "gt_present", t - 1
        later = [f for f in present if f >= t]
        if not later:
            return "gt_unverifiable", None
        f = min(later)
    side = "border" if touches(box(present[f]), shape, border_px) else "interior"
    return f"gt_{kind}_{side}", f


def boundary2d(kind: str, t: int, present: Mapping[int, np.ndarray], shape: tuple[int, int], border_px: float) -> str:
    """The pose-free baseline's label: the tracked mask's box on its last frame (exit) or first frame (enter)."""
    f = max(f for f in present if f < t) if kind == "exit" else t
    out, in_ = LABELS[kind][1], LABELS[kind][0]
    return out if touches(box(present[f]), shape, border_px) else in_


def rescore_clip(res: dict, dataset: str, data_root: str, tracks_root: str, tag: str, gt_tracks: str | None,
                 min_px: int, borders: Sequence[float]) -> dict:
    """The reference and baseline labels of one clip's predicted events, and its GT events with their border flags."""
    clip, classes, seed = res["clip"], res["classes"], res["seed"]
    table = load_table(dataset)
    inputs = read_clip(data_root, tracks_root, tag, clip, table)
    depth, _, _ = read_bundle(Path(data_root) / clip, "da3")
    shape = depth.shape[1:]
    if gt_tracks is None:
        gt_frames, gt_masks = gt_presence(inputs, table, classes, shape, min_px)
    else:
        _, gt_frames, gt_masks = read_gt_tracks(Path(gt_tracks) / clip, shape, min_px)
    gt_set = set(gt_frames)
    px = {b: b * shape[1] for b in borders}

    # The tracked masks of each node, on the frames where it is present.
    tracked: list[dict[int, np.ndarray]] = []
    for n in range(len(classes)):
        masks = {t: _resized(inputs.regions[t] == n + 1, shape) for t in inputs.regions}
        tracked.append({t: m for t, m in masks.items() if int(m.sum()) >= min_px})

    # Each predicted event once, with its labels under every pose set and margin.
    events: dict[tuple[str, int, int], dict] = {}
    for r in res["events"]:
        key = (r["kind"], r["node"], r["t"])
        events.setdefault(key, {"clip": clip, "video": video_of(clip), "kind": r["kind"], "node": r["node"],
                                "class_id": r["class_id"], "t": r["t"], "labels": {}})
        events[key]["labels"][f"{r['pose_set']}|{r['margin']}"] = r["label"]
    for (kind, n, t), e in events.items():
        e["ref"], e["ref_frame"], e[BOUNDARY2D] = {}, {}, {}
        for b in borders:
            e["ref"][str(b)], e["ref_frame"][str(b)] = reference(kind, t, gt_masks[n], gt_set, shape, px[b])
            e[BOUNDARY2D][str(b)] = boundary2d(kind, t, tracked[n], shape, px[b])

    # The GT events, each with the frame its mask was read on and whether that mask touched the border.
    gt_events = []
    for n in range(len(classes)):
        for g in gt_events_of(set(gt_masks[n]), gt_frames, seed):
            f = g.t - 1 if g.kind == "exit" else g.t
            gt_events.append({"clip": clip, "video": video_of(clip), "kind": g.kind, "node": n, "t": g.t,
                              "border": {str(b): touches(box(gt_masks[n][f]), shape, px[b]) for b in borders}})
    # How often a GT mask touches the border at all: the base rate the reference's border test works against.
    base = {str(b): [0, 0] for b in borders}
    for masks in gt_masks:
        for m in masks.values():
            bx = box(m)
            for b in borders:
                base[str(b)][0] += 1
                base[str(b)][1] += touches(bx, shape, px[b])
    return {"clip": clip, "events": list(events.values()), "gt_events": gt_events, "border_base": base}


def _job(args):
    return rescore_clip(*args)


def _boot(per_video: np.ndarray, n_videos: int, rng: np.random.Generator) -> np.ndarray:
    """Sums of `per_video` (videos x columns) over BOOT resamples of the videos."""
    idx = rng.integers(0, n_videos, size=(BOOT, n_videos))
    return per_video[idx].sum(axis=1)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def metrics(kept: np.ndarray, tp: np.ndarray, present: np.ndarray, in_view: np.ndarray, hit: Sequence[tuple],
            n_gt_true: int, groups: np.ndarray, base: dict | None, rng: np.random.Generator) -> dict:
    """Precision, recall, the share of false and of true exits called in view, with video-bootstrap intervals.

    The separation is the in-view share among the false exits minus that among the true ones: how much better than
    chance the condition's in-view label picks out a loss.

    Args:
        kept: Per event, whether the condition keeps it as an exit.
        tp: Per event, whether the reference calls it a true exit (`gt_*_border`).
        present: Per event, whether the reference calls it a false exit (`gt_present`).
        in_view: Per event, whether the condition's label is the in-view label (a loss).
        hit: Per event, the GT true event it would pick up, or None.
        n_gt_true: The GT true events in all.
        groups: Per event, its video.
        base: The `before` metrics' per-video sums, for the paired difference; None for `before` itself.
    """
    videos = sorted(set(groups))
    vi = {v: i for i, v in enumerate(videos)}
    # Per-video sums: kept, kept & tp, present, present & in_view, GT true events picked up (distinct), tp, tp & in_view.
    pv = np.zeros((len(videos), 7))
    picked = {h for h, k in zip(hit, kept) if k and h is not None}
    for i, g in enumerate(groups):
        pv[vi[g], 0] += kept[i]
        pv[vi[g], 1] += kept[i] and tp[i]
        pv[vi[g], 2] += present[i]
        pv[vi[g], 3] += present[i] and in_view[i]
        pv[vi[g], 5] += tp[i]
        pv[vi[g], 6] += tp[i] and in_view[i]
    for h in picked:
        pv[vi[h[0]], 4] += 1
    tot = pv.sum(axis=0)
    out = {"n_kept": int(tot[0]), "tp": int(tot[1]), "fp": int(tot[0] - tot[1]), "n_false": int(tot[2]),
           "false_in_view": int(tot[3]), "n_gt_true": int(n_gt_true), "gt_true_picked": len(picked)}
    out["precision"] = tot[1] / tot[0] if tot[0] else None
    out["recall"] = len(picked) / n_gt_true if n_gt_true else None
    out["false_exit_in_view_share"] = tot[3] / tot[2] if tot[2] else None
    out["true_exit_in_view_share"] = tot[6] / tot[5] if tot[5] else None
    out["separation"] = (out["false_exit_in_view_share"] - out["true_exit_in_view_share"]
                         if out["false_exit_in_view_share"] is not None and out["true_exit_in_view_share"] is not None else None)
    bs = _boot(pv, len(videos), rng)
    out["precision_ci95_video"] = _ci(_ratio(bs[:, 1], bs[:, 0]))
    out["false_exit_in_view_share_ci95_video"] = _ci(_ratio(bs[:, 3], bs[:, 2]))
    out["separation_ci95_video"] = _ci(_ratio(bs[:, 3], bs[:, 2]) - _ratio(bs[:, 6], bs[:, 5]))
    out["per_video"] = pv.tolist()
    out["videos"] = videos
    if base is not None:
        bpv = np.array(base["per_video"])
        idx = rng.integers(0, len(videos), size=(BOOT, len(videos)))
        a, b = pv[idx].sum(axis=1), bpv[idx].sum(axis=1)
        d_prec = _ratio(a[:, 1], a[:, 0]) - _ratio(b[:, 1], b[:, 0])
        d_rec = (a[:, 4] - b[:, 4]) / n_gt_true if n_gt_true else np.full(BOOT, np.nan)
        out["precision_diff_from_before"] = (out["precision"] or 0) - (base["precision"] or 0)
        out["precision_diff_ci95_video"] = _ci(d_prec)
        out["precision_diff_mark"] = mark_of(out["precision_diff_ci95_video"], +1, n_videos=len(videos))
        out["recall_diff_from_before"] = (out["recall"] or 0) - (base["recall"] or 0)
        out["recall_diff_ci95_video"] = _ci(d_rec)
        out["tp_retention"] = tot[1] / base["tp"] if base["tp"] else None
    return out


def separation_diff(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Separation of condition A minus that of B on the same videos, with the video bootstrap of the difference."""
    pa, pb = np.array(a["per_video"]), np.array(b["per_video"])
    n = len(pa)
    idx = rng.integers(0, n, size=(BOOT, n))
    sa, sb = pa[idx].sum(axis=1), pb[idx].sum(axis=1)
    d = (_ratio(sa[:, 3], sa[:, 2]) - _ratio(sa[:, 6], sa[:, 5])) - (_ratio(sb[:, 3], sb[:, 2]) - _ratio(sb[:, 6], sb[:, 5]))
    ci = _ci(d)
    diff = (a["separation"] - b["separation"]) if a["separation"] is not None and b["separation"] is not None else None
    return {"diff": diff, "diff_ci95_video": ci, "mark": mark_of(ci, +1, n_videos=n)}


def _ci(x: np.ndarray) -> list[float] | None:
    x = x[np.isfinite(x)]
    if x.size < BOOT // 2:
        return None
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarise(results: Sequence[dict], settings: dict, borders: Sequence[float], draws: int) -> tuple[dict, list[dict]]:
    """The confusion counts and the metrics of every condition, kind, margin and border."""
    rng = np.random.default_rng(SEED)
    margins = [str(float(m)) for m in settings["margins"]]
    shuffles = [f"shuffle_{i}" for i in range(settings["shuffles"])]
    summary, confusion = {"kinds": {}}, []
    for kind in ("exit", "enter"):
        ev = [e for res in results for e in res["events"] if e["kind"] == kind]
        gts = [g for res in results for g in res["gt_events"] if g["kind"] == kind]
        in_view, out_view, window = LABELS[kind]
        per_kind = {}
        for b in borders:
            bk = str(b)
            ver = [e for e in ev if e["ref"][bk] != "gt_unverifiable"]
            groups = np.array([e["video"] for e in ver])
            tp = np.array([e["ref"][bk].endswith("_border") for e in ver])
            present = np.array([e["ref"][bk] == "gt_present" for e in ver])
            gt_true = [(g["video"], g["clip"], g["node"], g["t"]) for g in gts if g["border"][bk]]
            n_gt_true = len(gt_true)
            gt_true_set = set(gt_true)
            # The GT true event a true predicted event picks up: the GT event on the frame after the mask's last
            # frame (exit) or on the mask's first frame (enter).
            hit = []
            for e, is_tp in zip(ver, tp):
                f = e["ref_frame"][bk]
                key = (e["video"], e["clip"], e["node"], (f + 1) if kind == "exit" else f) if is_tp else None
                hit.append(key if key in gt_true_set else None)
            ones = np.ones(len(ver), dtype=bool)
            conds = {}
            # The conditions: a label per event under each; the shuffles pooled; the pose-free baseline.
            def label_sets():
                for m in margins:
                    yield WITH_POSES, m, [e["labels"][f"{WITH_POSES}|{m}"] for e in ver], 1
                    yield STATIC, m, [e["labels"][f"{STATIC}|{m}"] for e in ver], 1
                    yield SHUFFLED, m, [e["labels"][f"{s}|{m}"] for s in shuffles for e in ver], len(shuffles)
                yield BOUNDARY2D, "-", [e[BOUNDARY2D][bk] for e in ver], 1
            for cond, m, labels, rep in label_sets():
                lab = np.array(labels)
                g_r, tp_r, pr_r = np.tile(groups, rep), np.tile(tp, rep), np.tile(present, rep)
                hit_r = hit * rep
                before = metrics(np.ones(len(lab), dtype=bool), tp_r, pr_r, lab == in_view, hit_r, n_gt_true * rep, g_r, None, rng)
                after = metrics(lab == out_view, tp_r, pr_r, lab == in_view, hit_r, n_gt_true * rep, g_r, before, rng)
                plus = metrics((lab == out_view) | (lab == window), tp_r, pr_r, lab == in_view, hit_r, n_gt_true * rep, g_r, before, rng)
                conds[f"{cond}|{m}"] = {"before": before, "after": after, "after_plus_window": plus, "n_pooled": len(lab), "replicates": rep}
                for p in LABELS[kind]:
                    row = {"kind": kind, "condition": cond, "margin": m, "border": bk, "pred": p}
                    for ref in REFS[kind][:3]:
                        row[ref] = int(sum((lab == p) & np.tile(np.array([e["ref"][bk] == ref for e in ver]), rep))) / rep
                    confusion.append(row)
            # The separation of the poses against the shuffles and against the baseline; the shuffles' per-video sums
            # are replicates, so they are scaled back to one.
            for m in margins:
                w = conds[f"{WITH_POSES}|{m}"]["before"]
                sh = dict(conds[f"{SHUFFLED}|{m}"]["before"])
                sh["per_video"] = (np.array(sh["per_video"]) / len(shuffles)).tolist()
                conds[f"with_poses_minus_shuffled_separation|{m}"] = separation_diff(w, sh, rng)
                conds[f"with_poses_minus_boundary2d_separation|{m}"] = separation_diff(w, conds[f"{BOUNDARY2D}|-"]["before"], rng)
            for v in conds.values():
                for part in ("before", "after", "after_plus_window"):
                    if part in v:
                        v[part].pop("per_video", None); v[part].pop("videos", None)
            # The random subset: keep as many events as the poses keep, at random, `draws` times.
            n_keep = int((np.array([e["labels"][f"{WITH_POSES}|{margins[0]}"] for e in ver]) == out_view).sum())
            precs, recs = [], []
            for _ in range(draws):
                keep = np.zeros(len(ver), dtype=bool)
                keep[rng.choice(len(ver), size=n_keep, replace=False)] = True
                precs.append(tp[keep].mean() if n_keep else np.nan)
                recs.append(len({h for h, k in zip(hit, keep) if k and h is not None}) / n_gt_true if n_gt_true else np.nan)
            conds[f"{RANDOM}|-"] = {"after": {"n_kept": n_keep, "precision": float(np.nanmean(precs)),
                                               "precision_p2.5_97.5": [float(np.nanpercentile(precs, 2.5)), float(np.nanpercentile(precs, 97.5))],
                                               "recall": float(np.nanmean(recs)), "draws": draws}}
            # The paired difference of the poses against the baseline, after.
            lp = np.array([e["labels"][f"{WITH_POSES}|{margins[0]}"] for e in ver]) == out_view
            lb = np.array([e[BOUNDARY2D][bk] for e in ver]) == out_view
            conds["with_poses_minus_boundary2d|" + margins[0]] = _paired_precision(lp, lb, tp, groups, rng)
            base = np.array([res["border_base"][bk] for res in results]).sum(axis=0)
            per_kind[bk] = {"n_pred": len(ev), "n_verifiable": len(ver), "n_gt": len(gts), "n_gt_true": n_gt_true,
                            "gt_masks_touching_border_share": float(base[1] / base[0]) if base[0] else None,
                            "reference_counts": {ref: int(sum(e["ref"][bk] == ref for e in ev)) for ref in REFS[kind]},
                            "conditions": conds}
        summary["kinds"][kind] = per_kind
    return summary, confusion


def _paired_precision(keep_a: np.ndarray, keep_b: np.ndarray, tp: np.ndarray, groups: np.ndarray,
                      rng: np.random.Generator) -> dict:
    """Precision of A minus precision of B on the same events, with the video bootstrap of the difference."""
    videos = sorted(set(groups))
    vi = {v: i for i, v in enumerate(videos)}
    pv = np.zeros((len(videos), 4))
    for i, g in enumerate(groups):
        pv[vi[g]] += [keep_a[i], keep_a[i] and tp[i], keep_b[i], keep_b[i] and tp[i]]
    tot = pv.sum(axis=0)
    pa = tot[1] / tot[0] if tot[0] else None
    pb = tot[3] / tot[2] if tot[2] else None
    bs = _boot(pv, len(videos), rng)
    d = _ratio(bs[:, 1], bs[:, 0]) - _ratio(bs[:, 3], bs[:, 2])
    ci = _ci(d)
    return {"precision_with_poses": pa, "precision_boundary2d": pb,
            "diff": (pa - pb) if pa is not None and pb is not None else None, "diff_ci95_video": ci,
            "mark": mark_of(ci, +1, n_videos=len(videos)), "n_videos": len(videos)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="the exit tool's output directory (clip_results.json, summary.json)")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tracks-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt-tracks")
    ap.add_argument("--borders", default=",".join(str(b) for b in BORDERS), help="as fractions of the frame's width")
    ap.add_argument("--random-draws", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    run = Path(a.run)
    settings = json.loads((run / "summary.json").read_text())["settings"]
    results = json.loads((run / "clip_results.json").read_text())
    borders = tuple(float(b) for b in a.borders.split(","))
    clips = set(read_population(a.clips))
    results = [r for r in results if r["clip"] in clips]
    if len(results) != len(clips):
        raise ValueError(f"{len(clips)} clips listed, {len(results)} in the run")
    jobs = [(r, a.dataset, a.data_root, a.tracks_root, a.tag, a.gt_tracks, settings["min_present_px"], borders) for r in results]
    with Pool(a.workers) as pool:
        scored = pool.map(_job, jobs, chunksize=1)
    summary, confusion = summarise(scored, settings, borders, a.random_draws)
    summary.update({"dataset": a.dataset, "run": str(run), "tag": a.tag, "gt_tracks": a.gt_tracks, "borders": borders,
                    "settings": settings, "clips": len(scored), "videos": len({video_of(r["clip"]) for r in scored}),
                    "tool_sha": tool_sha(__file__)})
    out = run / OUT_DIR
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "condition", "margin", "border", "pred"] + list(REFS["exit"][:3]) + list(REFS["enter"][1:3]))
        w.writeheader()
        for row in confusion:
            w.writerow(row)
    rows = [{"clip": e["clip"], "kind": e["kind"], "node": e["node"], "t": e["t"],
             **{f"ref_{b}": e["ref"][b] for b in e["ref"]}, **{f"boundary2d_{b}": e[BOUNDARY2D][b] for b in e[BOUNDARY2D]},
             **{k.replace("|", "_m"): v for k, v in e["labels"].items()}} for res in scored for e in res["events"]]
    with open(out / "per_event.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for kind in ("exit", "enter"):
        for b, d in summary["kinds"][kind].items():
            wp = d["conditions"][f"{WITH_POSES}|{str(float(settings['margins'][0]))}"]
            print(f"{kind} b={b}: refs {d['reference_counts']}; with_poses precision {wp['before']['precision']:.3f} -> "
                  f"{wp['after']['precision']:.3f}, false exits in view {wp['after']['false_exit_in_view_share']:.3f}")


if __name__ == "__main__":
    main()
