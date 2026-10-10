"""Write `SUMMARY.md`: the main numbers of every item, read from the tables.

Every number is copied by this script from a `summary.csv` or `tests.csv`,
never by hand. The page states values, intervals and p-values and does not
interpret them.

Usage:
    python3 -m aecai_rev.summary
"""
import argparse
import os

import pandas as pd

from aecai_rev.config import load
from aecai_rev.sweep import default_filter

MODES = ["rgb", "depth", "normal"]
PAIRS = ["normal - rgb", "normal - depth", "rgb - depth"]


def ci(row, nd=3):
    """Format a mean and its window-level interval."""
    return f"{row['mean']:.{nd}f} [{row['ci_low']:.{nd}f}, {row['ci_high']:.{nd}f}]"


def pval(p):
    """Format a p-value, `n/a` where the test had none."""
    if p is None or pd.isna(p):
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def diff(row, nd=3):
    """Format a paired difference: mean, window and video intervals, Holm p."""
    vid = (f"; video CI [{row['mean_ci_low_video']:.{nd}f}, {row['mean_ci_high_video']:.{nd}f}]"
           if not pd.isna(row.get("mean_ci_low_video")) else "")
    return (f"{row['mean_diff']:+.{nd}f} [{row['mean_ci_low']:+.{nd}f}, {row['mean_ci_high']:+.{nd}f}]"
            f"{vid}; median {row['median_diff']:+.{nd}f}; p_holm = {pval(row['p_holm'])}")


def table(summary, index, metrics, nd=3, where=None):
    """Return a Markdown table: one row per `index` value, one column per modality."""
    s = summary if where is None else summary.query(where)
    lines = ["| " + " | ".join([index, "metric"] + MODES) + " |",
             "|" + "---|" * (len(MODES) + 2)]
    for key in sorted(s[index].unique(), key=lambda v: (str(type(v)), v)):
        for m in metrics:
            cells = []
            for mode in MODES:
                r = s[(s[index] == key) & (s["metric"] == m) & (s["modality"] == mode)]
                cells.append(ci(r.iloc[0], nd) if len(r) else "")
            lines.append(f"| {key} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tests_block(tests, metrics, where=None, nd=3):
    """Return the paired differences of `metrics` as a bulleted list."""
    t = tests if where is None else tests.query(where)
    out = []
    for m in metrics:
        for pair in PAIRS:
            r = t[(t["metric"] == m) & (t["pair"] == pair)]
            if len(r) and not pd.isna(r.iloc[0].get("mean_diff")):
                out.append(f"- {m}, {pair}: {diff(r.iloc[0], nd)}")
    return "\n".join(out)


def read(cfg, item, name):
    """Read one item's table."""
    return pd.read_csv(os.path.join(cfg["paths"]["results"], item, name))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cfg = load()
    res = cfg["paths"]["results"]
    pps = cfg["track"]["default_points_per_side"]
    parts = ["# SUMMARY: main numbers of the revision experiments",
             "",
             "Written by `python3 -m aecai_rev.summary` from the tables; values only, "
             "no interpretation. Format: window mean [window-level bootstrap 95 % CI]. "
             "A difference is `a - b`: mean [window CI]; video CI (17 videos); median; "
             "Wilcoxon p, Holm-corrected over the three pairs of one metric. "
             f"Population: the {cfg['windows']['population']} windows. Area cut: "
             f"`{default_filter(cfg)}` of the valid area unless said otherwise. "
             f"SAM points_per_side {pps} unless said otherwise.",
             ""]

    # Reproduction.
    t1 = read(cfg, "00_check", "table1_reproduced.csv")
    parts += ["## Table 1 reproduced (`00_check/`)", "",
              "| SAM input | F1@.5 | F1@.75 | n_reg | oracle mIoU |", "|---|---|---|---|---|"]
    for _, r in t1.iterrows():
        parts.append(f"| {r['modality']} | {r['frozen_inst_F1_50']:.3f} | "
                     f"{r['frozen_inst_F1_75']:.3f} | {r['frozen_n_regions_mean']:.1f} | "
                     f"{r['frozen_GT_mIoU']:.3f} |")
    chk = read(cfg, "00_check", "legacy_check.csv")
    parts += ["", f"Tagged evaluator re-run equals the frozen JSONs on "
              f"{(chk['tagged_rerun_differs'].fillna('') == '').sum()} of {len(chk)} window x "
              f"modality pairs; `aecai_rev.score` (300 px) max per-frame difference "
              f"{chk['score_max_abs_diff'].max():.1e}.", ""]

    # Item 0.
    w = read(cfg, "00_windows", "windows.csv")
    ex = read(cfg, "00_windows", "excluded.csv")
    parts += ["## 0. Windows (`00_windows/`)", "",
              f"One window per CholecSeg8k clip: {len(w)} kept, {len(ex)} excluded "
              f"({', '.join(sorted(ex['reason'].unique()))}); "
              f"{int(w['n_annotated'].sum())} annotated frames in the kept windows. "
              f"Not extracted or tracked yet: the numbers below are on the AE-CAI windows.", ""]

    # Item 7.
    s7 = read(cfg, "07_filter", "summary.csv")
    t7 = read(cfg, "07_filter", "tests.csv")
    eq = read(cfg, "07_filter", "legacy_300px_as_fraction.csv").iloc[0]
    parts += ["## 7. Area cut (`07_filter/`)", "",
              f"300 px = {eq['median']:.5f} of the valid area (median; range "
              f"{eq['min']:.5f}-{eq['max']:.5f}, {int(eq['n_frames'])} frames).", "",
              table(s7, "filter", ["f1@0.5", "f1@0.75", "n_gt", "n_pred"]), "",
              "normal - rgb and normal - depth, F1@.5, per cut:", ""]
    for f in sorted(t7["filter"].unique()):
        r = t7[(t7["filter"] == f) & (t7["metric"] == "f1@0.5")]
        for pair in PAIRS[:2]:
            rr = r[r["pair"] == pair]
            if len(rr):
                parts.append(f"- {f}, {pair}: {diff(rr.iloc[0])}")
    parts.append("")

    # Item 1.
    s1 = read(cfg, "01_sweep", "summary.csv")
    t1s = read(cfg, "01_sweep", "tests.csv")
    m1 = read(cfg, "01_sweep", "matched.csv")
    gt = m1["gt_instances"].iloc[0]
    parts += ["## 1. Granularity sweep (`01_sweep/`)", "",
              f"Mean GT instances per frame: {gt:.2f}. Points marked * are outside the "
              f"specified grid {cfg['track']['points_per_side']}.", "",
              "| pps | " + " | ".join(f"{m} F1@.5" for m in MODES) + " | "
              + " | ".join(f"{m} n_reg" for m in MODES) + " |",
              "|" + "---|" * 7]
    for p in sorted(s1["pps"].unique()):
        mark = "" if p in cfg["track"]["points_per_side"] else "*"
        cells = []
        for metric, nd in (("f1@0.5", 3), ("n_reg", 1)):
            for mode in MODES:
                r = s1[(s1["pps"] == p) & (s1["metric"] == metric) & (s1["modality"] == mode)]
                cells.append(ci(r.iloc[0], nd))
        parts.append(f"| {p}{mark} | " + " | ".join(cells) + " |")
    parts += ["", "Matched granularity (basis n_reg):", ""]
    for _, r in m1[(m1["basis"] == "n_reg")].iterrows():
        if pd.isna(r["interp_value"]):
            interp = "outside the range of the points"
        elif pd.isna(r["interp_ci_low"]):
            interp = (f"{r['interp_value']:.3f} (interval undefined: "
                      f"{r['boot_in_range']:.0%} of resamples in range)")
        else:
            interp = f"{r['interp_value']:.3f} [{r['interp_ci_low']:.3f}, {r['interp_ci_high']:.3f}]"
        parts.append(f"- {r['grid']} grid, {r['modality']}, {r['metric']}: nearest pps "
                     f"{r['nearest_pps']} (n_reg {r['nearest_count']:.2f}) -> {r['nearest_value']:.3f}; "
                     f"interpolated at {gt:.2f}: {interp}")
    parts += ["", "normal - rgb, F1@.5, per pps:", ""]
    for p in sorted(t1s["pps"].unique()):
        r = t1s[(t1s["pps"] == p) & (t1s["metric"] == "f1@0.5") & (t1s["pair"] == "normal - rgb")]
        parts.append(f"- pps {p}: {diff(r.iloc[0])}")
    parts.append("")

    # Item 2.
    parts += ["## 2. Statistics", "",
              "Every `summary.csv` holds `ci_low`, `ci_high` (window bootstrap, 10,000, "
              "percentile) and `ci_low_video`, `ci_high_video`; every `tests.csv` holds "
              "`median_diff`, `ci_low`, `ci_high` (of the median), `mean_diff` with its "
              "intervals, `p_raw` (Wilcoxon, two-sided) and `p_holm`.", "",
              "Table 1 with intervals (300 px cut, as in the paper):", "",
              table(s7, "filter", ["f1@0.5", "f1@0.75", "n_reg", "miou"], where="filter == 'px300'"),
              "", tests_block(t7, ["f1@0.5", "f1@0.75", "miou"], where="filter == 'px300'"), ""]

    # Item 3.
    s3 = read(cfg, "03_merge", "summary.csv")
    t3 = read(cfg, "03_merge", "tests.csv")
    d3 = read(cfg, "03_merge", "delta.csv")
    parts += ["## 3. Merge per GT instance (`03_merge/`)", "",
              table(s3, "variant", ["f1@0.5", "f1@0.75", "miou", "n_reg"], where=f"pps == {pps}"),
              "", "merged - raw, per modality:", ""]
    for m in ["f1@0.5", "f1@0.75", "miou"]:
        for mode in MODES:
            r = d3[(d3["pps"] == pps) & (d3["metric"] == m) & (d3["modality"] == mode)]
            parts.append(f"- {m}, {mode}: {diff(r.iloc[0])}")
    parts += ["", "Between modalities after merging:", "",
              tests_block(t3, ["f1@0.5", "f1@0.75", "miou"],
                          where=f"pps == {pps} and variant == 'merged'"), ""]

    # Item 4.
    s4 = read(cfg, "04_identity", "summary.csv")
    t4 = read(cfg, "04_identity", "tests.csv")
    link = next(iter(cfg["identity"]["links"]))
    keys = ["id_switch_rate", "class_switch_rate", "fragmentation_mean", "fragmentation_ge2",
            "survival_forward", "survival_backward", "reappearance_rate", "reappear_same_gt",
            "reappear_same_class"]
    for subset in ["all", "frames_in_order"]:
        parts += [f"## 4. Identity, {subset} windows, GT linking `{link}` (`04_identity/`)", "",
                  table(s4, "subset", keys, where=f"subset == '{subset}' and link == '{link}'"), "",
                  tests_block(t4, keys, where=f"subset == '{subset}' and link == '{link}'"), ""]
    parts += [f"The same with GT linking only between adjacent samples is in "
              f"`04_identity/summary.csv` (`link == adjacent_samples`).", ""]

    # Items 5 and 6.
    for item, title, keys in [("05_merge_rate", "5. Instrument-tissue merges",
                               ["merge_region_rate", "merge_frame"]),
                              ("06_naming", "6. Nameability", ["named_one", "named_many", "named_none"])]:
        s = read(cfg, item, "summary.csv")
        t = read(cfg, item, "tests.csv")
        s = s.assign(level="all")
        parts += [f"## {title} (`{item}/`)", "", table(s, "level", keys), "",
                  tests_block(t, keys), ""]
    ev = open(os.path.join(res, "06_naming", "fig6_named_events.txt")).read().strip()
    parts += ["Named events (`06_naming/fig6_named_events.txt`; not Fig. 6's window, "
              "whose graph is not on this machine, see `00_inventory.md`):", "", "```", ev, "```", ""]
    with open(os.path.join(res, "SUMMARY.md"), "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {os.path.join(res, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()
