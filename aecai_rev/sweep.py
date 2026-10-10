"""Item 1: instance F1 against region count over the SAM granularity sweep.

The script reads the window scores at the default area cut, writes the
long, summary and test tables over modality and points_per_side, plots F1
against the region count with the mean GT instance count as a dashed line,
and compares modalities at matched granularity: for each modality, the
condition whose region count is nearest the GT instance count, and F1
interpolated linearly to that count (bootstrap interval over windows), on
the specified points_per_side grid and on the grid with the added points.

Usage:
    python3 -m aecai_rev.sweep
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aecai_rev.config import load, write_meta  # noqa: E402
from aecai_rev.score import all_pps  # noqa: E402
from aecai_rev.stats import versions  # noqa: E402
from aecai_rev.tables import load_window_scores, write_tables  # noqa: E402
from evalkit.tools.paired_stats import N_BOOT, SEED  # noqa: E402

METRICS = ["f1@0.5", "f1@0.75", "n_reg", "n_pred", "n_gt", "miou"]
BASES = ["n_reg", "n_pred"]


def default_filter(cfg):
    """Return the name of the default area cut."""
    return f"frac{cfg['score']['min_area_frac']:g}"


def spec_in(index, spec):
    """Return the points_per_side values of `index` in the specified grid."""
    return [p for p in index if p in spec]


def interp_at(x, y, x0):
    """Interpolate y at x0 along points sorted by x; None outside their range."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if x0 < x[0] or x0 > x[-1]:
        return None
    return float(np.interp(x0, x, y))


def matched(wide, cfg, grid):
    """Compare modalities at the granularity of the GT.

    Args:
        wide: Window-level table, one row per window, modality and pps,
            one column per metric.
        cfg: The loaded configuration.
        grid: The points_per_side values to use.

    Returns:
        One row per modality, basis (`n_reg` or `n_pred`) and F1 threshold.
    """
    wide = wide[wide["pps"].isin(grid)]
    rng = np.random.default_rng(SEED)
    windows = sorted(wide["window_id"].unique())
    n_gt = wide.groupby("window_id")["n_gt"].first()
    boot = rng.integers(0, len(windows), size=(N_BOOT, len(windows)))
    rows = []
    for mode, g in wide.groupby("modality"):
        piv = {m: g.pivot(index="window_id", columns="pps", values=m).loc[windows]
               for m in METRICS}
        pps = list(piv["n_reg"].columns)
        gt = float(n_gt.loc[windows].mean())
        for basis in BASES:
            nb = piv[basis].mean()
            near = int(min(pps, key=lambda p: abs(nb[p] - gt)))
            for f1 in ["f1@0.5", "f1@0.75"]:
                fv = piv[f1].mean()
                point = interp_at(nb.values, fv.values, gt)
                draws = []
                for idx in boot:
                    sel = [windows[i] for i in idx]
                    v = interp_at(piv[basis].loc[sel].mean().values,
                                  piv[f1].loc[sel].mean().values,
                                  float(n_gt.loc[sel].mean()))
                    if v is not None:
                        draws.append(v)
                ok = len(draws) / N_BOOT
                rows.append(dict(
                    modality=mode, basis=basis, metric=f1, gt_instances=gt,
                    nearest_pps=near, nearest_count=float(nb[near]),
                    nearest_value=float(fv[near]),
                    interp_value=point,
                    interp_ci_low=float(np.percentile(draws, 2.5)) if ok >= 0.95 else None,
                    interp_ci_high=float(np.percentile(draws, 97.5)) if ok >= 0.95 else None,
                    boot_in_range=ok))
    return pd.DataFrame(rows)


def plot(summary, out, metric, basis, gt_mean, name, spec):
    """Plot F1 against the region count, one line per modality.

    Points of the specified grid are filled; the added points are hollow.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for mode in ["rgb", "depth", "normal"]:
        s = summary[summary["modality"] == mode].sort_values("pps")
        x = s[s["metric"] == basis].set_index("pps")
        y = s[s["metric"] == metric].set_index("pps")
        if x.empty:
            continue
        line = ax.errorbar(x["mean"], y["mean"],
                           yerr=[y["mean"] - y["ci_low"], y["ci_high"] - y["mean"]],
                           xerr=[x["mean"] - x["ci_low"], x["ci_high"] - x["mean"]],
                           marker="", capsize=2, lw=1.2, label=mode)
        color = line[0].get_color()
        added = [p for p in x.index if p not in spec]
        ax.plot(x.loc[spec_in(x.index, spec), "mean"], y.loc[spec_in(x.index, spec), "mean"],
                "o", ms=4, color=color)
        ax.plot(x.loc[added, "mean"], y.loc[added, "mean"], "o", ms=4, mfc="white", color=color)
        for p in x.index:
            ax.annotate(str(p), (x.loc[p, "mean"], y.loc[p, "mean"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axvline(gt_mean, ls="--", c="0.4", lw=1, label="GT instances")
    ax.set_xlabel({"n_reg": "regions per frame ($n_{reg}$)",
                   "n_pred": "regions entering F1 per frame"}[basis])
    ax.set_ylabel({"f1@0.5": "instance F1@0.5", "f1@0.75": "instance F1@0.75"}[metric])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    # Select: the default area cut, the regions as tracked.
    cfg = load()
    df = load_window_scores(cfg)
    df = df[(df["filter"] == default_filter(cfg)) & (df["variant"] == "raw")
            & df["metric"].isin(METRICS)]

    # Tables: long, summary with intervals, tests between modalities per pps.
    out = os.path.join(cfg["paths"]["results"], "01_sweep")
    summary, _ = write_tables(out, df, ["modality", "pps", "metric"], cfg)

    # Matched granularity, then the figures.
    wide = df.pivot_table(index=["window_id", "modality", "pps"], columns="metric",
                          values="value").reset_index()
    spec = list(cfg["track"]["points_per_side"])
    both = [matched(wide, cfg, spec).assign(grid="specified"),
            matched(wide, cfg, all_pps(cfg)).assign(grid="extended")]
    pd.concat(both).to_csv(os.path.join(out, "matched.csv"), index=False)
    summary["in_specified_grid"] = summary["pps"].isin(spec)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    gt_mean = float(wide.groupby("window_id")["n_gt"].first().mean())
    for metric, tag in [("f1@0.5", ""), ("f1@0.75", "_f1_75")]:
        plot(summary, out, metric, "n_reg", gt_mean, f"f1_vs_nreg{tag}", spec)
        plot(summary, out, metric, "n_pred", gt_mean, f"f1_vs_npred{tag}", spec)
    write_meta(out, cfg, what="item 1: granularity sweep", filter=default_filter(cfg),
               variant="raw", libraries=versions())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
