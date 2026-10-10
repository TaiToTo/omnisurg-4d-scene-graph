"""Item 7: the small-region cut as a fraction of the valid area, and its sensitivity.

The script reports, at the AE-CAI setting (points_per_side 24), every
metric under each `min_area_frac` and under the AE-CAI 300 px cut, with
window-level intervals and the paired tests between modalities. It also
records what fraction of a frame's valid area 300 px was, frame by frame.

Usage:
    python3 -m aecai_rev.filter_sensitivity
"""
import argparse
import os

import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.stats import versions
from aecai_rev.tables import load_window_scores, write_tables

METRICS = ["f1@0.5", "f1@0.75", "n_reg", "n_pred", "n_gt", "miou"]


def legacy_equivalent(cfg):
    """Describe 300 px as a fraction of each annotated frame's valid area."""
    frames = pd.read_csv(os.path.join(cfg["paths"]["work"], "scores", "frames.csv.gz"),
                         usecols=["clip", "frame", "valid_px"]).drop_duplicates()
    frac = cfg["score"]["legacy_min_area_px"] / frames["valid_px"]
    q = frac.quantile([0, 0.05, 0.5, 0.95, 1]).to_dict()
    return pd.DataFrame([dict(n_frames=len(frac), mean=frac.mean(), min=q[0], p05=q[0.05],
                              median=q[0.5], p95=q[0.95], max=q[1],
                              valid_px_median=frames["valid_px"].median())])


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cfg = load()
    df = load_window_scores(cfg)
    df = df[(df["pps"] == cfg["track"]["default_points_per_side"]) & (df["variant"] == "raw")
            & df["metric"].isin(METRICS)]
    out = os.path.join(cfg["paths"]["results"], "07_filter")
    summary, tests = write_tables(out, df, ["filter", "modality", "metric"], cfg)
    summary.to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    legacy_equivalent(cfg).to_csv(os.path.join(out, "legacy_300px_as_fraction.csv"), index=False)
    write_meta(out, cfg, what="item 7: area cut sensitivity",
               points_per_side=cfg["track"]["default_points_per_side"], variant="raw",
               libraries=versions())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
