"""Item 3: scores after merging the regions of each GT instance into one.

The script compares, at the default area cut, every condition's scores on
the regions as tracked (`raw`) and after the regions that overlap one GT
instance most are merged (`merged`, see `score.merge_by_instance`). It
writes the long, summary and test tables over modality, points_per_side and
variant, and the paired change merged - raw per condition (`delta.csv`).

Usage:
    python3 -m aecai_rev.merge_rescore
"""
import argparse
import os

import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.stats import paired_tests, versions
from aecai_rev.sweep import default_filter
from aecai_rev.tables import load_window_scores, write_tables

METRICS = ["f1@0.5", "f1@0.75", "n_reg", "n_pred", "miou"]


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cfg = load()
    df = load_window_scores(cfg)
    df = df[(df["filter"] == default_filter(cfg)) & df["metric"].isin(METRICS)]
    out = os.path.join(cfg["paths"]["results"], "03_merge")
    write_tables(out, df, ["variant", "modality", "pps", "metric"], cfg)
    delta = paired_tests(df, ["modality", "pps", "metric"], [("merged", "raw")], side="variant")
    delta.to_csv(os.path.join(out, "delta.csv"), index=False)
    write_meta(out, cfg, what="item 3: merge per GT instance, then rescore",
               filter=default_filter(cfg), libraries=versions())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
