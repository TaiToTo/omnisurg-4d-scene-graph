"""Read the saved scores and write the long, summary and test tables.

Every item's table goes through `write_tables`: the window-level long
table, its per-condition summary with intervals, and the paired tests
between modalities. A window is named by `window_id` (the clip name for the
AE-CAI windows, `W<nnn>` for enumerated ones) and its video by `video_id`.

Usage:
    from aecai_rev.tables import load_window_scores, write_tables
"""
import csv
import os

import pandas as pd

from aecai_rev.stats import paired_tests, summarize
from evalkit.tools.paired_stats import video_of


def window_ids(cfg):
    """Return a dict from clip directory name to window id."""
    if cfg["windows"]["population"] == "legacy27":
        return {c: c for c in cfg["windows"]["legacy27"]}
    path = os.path.join(cfg["paths"]["results"], "00_windows", "windows.csv")
    with open(path) as f:
        return {row["clip"]: row["window_id"] for row in csv.DictReader(f)}


def load_window_scores(cfg):
    """Read the window-level scores written by `aecai_rev.score`.

    Raises:
        FileNotFoundError: `aecai_rev.score` has not been run.
    """
    path = os.path.join(cfg["paths"]["work"], "scores", "windows.csv")
    df = pd.read_csv(path)
    ids = window_ids(cfg)
    missing = sorted(set(df["clip"]) - set(ids))
    if missing:
        raise ValueError(f"scores hold clips outside the population: {missing[:3]}")
    df["window_id"] = df["clip"].map(ids)
    df["video_id"] = df["clip"].map(video_of)
    return df


def write_tables(out, long, by, cfg, test_by=None):
    """Write `long.csv`, `summary.csv` and `tests.csv` into `out`.

    Args:
        out: The output directory.
        long: Window-level table with `clip`, `window_id`, `metric`,
            `value` and the columns in `by`.
        by: The columns naming a condition; `modality` among them.
        cfg: The loaded configuration (pairs to test).
        test_by: The columns a test is run within; `by` minus `modality`
            by default.

    Returns:
        (summary, tests) data frames.
    """
    os.makedirs(out, exist_ok=True)
    cols = ["window_id"] + [c for c in by if c != "metric"] + ["metric", "value"]
    long.sort_values(cols[:-1])[cols].to_csv(os.path.join(out, "long.csv"), index=False)
    summary = summarize(long, by)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    test_by = test_by or [c for c in by if c != "modality"]
    pairs = [tuple(p) for p in cfg["stats"]["pairs"]]
    tests = paired_tests(long, test_by, pairs)
    tests.to_csv(os.path.join(out, "tests.csv"), index=False)
    return summary, tests
