"""Name the windows of the configured population.

`window_clips` returns the `outputs/cholec_gt` directory of each window:
the AE-CAI list for `legacy27`, or the `clip` column of
`00_windows/windows.csv` for `enumerated`.

Usage:
    from aecai_rev.windows_io import window_clips
"""
import csv
import os


def window_clips(cfg):
    """Return the clip directory name of every window, in a fixed order.

    Args:
        cfg: The loaded configuration.

    Returns:
        A list of directory names under `paths.cholec_gt`.

    Raises:
        ValueError: The population is neither `legacy27` nor `enumerated`.
    """
    pop = cfg["windows"]["population"]
    if pop == "legacy27":
        return list(cfg["windows"]["legacy27"])
    if pop == "enumerated":
        path = os.path.join(cfg["paths"]["results_root"], "00_windows", "windows.csv")
        with open(path) as f:
            return [row["clip"] for row in csv.DictReader(f)]
    raise ValueError(f"unknown window population {pop!r}")
