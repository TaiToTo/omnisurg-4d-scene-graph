"""The keys of a score JSON: which metrics there are, how a key is spelled, and which way is better.

The metrics, their order and which way each is better are those of the
table in `docs/evaluation.md`, "Metrics". Each per-frame metric is written
once per view, as `metric_key(metric, view)`, and each clip-level metric
once per clip, under its own name.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# The per-frame metrics, written once per view, in the table's order.
FRAME_METRICS = ("F1_50", "SQ", "inst_BF", "mIoU", "boundary_F", "boundary_R_raw",
                 "VI_split", "VI_merge", "unlabelled_share")

# The metrics that are one value per clip, written once, under no view.
CLIP_METRICS = ("time_IoU",)

# The table's `better` column: +1 higher, -1 lower, 0 a reference value that
# is reported and never marked.
SIGNS: Mapping[str, int] = MappingProxyType({
    "F1_50": +1, "SQ": +1, "inst_BF": +1, "mIoU": +1, "boundary_F": +1,
    "boundary_R_raw": +1, "VI_split": -1, "VI_merge": -1,
    "unlabelled_share": 0, "time_IoU": 0,
})


def metric_key(metric: str, view: str) -> str:
    """The per-clip key of one metric in one view, as the evaluator writes it."""
    return f"{metric}/{view}"


def split_key(key: str) -> tuple[str, str | None]:
    """A key back into its metric and its view; the view is None for a clip-level key or a pilot key."""
    metric, sep, view = key.partition("/")
    return (metric, view) if sep else (key, None)
