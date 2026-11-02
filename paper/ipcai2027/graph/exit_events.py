"""Read the event rows of an exit check, and sort what lies under the point an event's node was projected to.

`evalkit.tools.exit_vs_loss` writes `events.csv`, one row per tracked event, pose set and margin, and
`gt_events.csv`, one row per reference event and margin. The paper reads margin 0 and the estimated poses. In a row,
`region_under` is the tracked region under the projected point, a node's region id being its index plus one and -1
no region; `gt_class_under` is the reference mask id there.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from evalkit.classes import ClassTable
from evalkit.tools.exit_vs_loss import LABELS, WITH_POSES

MARGIN = 0.0
POINT_KINDS = ("instrument", "own_class", "other_class", "background", "no_reference")
NODE_KINDS = ("another_node", "own_node", "no_node")


def read_events(run: Path, pose_set: str | None = WITH_POSES) -> list[dict]:
    """Return the rows of `events.csv` at margin 0, under one pose set, or under every pose set when None."""
    rows = [r for r in csv.DictReader((run / "events.csv").open()) if float(r["margin"]) == MARGIN]
    return [r for r in rows if pose_set is None or r["pose_set"] == pose_set]


def read_reference(run: Path, kind: str) -> list[dict]:
    """Return the rows of `gt_events.csv` of one kind at margin 0."""
    return [r for r in csv.DictReader((run / "gt_events.csv").open())
            if r["kind"] == kind and float(r["margin"]) == MARGIN]


def within_view(rows: list[dict], kind: str) -> list[dict]:
    """Return the rows of one kind that the poses classify within the view."""
    return [r for r in rows if r["kind"] == kind and r["label"] == LABELS[kind][0]]


def node_kind(row: dict) -> str:
    """Return whether another tracked node, the event's own node, or no node lies under the projected point."""
    under = int(row["region_under"])
    if under < 0:
        return "no_node"
    return "own_node" if under == int(row["node"]) + 1 else "another_node"


def point_kind(table: ClassTable, node_class: int, under: str) -> str:
    """Return the kind of the reference mask id under the projected point, for a node of class `node_class`.

    An instrument class is named before the node's own class: an instrument lost over an instrument counts as an
    instrument. A mask id of no class, or of a class that is not a thing, counts as background.
    """
    if under in ("", "None"):
        return "no_reference"
    mid = int(under)
    if mid in table.background_mask_ids or mid in table.excluded_mask_ids or mid not in table.mask_id_to_class:
        return "background"
    c = table.mask_id_to_class[mid]
    kind = str(table.type_of(c))
    if kind in ("ignored", "background", "excluded"):
        return "background"
    if kind == "tool":
        return "instrument"
    return "own_class" if c == node_class else "other_class"


def counts_of(kinds: list[str], names: tuple[str, ...]) -> dict:
    """Return the count and the share of each name among `kinds`."""
    c = Counter(kinds)
    return {k: {"n": c[k], "share": c[k] / len(kinds) if kinds else None} for k in names}


def per_video(rows: list[dict], columns) -> tuple[list[str], np.ndarray]:
    """Return the videos in sorted order and, per video, the sum of each column function over its rows."""
    videos = sorted({r["video"] for r in rows})
    at = {v: i for i, v in enumerate(videos)}
    out = np.zeros((len(videos), len(columns)))
    for r in rows:
        out[at[r["video"]]] += [float(f(r)) for f in columns]
    return videos, out
