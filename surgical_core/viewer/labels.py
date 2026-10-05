"""`LabelTable`: label id to display name and colour, for GT classes and instances.

The segmentation and graph builders depend on this table alone and do not
know whether a label is a GT class or a tracked instance. `label_table`
builds one from plain data, so a video with no dataset class table needs
nothing else. `gt_tables` builds one from the evaluator's class tables, and
`instance_table` from the ids a tracker produced.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from surgical_core.viewer.palette import BACKGROUND_COLOR, instance_color


@dataclass(frozen=True)
class LabelEntry:
    """How one label is shown.

    Attributes:
        name: the name in the legend and on the graph node.
        color: float RGB, each in 0 to 1.
    """
    name: str
    color: list[float]


@dataclass
class LabelTable:
    """Label id to `LabelEntry`, plus the ids that are background.

    An id in neither `entries` nor `background_ids` is not refused: it is
    named `id N`, listed by `present_ids`, and drawn in the background
    colour. The exporter decides whether such an id is a fault; this table
    only has to show every id it is given.

    Attributes:
        entries: foreground ids only.
        background_ids: ids drawn dark grey and left out of the legend and
            the graph.
    """
    entries: dict[int, LabelEntry]
    background_ids: set[int]

    def name(self, i: int) -> str:
        e = self.entries.get(int(i))
        return e.name if e else f"id {int(i)}"

    def color(self, i: int) -> list[float]:
        e = self.entries.get(int(i))
        return list(e.color) if e else list(BACKGROUND_COLOR)

    def is_background(self, i: int) -> bool:
        return int(i) in self.background_ids

    def present_ids(self, id_map: np.ndarray) -> list[int]:
        """The foreground ids that occur in `id_map`, ascending."""
        return [int(v) for v in np.unique(id_map) if int(v) not in self.background_ids]


def label_table(
    labels: Iterable[tuple[int, str, Sequence[int]]], background_ids: Iterable[int] = (),
) -> LabelTable:
    """A table from plain data: one `(id, name, (r, g, b))` per foreground label.

    Args:
        labels: The foreground labels, each colour three integers in 0 to 255.
        background_ids: The ids drawn as background and left out of the
            legend and the graph.

    Raises:
        ValueError: An id is given twice, is both a label and background, or
            has a colour that is not three integers in 0 to 255.
    """
    background = {int(i) for i in background_ids}
    entries = {}
    for i, name, colour in labels:
        i = int(i)
        if i in entries or i in background:
            where = "twice" if i in entries else "as a label and as background"
            raise ValueError(f"id {i} ({name}) is given {where}")
        rgb = tuple(colour)
        if len(rgb) != 3 or not all(isinstance(c, (int, np.integer)) and 0 <= c <= 255 for c in rgb):
            raise ValueError(f"id {i} ({name}): a colour is three integers in 0 to 255, got {colour!r}")
        entries[i] = LabelEntry(name, [c / 255.0 for c in rgb])
    return LabelTable(entries=entries, background_ids=background)


def instance_table(ids: Iterable[int]) -> LabelTable:
    """A table for instance ids: `obj N` and an instance colour each; 0 is background."""
    entries = {int(i): LabelEntry(f"obj {int(i)}", instance_color(int(i)))
               for i in ids if int(i) != 0}
    return LabelTable(entries=entries, background_ids={0})


def remap_to_instances(
    label_maps: list[np.ndarray],
) -> tuple[list[np.ndarray], LabelTable, dict[int, int]]:
    """Renumber raw instance labels to 1.. consistently across frames.

    A tracker's raw labels are -1 for unassigned and 0.. for instances. The
    viewer reserves 0 for background, so the instances are packed into 1..
    with one mapping for all frames: the same raw id gets the same new id,
    and so the same colour, in every frame, which is what lets an instance be
    followed over time by its colour.

    Args:
        label_maps: (H, W) int arrays, -1 for unassigned.

    Returns:
        `(remapped, table, orig_to_new)`: the (H, W) arrays with 0 as
        background and instances from 1, their `LabelTable`, and the raw to
        new id mapping, for a legend that wants to show the raw id.
    """
    ids = set()
    for L in label_maps:
        ids.update(int(v) for v in np.unique(L) if v >= 0)
    orig_to_new = {orig: i + 1 for i, orig in enumerate(sorted(ids))}
    remapped = []
    for L in label_maps:
        r = np.zeros_like(L, dtype=np.int64)
        for orig, new in orig_to_new.items():
            r[L == orig] = new
        remapped.append(r)
    return remapped, instance_table(orig_to_new.values()), orig_to_new
