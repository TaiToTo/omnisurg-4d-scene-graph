"""`LabelTable`: label id to display name and colour, for GT classes and instances.

The segmentation and graph builders depend on this table alone and do not
know whether a label is a GT class or a tracked instance. A GT table is made
from the dataset's class table in `evalkit.classes`, so that the viewer shows
exactly the classes and colours the evaluator scores; an instance table is
made from the ids a tracker produced.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from evalkit.classes import ClassTable, ClassType, load_table
from surgical_core.viewer.palette import BACKGROUND_COLOR, instance_color

# The types that are not objects: drawn as background and left out of the
# legend and the graph. The same three the evaluator removes from every view.
_NOT_AN_OBJECT = frozenset({ClassType.IGNORED, ClassType.BACKGROUND, ClassType.EXCLUDED})


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


def label_table_of(table: ClassTable) -> LabelTable:
    """The display table of a dataset's class table.

    Every class whose type is an object is a foreground entry with the
    colour the class table gives it; the ignored, background and excluded
    types become background ids.

    Raises:
        ValueError: a foreground class has no colour. ATLAS-120k's benchmark
            set has none, because a mask reaches it only through the mapping;
            it cannot be drawn without choosing colours, and that choice is
            not made silently here.
    """
    entries = {}
    background = set()
    for cid, e in table.entries.items():
        if e.type in _NOT_AN_OBJECT:
            background.add(int(cid))
            continue
        if e.colour is None:
            raise ValueError(
                f"{table.dataset}/{table.class_set}: class {cid} ({e.name}) has no colour, "
                "so the set cannot be drawn")
        entries[int(cid)] = LabelEntry(e.name, [c / 255.0 for c in e.colour])
    return LabelTable(entries=entries, background_ids=background)


def cholec_gt_table() -> LabelTable:
    """The CholecSeg8k classes, as the evaluator's class table defines them."""
    return label_table_of(load_table("cholecseg8k"))


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
