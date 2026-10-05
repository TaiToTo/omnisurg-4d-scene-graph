"""The label tables of the evaluator's class tables, so the viewer shows the classes and colours it scores.

The one module of the viewer that imports `evalkit`. A video with no dataset
class table builds its table with `labels.label_table` and does not need it.
"""

from evalkit.classes import ClassTable, ClassType, load_table
from surgical_core.viewer.labels import LabelTable, label_table

# The types that are not objects: drawn as background and left out of the
# legend and the graph. The same three the evaluator removes from every view.
_NOT_AN_OBJECT = frozenset({ClassType.IGNORED, ClassType.BACKGROUND, ClassType.EXCLUDED})


def label_table_of(table: ClassTable) -> LabelTable:
    """The label table of a dataset's class table.

    Every class whose type is an object is a foreground entry with the
    colour the class table gives it; the ignored, background and excluded
    types become background ids.

    Raises:
        ValueError: a foreground class has no colour. ATLAS-120k's benchmark
            set has none, because a mask reaches it only through the mapping;
            it cannot be drawn without choosing colours, and that choice is
            not made silently here.
    """
    labels = []
    background = set()
    for cid, e in table.entries.items():
        if e.type in _NOT_AN_OBJECT:
            background.add(int(cid))
            continue
        if e.colour is None:
            raise ValueError(
                f"{table.dataset}/{table.class_set}: class {cid} ({e.name}) has no colour, "
                "so the set cannot be drawn")
        labels.append((int(cid), e.name, e.colour))
    return label_table(labels, background)


def cholec_gt_table() -> LabelTable:
    """The CholecSeg8k classes, as the evaluator's class table defines them."""
    return label_table_of(load_table("cholecseg8k"))
