"""The viewer's data model, shared by the exporter and the evaluation toolkit.

Only the label table and the palette it draws from are re-exported here. The
viewer's other modules (the point-cloud, graph and hierarchy frames) pull in
the pipeline's dependencies, and importing them from this package
initialiser would make every import of `surgical_core.viewer` need them. They
are imported by their full path when they are ported.
"""

from surgical_core.viewer.labels import (  # noqa: F401
    LabelEntry, LabelTable, cholec_gt_table, instance_table, label_table_of, remap_to_instances)
from surgical_core.viewer.palette import BACKGROUND_COLOR, INSTANCE_PALETTE, instance_color  # noqa: F401

__all__ = [
    "BACKGROUND_COLOR",
    "INSTANCE_PALETTE",
    "LabelEntry",
    "LabelTable",
    "cholec_gt_table",
    "instance_color",
    "instance_table",
    "label_table_of",
    "remap_to_instances",
]
