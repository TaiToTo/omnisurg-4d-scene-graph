"""Viewer-side data model, shared by the exporter and the evaluation toolkit.

Only the label table and the palette it draws from are re-exported here.
Importing the viewer's geometry modules from this package initialiser would
pull trimesh into every import of `surgical_core.atlas`, and that import is on
the path the frozen evaluation code takes to compute `eval_code_sha` — so the
eval-only install would need the pipeline's dependencies to score a single
clip. The remaining modules are imported by their full path instead.
"""
from .labels import LabelEntry, LabelTable, cholec_gt_table, instance_table, remap_to_instances
from .palette import BACKGROUND_COLOR, INSTANCE_PALETTE, instance_color

__all__ = [
    "BACKGROUND_COLOR",
    "INSTANCE_PALETTE",
    "LabelEntry",
    "LabelTable",
    "cholec_gt_table",
    "instance_color",
    "instance_table",
    "remap_to_instances",
]
