"""The viewer's label table, built from plain data, from the evaluator's class tables, and from instances.

A GT label table is derived from `evalkit.classes`, so the viewer cannot
drift from the evaluator on a name or a colour; what is tested is the
derivation (which types are background, that a colourless foreground class
is refused), the table a video with no class table builds, that the base
of the viewer does not need the evaluator, and the instance side (stable
colours, cross-frame renumbering, the background id never coloured).
"""

import subprocess
import sys

import numpy as np
import pytest

from evalkit.classes import ClassType, load_table
from surgical_core.viewer.gt_tables import cholec_gt_table, label_table_of
from surgical_core.viewer.labels import LabelTable, instance_table, label_table, remap_to_instances
from surgical_core.viewer.palette import BACKGROUND_COLOR, INSTANCE_PALETTE, instance_color


def loads_evalkit(module: str) -> bool:
    """Whether importing `module` in a fresh interpreter imports `evalkit` too."""
    code = f"import sys, {module}; print('evalkit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return out.stdout.strip() == "True"


def test_the_viewer_does_not_need_the_evaluator():
    assert not loads_evalkit("surgical_core.viewer")
    assert not loads_evalkit("surgical_core.viewer.labels")


def test_the_check_sees_the_evaluator_where_it_is_imported():
    assert loads_evalkit("surgical_core.viewer.gt_tables")


def test_a_video_with_no_class_table_gets_a_label_table():
    table = label_table([(1, "liver", (200, 80, 60)), (2, "tool", (0, 120, 255))], background_ids={0})
    assert table.name(1) == "liver" and table.color(2) == [0.0, 120 / 255, 1.0]
    assert table.is_background(0) and table.present_ids(np.array([[0, 1], [2, 2]])) == [1, 2]


@pytest.mark.parametrize("labels, background, match", [
    ([(1, "a", (0, 0, 0)), (1, "b", (1, 1, 1))], (), "twice"),
    ([(0, "a", (0, 0, 0))], {0}, "as a label and as background"),
    ([(1, "a", (0, 0))], (), "three integers"),
    ([(1, "a", (-1, 0, 0))], (), "three integers"),
    ([(1, "a", (0, 0, 256))], (), "three integers"),
    ([(1, "a", (0.5, 0.5, 0.5))], (), "three integers"),
    ([(1, "a", None)], (), "three integers"),
])
def test_a_label_given_twice_or_with_a_colour_that_is_not_rgb_is_refused(labels, background, match):
    with pytest.raises(ValueError, match=match):
        label_table(labels, background)


def test_cholecseg8k_table_follows_the_class_table():
    classes = load_table("cholecseg8k")
    labels = cholec_gt_table()
    # The ignored classes (the black surround, the annotation tool's region
    # line) are background; every other class is a foreground entry.
    assert labels.background_ids == {0, 13}
    assert set(labels.entries) == set(classes.entries) - {0, 13}
    for cid, e in labels.entries.items():
        assert e.name == classes.entries[cid].name
        assert e.color == [c / 255.0 for c in classes.entries[cid].colour]
    assert labels.name(2) == "Liver"
    assert labels.color(0) == BACKGROUND_COLOR
    assert labels.is_background(13) and not labels.is_background(2)


def test_atlas120k_original_table_makes_the_markers_background():
    classes = load_table("atlas120k", "original")
    labels = label_table_of(classes)
    # Exactly the three types the evaluator removes from every view are
    # background; every other class is a foreground entry.
    assert labels.background_ids == classes.ids_of_type(
        ClassType.IGNORED, ClassType.BACKGROUND, ClassType.EXCLUDED)
    assert set(labels.entries) == set(classes.entries) - labels.background_ids


def test_a_set_without_colours_is_refused():
    """ATLAS-120k's benchmark classes have no colours of their own; drawing
    them would mean inventing some, which is not done silently."""
    with pytest.raises(ValueError, match="no colour"):
        label_table_of(load_table("atlas120k", "benchmark"))


def test_present_ids_leaves_background_out():
    t = LabelTable(entries={}, background_ids={0, 7})
    assert t.present_ids(np.array([[0, 3], [7, 3], [5, 0]])) == [3, 5]
    # An id the table does not know is listed, named by its number, and
    # drawn in the background colour.
    assert t.name(3) == "id 3"
    assert t.color(3) == BACKGROUND_COLOR


def test_instance_colours_are_stable_and_distinct():
    assert instance_color(1) == INSTANCE_PALETTE[0].tolist()
    assert instance_color(25) == instance_color(1)
    assert len({tuple(c) for c in INSTANCE_PALETTE.tolist()}) == len(INSTANCE_PALETTE)


def test_background_id_is_never_given_an_instance_colour():
    """Through the modulo, id 0 would silently get the palette's last colour."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="start at 1"):
            instance_color(bad)
    assert 0 not in instance_table([0, 1, 2]).entries


def test_remap_to_instances_is_consistent_across_frames():
    a = np.array([[-1, 4], [4, 9]])
    b = np.array([[9, 9], [-1, 2]])
    remapped, table, orig_to_new = remap_to_instances([a, b])
    assert orig_to_new == {2: 1, 4: 2, 9: 3}
    assert remapped[0].tolist() == [[0, 2], [2, 3]]
    assert remapped[1].tolist() == [[3, 3], [0, 1]]
    assert set(table.entries) == {1, 2, 3}
    assert table.name(3) == "obj 3"
    assert table.color(3) == instance_color(3)
    assert table.is_background(0)
