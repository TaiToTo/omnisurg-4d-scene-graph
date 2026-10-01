"""The viewer's label table shows what the evaluator's class table defines.

A GT label table is derived from `evalkit.classes`, so the viewer cannot
drift from the evaluator on a name or a colour; what is tested is the
derivation (which types are background, that a colourless foreground class
is refused) and the instance side (stable colours, cross-frame renumbering,
the background id never coloured).
"""

import numpy as np
import pytest

from evalkit.classes import load_table
from surgical_core.viewer.labels import (
    LabelTable, cholec_gt_table, instance_table, label_table_of, remap_to_instances)
from surgical_core.viewer.palette import BACKGROUND_COLOR, INSTANCE_PALETTE, instance_color


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
    markers = classes.excluded_mask_ids | classes.background_mask_ids
    assert markers <= labels.background_ids
    assert not (set(labels.entries) & labels.background_ids)
    assert all(e.color for e in labels.entries.values())


def test_a_set_without_colours_is_refused():
    """ATLAS-120k's benchmark classes have no colours of their own; drawing
    them would mean inventing some, which is not done silently."""
    with pytest.raises(ValueError, match="no colour"):
        label_table_of(load_table("atlas120k", "benchmark"))


def test_present_ids_leaves_background_out():
    t = LabelTable(entries={}, background_ids={0, 7})
    assert t.present_ids(np.array([[0, 3], [7, 3], [5, 0]])) == [3, 5]
    assert t.name(3) == "id 3"


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
