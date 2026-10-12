"""Every name in the table follows the rule of names, and the reader refuses what it does not hold.

Each check of `parse` and `read_table` is shown refusing a planted fault.
"""
from pathlib import Path

import pytest

from evalkit.evaluate import PROPAGATION_RULES
from evalkit.tools import condition_names as CN

HEADER = "\t".join(CN.COLUMNS)


def test_every_name_in_the_table_parses_back_into_its_parts():
    table = CN.read_table()
    names = [row.name for row in table.values() if row.name]
    assert names, "the table names no condition; the check reads nothing"
    for name in names:
        assert CN.parse(name).name() == name


def test_the_table_covers_both_datasets_and_says_why_a_row_has_no_name():
    table = CN.read_table()
    assert {d for d, _ in table} == set(CN.DATASETS)
    assert all(row.unresolved for row in table.values() if not row.name)


@pytest.mark.parametrize("name, parts", [
    ("rgb_pps24_sam_centre", CN.Parts("rgb", None, 24, "sam", "centre")),
    ("normal-edge_da3_pps24_sam_forward", CN.Parts("normal-edge", "da3", 24, "sam", "forward")),
    ("normal-edge_pi3x_pps8_perframe", CN.Parts("normal-edge", "pi3x", 8, None, "perframe")),
    ("rgb_pps24_kmerge10_centre", CN.Parts("rgb", None, 24, "kmerge10", "centre")),
    ("rgb_gt_centre", CN.Parts("rgb", None, None, "gt", "centre")),
    ("normal-edge_da3_pps24_sam_centre_keepring_instrument-text",
     CN.Parts("normal-edge", "da3", 24, "sam", "centre", ("keepring", "instrument-text"))),
])
def test_a_name_is_read_into_its_parts(name, parts):
    assert CN.parse(name) == parts and parts.name() == name


@pytest.mark.parametrize("name, said", [
    ("normal_edge_da3_pps24_sam_centre", "needs a depth model"),
    ("normal_pps24_sam_centre", "needs a depth model"),
    ("rgb_da3_pps24_sam_centre", "takes no depth model"),
    ("rgb_pps24_centre", "a seed is named under a propagated rule"),
    ("rgb_pps24_sam_perframe", "a seed is named under a propagated rule"),
    ("rgb_sam_centre", "the grid is named"),
    ("rgb_pps24_gt_centre", "the grid is named"),
    ("rgb_perframe", "the grid is named"),
    ("rgb_pps24_sam", "is no propagation rule"),
    ("rgb_pps24_sam_center", "is no propagation rule"),
    ("rgb_pps0_sam_centre", "'pps0' is no grid and no seed"),
    ("rgb_pps24_kmerge0_centre", "'kmerge0' is no grid and no seed"),
    ("rgbd_pps24_sam_centre", "is no input"),
    ("rgb__pps24_sam_centre", "an empty part"),
    ("rgb_pps24_sam_centre_keep-ring", "are no settings"),
    ("normal-edge_da3_pps24_sam_centre_instrument-text_keepring", "in the order"),
    ("normal-edge_da3_pps24_sam_centre_keepring_keepring", "in the order"),
    ("rgb_pps24_sam_centre_keepring", "has a ring to keep"),
    ("rgb_pps24_perframe_instrument-text", "adds instruments"),
])
def test_a_name_that_breaks_the_rule_is_refused(name, said):
    with pytest.raises(ValueError, match=said):
        CN.parse(name)


def test_the_rules_are_the_ones_a_score_records():
    assert set(CN.RULES.values()) == set(PROPAGATION_RULES)


def test_the_inputs_are_the_segmenter_s_input_modes():
    render = pytest.importorskip("surgical_core.geometry.render", reason="the input modes need the `render` extra")
    assert CN.INPUTS == tuple(m.replace("_", "-") for m in render.SAM_INPUT_MODES)
    assert CN.EDGE_INPUTS == tuple(m.replace("_", "-") for m in render.SAM_INPUT_MODES if render.uses_geom_edge(m))


def test_the_reader_gives_a_name_and_refuses_an_unknown_or_unresolved_condition():
    assert CN.name_of("atlas120k", "op_edge_center") == "normal-edge_da3_pps24_sam_centre"
    assert CN.name_of("cholecseg8k", "t5_k10") == CN.name_of("atlas120k", "t5_k10") == "rgb_pps24_kmerge10_centre"
    with pytest.raises(ValueError, match="no workbench condition of cholecseg8k"):
        CN.name_of("cholecseg8k", "op_edge_center")
    with pytest.raises(ValueError, match="its records show no name: the GT seed sits one frame before the centre"):
        CN.name_of("atlas120k", "t12_rgb")


GOOD = ["atlas120k\top_edge_center\tnormal-edge_da3_pps24_sam_centre\t\tseed_info.json",
        "atlas120k\top_edge\t\tthe seed frame is off the centre\tseed_info.json"]


def write(tmp_path: Path, lines: list[str], header: str = HEADER) -> Path:
    path = tmp_path / "names.tsv"
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    return path


def test_a_good_table_is_read(tmp_path):
    table = CN.read_table(write(tmp_path, GOOD))
    assert CN.name_of("atlas120k", "op_edge_center", table) == "normal-edge_da3_pps24_sam_centre"


@pytest.mark.parametrize("lines, header, said", [
    (GOOD, "dataset\tworkbench\tname\trecord", "the header is not"),
    (GOOD + ["atlas120k\tx\trgb_pps24_sam_forward\t\tseed_info.json\textra"], HEADER, "6 fields, not 5"),
    (GOOD + ["stereomis\tx\trgb_pps24_sam_forward\t\tseed_info.json"], HEADER, "no dataset of"),
    (GOOD + ["atlas120k\tx\trgb_pps24_sam_forward\t\tthe score"], HEADER, "is no record of"),
    (GOOD + ["atlas120k\tx\trgb_pps24_sam_forward\tunclear\tseed_info.json"], HEADER, "not both and not neither"),
    (GOOD + ["atlas120k\tx\t\t\tseed_info.json"], HEADER, "not both and not neither"),
    (GOOD + ["atlas120k\tx\trgb_pps24_sam_center\t\tseed_info.json"], HEADER, "is no propagation rule"),
    (GOOD + ["atlas120k\top_edge\trgb_pps24_sam_forward\t\tseed_info.json"], HEADER, "comes twice"),
    (GOOD + ["atlas120k\tx\tnormal-edge_da3_pps24_sam_centre\t\tseed_info.json"], HEADER,
     "also names op_edge_center"),
])
def test_a_table_that_breaks_its_rules_is_refused(tmp_path, lines, header, said):
    with pytest.raises(ValueError, match=said):
        CN.read_table(write(tmp_path, lines, header))


def test_one_name_in_two_datasets_is_one_condition(tmp_path):
    table = CN.read_table(write(tmp_path, GOOD + [
        "cholecseg8k\tch_edge_center_noring\tnormal-edge_da3_pps24_sam_centre\t\tseed_info.json, run log"]))
    assert CN.name_of("cholecseg8k", "ch_edge_center_noring", table) == CN.name_of("atlas120k", "op_edge_center", table)
