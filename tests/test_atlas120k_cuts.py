"""The judged cut candidates of the ATLAS-120k population hold what the README says.

`atlas120k_meta/cuts.csv` is a copy of a table judged in the private
workbench. Nothing reads it yet, so these checks are what keeps an edit to
it from quietly contradicting the README: every row is a candidate inside a
clip of the population, a cut is one pair of adjacent frames at its
candidate's boundary, no cut is listed twice, the rows are in the
population's order, and the counts are the README's.
"""

import csv
from pathlib import Path

import pytest

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"

COLUMNS = ["clip", "spot_first", "spot_last", "d_pix", "boundary_before", "boundary_after",
           "cut", "cut_before", "cut_after", "rechecked"]


@pytest.fixture(scope="module")
def rows() -> list:
    text = (META / "cuts.csv").read_bytes()
    text.decode("ascii")
    table = list(csv.DictReader(text.decode("ascii").splitlines()))
    assert list(table[0]) == COLUMNS
    return table


@pytest.fixture(scope="module")
def order() -> dict:
    return {clip: i for i, clip in enumerate((META / "clips.txt").read_text().split())}


def test_every_row_is_a_candidate_inside_a_clip_of_the_population(rows, order):
    for r in rows:
        assert r["clip"] in order
        first, last = int(r["spot_first"]), int(r["spot_last"])
        before, after = int(r["boundary_before"]), int(r["boundary_after"])
        assert first <= before < after <= last
        assert after == before + 1
        assert float(r["d_pix"]) > 0
        assert r["rechecked"] in ("0", "1")


def test_a_cut_is_its_candidate_pair_and_a_row_that_is_not_a_cut_names_no_frames(rows):
    for r in rows:
        assert r["cut"] in ("yes", "no")
        if r["cut"] == "yes":
            assert (r["cut_before"], r["cut_after"]) == (r["boundary_before"], r["boundary_after"])
        else:
            assert (r["cut_before"], r["cut_after"]) == ("", "")


def test_the_rows_are_in_the_populations_order_and_no_boundary_is_listed_twice(rows, order):
    keys = [(order[r["clip"]], int(r["boundary_after"])) for r in rows]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_the_counts_are_the_readmes(rows):
    cuts = [r for r in rows if r["cut"] == "yes"]
    assert len(rows) == 433
    assert len(cuts) == 72
    assert len({r["clip"] for r in cuts}) == 39
    assert sum(r["rechecked"] == "1" for r in rows) == 46
    assert sum(r["rechecked"] == "1" for r in rows if r["cut"] == "no") == 1
    assert sum(float(r["d_pix"]) <= 30 for r in rows) == 2
