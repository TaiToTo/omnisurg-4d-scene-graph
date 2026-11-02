"""The judged cut table of ATLAS-120k, `atlas120k_meta/cuts.csv`, and `load_cuts`, which reads it.

The first tests read the committed table and check that it holds what the
README says: every row is a candidate inside a clip of the population, a
cut is one pair of adjacent frames at its candidate's boundary, no cut is
listed twice, the rows are in the population's order, and the counts are
the README's, those of the split at the cuts included. The others plant
each fault `load_cuts` refuses in a table made up for the test.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from surgical_core.atlas120k.cuts import COLUMNS, load_cuts, whole_clip

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"
CLIP = "proc__vid__gt_0001"
# The clips whose every piece fell below eight frames when split at their
# cuts, so that the population holds none of them.
DROPPED_AT_CUTS = frozenset({
    "cholecystectomy__JedzslyadF8__gt_0004", "cholecystectomy__m0j9Us_4zR8__gt_0003",
    "hemicolectomy__5YDMlxTl0k8__gt_0026", "hemicolectomy__6qb-oFqRjbE__gt_0019",
    "hemicolectomy__SbqubK1FX50__gt_0001", "hemicolectomy__SbqubK1FX50__gt_0008",
})


@pytest.fixture(scope="module")
def rows() -> list:
    text = (META / "cuts.csv").read_bytes()
    text.decode("ascii")
    table = list(csv.DictReader(text.decode("ascii").splitlines()))
    assert tuple(table[0]) == COLUMNS
    return table


def _whole_clips() -> set[str]:
    """The clips of the population before the split: each clip, or the clip each piece came from."""
    return {whole_clip(c) for c in (META / "clips.txt").read_text().split()}


@pytest.fixture(scope="module")
def order() -> dict:
    # The table judged the clips before the split, the dropped ones included, in name order.
    return {clip: i for i, clip in enumerate(sorted(_whole_clips() | DROPPED_AT_CUTS))}


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


def _row(**kw) -> dict:
    row = dict(clip=CLIP, spot_first="100", spot_last="120", d_pix="35.0", boundary_before="110",
               boundary_after="111", cut="yes", cut_before="110", cut_after="111", rechecked="0")
    row.update(kw)
    return row


def _write(tmp_path, rows, columns=COLUMNS) -> str:
    p = tmp_path / "cuts.csv"
    lines = [",".join(columns)] + [",".join(r[c] for c in columns) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_the_cuts_are_the_rows_judged_a_cut_ascending_per_clip(tmp_path):
    rows = [_row(boundary_before="300", boundary_after="301", cut_before="300", cut_after="301"),
            _row(cut="no", cut_before="", cut_after=""),
            _row(boundary_before="50", boundary_after="51", cut_before="52", cut_after="53"),
            _row(clip="proc__vid__gt_0002", cut="no", cut_before="", cut_after="")]
    assert load_cuts(_write(tmp_path, rows)) == {CLIP: (52, 300)}


def test_a_missing_table_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cut table"):
        load_cuts(str(tmp_path / "cuts.csv"))


def test_other_columns_are_refused(tmp_path):
    with pytest.raises(ValueError, match="the columns are"):
        load_cuts(_write(tmp_path, [_row()], columns=COLUMNS[:-1]))


@pytest.mark.parametrize("fault, match", [
    (dict(cut="maybe"), "cut is 'maybe', not 'yes' or 'no'"),
    (dict(cut_before=""), "cut_before is '', not a frame number"),
    (dict(cut_after="x"), "cut_after is 'x', not a frame number"),
    (dict(cut_after="112"), "not between 110 and 112"),
    (dict(cut="no"), "a candidate that is not a cut names a cut's frames"),
    (dict(clip="proc__vid__clip_0001"), "is not a clip name"),
])
def test_a_row_that_cannot_be_read_for_certain_is_refused(tmp_path, fault, match):
    with pytest.raises(ValueError, match=match):
        load_cuts(_write(tmp_path, [_row(), _row(**fault)]))


def test_a_cut_listed_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="listed twice"):
        load_cuts(_write(tmp_path, [_row(), _row(boundary_before="109", boundary_after="110")]))


def test_the_committed_table_holds_72_cuts_in_39_clips():
    cuts = load_cuts(str(META / "cuts.csv"))
    assert sum(map(len, cuts.values())) == 72 and len(cuts) == 39
    assert set(cuts) <= _whole_clips() | DROPPED_AT_CUTS


def test_the_split_at_the_cuts_gives_the_readmes_counts():
    clips = (META / "clips.txt").read_text().split()
    pieces = Counter(whole_clip(c) for c in clips)
    with_cut = set(load_cuts(str(META / "cuts.csv")))
    split = [pieces[c] for c in with_cut if pieces[c] > 1]
    # Of the 315 clips before the split: 276 hold no cut and stay whole, 17 keep one piece, 16 give 46 pieces,
    # and 6 lose every piece.
    assert [n for c, n in pieces.items() if c not in with_cut] == [1] * 276
    assert sum(pieces[c] == 1 for c in with_cut) == 17
    assert (len(split), sum(split)) == (16, 46)
    assert with_cut - set(pieces) == DROPPED_AT_CUTS
    manifest = json.loads((META / "depth_manifest.json").read_text())
    assert (len(clips), len({tuple(c.split("__")[:2]) for c in clips}), manifest["n_frames"]) == (339, 91, 7547)


def test_whole_clip_names_the_clip_a_piece_came_from():
    assert whole_clip("proc__vid__gt_0001") == "proc__vid__gt_0001"
    assert whole_clip("proc__vid__gt_0001s2") == "proc__vid__gt_0001"
    for name in ("proc__vid__clip_0001", "proc__vid__gt_0001s", "proc__vid__gt_0001x2"):
        with pytest.raises(ValueError, match="not a clip of the population"):
            whole_clip(name)
