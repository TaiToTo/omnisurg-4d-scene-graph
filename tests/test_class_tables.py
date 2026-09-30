"""The class tables say what `docs/evaluation.md` says, and refuse what they cannot trust."""
import json
from pathlib import Path

import pytest

from evalkit.classes import VIEWS, ClassType, load_table, table_paths

# The CholecSeg8k table of docs/evaluation.md, copied by hand so that the JSON
# and the document are checked against each other, not against themselves.
CHOLEC = {
    0: ("Black Background", (127, 127, 127), "ignored"),
    1: ("Abdominal Wall", (210, 140, 140), "backdrop"),
    2: ("Liver", (255, 114, 114), "tissue"),
    3: ("Gastrointestinal Tract", (231, 70, 156), "tissue"),
    4: ("Fat", (186, 183, 75), "tissue"),
    5: ("Grasper", (170, 255, 0), "tool"),
    6: ("Connective Tissue", (255, 85, 0), "appearance"),
    7: ("Blood", (255, 0, 0), "appearance"),
    8: ("Cystic Duct", (255, 255, 0), "expert"),
    9: ("L-hook Electrocautery", (169, 255, 184), "tool"),
    10: ("Gallbladder", (255, 160, 165), "tissue"),
    11: ("Hepatic Vein", (0, 50, 128), "expert"),
    12: ("Liver Ligament", (111, 74, 0), "tissue"),
    13: ("Region line", (255, 255, 255), "ignored"),
}


@pytest.fixture
def cholec():
    return load_table("cholecseg8k")


def test_cholecseg8k_matches_the_specification(cholec):
    assert set(cholec.entries) == set(CHOLEC)
    for cid, (name, colour, class_type) in CHOLEC.items():
        e = cholec.entries[cid]
        assert (e.name, e.colour, e.type) == (name, colour, ClassType(class_type))


def test_cholecseg8k_reads_hepatic_vein_by_the_colour_the_masks_use(cholec):
    # The pilot evaluator's table had (0, 255, 0), a colour no mask contains.
    assert cholec.id_of_colour((0, 50, 128)) == 11
    with pytest.raises(KeyError):
        cholec.id_of_colour((0, 255, 0))


def test_an_unknown_colour_raises_rather_than_becoming_background(cholec):
    with pytest.raises(KeyError, match=r"\(0, 0, 0\)"):
        cholec.id_of_colour((0, 0, 0))


def test_an_unknown_id_raises(cholec):
    with pytest.raises(KeyError, match="14"):
        cholec.type_of(14)


def test_the_views_nest_as_specified():
    assert VIEWS["geometric"] < VIEWS["tissue"] < VIEWS["all"]
    assert VIEWS["all"] - VIEWS["tissue"] == {ClassType.TOOL}
    assert VIEWS["geometric"] == {ClassType.TISSUE}
    for view in VIEWS.values():
        assert not view & {ClassType.IGNORED, ClassType.BACKGROUND, ClassType.EXCLUDED}


def test_cholecseg8k_views(cholec):
    assert cholec.ids_in_view("all") == frozenset(range(1, 13))
    assert cholec.ids_in_view("tissue") == frozenset(range(1, 13)) - {5, 9}
    assert cholec.ids_in_view("geometric") == {2, 3, 4, 10, 12}
    with pytest.raises(KeyError, match="unknown view"):
        cholec.ids_in_view("full")


def test_the_table_files_are_listed_for_hashing(cholec):
    assert cholec.path in table_paths()
    assert all(p.suffix == ".json" for p in table_paths())


# A check earns its place by failing when it should: each fault below is
# planted in a copy of the real table, and the loader has to refuse it.

def _plant(tmp_path: Path, edit) -> Path:
    raw = json.loads(Path(load_table("cholecseg8k").path).read_text(encoding="utf-8"))
    edit(raw)
    out = tmp_path / "cholecseg8k.json"
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def _class(raw, cid):
    return next(c for c in raw["classes"] if c["id"] == cid)


@pytest.mark.parametrize("fault, message", [
    (lambda r: _class(r, 7).update(type="tisue"), "type 'tisue'"),
    (lambda r: _class(r, 7).pop("type"), "fields"),
    (lambda r: _class(r, 7).update(note="x"), "fields"),
    (lambda r: _class(r, 7).update(id=2), "id 2 is given to two"),
    (lambda r: _class(r, 7).update(name="Liver"), "name 'Liver' is given to two"),
    (lambda r: _class(r, 7).update(colour=[255, 114, 114]), "colour .* is given to two"),
    (lambda r: _class(r, 7).update(colour=[255, 0]), "expected three ints"),
    (lambda r: _class(r, 7).update(colour=[255, 0, 256]), "expected three ints"),
    (lambda r: _class(r, 7).update(id=-1), "not a non-negative integer"),
    (lambda r: _class(r, 7).update(id=True), "not a non-negative integer"),
    (lambda r: r.update(dataset="cholec"), "file says dataset"),
    (lambda r: r.update(extra=1), "top-level keys"),
    (lambda r: r.update(classes=[]), "no classes"),
])
def test_a_planted_fault_is_refused(tmp_path, fault, message):
    path = _plant(tmp_path, fault)
    with pytest.raises(ValueError, match=message):
        load_table("cholecseg8k", path=path)


def test_the_planted_copy_loads_when_nothing_is_planted(tmp_path):
    # Otherwise the faults above could be refused for a reason other than the fault.
    path = _plant(tmp_path, lambda r: None)
    assert load_table("cholecseg8k", path=path).entries.keys() == CHOLEC.keys()


def test_a_dataset_without_a_table_format_is_refused(tmp_path):
    path = tmp_path / "lapex.json"
    path.write_text(json.dumps({"dataset": "lapex", "classes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no table format"):
        load_table("lapex", path=path)
