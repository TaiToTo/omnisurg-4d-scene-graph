"""The class tables say what `docs/evaluation.md` says, and refuse what they cannot trust."""
import json
import re
from pathlib import Path

import pytest

from evalkit import classes
from evalkit.classes import VIEWS, ClassType, load_table, table_paths

REPO = Path(__file__).resolve().parent.parent
EVALUATION_MD = REPO / "docs" / "evaluation.md"


def cholec_table_of_the_specification() -> dict[int, tuple[str, tuple[int, int, int], str]]:
    """Read the CholecSeg8k table out of docs/evaluation.md, so that the JSON
    and the document are checked against each other and not against a third
    copy typed by hand."""
    text = EVALUATION_MD.read_text(encoding="utf-8")
    section = text.split("### CholecSeg8k", 1)[1].split("\n### ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("id", "---:") or set(cells[0]) <= {"-", ":"}:
            continue
        cid, name, colour, type_cell = cells
        if not cid.isdigit():
            raise AssertionError(f"the document gives {name!r} no id: {line!r}")
        m = re.fullmatch(r"\((\d+), (\d+), (\d+)\)", colour)
        assert m, line
        rows[int(cid)] = (name, tuple(int(v) for v in m.groups()), re.match(r"[a-z]+", type_cell).group())
    assert len(rows) >= 13, "the CholecSeg8k table was not found in docs/evaluation.md"
    return rows


@pytest.fixture
def cholec():
    return load_table("cholecseg8k")


def test_cholecseg8k_matches_the_specification(cholec):
    spec = cholec_table_of_the_specification()
    assert set(cholec.entries) == set(spec)
    for cid, (name, colour, class_type) in spec.items():
        e = cholec.entries[cid]
        assert (e.name, e.colour, e.type) == (name, colour, ClassType(class_type)), cid


def test_the_specification_parser_reads_the_table_it_claims_to():
    spec = cholec_table_of_the_specification()
    assert spec[11] == ("Hepatic Vein", (0, 50, 128), "expert")
    assert spec[13][1] == (255, 255, 255)


def test_cholecseg8k_reads_hepatic_vein_by_the_colour_the_masks_use(cholec):
    # The pilot evaluator's table had (0, 255, 0), a colour no mask contains.
    assert cholec.id_of_colour((0, 50, 128)) == 11
    with pytest.raises(KeyError):
        cholec.id_of_colour((0, 255, 0))


def test_an_unknown_colour_raises_rather_than_becoming_background(cholec):
    with pytest.raises(KeyError, match=r"\(0, 0, 0\)"):
        cholec.id_of_colour((0, 0, 0))


def test_no_colour_is_another_class_read_in_bgr(cholec):
    # A mask read with the channels reversed must not name a wrong class
    # silently; it may only hit the same grey or raise.
    for e in cholec.entries.values():
        r, g, b = e.colour
        if (b, g, r) in cholec.colour_to_id:
            assert cholec.colour_to_id[(b, g, r)] == e.id


def test_an_unknown_id_raises(cholec):
    with pytest.raises(KeyError, match="14"):
        cholec.type_of(14)


def test_ids_of_type_without_a_type_raises(cholec):
    with pytest.raises(TypeError):
        cholec.ids_of_type()


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


def test_the_hashed_files_are_exactly_the_tables_the_loader_reads(cholec):
    assert table_paths() == [cholec.path]


def test_a_stray_file_in_the_table_directory_is_refused(tmp_path, monkeypatch):
    # Otherwise eval_code_sha would differ on the one machine that has the file.
    for p in table_paths():
        (tmp_path / p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(classes, "TABLE_DIR", tmp_path)
    assert table_paths() == [tmp_path / "cholecseg8k.json"]
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="notes.json"):
        table_paths()


def test_a_missing_table_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(classes, "TABLE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="cholecseg8k.json"):
        table_paths()


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
    (lambda r: _class(r, 7).update(name="Liver"), "name 'liver' is given to two"),
    (lambda r: _class(r, 7).update(name="LIVER"), "name 'liver' is given to two"),
    (lambda r: _class(r, 7).update(name=7), "expected a non-empty string"),
    (lambda r: _class(r, 7).update(name=" "), "expected a non-empty string"),
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


def test_a_key_given_twice_is_refused(tmp_path):
    # json.load would keep the second type and make Blood geometric, silently.
    # json.dumps cannot write a duplicate key, so the text is edited directly.
    text = Path(load_table("cholecseg8k").path).read_text(encoding="utf-8")
    planted = text.replace('"name": "Blood",', '"name": "Blood", "type": "tissue",', 1)
    assert planted != text
    path = tmp_path / "cholecseg8k.json"
    path.write_text(planted, encoding="utf-8")
    with pytest.raises(ValueError, match="'type' is given twice"):
        load_table("cholecseg8k", path=path)


def test_the_planted_copy_loads_when_nothing_is_planted(tmp_path):
    # Otherwise the faults above could be refused for a reason other than the fault.
    path = _plant(tmp_path, lambda r: None)
    assert load_table("cholecseg8k", path=path).entries.keys() == load_table("cholecseg8k").entries.keys()


def test_a_dataset_without_a_table_format_is_refused(tmp_path):
    path = tmp_path / "lapex.json"
    path.write_text(json.dumps({"dataset": "lapex", "classes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no table format"):
        load_table("lapex", path=path)
