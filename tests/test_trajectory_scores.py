"""The reader of camera trajectory score files compares only rows of one scorer, each clip once per condition."""

import json

import pytest

from trajectory_eval.tools.scores import HASH_KEY, read_rows

SHA_A, SHA_B = "a" * 64, "b" * 64


def row(clip: str, cond: str, sha: str | None = SHA_A) -> dict:
    r = dict(clip=clip, seq=clip.split("__")[0], cond=cond, ate_rel=0.5)
    if sha is not None:
        r[HASH_KEY] = sha
    return r


def write(path, rows) -> str:
    path.write_text(json.dumps(rows))
    return str(path)


def test_the_rows_of_one_scorer_are_read_in_order(tmp_path):
    a = write(tmp_path / "controls.json", [row("P1__clip_0001", "floor_static"), row("P1__clip_0001", "floor_line")])
    b = write(tmp_path / "methods.json", [row("P1__clip_0001", "da3")])
    assert [r["cond"] for r in read_rows([a, b])] == ["floor_static", "floor_line", "da3"]


def test_rows_of_two_versions_of_the_scorer_are_refused(tmp_path):
    a = write(tmp_path / "controls.json", [row("P1__clip_0001", "floor_static")])
    b = write(tmp_path / "methods.json", [row("P1__clip_0001", "da3", SHA_B)])
    with pytest.raises(ValueError, match=r"the rows record 2 versions of the scorer, which are not compared: "
                                         r"aaaaaaaaaaaa in \[.*controls.json'\]; bbbbbbbbbbbb in \[.*methods.json'\]"):
        read_rows([a, b])


def test_a_row_without_the_hash_is_refused(tmp_path):
    # The workbench's rows record no hash.
    a = write(tmp_path / "methods.json", [row("P1__clip_0001", "da3"), row("P1__clip_0001", "pi3x", None)])
    with pytest.raises(ValueError, match=r"methods.json: row 1 records no trajectory_code_sha"):
        read_rows([a])


def test_a_clip_scored_twice_under_one_condition_is_refused(tmp_path):
    # Two clip roots scored without a suffix both name their conditions `da3`.
    a = write(tmp_path / "methods.json", [row("P1__clip_0001", "da3")])
    b = write(tmp_path / "methods_masked.json", [row("P1__clip_0001", "da3")])
    with pytest.raises(ValueError, match=r"P1__clip_0001 is scored twice under da3, in .*methods.json and "
                                         r".*methods_masked.json"):
        read_rows([a, b])


@pytest.mark.parametrize("content", [{"rows": []}, [["P1__clip_0001", "da3"]]])
def test_a_file_that_is_not_a_list_of_rows_is_refused(tmp_path, content):
    with pytest.raises(ValueError, match=r"is not a list of rows"):
        read_rows([write(tmp_path / "x.json", content)])
