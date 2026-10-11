"""Compare a copy of ATLAS-bench's `datasets/class_mapping.py`, written from the class table, with the class table.

The copy agrees on every id, and each fault is planted in it: an id mapped to another class, an id missing, an id
too many, and files the tool cannot read the mapping from. The copy raises when run, as the real file would
without its imports, so a tool that ran it would fail every test.
"""

import sys

import pytest

from evalkit.classes import load_table
from evalkit.tools.check_benchmark_mapping import disagreements, main, read_upstream_mapping

OURS = dict(load_table("atlas120k", "benchmark").mask_id_to_class)


def write_upstream(tmp_path, mapping: dict, text: str | None = None):
    """Write a `class_mapping.py` that assigns `mapping`, or holds `text` in its place."""
    entries = "\n".join(f"    {k}: {v}," for k, v in mapping.items())
    body = text if text is not None else f"mapping = {{\n{entries}\n}}\n"
    path = tmp_path / "class_mapping.py"
    path.write_text(f'raise SystemExit("run, not parsed")\nmapped_class_names = {{0: "Background"}}\n{body}')
    return path


def run(monkeypatch, capsys, path) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["check_benchmark_mapping", str(path)])
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def test_a_copy_that_agrees_on_every_id_passes(tmp_path, monkeypatch, capsys):
    assert len(OURS) == 47
    path = write_upstream(tmp_path, OURS)
    assert read_upstream_mapping(path) == OURS
    assert run(monkeypatch, capsys, path) == (0, "all 47 ids map alike\n")


def test_an_id_mapped_to_another_class_is_named_and_fails(tmp_path, monkeypatch, capsys):
    # Kidney, which the benchmark sends to background, mapped to Lung instead.
    code, out = run(monkeypatch, capsys, write_upstream(tmp_path, {**OURS, 32: 26}))
    assert (code, out) == (1, "id 32: ATLAS-bench maps it to 26, the class table to 0\n")


def test_an_id_on_one_side_only_is_named():
    upstream = {k: v for k, v in OURS.items() if k != 46} | {47: 0}
    assert disagreements(upstream, OURS) == [
        "id 46: the class table maps it to 0, and ATLAS-bench has no such id",
        "id 47: ATLAS-bench maps it to 0, and the class table has no such id",
    ]


@pytest.mark.parametrize("text, why", [
    ("mapped = {0: 0}\n", "assigns `mapping` 0 times"),
    ("mapping = {0: 0}\nmapping = {0: 0}\n", "assigns `mapping` 2 times"),
    ("mapping = dict(zip(range(47), range(47)))\n", "`mapping` is not a literal"),
    ("mapping = {0: True}\n", "not a dict of integer ids to integer classes"),
    ("mapping = [0, 1]\n", "not a dict of integer ids to integer classes"),
    ("mapping = {\n", "is not Python"),
])
def test_a_file_the_mapping_cannot_be_read_from_is_refused(tmp_path, monkeypatch, text, why):
    path = write_upstream(tmp_path, {}, text)
    with pytest.raises(ValueError, match=why):
        read_upstream_mapping(path)
    monkeypatch.setattr(sys, "argv", ["check_benchmark_mapping", str(path)])
    with pytest.raises(SystemExit, match=why):
        main()
