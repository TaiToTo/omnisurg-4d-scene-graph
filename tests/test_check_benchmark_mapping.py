"""Compare a copy of ATLAS-bench's `datasets/class_mapping.py`, written from the class table, with the class table.

The copy agrees on every id, and each fault is planted in it: an id mapped to another class, an id missing, an id
too many, a statement that could change the mapping after it is assigned, and files the tool cannot read the
mapping from. The copy imports a module that does not exist, as the real file cannot run without its imports, so
a tool that ran it would fail every test.
"""

import sys

import pytest

from evalkit.classes import load_table
from evalkit.tools.check_benchmark_mapping import disagreements, main, read_upstream_mapping

OURS = dict(load_table("atlas120k", "benchmark").mask_id_to_class)

# The real file's statements around the dict, and a docstring: an import, another dict, a function and a
# `__main__` block, whose change of the dict runs only when the file runs as a script.
HEAD = ('"""Class mappings."""\nfrom no_such_package_for_this_test import Image\n'
        'mapped_class_names = {0: "Background"}\n')
TAIL = ('def remap_mask(mask, mapping, default_value=0):\n    """Remap a mask."""\n    return mask\n\n'
        'if __name__ == "__main__":\n    mapping[32] = 26\n')


def write_upstream(tmp_path, mapping: dict, text: str | None = None, after: str = ""):
    """Write a `class_mapping.py` that assigns `mapping`, or holds `text` in its place, followed by `after`."""
    entries = "\n".join(f"    {k}: {v}," for k, v in mapping.items())
    body = text if text is not None else f"mapping = {{\n{entries}\n}}\n"
    path = tmp_path / "class_mapping.py"
    path.write_text(HEAD + body + after + TAIL)
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


@pytest.mark.parametrize("after", [
    "mapping[32] = 26\n",
    "mapping.update({46: 5})\n",
    "for k in (5, 6):\n    mapping[k] = 0\n",
    "mapping |= {47: 3}\n",
    "if True:\n    mapping[32] = 26\n",
    'if __name__ != "__main__":\n    mapping[32] = 26\n',
    "other = mapping.update({46: 5})\n",
    "from other import mapping\n",
    "from other import *\n",
    "from other import table as mapping\n",
    "class Remap:\n    mapping[32] = 26\n",
    "@register\ndef remap(mask):\n    return mask\n",
    "def remap(mask, m=mapping.update({46: 5})):\n    return mask\n",
])
def test_a_statement_that_could_change_the_mapping_after_it_is_assigned_is_refused(tmp_path, monkeypatch, after):
    path = write_upstream(tmp_path, OURS, after=after)
    with pytest.raises(ValueError, match="could change `mapping` after it is assigned"):
        read_upstream_mapping(path)
    monkeypatch.setattr(sys, "argv", ["check_benchmark_mapping", str(path)])
    with pytest.raises(SystemExit, match="could change `mapping` after it is assigned"):
        main()
