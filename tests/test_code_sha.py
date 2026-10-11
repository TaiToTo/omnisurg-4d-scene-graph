"""`eval_code_sha` covers every module and every table, moves when one of them does, and refuses what it cannot place."""
import re
import shutil
from pathlib import Path

import pytest

import evalkit
from evalkit.code_sha import (
    DOCUMENTS, HASHED_MODULES, PACKAGE_DIR, RECORDED_LIBRARIES, TOOLS_DIR_NAME, _check_covered, eval_code_files,
    eval_code_sha, hashed_files)


def copy_package(tmp_path: Path) -> Path:
    """A copy of the installed package, for planting faults in."""
    dst = tmp_path / "evalkit"
    shutil.copytree(PACKAGE_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_the_sha_is_a_full_sha256_and_the_same_on_two_calls():
    sha = eval_code_sha()
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert eval_code_sha() == sha


def test_every_module_of_the_package_is_listed_and_the_tools_are_not():
    package = Path(evalkit.__file__).parent
    assert {p.name for p in package.glob("*.py")} == set(HASHED_MODULES)
    assert "code_sha.py" in HASHED_MODULES
    assert (package / TOOLS_DIR_NAME / "__init__.py").is_file()
    assert not any(f.startswith(TOOLS_DIR_NAME + "/") for f in eval_code_files())


def test_the_hashed_files_are_the_modules_then_the_tables_in_a_fixed_order():
    files = eval_code_files()
    assert files == sorted(HASHED_MODULES) + ["class_tables/atlas120k.json", "class_tables/cholecseg8k.json"]
    assert [p.name for p in hashed_files()] == [Path(f).name for f in files]


def test_a_copy_of_the_package_has_the_same_sha_wherever_it_lies(tmp_path, monkeypatch):
    # The paths hashed are package-relative, so neither the install location
    # nor how the package is named on the way there moves the sha.
    assert eval_code_sha(copy_package(tmp_path)) == eval_code_sha()
    monkeypatch.chdir(PACKAGE_DIR.parent)
    assert eval_code_sha(Path(PACKAGE_DIR.name)) == eval_code_sha()


@pytest.mark.parametrize("name", ["class_tables/cholecseg8k.json", "objects.py", "code_sha.py"])
def test_one_changed_byte_in_a_table_or_a_module_is_a_new_evaluator(tmp_path, name):
    pkg = copy_package(tmp_path)
    path = pkg / name
    data = path.read_bytes()
    path.write_bytes(data + b"\n")
    assert eval_code_sha(pkg) != eval_code_sha()
    path.write_bytes(data)
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_tool_can_change_or_appear_without_moving_the_sha(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / TOOLS_DIR_NAME / "__init__.py").write_text("x = 1\n")
    (pkg / TOOLS_DIR_NAME / "paired_stats.py").write_text("MARK_RULE = 'ci'\n")
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_document_can_change_or_be_absent_without_moving_the_sha(tmp_path):
    # A wheel ships the modules and the tables and no README, and must give
    # the same sha as the checkout it was built from.
    assert DOCUMENTS == {"README.md"}
    pkg = copy_package(tmp_path)
    (pkg / "README.md").write_text("# rewritten\n")
    assert eval_code_sha(pkg) == eval_code_sha()
    (pkg / "README.md").unlink()
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_document_not_named_is_refused(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "NOTES.md").write_text("# notes\n")
    with pytest.raises(ValueError, match="NOTES.md"):
        eval_code_sha(pkg)


def test_a_module_on_no_list_is_refused_rather_than_left_out(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "stray.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="stray.py"):
        eval_code_sha(pkg)
    with pytest.raises(ValueError, match="stray.py"):
        _check_covered(pkg)


def test_a_broken_symlink_is_refused_like_any_other_entry(tmp_path):
    # It is neither a file nor a directory, so a check that sorts entries by
    # `is_file()` would never see it.
    pkg = copy_package(tmp_path)
    (pkg / "ghost.py").symlink_to("nowhere")
    with pytest.raises(ValueError, match="ghost.py"):
        eval_code_sha(pkg)


def test_a_listed_module_that_is_missing_is_refused(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "vi.py").unlink()
    with pytest.raises(FileNotFoundError, match="vi.py"):
        eval_code_sha(pkg)


def test_a_file_that_is_not_a_module_is_refused(tmp_path):
    # A data file a module read would decide scores without being hashed.
    pkg = copy_package(tmp_path)
    (pkg / "weights.npy").write_bytes(b"\0")
    with pytest.raises(ValueError, match="weights.npy"):
        eval_code_sha(pkg)


def test_what_the_filesystem_writes_on_its_own_is_skipped(tmp_path):
    # Finder leaves `.DS_Store` in any directory it opens; it decides nothing,
    # and raising on it would make the evaluator unusable on a Mac for no gain.
    pkg = copy_package(tmp_path)
    (pkg / ".DS_Store").write_bytes(b"\0")
    (pkg / "class_tables" / ".DS_Store").write_bytes(b"\0")
    (pkg / "__pycache__").mkdir()
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_directory_other_than_the_tables_and_the_tools_is_refused(tmp_path):
    # A subpackage would be neither listed nor hashed, and not even seen.
    pkg = copy_package(tmp_path)
    (pkg / "scoring").mkdir()
    (pkg / "scoring" / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="scoring"):
        eval_code_sha(pkg)


def test_the_tables_of_a_copy_are_checked_as_the_packages_own(tmp_path):
    # A renamed or added table is refused, not hashed under its new name.
    pkg = copy_package(tmp_path)
    (pkg / "class_tables" / "cholecseg8k.json").rename(pkg / "class_tables" / "cholec.json")
    with pytest.raises(FileNotFoundError, match="cholecseg8k.json"):
        eval_code_sha(pkg)
    (pkg / "class_tables" / "cholec.json").rename(pkg / "class_tables" / "cholecseg8k.json")
    (pkg / "class_tables" / "notes.json").write_text("{}")
    with pytest.raises(ValueError, match="notes.json"):
        eval_code_sha(pkg)


def plant_import(pkg: Path, line: str) -> None:
    """Append an import to a hashed module of the copy, inside a function so nothing runs."""
    with open(pkg / "objects.py", "a", encoding="utf-8") as f:
        f.write(f"\n\ndef _planted():\n    {line}\n")


@pytest.mark.parametrize("line, named", [
    ("from surgical_core.cholec import *", "surgical_core.cholec"),
    ("import surgical_core.atlas120k as a", "surgical_core.atlas120k"),
    ("from evalkit.tools import paired_stats", "evalkit.tools"),
    ("from evalkit.tools.paired_stats import mark_of", "evalkit.tools.paired_stats"),
    ("from . import tools", "evalkit.tools"),
    ("from .tools import paired_stats", "evalkit.tools"),
    ("import scipy", "scipy"),
    ("import importlib", "importlib"),
    ("from importlib import import_module", "importlib"),
    ("mod = __import__('scipy')", "__import__"),
])
def test_an_import_to_what_is_neither_hashed_nor_recorded_is_refused(tmp_path, line, named):
    # A GT reader outside the package, a tool, or a library whose version no
    # score records would each decide scores without a trace in the sha.
    pkg = copy_package(tmp_path)
    plant_import(pkg, line)
    with pytest.raises(ValueError, match=re.escape(named)):
        eval_code_sha(pkg)


@pytest.mark.parametrize("line", [
    "from evalkit.classes import VIEWS",
    "from evalkit import classes",
    "from . import classes",
    "from .classes import VIEWS",
    "import evalkit.vi",
    "import json, struct",
    "import numpy, cv2, PIL.Image",
])
def test_an_import_of_a_hashed_module_the_standard_library_or_a_recorded_library_passes(tmp_path, line):
    assert RECORDED_LIBRARIES == {"numpy", "cv2", "PIL"}
    pkg = copy_package(tmp_path)
    plant_import(pkg, line)
    assert eval_code_sha(pkg) != eval_code_sha()
