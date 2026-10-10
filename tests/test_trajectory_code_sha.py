"""`trajectory_code_sha` covers every module of the package, moves when one does, and refuses what it cannot place."""

import re
import shutil
from pathlib import Path

import pytest

from evalkit.code_sha import _sha_of as eval_sha_of
from trajectory_eval.code_sha import (HASHED_MODULES, PACKAGE_DIR, TOOLS_DIR_NAME, _sha_of, hashed_files,
                                      trajectory_code_files, trajectory_code_sha)


def copy_package(tmp_path: Path) -> Path:
    """Return a copy of the package, for planting faults in."""
    dst = tmp_path / "trajectory_eval"
    shutil.copytree(PACKAGE_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_the_hash_is_a_full_sha256_and_the_same_on_two_calls():
    sha = trajectory_code_sha()
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert trajectory_code_sha() == sha


def test_every_module_of_the_package_is_hashed_and_the_tools_are_not():
    assert {p.name for p in PACKAGE_DIR.glob("*.py")} == set(HASHED_MODULES)
    assert {"code_sha.py", "pose_metrics.py", "score.py"} <= set(HASHED_MODULES)
    assert (PACKAGE_DIR / TOOLS_DIR_NAME / "__init__.py").is_file()
    assert trajectory_code_files() == sorted(HASHED_MODULES)


def test_the_bytes_are_hashed_as_eval_code_sha_hashes_the_evaluator():
    files = hashed_files()
    assert _sha_of(files, PACKAGE_DIR) == eval_sha_of(files, PACKAGE_DIR) == trajectory_code_sha()


def test_a_copy_of_the_package_has_the_same_hash_wherever_it_lies(tmp_path):
    assert trajectory_code_sha(copy_package(tmp_path)) == trajectory_code_sha()


@pytest.mark.parametrize("name", HASHED_MODULES)
def test_one_changed_byte_in_a_module_is_a_new_scorer(tmp_path, name):
    pkg = copy_package(tmp_path)
    data = (pkg / name).read_bytes()
    (pkg / name).write_bytes(data + b"\n")
    assert trajectory_code_sha(pkg) != trajectory_code_sha()
    (pkg / name).write_bytes(data)
    assert trajectory_code_sha(pkg) == trajectory_code_sha()


def test_a_tool_can_change_or_appear_without_moving_the_hash(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / TOOLS_DIR_NAME / "scores.py").write_text("# changed\n")
    (pkg / TOOLS_DIR_NAME / "new_tool.py").write_text("import surgical_core\n")
    assert trajectory_code_sha(pkg) == trajectory_code_sha()


@pytest.mark.parametrize("entry, is_dir", [("extra.py", False), ("notes.txt", False), ("data", True)])
def test_an_entry_the_list_does_not_name_is_refused(tmp_path, entry, is_dir):
    pkg = copy_package(tmp_path)
    (pkg / entry).mkdir() if is_dir else (pkg / entry).write_text("")
    with pytest.raises(ValueError, match=rf"neither hashed modules nor tools/: \['{entry}'\]"):
        trajectory_code_sha(pkg)


def test_a_listed_module_that_is_missing_is_refused(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "score.py").unlink()
    with pytest.raises(FileNotFoundError, match=r"listed modules are missing .*\['score.py'\]"):
        trajectory_code_sha(pkg)


@pytest.mark.parametrize("line, target", [
    ("import surgical_core.stereomis", "surgical_core.stereomis"),
    ("from evalkit.tools import pose_controls", "evalkit.tools"),
    ("from trajectory_eval.tools import scores", "trajectory_eval.tools"),
    ("from .tools import scores", "trajectory_eval.tools"),
    ("from .. import evalkit", r"\.\."),
    ("import scipy", "scipy"),
])
def test_an_import_of_what_is_not_hashed_is_refused(tmp_path, line, target):
    pkg = copy_package(tmp_path)
    (pkg / "score.py").write_text((pkg / "score.py").read_text() + f"\n{line}\n")
    with pytest.raises(ValueError, match=rf"score.py imports {target}, which would decide scores while not hashed"):
        trajectory_code_sha(pkg)


@pytest.mark.parametrize("line", ["import importlib", "x = __import__('surgical_core')"])
def test_an_import_by_a_string_is_refused(tmp_path, line):
    pkg = copy_package(tmp_path)
    (pkg / "score.py").write_text((pkg / "score.py").read_text() + f"\n{line}\n")
    with pytest.raises(ValueError, match=r"score.py imports through (importlib|__import__)"):
        trajectory_code_sha(pkg)


@pytest.mark.parametrize("line", ["import json", "from numpy import linalg", "from . import pose_metrics",
                                  "from trajectory_eval import pose_metrics"])
def test_the_standard_library_numpy_and_the_hashed_modules_may_be_imported(tmp_path, line):
    pkg = copy_package(tmp_path)
    (pkg / "score.py").write_text((pkg / "score.py").read_text() + f"\n{line}\n")
    assert re.fullmatch(r"[0-9a-f]{64}", trajectory_code_sha(pkg))
