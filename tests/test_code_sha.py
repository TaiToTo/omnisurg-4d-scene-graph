"""`eval_code_sha` covers every module and every table, moves when one of them does, and refuses what it cannot place."""
import re
import shutil
from pathlib import Path

import pytest

import evalkit
from evalkit.code_sha import (
    HASHED_MODULES, PACKAGE_DIR, TOOLS_DIR_NAME, _check_covered, eval_code_files, eval_code_sha, hashed_files)


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


@pytest.mark.parametrize("name", ["class_tables/cholecseg8k.json", "objects.py"])
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
    (pkg / TOOLS_DIR_NAME / "paired_stats.py").write_text("VERDICT_RULE = 'ci'\n")
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_module_on_no_list_is_refused_rather_than_left_out(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "stray.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="stray.py"):
        eval_code_sha(pkg)
    with pytest.raises(ValueError, match="stray.py"):
        _check_covered(pkg)


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
