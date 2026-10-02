"""`eval_code_sha` covers every scoring module and every table, and moves when one of them does."""
import re
import shutil
from pathlib import Path

import pytest

import evalkit
from evalkit.code_sha import (
    PACKAGE_DIR, READER_MODULES, SCORING_MODULES, _check_covered, eval_code_files, eval_code_sha,
    hashed_files)


def copy_package(tmp_path: Path) -> Path:
    """A copy of the installed package, for planting faults in."""
    dst = tmp_path / "evalkit"
    shutil.copytree(PACKAGE_DIR, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_the_sha_is_a_full_sha256_and_the_same_on_two_calls():
    sha = eval_code_sha()
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert eval_code_sha() == sha


def test_every_module_of_the_package_is_on_exactly_one_list():
    present = {p.name for p in Path(evalkit.__file__).parent.glob("*.py")}
    assert present == set(SCORING_MODULES) | set(READER_MODULES)
    assert not set(SCORING_MODULES) & set(READER_MODULES)
    assert "code_sha.py" in SCORING_MODULES


def test_the_hashed_files_are_the_scoring_modules_then_the_tables_in_a_fixed_order():
    files = eval_code_files()
    assert files == sorted(SCORING_MODULES) + ["class_tables/atlas120k.json", "class_tables/cholecseg8k.json"]
    assert [p.name for p in hashed_files()] == [Path(f).name for f in files]
    assert not any(name in files for name in READER_MODULES)


def test_a_copy_of_the_package_has_the_same_sha_wherever_it_lies(tmp_path):
    # The paths hashed are package-relative, so the install location does
    # not move the sha.
    assert eval_code_sha(copy_package(tmp_path)) == eval_code_sha()


@pytest.mark.parametrize("name", ["class_tables/cholecseg8k.json", "objects.py"])
def test_one_changed_byte_in_a_table_or_a_module_is_a_new_evaluator(tmp_path, name):
    pkg = copy_package(tmp_path)
    path = pkg / name
    data = path.read_bytes()
    path.write_bytes(data + b"\n")
    assert eval_code_sha(pkg) != eval_code_sha()
    path.write_bytes(data)
    assert eval_code_sha(pkg) == eval_code_sha()


def test_a_renamed_table_is_a_new_evaluator_even_with_the_same_bytes(tmp_path):
    pkg = copy_package(tmp_path)
    (pkg / "class_tables" / "cholecseg8k.json").rename(pkg / "class_tables" / "cholec.json")
    assert eval_code_sha(pkg) != eval_code_sha()


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


def test_a_module_cannot_be_both_hashed_and_a_reader(monkeypatch):
    import evalkit.code_sha as m
    monkeypatch.setattr(m, "READER_MODULES", ("__init__.py", "vi.py"))
    with pytest.raises(ValueError, match="both"):
        _check_covered()
