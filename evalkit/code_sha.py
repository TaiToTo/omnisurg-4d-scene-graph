"""`eval_code_sha`: the hash that says which evaluator made a score.

Every score carries it, and two scores are compared only when it matches
(`docs/evaluation.md`, "Recorded with every score"). It is a sha256 over the
files that decide a score: every module of this package, and the class
tables, since a changed type is a new evaluator. The tools that only read
scores live under `tools/` and are not hashed, so that fixing one of them,
or adding one, does not make old and new scores incomparable. The files
under the freeze rule of `AGENTS.md` are exactly the hashed ones.

The modules are listed here by name, and the package may hold nothing else:
a module the list does not name raises, as does a listed one that is
missing, a file that is not a module, or a directory other than the tables
and the tools. Listing them rather than hashing whatever is there costs one
line per new module, and in return a scratch file cannot move the sha
without a word, and a module that computes a score cannot be added without
being hashed. The pilot evaluator's sha covered its two scripts, and the GT
loaders that decided every score lived outside it until a check was added;
here the check is in from the start. This module is on the list: a change
to what is hashed is a change to the evaluator.

The hash reads the bytes as they are on disk, each file as its
package-relative path, a NUL, its content and a NUL, in a fixed order. The
path binds each content to its name, and it is relative so that where the
package is installed does not matter. `.gitattributes` turns line-ending
conversion off for every file, so a checkout has the bytes that were
measured.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from evalkit.classes import TABLE_DIR, table_paths

PACKAGE_DIR = Path(__file__).resolve().parent

# The directories the package has besides its modules: the class tables,
# hashed through `table_paths`, and the tools, never hashed.
TOOLS_DIR_NAME = "tools"
_DIRECTORIES = frozenset({TABLE_DIR.name, TOOLS_DIR_NAME, "__pycache__"})

# Every module of the package. Hashed, and under the freeze rule once the
# evaluator is frozen.
HASHED_MODULES = (
    "__init__.py",
    "boundary.py",
    "classes.py",
    "classmap.py",
    "code_sha.py",
    "objects.py",
    "scored.py",
    "time_iou.py",
    "unlabelled.py",
    "vi.py",
)


def _check_covered(package_dir: Path) -> None:
    """Refuse a module the list does not name, a listed one that is missing, and anything else.

    Raises:
        FileNotFoundError: A listed module is not there.
        ValueError: The package holds a module the list does not name, a
            file that is not a module, or a directory it should not have.
    """
    files = {p.name for p in package_dir.iterdir() if p.is_file()}
    present = {name for name in files if name.endswith(".py")}
    listed = set(HASHED_MODULES)
    missing = listed - present
    if missing:
        raise FileNotFoundError(f"listed modules are missing from {package_dir}: {sorted(missing)}")
    unlisted = present - listed
    if unlisted:
        raise ValueError(
            f"{package_dir} holds modules that are not hashed; add each to HASHED_MODULES, or "
            f"move it under {TOOLS_DIR_NAME}/ if it only reads scores: {sorted(unlisted)}"
        )
    other = files - present
    if other:
        raise ValueError(f"{package_dir} holds files that are not modules, which nothing hashes: {sorted(other)}")
    directories = {p.name for p in package_dir.iterdir() if p.is_dir()} - _DIRECTORIES
    if directories:
        raise ValueError(
            f"{package_dir} holds directories that are neither the class tables nor {TOOLS_DIR_NAME}/, "
            f"which nothing hashes: {sorted(directories)}"
        )


def hashed_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    """The files `eval_code_sha` covers, in the order they are hashed.

    The modules by name, then the class tables as `table_paths` lists them.
    Checks the package first, so a module on no list raises here rather than
    being left out; a copy of the package elsewhere is checked the same way.
    """
    package_dir = Path(package_dir).resolve()
    _check_covered(package_dir)
    modules = [package_dir / name for name in sorted(HASHED_MODULES)]
    return modules + table_paths(package_dir / TABLE_DIR.name)


def _sha_of(paths: list[Path], relative_to: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def eval_code_sha(package_dir: Path = PACKAGE_DIR) -> str:
    """The sha256 hex digest of the evaluator: its modules and class tables."""
    package_dir = Path(package_dir).resolve()
    return _sha_of(hashed_files(package_dir), package_dir)


def eval_code_files(package_dir: Path = PACKAGE_DIR) -> list[str]:
    """The hashed files, package-relative, in the order hashed; written beside the sha so a reader can see what it covered."""
    package_dir = Path(package_dir).resolve()
    return [p.relative_to(package_dir).as_posix() for p in hashed_files(package_dir)]
