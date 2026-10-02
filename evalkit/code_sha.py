"""`eval_code_sha`: the hash that says which evaluator made a score.

Every score carries it, and two scores are compared only when it matches
(`docs/evaluation.md`, "Recorded with every score"). It is a sha256 over the
files that decide a score: the modules that compute one, and the class
tables, since a changed type is a new evaluator. The tools that only read
scores are not hashed, so that fixing one of them does not make old and
new scores incomparable. The files under the freeze rule of `AGENTS.md`
are exactly the hashed ones.

Which module is which is written here, in two lists, and the package may
hold nothing else: a module on neither list raises, as does a listed one
that is missing. The pilot evaluator's sha covered its two scripts, and the
GT loaders that decided every score lived outside it until a check was
added; here the check is in from the start, so that a module that computes
a score cannot be added without being hashed.

The hash reads the bytes as they are on disk, each file as its
package-relative path, a NUL, its content and a NUL, in a fixed order. The
path is included so that a renamed table or module is a new evaluator, and
it is relative so that where the package is installed does not matter.
`.gitattributes` turns line-ending conversion off for every file, so a
checkout has the bytes that were measured.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from evalkit.classes import TABLE_DIR, table_paths

PACKAGE_DIR = Path(__file__).resolve().parent

# The modules that compute scores, or decide what is scored. Hashed, and
# under the freeze rule once the evaluator is frozen. This module is among
# them: a change to what is hashed is a change to the evaluator.
SCORING_MODULES = (
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

# The modules that compute no score: the package's own docstring, and the
# tools that read scores once they are ported. Not hashed, so that a fix to
# one of them leaves every sha where it is.
READER_MODULES = (
    "__init__.py",
)


def _modules_present(package_dir: Path) -> set[str]:
    return {p.name for p in package_dir.iterdir() if p.suffix == ".py"}


def _check_covered(package_dir: Path = PACKAGE_DIR) -> None:
    """Refuse a module that is on neither list, on both, or listed but missing.

    Raises:
        ValueError: The package holds a module the lists do not name, or a
            name is on both lists.
        FileNotFoundError: A listed module is not there.
    """
    scoring, readers = set(SCORING_MODULES), set(READER_MODULES)
    both = scoring & readers
    if both:
        raise ValueError(f"listed as both scoring and reading, which cannot be: {sorted(both)}")
    present = _modules_present(package_dir)
    missing = (scoring | readers) - present
    if missing:
        raise FileNotFoundError(f"listed modules are missing from {package_dir}: {sorted(missing)}")
    unlisted = present - scoring - readers
    if unlisted:
        raise ValueError(
            f"{package_dir} holds modules that are neither hashed nor declared to read scores only; "
            f"add each to SCORING_MODULES or READER_MODULES: {sorted(unlisted)}"
        )


def hashed_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    """The files `eval_code_sha` covers, in the order they are hashed.

    The scoring modules by name, then the class tables as `table_paths`
    lists them. Checks the package first, so a module on no list raises
    here rather than being left out.
    """
    _check_covered(package_dir)
    modules = [package_dir / name for name in sorted(SCORING_MODULES)]
    if package_dir == PACKAGE_DIR:
        return modules + table_paths()
    # A package copied elsewhere, as a test plants it: its own tables.
    return modules + sorted((package_dir / TABLE_DIR.name).glob("*.json"))


def _sha_of(paths: list[Path], relative_to: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def eval_code_sha(package_dir: Path = PACKAGE_DIR) -> str:
    """The sha256 hex digest of the evaluator: its scoring modules and class tables."""
    return _sha_of(hashed_files(package_dir), package_dir)


def eval_code_files(package_dir: Path = PACKAGE_DIR) -> list[str]:
    """The hashed files, package-relative, in the order hashed; written beside the sha so a reader can see what it covered."""
    return [p.relative_to(package_dir).as_posix() for p in hashed_files(package_dir)]
