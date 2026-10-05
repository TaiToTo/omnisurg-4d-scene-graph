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
missing, a file that is neither a module nor a document named here, and a
directory other than the tables and the tools. Listing them rather than
hashing whatever is there costs one line per new module, and in return a
scratch file cannot move the sha without a word, and a module that computes
a score cannot be added without being hashed. This module is on the list: a
change to what is hashed is a change to the evaluator.

A hashed module may import only the other hashed modules, the standard
library, and the three libraries whose versions every score records:
numpy, OpenCV and Pillow. Anything else it imported would decide scores
while being neither hashed nor recorded, so the import statements of every
hashed module are read and any other target raises, as do `importlib` and
`__import__`, whose targets cannot be read. The pilot evaluator's sha
covered its two scripts, and the GT loaders that decided every score lived
outside it until a check on its imports was added. Here the check is on
every module from the start, and it is stricter: a reader the evaluator
needs goes on the list and is hashed, or stays outside and hands the
evaluator arrays. The check reads statements, so it catches a dependency
added without thought, which is what it is for; it does not try to see
through a module written to evade it.

The hash reads the bytes as they are on disk: for each file, in a fixed
order, its package-relative path and then its content, each preceded by its
length as eight big-endian bytes. The path binds each content to its name,
and it is relative so that where the package is installed does not matter.
`.gitattributes` turns line-ending conversion off for every file, so a
checkout has the bytes that were measured.
"""
from __future__ import annotations

import ast
import hashlib
import struct
import sys
from pathlib import Path

from evalkit.classes import TABLE_DIR, UNTRACKED_NAMES, table_paths

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = __name__.rpartition(".")[0]

# The directories the package has besides its modules: the class tables,
# hashed through `table_paths`, and the tools, never hashed.
TOOLS_DIR_NAME = "tools"
_DIRECTORIES = frozenset({TABLE_DIR.name, TOOLS_DIR_NAME})

# Documents the package may carry beside its modules. They are read by
# people, not by the evaluator, so they are not hashed: a fixed sentence must
# not make old and new scores incomparable, and the freeze must not stop a
# sentence being fixed. Each may be absent, since a wheel ships no document.
DOCUMENTS = frozenset({"README.md"})

# Every module of the package. Hashed, and under the freeze rule once the
# evaluator is frozen.
HASHED_MODULES = (
    "__init__.py",
    "boundary.py",
    "classes.py",
    "classmap.py",
    "clip.py",
    "code_sha.py",
    "frame.py",
    "inst_bf.py",
    "keys.py",
    "objects.py",
    "pilot.py",
    "scored.py",
    "time_iou.py",
    "unlabelled.py",
    "vi.py",
)

# What a hashed module may import besides the standard library and the other
# hashed modules: the libraries whose versions every score records.
RECORDED_LIBRARIES = frozenset({"numpy", "cv2", "PIL"})

# Standard-library ways of importing by a string this check cannot read.
_DYNAMIC_IMPORTS = frozenset({"importlib", "__import__"})


def _check_covered(package_dir: Path) -> None:
    """Refuse a module the list does not name, a listed one that is missing, and anything else.

    One pass over the directory places every entry by name: a listed module,
    a document, one of the two directories, a name the filesystem writes on
    its own, or a fault. Going by name rather than by `is_file()` means a
    broken symlink, which is neither file nor directory, is placed too.

    Raises:
        FileNotFoundError: A listed module is not there.
        ValueError: The package holds a module the list does not name, a
            file that is neither a module nor a document, or a directory it
            should not have.
    """
    listed = set(HASHED_MODULES)
    present: set[str] = set()
    unlisted: list[str] = []
    other: list[str] = []
    for entry in package_dir.iterdir():
        name = entry.name
        if name in listed:
            present.add(name)
        elif name in DOCUMENTS or name in UNTRACKED_NAMES:
            continue
        elif name in _DIRECTORIES and entry.is_dir():
            continue
        elif name.endswith(".py"):
            unlisted.append(name)
        else:
            other.append(name)
    missing = listed - present
    if missing:
        raise FileNotFoundError(f"listed modules are missing from {package_dir}: {sorted(missing)}")
    if unlisted:
        raise ValueError(
            f"{package_dir} holds modules that are not hashed; add each to HASHED_MODULES, or "
            f"move it under {TOOLS_DIR_NAME}/ if it only reads scores: {sorted(unlisted)}"
        )
    if other:
        raise ValueError(
            f"{package_dir} holds entries that are neither listed modules, documents {sorted(DOCUMENTS)}, "
            f"nor the directories {sorted(_DIRECTORIES)}, which nothing hashes: {sorted(other)}"
        )


def _import_targets(module: Path) -> list[str]:
    """The absolute module paths the import statements of `module` reach, plus `__import__` where it is named.

    `from X import a, b` reaches `X`, except when `X` is this package or a
    relative `.`: then it reaches `X.a` and `X.b`, since those are modules.
    A relative import beyond this package (`from .. import x`) is reported
    as it is written and refused by the caller.
    """
    tree = ast.parse(module.read_bytes(), filename=module.name)
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                targets.append("." * node.level + (node.module or ""))
                continue
            base = node.module or ""
            if node.level == 1:
                base = f"{PACKAGE_NAME}.{base}" if base else PACKAGE_NAME
            if base == PACKAGE_NAME:
                targets += [f"{PACKAGE_NAME}.{alias.name}" for alias in node.names]
            else:
                targets.append(base)
        elif isinstance(node, ast.Name) and node.id == "__import__":
            targets.append("__import__")
    return targets


def _check_imports(package_dir: Path) -> None:
    """Refuse an import from a hashed module to anything that is neither hashed, standard, nor recorded.

    Raises:
        ValueError: A hashed module imports a module outside this package
            that is not in the standard library or `RECORDED_LIBRARIES`,
            imports something of this package that is not hashed (the
            tools), or imports by a string through `importlib` or
            `__import__`.
    """
    hashed = {f"{PACKAGE_NAME}.{Path(name).stem}" for name in HASHED_MODULES if name != "__init__.py"}
    hashed.add(PACKAGE_NAME)
    for name in sorted(HASHED_MODULES):
        for target in _import_targets(package_dir / name):
            top = target.split(".")[0]
            if top == PACKAGE_NAME:
                if target not in hashed:
                    raise ValueError(
                        f"{name} imports {target}, which is not hashed; a module a hashed one needs goes "
                        f"on HASHED_MODULES, not under {TOOLS_DIR_NAME}/"
                    )
            elif top in _DYNAMIC_IMPORTS:
                raise ValueError(f"{name} imports through {top}, whose target this check cannot read")
            elif top in sys.stdlib_module_names or top in RECORDED_LIBRARIES:
                continue
            else:
                raise ValueError(
                    f"{name} imports {target}, which would decide scores while neither hashed nor recorded; "
                    f"a hashed module may import the standard library, {sorted(RECORDED_LIBRARIES)}, "
                    f"and the other hashed modules"
                )


def hashed_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    """The files `eval_code_sha` covers, in the order they are hashed.

    The modules by name, then the class tables as `table_paths` lists them.
    Checks the package and its imports first, so a module on no list raises
    here rather than being left out; a copy of the package elsewhere is
    checked the same way.
    """
    package_dir = Path(package_dir).resolve()
    _check_covered(package_dir)
    _check_imports(package_dir)
    modules = [package_dir / name for name in sorted(HASHED_MODULES)]
    return modules + table_paths(package_dir / TABLE_DIR.name)


def _sha_of(paths: list[Path], relative_to: Path) -> str:
    # Each part is preceded by its length rather than followed by a
    # delimiter: a delimiter can occur inside a part, a length cannot be
    # mistaken for one, so the hashed bytes spell out one file list only.
    h = hashlib.sha256()
    for path in paths:
        for part in (path.relative_to(relative_to).as_posix().encode("utf-8"), path.read_bytes()):
            h.update(struct.pack(">Q", len(part)))
            h.update(part)
    return h.hexdigest()


def eval_code_sha(package_dir: Path = PACKAGE_DIR) -> str:
    """The sha256 hex digest of the evaluator: its modules and class tables."""
    package_dir = Path(package_dir).resolve()
    return _sha_of(hashed_files(package_dir), package_dir)


def eval_code_files(package_dir: Path = PACKAGE_DIR) -> list[str]:
    """The hashed files, package-relative, in the order hashed; written beside the sha so a reader can see what it covered."""
    package_dir = Path(package_dir).resolve()
    return [p.relative_to(package_dir).as_posix() for p in hashed_files(package_dir)]
