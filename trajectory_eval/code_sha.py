"""Hash the modules that turn a camera trajectory into a score: `trajectory_code_sha`.

Every StereoMIS score records the hash, and two scores are compared only when
it matches. It covers the modules of this package, listed by name; `tools/`
is not hashed. The package may hold nothing else, and a hashed module may
import only the standard library, numpy and the other hashed modules. The
bytes are hashed as `eval_code_sha` hashes the evaluator: for each file, its
package-relative path and then its content, each preceded by its length as
eight big-endian bytes.
"""

import ast
import hashlib
import struct
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = __name__.rpartition(".")[0]

# The directory of what reads scores or makes a score's inputs. It is not hashed, so that fixing a tool does not
# make old and new scores incomparable.
TOOLS_DIR_NAME = "tools"

# Every module of the package, hashed.
HASHED_MODULES = ("__init__.py", "code_sha.py", "pose_metrics.py", "score.py")

# What a hashed module may import besides the standard library and the other hashed modules. A reader of the
# dataset or a control's trajectory imported here would decide scores while not hashed.
LIBRARIES = frozenset({"numpy"})

# The names a filesystem or the interpreter writes into a directory on its own.
UNTRACKED_NAMES = frozenset({"__pycache__", ".DS_Store"})


def _check_covered(package_dir: Path) -> None:
    """Refuse a module the list does not name, a listed one that is missing, and any other entry but `tools/`.

    Raises:
        FileNotFoundError: a listed module is not there.
        ValueError: the package holds a module the list does not name, or another file or directory.
    """
    names = {p.name for p in package_dir.iterdir()} - UNTRACKED_NAMES
    missing = set(HASHED_MODULES) - names
    if missing:
        raise FileNotFoundError(f"listed modules are missing from {package_dir}: {sorted(missing)}")
    others = sorted(n for n in names - set(HASHED_MODULES)
                    if not (n == TOOLS_DIR_NAME and (package_dir / n).is_dir()))
    if others:
        raise ValueError(f"{package_dir} holds entries that are neither hashed modules nor {TOOLS_DIR_NAME}/: "
                         f"{others}; a module that scores goes on HASHED_MODULES, one that reads scores under "
                         f"{TOOLS_DIR_NAME}/")


def _import_targets(module: Path) -> list[str]:
    """Return the module paths the import statements of `module` reach, and `__import__` where it is named.

    `from X import a` reaches `X`, except when `X` is this package or `.`: then it reaches `X.a`, a module. A
    relative import beyond this package is returned as written, and refused by the caller.
    """
    targets = []
    for node in ast.walk(ast.parse(module.read_bytes(), filename=module.name)):
        if isinstance(node, ast.Import):
            targets += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                targets.append("." * node.level + (node.module or ""))
                continue
            base = node.module or ""
            if node.level == 1:
                base = f"{PACKAGE_NAME}.{base}" if base else PACKAGE_NAME
            targets += [f"{base}.{a.name}" for a in node.names] if base == PACKAGE_NAME else [base]
        elif isinstance(node, ast.Name) and node.id == "__import__":
            targets.append("__import__")
    return targets


def _check_imports(package_dir: Path) -> None:
    """Refuse an import from a hashed module to anything but the standard library, `LIBRARIES` and a hashed module.

    Raises:
        ValueError: a hashed module imports something else, or imports through `importlib` or `__import__`,
            whose target cannot be read.
    """
    hashed = {PACKAGE_NAME} | {f"{PACKAGE_NAME}.{Path(n).stem}" for n in HASHED_MODULES if n != "__init__.py"}
    for name in HASHED_MODULES:
        for target in _import_targets(package_dir / name):
            top = target.split(".")[0]
            if top in ("importlib", "__import__"):
                raise ValueError(f"{name} imports through {top}, whose target cannot be read")
            if top == PACKAGE_NAME and target in hashed:
                continue
            if top != PACKAGE_NAME and (top in sys.stdlib_module_names or top in LIBRARIES):
                continue
            raise ValueError(f"{name} imports {target}, which would decide scores while not hashed; a hashed module "
                             f"may import the standard library, {sorted(LIBRARIES)} and the other hashed modules")


def hashed_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    """Return the files `trajectory_code_sha` covers, in the order hashed, once the package and imports are checked."""
    package_dir = Path(package_dir).resolve()
    _check_covered(package_dir)
    _check_imports(package_dir)
    return [package_dir / name for name in sorted(HASHED_MODULES)]


def _sha_of(paths: list[Path], relative_to: Path) -> str:
    # A length before each part, not a delimiter after it: a delimiter can occur inside a part.
    h = hashlib.sha256()
    for path in paths:
        for part in (path.relative_to(relative_to).as_posix().encode("utf-8"), path.read_bytes()):
            h.update(struct.pack(">Q", len(part)))
            h.update(part)
    return h.hexdigest()


def trajectory_code_sha(package_dir: Path = PACKAGE_DIR) -> str:
    """Return the sha256 hex digest of the hashed modules."""
    package_dir = Path(package_dir).resolve()
    return _sha_of(hashed_files(package_dir), package_dir)


def trajectory_code_files(package_dir: Path = PACKAGE_DIR) -> list[str]:
    """Return the hashed files, package-relative, in the order hashed."""
    package_dir = Path(package_dir).resolve()
    return [p.relative_to(package_dir).as_posix() for p in hashed_files(package_dir)]
