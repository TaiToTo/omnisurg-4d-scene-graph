"""Hold the mark rule of `AGENTS.md` over every tracked Python file.

A star is decided by `paired_stats.mark_of` and nothing else: never by a
p-value, never by a truth value that has dropped the direction, and never
printed by a module that does not borrow `mark_of`. Each scan is shown to
catch a planted fault, and to let the allowed forms through. The scans need
no library beyond the standard one, so they run without the extras too.
"""
import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The scans read syntax trees, not text, so a statement split over lines,
# an `if`, an f-string or a name standing in for 0.05 are one statement
# each. What they do not see is a decision spread over two statements
# (`sig = p < 0.05`, then `mark = "★" if sig else ""`); they are a
# tripwire, not a proof.


def _tracked_python_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "*.py"], check=True, capture_output=True).stdout
    for rel in out.decode().split("\0"):
        if rel and rel != "evalkit/tools/paired_stats.py":
            yield rel, (REPO / rel).read_text(encoding="utf-8")


def _package_of(rel: str) -> str:
    """The package a tracked file belongs to, for its relative imports."""
    return ".".join(Path(rel).parts[:-1])


def _alpha_names(tree) -> set[str]:
    """Names bound to 0.05 anywhere in the module, in a tuple assignment too, so that `ALPHA = 0.05` hides nothing."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = []
            for t in node.targets:
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    pairs += zip(t.elts, node.value.elts)
                else:
                    pairs.append((t, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        else:
            continue
        names |= {t.id for t, v in pairs
                  if isinstance(t, ast.Name) and isinstance(v, ast.Constant) and v.value == 0.05}
    return names


def _compares_with_alpha(node, alpha: set[str]) -> bool:
    """Whether a comparison in `node` has 0.05, or a name for it, on either side, whichever way it reads."""
    def is_alpha(x):
        return (isinstance(x, ast.Constant) and x.value == 0.05) or (isinstance(x, ast.Name) and x.id in alpha)

    return any(any(is_alpha(part) for part in (cmp_.left, *cmp_.comparators))
               for cmp_ in ast.walk(node) if isinstance(cmp_, ast.Compare))


MARK_NAMES = {"star", "mark", "sig", "beats", "verdict", "sep", "clear", "p_clear"}


def _names_in(node) -> set[str]:
    """The names a statement touches: its variables, and the string keys it subscripts (`row["star"]`)."""
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    names |= {n.slice.value for n in ast.walk(node)
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    return names


def marks_decided_by_p(src: str) -> list[str]:
    """`<line>  <statement>` for every statement that decides a mark from a p-value.

    Printing p beside a mark is allowed; comparing p with 0.05 in a
    statement that also touches a mark, by name or by the character, is
    not, whichever way the comparison reads.
    """
    tree = ast.parse(src)
    alpha = _alpha_names(tree)
    bad = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr)):
            test = node
        elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
            test = node.test
        else:
            continue
        if not _compares_with_alpha(test, alpha):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "★" in seg or "✗" in seg or (_names_in(node) & MARK_NAMES):
            bad.setdefault(node.lineno, f"{node.lineno}  {seg.splitlines()[0].strip()}")
    return [bad[k] for k in sorted(bad)]


# The names a mark from `mark_of` is kept under: a variable, or a row's key.
MARK_HOLDERS = {"mark", "verdict"}


def _reads_a_mark(node) -> bool:
    """Whether `node` calls `mark_of`, bare or as an attribute (`PS.mark_of`), or reads a name in `MARK_HOLDERS`."""
    calls = any(isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == "mark_of")
                                            or (isinstance(n.func, ast.Attribute) and n.func.attr == "mark_of"))
                for n in ast.walk(node))
    return calls or bool(_names_in(node) & MARK_HOLDERS)


def marks_bound_to_a_truth_value(src: str) -> list[str]:
    """`<line>  <statement>` for every assignment that binds `bool()` of a mark to a mark's name.

    `star = bool(mark_of(ci))` is True for a cross too, and so is
    `star = bool(row["mark"])`; four readers once drew a loss as a star
    that way. Counting with bool() is fine; binding it to a mark's name is not.
    """
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        for t in node.targets:
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                targets.add(t.slice.value)
        if not (targets & MARK_NAMES):
            continue
        for call in (n for n in ast.walk(node.value) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == "bool" and _reads_a_mark(call):
                seg = ast.get_source_segment(src, node) or ""
                bad.append(f"{node.lineno}  {seg.splitlines()[0].strip()}")
    return bad


# Every glyph that reads as "better" or "worse" in a table: the two of
# `mark_of`, and the circle and cross that read as a mark on a sign alone.
MARKS = ("★", "✗", "○", "×")


def _emits_a_mark(tree) -> bool:
    docstrings = set()
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and any(m in n.value for m in MARKS)
               and id(n) not in docstrings for n in ast.walk(tree))


def _dotted(node) -> str | None:
    """`a.b.c` for an attribute chain that starts at a name; None for any other expression."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _module_of(node: ast.ImportFrom, package: str) -> str:
    """The module a `from` import names, a relative one resolved against `package`."""
    if not node.level:
        return node.module
    base = package.split(".")
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


BORROWED = {"evalkit.tools.paired_stats.mark_of", "evalkit.tools.paired_stats.is_star"}


def _borrows_mark_of(tree, package: str) -> bool:
    """Whether the module calls `paired_stats.mark_of` or `is_star`, through whatever import form binds it.

    Every name an import binds is resolved to the dotted thing it stands
    for, and every name or attribute chain the module uses is read through
    that table; so `from evalkit.tools import paired_stats as PS` and
    `PS.mark_of(...)` count, as `import evalkit.tools.paired_stats` and the
    full chain do.
    """
    bound = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                bound[a.asname] = a.name
                if a.asname is None:
                    bound[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                bound[a.asname or a.name] = f"{_module_of(n, package)}.{a.name}"
    bound.pop(None, None)
    for n in ast.walk(tree):
        dotted = _dotted(n) if isinstance(n, (ast.Attribute, ast.Name)) else None
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        if bound.get(head, head) + (f".{rest}" if rest else "") in BORROWED:
            return True
    return False


def prints_a_mark_without_mark_of(src: str, package: str = "") -> bool:
    """Whether a module writes a star or a cross outside a docstring without borrowing `mark_of`."""
    tree = ast.parse(src)
    return _emits_a_mark(tree) and not _borrows_mark_of(tree, package)


def test_no_statement_decides_a_mark_from_a_p_value():
    bad = [f"{rel}:{hit}" for rel, src in _tracked_python_files() for hit in marks_decided_by_p(src)]
    assert not bad, "a mark is decided by p; the rule is paired_stats.MARK_RULE:\n  " + "\n  ".join(bad)


def test_no_mark_is_bound_to_a_truth_value_that_dropped_the_direction():
    bad = [f"{rel}:{hit}" for rel, src in _tracked_python_files() for hit in marks_bound_to_a_truth_value(src)]
    assert not bad, "bool(mark_of(...)) bound to a mark's name drops the direction:\n  " + "\n  ".join(bad)


def test_every_file_that_prints_a_mark_borrows_mark_of():
    # The tests are left out: they spell the marks to check them.
    bad = [rel for rel, src in _tracked_python_files()
           if not rel.startswith("tests/") and prints_a_mark_without_mark_of(src, _package_of(rel))]
    assert not bad, "a mark is printed without borrowing paired_stats.mark_of:\n  " + "\n  ".join(bad)


# The planted faults. Each scan is shown to catch the forms it is for, the
# negated and the indirect ones included, and to let the allowed ones pass.

DECIDED_BY_P = [
    'mark = "★" if p < 0.05 else ""',
    'mark = "" if p > 0.05 else "★"',
    'mark = "★" if 0.05 > p else ""',
    'mark = "" if 0.05 < p else "★"',
    'if p <= 0.05:\n    mark = "★"',
    'while p < 0.05:\n    star = True',
    'row["star"] = p < 0.05',
    'print(f"{k} {\'★\' if p < 0.05 else \'\'}")',
    'ALPHA, BETA = 0.05, 0.1\nmark = "★" if p < ALPHA else ""',
    'ALPHA: float = 0.05\nmark = "★" if p < ALPHA else ""',
    'def f(p):\n    alpha = 0.05\n    return "★" if p < alpha else ""',
    'mark = ("★"\n        if p\n        < 0.05 else "")',
]

P_BESIDE_A_MARK = [
    'print(f"{mark} p={p:.4f}")',
    'mark = mark_of(ci)',
    'row["mark"] = mark_of(ci)',
    'if p < 0.05:\n    print("a small p, printed, not decisive")',
    'alpha = 0.05\nprint(f"alpha={alpha}")',
    'sig = wilcoxon_p_video',
    'row["wilcoxon_p"] = round(p, 5)',
]


@pytest.mark.parametrize("src", DECIDED_BY_P)
def test_the_p_scan_sees_a_planted_decision(src):
    assert marks_decided_by_p(src), src


@pytest.mark.parametrize("src", P_BESIDE_A_MARK)
def test_the_p_scan_lets_p_be_printed_beside_a_mark(src):
    assert not marks_decided_by_p(src), src


BOUND_TO_A_TRUTH_VALUE = [
    'star = bool(mark_of(ci))',
    'row["star"] = bool(PS.mark_of(ci))',
    'sig = bool(paired_stats.mark_of(r["ci95_video"], r["sign"]))',
    'star = bool(row["mark"])',
    'star = bool(mark)',
]

KEEPS_THE_DIRECTION = [
    'n_marked = sum(bool(mark_of(c)) for c in cis)',
    'star = mark_of(ci) == "★"',
    'star = is_star(ci)',
    'mark = mark_of(ci)',
    'row["mark"] = mark_of(ci)',
]


@pytest.mark.parametrize("src", BOUND_TO_A_TRUTH_VALUE)
def test_the_bool_scan_sees_a_planted_binding(src):
    assert marks_bound_to_a_truth_value(src), src


@pytest.mark.parametrize("src", KEEPS_THE_DIRECTION)
def test_the_bool_scan_lets_counting_and_comparing_pass(src):
    assert not marks_bound_to_a_truth_value(src), src


PRINTS_WITHOUT_BORROWING = [
    'print("★" if lo > 0 else "")',
    'mark = "○" if d * sign > 0 else "×"',
    'from evalkit.tools.paired_stats import boot_ci\nmark = "★" if boot_ci(d)[0] > 0 else ""',
    'import evalkit.tools.paired_stats\nmark = "★"',
    'from evalkit.tools import paired_stats as PS\nmark = "★"\nvideo = PS.video_of(c)',
]

LEGEND = 'print("  ★ better   ✗ worse")\n'

BORROWS = [
    ('from evalkit.tools.paired_stats import mark_of\n' + LEGEND + 'print(mark_of(ci, sign))', ""),
    ('from evalkit.tools.paired_stats import is_star\n' + LEGEND + 'print(is_star(ci))', ""),
    ('from evalkit.tools import paired_stats as PS\n' + LEGEND + 'print(PS.mark_of(ci, sign))', ""),
    ('from evalkit.tools import paired_stats\n' + LEGEND + 'print(paired_stats.mark_of(ci))', ""),
    ('import evalkit.tools.paired_stats as P\n' + LEGEND + 'print(P.is_star(ci))', ""),
    ('import evalkit.tools.paired_stats\n' + LEGEND + 'print(evalkit.tools.paired_stats.mark_of(ci))', ""),
    ('from evalkit import tools\n' + LEGEND + 'print(tools.paired_stats.mark_of(ci))', ""),
    ('from .paired_stats import mark_of\n' + LEGEND + 'print(mark_of(ci))', "evalkit.tools"),
    ('from . import paired_stats\n' + LEGEND + 'print(paired_stats.mark_of(ci))', "evalkit.tools"),
    ('from ..tools import paired_stats\n' + LEGEND + 'print(paired_stats.mark_of(ci))', "evalkit.other"),
]


@pytest.mark.parametrize("src", PRINTS_WITHOUT_BORROWING)
def test_the_mark_scan_sees_a_planted_mark_without_mark_of(src):
    assert prints_a_mark_without_mark_of(src), src


@pytest.mark.parametrize("src, package", BORROWS)
def test_the_mark_scan_knows_every_import_form(src, package):
    assert not prints_a_mark_without_mark_of(src, package), src


def test_the_mark_scan_ignores_a_mark_in_a_docstring():
    assert not prints_a_mark_without_mark_of('"""A ★ is a star and a ✗ a cross."""\nx = 1')
    assert not prints_a_mark_without_mark_of('def f():\n    """★"""\n    return 1')
