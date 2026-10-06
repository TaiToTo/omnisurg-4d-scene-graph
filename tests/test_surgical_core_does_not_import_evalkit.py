"""`surgical_core` is the layer the pipeline and the toolkit share, so no module in it may import `evalkit`.

Every `.py` file under `surgical_core/` is read statically, so an import
inside a function is found too, which a check on loaded modules misses.
The one exemption is `viewer/gt_tables.py`, the adapter that reads the
evaluator's class tables. `tests/test_viewer_labels.py` checks the other
half: that importing the viewer's base loads no `evalkit` module.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The one module allowed to read the evaluator's class tables.
ALLOWED = {Path("surgical_core/viewer/gt_tables.py")}


def evalkit_imports(source: str, filename: str) -> list[tuple[int, str]]:
    """Every import statement in `source` that reaches `evalkit`, as `(line, target)`.

    `ast.walk` visits function bodies, so an import written inside a
    function is found before it ever runs. A relative import stays inside
    `surgical_core` and cannot reach `evalkit`.
    """
    hits = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, alias.name) for alias in node.names
                     if alias.name.split(".")[0] == "evalkit"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".")[0] == "evalkit":
                hits.append((node.lineno, node.module))
    return hits


def test_surgical_core_does_not_import_evalkit():
    files = sorted((REPO / "surgical_core").rglob("*.py"))
    assert files, "surgical_core holds no Python files; this check scans nothing"
    hits = []
    for f in files:
        if f.relative_to(REPO) in ALLOWED:
            continue
        hits += [f"{f.relative_to(REPO)}:{line} imports {target}"
                 for line, target in evalkit_imports(f.read_text(), str(f))]
    assert not hits, "surgical_core must not import evalkit:\n" + "\n".join(hits)


def test_a_stale_exemption_is_refused():
    for path in ALLOWED:
        assert (REPO / path).is_file(), f"exempted file {path} no longer exists"


def test_the_scan_finds_an_import_inside_a_function():
    planted = (
        "def f():\n"
        "    import evalkit.classes\n"
        "    from evalkit.classes import VIEWS\n"
    )
    assert evalkit_imports(planted, "planted.py") == [
        (2, "evalkit.classes"), (3, "evalkit.classes")]
