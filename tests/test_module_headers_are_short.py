"""A module's header says what the module does, in at most `HEADER_LINES` lines, then how to run it.

The header is the module docstring, the summary line and blank lines
counted, less a `Usage:` section that ends it. A reader decides from it
whether this is the file they want; history, definitions and the defence
of choices go where `docs/review.md` sends them. The modules that were
over the cap when the rule was made are listed in `STILL_LONG`; one leaves
the list when it is shortened, cannot stay in it once short, and cannot be
added after the fact: the list is read back as the commit that introduced
this file wrote it.
"""

import ast
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The most lines a header may have before its Usage.
HEADER_LINES = 12

# The modules longer than the cap when the rule was made, shortened by the
# workstream `docs/short-headers`. A module is removed here when it is
# shortened; a module that is short and still listed fails the ratchet.
STILL_LONG = frozenset({
    "evalkit/boundary.py",
    "evalkit/classmap.py",
    "evalkit/code_sha.py",
    "evalkit/objects.py",
    "evalkit/pilot.py",
    "evalkit/time_iou.py",
    "evalkit/tools/paired_stats.py",
    "evalkit/tools/scores.py",
    "surgical_core/atlas120k/clip_rects.py",
    "surgical_core/atlas120k/frame_ratio.py",
    "surgical_core/geometry/__init__.py",
    "tests/scenes.py",
    "tests/test_no_personal_email.py",
    "tests/test_no_todo_in_code.py",
})


def tracked_modules(repo: Path) -> list[str]:
    """Every `.py` file git tracks under `repo`, as repository-relative paths."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "*.py"], check=True, capture_output=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p and not os.path.islink(repo / p)]


def header_length(source: str) -> int:
    """The lines of `source`'s module docstring, less a `Usage:` section that ends it; 0 without a docstring.

    A `Usage:` section is one that starts at the left margin and runs to the
    end of the docstring with every later line indented or blank. Text after
    it at the margin is not usage, so the whole docstring is counted then,
    and so is a `Usage:` in the middle of the prose. Blank lines at the end
    of what is counted are left out.
    """
    doc = ast.get_docstring(ast.parse(source), clean=False) or ""
    lines = doc.strip("\n").splitlines()
    usage = next((i for i, line in enumerate(lines) if line.startswith("Usage:")), None)
    head = list(lines)
    if usage is not None and all(not line.strip() or line[0] in " \t" for line in lines[usage + 1:]):
        head = lines[:usage]
    while head and not head[-1].strip():
        head.pop()
    return len(head)


def first_listed(repo: Path) -> frozenset[str]:
    """`STILL_LONG` as the commit that added this file wrote it.

    Raises:
        RuntimeError: The history does not reach that commit (a shallow
            clone), so what the list may hold cannot be known.
    """
    rel = Path(__file__).resolve().relative_to(repo).as_posix()
    log = subprocess.run(["git", "-C", str(repo), "log", "--diff-filter=A", "--format=%H", "--", rel],
                         check=True, capture_output=True).stdout.decode("utf-8").split()
    if not log:
        raise RuntimeError(f"the commit that added {rel} is not in the history; clone with the full history")
    source = subprocess.run(["git", "-C", str(repo), "show", f"{log[-1]}:{rel}"],
                            check=True, capture_output=True).stdout.decode("utf-8")
    return listed_in(source)


def listed_in(source: str) -> frozenset[str]:
    """The names `STILL_LONG` holds in `source`, a version of this file."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STILL_LONG" for t in node.targets):
            return frozenset(ast.literal_eval(node.value.args[0]))
    raise RuntimeError("no STILL_LONG in that version of the file")


def test_no_header_is_longer_than_the_cap():
    over = {}
    for rel in tracked_modules(REPO):
        if rel in STILL_LONG:
            continue
        n = header_length((REPO / rel).read_text(encoding="utf-8"))
        if n > HEADER_LINES:
            over[rel] = n
    assert not over, (
        f"a module header is longer than {HEADER_LINES} lines; say what the module does and send the "
        "rest where docs/review.md says:\n  " + "\n  ".join(f"{f}: {n} lines" for f, n in sorted(over.items())))


def test_a_listed_module_is_tracked_and_still_long():
    """The ratchet: a module leaves `STILL_LONG` when shortened, and a stale entry is refused."""
    tracked = set(tracked_modules(REPO))
    assert "tests/test_module_headers_are_short.py" in tracked, "the scan sees no module; this file is one"
    missing = STILL_LONG - tracked
    assert not missing, f"listed but not tracked: {sorted(missing)}"
    short = {rel: header_length((REPO / rel).read_text(encoding="utf-8")) for rel in STILL_LONG}
    short = {rel: n for rel, n in short.items() if n <= HEADER_LINES}
    assert not short, f"shortened, so remove from STILL_LONG: {short}"


def test_the_measure_counts_to_the_usage_line_and_a_planted_long_header_is_over():
    """The check earns its place by failing: a header of thirteen lines is over, one of twelve plus Usage is not."""
    thirteen = '"""' + "\n".join(f"line {i}" for i in range(13)) + '\n"""\nx = 1\n'
    assert header_length(thirteen) == 13 > HEADER_LINES
    twelve_and_usage = '"""' + "\n".join(f"line {i}" for i in range(12)) + '\n\nUsage:\n    run it\n    twice\n"""\n'
    assert header_length(twelve_and_usage) == 12 <= HEADER_LINES
    assert header_length("x = 1\n") == 0
    assert header_length('"""One line."""\n') == 1


def test_no_module_was_added_to_the_list_after_the_rule_was_made():
    """The list can only empty: a name the introducing commit did not write is refused."""
    added = STILL_LONG - first_listed(REPO)
    assert not added, f"added to STILL_LONG after the rule was made; shorten the header instead: {sorted(added)}"


def test_the_measure_counts_what_follows_a_usage_that_is_not_the_last_section():
    """The planted loopholes: prose after Usage, and a Usage in the middle of the prose, are counted."""
    usage_then_prose = '"""Short.\n\nUsage:\n    run it\n\n' + "\n".join(f"history {i}" for i in range(40)) + '\n"""\n'
    assert header_length(usage_then_prose) == 45 > HEADER_LINES
    usage_in_prose = '"""Short.\n\nUsage: of the word in a sentence\n' + "\n".join(f"line {i}" for i in range(12)) + '\n"""\n'
    assert header_length(usage_in_prose) == 15 > HEADER_LINES
    indented_usage_is_prose = '"""Short.\n\n  Usage:\n    run it\n"""\n'
    assert header_length(indented_usage_is_prose) == 4


def test_the_list_is_read_back_from_a_version_of_this_file():
    """The reader of `STILL_LONG` finds the set in a file's source and refuses a file without one."""
    assert listed_in('STILL_LONG = frozenset({"a.py", "b.py"})\n') == {"a.py", "b.py"}
    with pytest.raises(RuntimeError, match="no STILL_LONG"):
        listed_in("x = 1\n")
