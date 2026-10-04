"""A module's header says what the module does, in at most `HEADER_LINES` lines, then how to run it.

The header is the module docstring up to its `Usage:` line. A reader decides
from it whether this is the file they want; history, definitions and the
defence of choices go where `docs/review.md` sends them. The modules that
were over the cap when the rule was made are listed in `STILL_LONG`; one
leaves the list when it is shortened, and cannot stay in it once short.
"""

import ast
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The most lines a header may have before its Usage.
HEADER_LINES = 12

# The modules longer than the cap when the rule was made, shortened by the
# workstream `docs/short-headers`. A module is removed here when it is
# shortened; a module that is short and still listed fails the ratchet.
STILL_LONG = frozenset({
    "evalkit/boundary.py",
    "evalkit/classes.py",
    "evalkit/classmap.py",
    "evalkit/clip.py",
    "evalkit/code_sha.py",
    "evalkit/frame.py",
    "evalkit/inst_bf.py",
    "evalkit/objects.py",
    "evalkit/pilot.py",
    "evalkit/scored.py",
    "evalkit/time_iou.py",
    "evalkit/tools/paired_stats.py",
    "evalkit/tools/scores.py",
    "evalkit/unlabelled.py",
    "evalkit/vi.py",
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
    """The lines of `source`'s module docstring before its `Usage:` line, blank lines at the end left out; 0 without a docstring."""
    doc = ast.get_docstring(ast.parse(source), clean=False) or ""
    lines = doc.strip("\n").splitlines()
    head = lines[:next((i for i, line in enumerate(lines) if line.strip().startswith("Usage:")), len(lines))]
    while head and not head[-1].strip():
        head.pop()
    return len(head)


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
