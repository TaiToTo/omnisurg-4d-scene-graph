"""No TODO marker in the repository's files.

A path known to give a wrong answer for some input raises on that input, and
a decision not yet made is written under "Open questions" in `docs/porting.md`.
Neither lives in a comment. A comment is read only by whoever opens that file;
the clip extractor whose gap frames carried the wrong frame number said so in
a comment for two months while the data it produced was measured, and the
fault was found again downstream and explained wrongly.

Every tracked text file is scanned, except the planning documents that hold
the open items and the file that states the rule.
"""

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

# The only files that may carry the words. The planning documents are where
# an open item is written down; `AGENTS.md` (and its symlink) states the rule;
# this file defines the pattern.
ALLOWED = frozenset({
    "docs/porting.md",
    "docs/workstreams.md",
    "AGENTS.md",
    "CLAUDE.md",
    "tests/test_no_todo_in_code.py",
})


def tracked_files(repo: Path) -> list[Path]:
    """Every file git tracks under `repo`, symlinks left out (their targets are tracked too)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                         check=True, capture_output=True).stdout
    paths = [repo / p for p in out.decode("utf-8").split("\0") if p]
    return [p for p in paths if not os.path.islink(p)]


def markers_in(path: Path) -> list[tuple[int, str]]:
    """The `(line number, line)` pairs of `path` that carry a marker.

    Raises:
        UnicodeDecodeError: the file is not UTF-8 text. Nothing in this
            repository should be, and a file that cannot be read cannot be
            passed as clean.
    """
    text = path.read_bytes().decode("utf-8")
    return [(i, line.rstrip()) for i, line in enumerate(text.splitlines(), 1) if MARKER.search(line)]


def test_no_marker_in_tracked_files():
    hits = {}
    for p in tracked_files(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in ALLOWED:
            continue
        found = markers_in(p)
        if found:
            hits[rel] = found
    report = "\n".join(f"{f}:{n}: {line}" for f, lines in sorted(hits.items()) for n, line in lines)
    assert not hits, (
        "TODO markers in code. Make the path raise on the input it is wrong for, "
        "or write the open question in docs/porting.md:\n" + report)


def test_the_scanner_sees_a_planted_marker(tmp_path):
    """The check earns its place by failing: a planted marker is reported, with its line."""
    p = tmp_path / "x.py"
    p.write_text("a = 1\n# TODO: decide later\nb = 2  # FIXME\n", encoding="utf-8")
    assert markers_in(p) == [(2, "# TODO: decide later"), (3, "b = 2  # FIXME")]
    (tmp_path / "clean.py").write_text("todos = []  # the word inside another word is not a marker\n",
                                       encoding="utf-8")
    assert markers_in(tmp_path / "clean.py") == []


def test_every_exemption_names_a_tracked_file():
    """A stale exemption would silently widen the rule; each must still exist."""
    tracked = {p.relative_to(REPO).as_posix() for p in tracked_files(REPO)}
    tracked |= {"CLAUDE.md"}                 # the symlink, left out of tracked_files on purpose
    missing = ALLOWED - tracked
    assert not missing, missing
