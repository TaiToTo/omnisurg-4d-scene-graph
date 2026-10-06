"""Check that no TODO marker is in the repository's files.

A path known to give a wrong answer raises on that input, and a decision not
yet made goes under "Open questions" in `docs/porting.md`. Neither lives in
a comment, which only whoever opens the file reads. `docs/review.md` gives
the incident behind the rule ("Code"). Every tracked text file is scanned,
except the planning documents that hold the open items and the files that
state the rule. A binary file is told from text as git tells it, by a NUL
byte in its first 8,000 bytes, so no text file can pass as binary by being
saved in another encoding.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

MARKER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

# The only files that may carry the words. The planning documents are where
# an open item is written down; `AGENTS.md` (and its symlink) states the rule;
# this file defines the pattern.
ALLOWED = frozenset({
    "docs/porting.md",
    "docs/workstreams.md",
    "docs/review.md",
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


def is_binary(path: Path) -> bool:
    """Return whether git would treat `path` as binary: a NUL byte in its first 8,000 bytes."""
    with path.open("rb") as f:
        return b"\0" in f.read(8000)


def markers_in(path: Path) -> list[tuple[int, str]]:
    """The `(line number, line)` pairs of `path` that carry a marker; none for a binary file.

    Raises:
        UnicodeDecodeError: the file is text but not UTF-8. Nothing in this
            repository should be, and a file that cannot be read cannot be
            passed as clean.
    """
    if is_binary(path):
        return []
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


def test_a_binary_file_is_skipped_and_a_text_file_in_another_encoding_is_refused(tmp_path):
    """A PNG is not text, so the word inside it is no marker; text that is not UTF-8 still raises."""
    png = tmp_path / "figure.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR TODO in a chunk")
    assert is_binary(png) and markers_in(png) == []
    latin = tmp_path / "notes.txt"
    latin.write_bytes("caf\xe9  # TODO\n".encode("latin-1"))
    assert not is_binary(latin)
    with pytest.raises(UnicodeDecodeError):
        markers_in(latin)


def test_every_exemption_names_a_tracked_file():
    """A stale exemption would silently widen the rule; each must still exist."""
    tracked = {p.relative_to(REPO).as_posix() for p in tracked_files(REPO)}
    tracked |= {"CLAUDE.md"}                 # the symlink, left out of tracked_files on purpose
    missing = ALLOWED - tracked
    assert not missing, missing
