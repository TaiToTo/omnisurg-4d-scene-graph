"""Every module under `evalkit/tools/` is named in the target layout of `AGENTS.md`.

A layout written in a document goes out of date unless something reads it.
This test reads the `tools/` entry of the layout and compares it with the
modules on disk.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "evalkit" / "tools"
HEADING = "## Target layout"


def tools_entry(text: str) -> str:
    """The names the `tools/` entry of the layout under `HEADING` lists, as one string.

    The entry is its first line after the colon, which ends the description,
    and the lines that continue it, up to the next entry.

    Raises:
        ValueError: No heading, no fenced block under it, or no `tools/` entry in the block.
    """
    _, sep, rest = text.partition(HEADING)
    block = re.search(r"^```\n(.*?)^```$", rest, re.S | re.M) if sep else None
    if block is None:
        raise ValueError(f"no fenced layout under {HEADING!r}")
    lines = block.group(1).split("\n")
    start = next((i for i, line in enumerate(lines) if "tools/" in line), None)
    if start is None:
        raise ValueError("the layout has no tools/ entry")
    entry = [lines[start].partition(":")[2]]
    for line in lines[start + 1:]:
        if "──" in line:
            break
        entry.append(line)
    return " ".join(entry)


def unnamed(entry: str, names: list[str]) -> list[str]:
    """The names that `entry` does not list as a word."""
    words = set(re.findall(r"\w+", entry))
    return sorted(name for name in names if name not in words)


def test_every_tool_is_named_in_the_layout():
    tools = [p.stem for p in TOOLS.glob("*.py") if p.stem != "__init__"]
    assert tools, f"no module found under {TOOLS}; the scan reads nothing"
    assert not unnamed(tools_entry((REPO / "AGENTS.md").read_text(encoding="utf-8")), tools)


def test_a_tool_missing_from_the_entry_is_reported_and_the_description_does_not_count():
    page = (f"{HEADING}\n\n```\n├── evalkit/\n│   └── tools/   what only reads scores: paired_stats\n"
            "│                compare_eval\n├── pipeline/   pilot_check\n```\n")
    assert unnamed(tools_entry(page), ["paired_stats", "compare_eval", "scores", "pilot_check"]) \
        == ["pilot_check", "scores"]


def test_a_page_without_the_layout_is_refused():
    with pytest.raises(ValueError, match="no fenced layout"):
        tools_entry("# AGENTS\n\nno layout\n")
    with pytest.raises(ValueError, match="no tools/ entry"):
        tools_entry(f"{HEADING}\n\n```\n├── evalkit/\n```\n")
