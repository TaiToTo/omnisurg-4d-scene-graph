"""The `tools/` entry of the layout in `AGENTS.md` names exactly the modules under `evalkit/tools/`.

A layout written in a document goes out of date unless something reads it.
This test reads the `tools/` entry and compares it with the modules on disk,
both ways: a module the entry leaves out, a name it keeps for a module that
is gone, and a module it still lists in parentheses as waiting.
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


def tools_named(entry: str) -> tuple[set[str], set[str]]:
    """The modules `entry` names as ported, and those it names in parentheses as waiting.

    A parenthesis lists names and then, after a colon, what they wait on;
    the words after that colon are not names.
    """
    waiting = set()
    for inside in re.findall(r"\(([^)]*)\)", entry):
        waiting |= set(re.findall(r"\w+", inside.partition(":")[0]))
    ported = set(re.findall(r"\w+", re.sub(r"\([^)]*\)", " ", entry)))
    return ported, waiting


def drift(entry: str, modules: set[str]) -> list[str]:
    """Every way `entry` disagrees with `modules`, the tools on disk, one line each."""
    ported, waiting = tools_named(entry)
    return ([f"{m}: under evalkit/tools/, not named in the layout" for m in sorted(modules - ported - waiting)]
            + [f"{m}: named in the layout, not under evalkit/tools/" for m in sorted(ported - modules)]
            + [f"{m}: under evalkit/tools/, but the layout says it waits" for m in sorted(modules & waiting)])


def test_the_layout_names_exactly_the_tools_on_disk():
    modules = {p.stem for p in TOOLS.glob("*.py") if p.stem != "__init__"}
    assert modules, f"no module found under {TOOLS}; the scan reads nothing"
    problems = drift(tools_entry((REPO / "AGENTS.md").read_text(encoding="utf-8")), modules)
    assert not problems, "AGENTS.md's layout disagrees with evalkit/tools/:\n" + "\n".join(problems)


PAGE = (f"{HEADING}\n\n```\n├── evalkit/\n"
        "│   └── tools/   what only reads scores: paired_stats\n"
        "│                compare_eval\n"
        "│                (kmerge: which side waits on docs/porting.md)\n"
        "├── pipeline/   pilot_check\n```\n")


def test_a_tool_left_out_is_reported_and_the_description_does_not_count():
    assert drift(tools_entry(PAGE), {"paired_stats", "compare_eval", "scores", "pilot_check"}) == [
        "pilot_check: under evalkit/tools/, not named in the layout",
        "scores: under evalkit/tools/, not named in the layout",
    ]


def test_a_name_kept_for_a_removed_tool_is_reported():
    # The words after the colon in the parenthesis are not names, so they are not reported as gone.
    assert drift(tools_entry(PAGE), {"paired_stats"}) == ["compare_eval: named in the layout, not under evalkit/tools/"]


def test_a_tool_the_layout_says_waits_is_reported_once_it_arrives():
    assert drift(tools_entry(PAGE), {"paired_stats", "compare_eval", "kmerge"}) == [
        "kmerge: under evalkit/tools/, but the layout says it waits",
    ]


def test_a_page_without_the_layout_is_refused():
    with pytest.raises(ValueError, match="no fenced layout"):
        tools_entry("# AGENTS\n\nno layout\n")
    with pytest.raises(ValueError, match="no tools/ entry"):
        tools_entry(f"{HEADING}\n\n```\n├── evalkit/\n```\n")
