"""The layout in `AGENTS.md` names what is on disk, entry by entry.

A layout written in a document goes out of date unless something reads it.
Every directory and file the layout names must exist. Three entries list
names: the root's files, the packages of `surgical_core/` and the modules of
`evalkit/tools/`, which must match the disk both ways, and `docs/`, whose
documents must exist. A name in parentheses waits, and must not exist yet;
the words after the colon in a parenthesis say what it waits on.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "evalkit" / "tools"
HEADING = "## Target layout"
ENTRY = re.compile(r"^((?:│   |    )*)[├└]── (\S+)\s*(.*)$")
ROOT = ""


def entries(text: str) -> dict[str, str]:
    """Each entry of the layout under `HEADING`: its repository-relative path, and its description.

    An entry runs from its line to the next entry, through the lines that
    continue it. The root's files are one entry, under the path "".

    Raises:
        ValueError: No heading, or no fenced block under it.
    """
    _, sep, rest = text.partition(HEADING)
    block = re.search(r"^```\n(.*?)^```$", rest, re.S | re.M) if sep else None
    if block is None:
        raise ValueError(f"no fenced layout under {HEADING!r}")
    out: dict[str, str] = {}
    parents: list[str] = []
    path = None
    for line in block.group(1).split("\n"):
        m = ENTRY.match(line)
        if m:
            depth, name = len(m.group(1)) // 4, m.group(2)
            parents = parents[:depth]
            if name.endswith("/"):
                path = "".join(parents) + name
                parents.append(name)
                out[path] = m.group(3)
            elif "/" in name:
                path = name
                out[path] = m.group(3)
            else:
                path = ROOT
                out[path] = f"{name} {m.group(3)}"
        elif path is not None and line.strip("│ "):
            out[path] += " " + line.strip("│ ")
    return out


def tools_entry(text: str) -> str:
    """The description of the `tools/` entry, which lists the tools.

    Raises:
        ValueError: The layout has no `tools/` entry.
    """
    found = entries(text).get("evalkit/tools/")
    if found is None:
        raise ValueError("the layout has no tools/ entry")
    return found.partition(":")[2]


def names_in(description: str) -> tuple[set[str], set[str]]:
    """The names `description` lists as present, and those it lists in parentheses as waiting.

    A parenthesis lists names and then, after a colon, what they wait on;
    the words after that colon are not names.
    """
    waiting = set()
    for inside in re.findall(r"\(([^)]*)\)", description):
        waiting |= set(re.findall(r"[\w.-]+", inside.partition(":")[0]))
    present = set(re.findall(r"[\w.-]+", re.sub(r"\([^)]*\)", " ", description)))
    return present, waiting


def drift(description: str, on_disk: set[str], where: str = "evalkit/tools/") -> list[str]:
    """Every way the names `description` lists disagree with `on_disk`, the names under `where`."""
    present, waiting = names_in(description)
    return ([f"{m}: under {where}, not named in the layout" for m in sorted(on_disk - present - waiting)]
            + [f"{m}: named in the layout, not under {where}" for m in sorted(present - on_disk)]
            + [f"{m}: under {where}, but the layout says it waits" for m in sorted(on_disk & waiting)])


def missing_paths(layout: dict[str, str], repo: Path) -> list[str]:
    """The directories and files the layout names that `repo` does not hold, the root's files included."""
    present, waiting = names_in(layout.get(ROOT, ""))
    return ([f"{p}: named in the layout, not in the repository" for p in sorted(layout) if p and not (repo / p).exists()]
            + [f"{n}: named in the layout, not in the repository" for n in sorted(present) if not (repo / n).exists()]
            + [f"{n}: in the repository, but the layout says it waits" for n in sorted(waiting) if (repo / n).exists()])


def packages(directory: Path) -> set[str]:
    """The modules and packages directly under `directory`, by name."""
    return ({p.stem for p in directory.glob("*.py") if p.stem != "__init__"}
            | {p.name for p in directory.iterdir() if (p / "__init__.py").is_file()})


@pytest.fixture(scope="module")
def layout() -> dict[str, str]:
    return entries((REPO / "AGENTS.md").read_text(encoding="utf-8"))


def test_every_directory_and_file_the_layout_names_exists(layout):
    assert {ROOT, "surgical_core/", "evalkit/tools/", "docs/"} <= set(layout), "the parser lost an entry"
    problems = missing_paths(layout, REPO)
    assert not problems, "AGENTS.md's layout disagrees with the repository:\n" + "\n".join(problems)


def test_the_layout_names_exactly_the_tools_on_disk(layout):
    modules = {p.stem for p in TOOLS.glob("*.py") if p.stem != "__init__"}
    assert modules, f"no module found under {TOOLS}; the scan reads nothing"
    problems = drift(layout["evalkit/tools/"].partition(":")[2], modules)
    assert not problems, "AGENTS.md's layout disagrees with evalkit/tools/:\n" + "\n".join(problems)


def test_the_layout_names_exactly_the_packages_of_surgical_core(layout):
    on_disk = packages(REPO / "surgical_core")
    assert on_disk, "no module found under surgical_core/; the scan reads nothing"
    problems = drift(layout["surgical_core/"], on_disk, "surgical_core/")
    assert not problems, "AGENTS.md's layout disagrees with surgical_core/:\n" + "\n".join(problems)


def test_every_document_the_layout_names_exists(layout):
    present, _ = names_in(layout["docs/"])
    assert present, "the docs/ entry names no document"
    gone = sorted(n for n in present if not (REPO / "docs" / n).is_file())
    assert not gone, f"named in the layout's docs/ entry, not under docs/: {gone}"


PAGE = (f"{HEADING}\n\n```\nrepo/\n├── LICENSE  README.md  (CITATION.cff: at the release)\n├── evalkit/\n"
        "│   └── tools/   what only reads scores: paired_stats\n"
        "│                compare_eval\n"
        "│                (kmerge: which side waits on docs/porting.md)\n"
        "├── pipeline/   pilot_check\n└── .github/workflows/ci.yml\n```\n")


def test_the_parser_reads_each_entry_with_its_path():
    assert set(entries(PAGE)) == {ROOT, "evalkit/", "evalkit/tools/", "pipeline/", ".github/workflows/ci.yml"}
    assert names_in(entries(PAGE)[ROOT]) == ({"LICENSE", "README.md"}, {"CITATION.cff"})


def test_a_missing_directory_a_missing_root_file_and_an_arrived_waiting_file_are_reported(tmp_path):
    (tmp_path / "evalkit" / "tools").mkdir(parents=True)
    (tmp_path / "LICENSE").write_text("")
    (tmp_path / "CITATION.cff").write_text("")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("")
    assert missing_paths(entries(PAGE), tmp_path) == [
        "pipeline/: named in the layout, not in the repository",
        "README.md: named in the layout, not in the repository",
        "CITATION.cff: in the repository, but the layout says it waits",
    ]


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
