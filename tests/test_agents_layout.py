"""The layout in `AGENTS.md` names what is on disk, entry by entry.

A layout written in a document goes out of date unless something reads it.
This test checks the layout against the disk:

- every directory and file it names exists;
- every file that the root's entry and the `docs/` entry list exists;
- the packages of `surgical_core/` and the modules of `evalkit/tools/`
  match their entries both ways;
- a name in parentheses waits, and does not exist yet.

The words after the colon in a parenthesis say what the name waits on.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "evalkit" / "tools"
HEADING = "## Target layout"
ENTRY = re.compile(r"^((?:│   |    )*)[├└]── (\S+)\s*(.*)$")
ROOT = ""
# The entries whose description lists files under their path.
LISTS_FILES = (ROOT, "docs/")


def entries(text: str) -> dict[str, str]:
    """Each entry of the layout under `HEADING`: its repository-relative path, and its description.

    An entry runs from its line to the next entry, through the lines that
    continue it. A top-level name without a slash starts the root's entry,
    which lists the root's files under the path "".

    Raises:
        ValueError: No heading, no fenced block under it, a line that draws a
            branch but is not an entry, an entry deeper than its parent, or
            two entries for one path.
    """
    # Find the fenced block under the heading.
    _, sep, rest = text.partition(HEADING)
    block = re.search(r"^```\n(.*?)^```$", rest, re.S | re.M) if sep else None
    if block is None:
        raise ValueError(f"no fenced layout under {HEADING!r}")
    out: dict[str, str] = {}
    parents: list[str] = []
    path = None
    for line in block.group(1).split("\n"):
        m = ENTRY.match(line)
        # Add a line that continues the entry above it.
        if m is None:
            if "──" in line:
                raise ValueError(f"a branch that is not an entry: {line!r}")
            if path is not None and line.strip("│ "):
                out[path] += " " + line.strip("│ ")
            continue
        # Place a new entry under the directories above it.
        depth, name, description = len(m.group(1)) // 4, m.group(2), m.group(3)
        if depth > len(parents):
            raise ValueError(f"an entry deeper than its parent: {line!r}")
        parents = parents[:depth]
        if depth == 0 and "/" not in name:
            path, description = ROOT, f"{name} {description}"
        else:
            path = "".join(parents) + name
        if path in out:
            raise ValueError(f"two entries for one path: {line!r}")
        if name.endswith("/"):
            parents.append(name)
        out[path] = description
    return out


def tools_entry(layout: dict[str, str]) -> str:
    """The part of the `tools/` entry's description that lists the tools, after its colon.

    Raises:
        ValueError: The layout has no `tools/` entry.
    """
    found = layout.get("evalkit/tools/")
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
    """Every way the layout's paths, and the files its `LISTS_FILES` entries list, disagree with `repo`.

    A path or a listed file that `repo` does not hold is reported, and so is
    a file in parentheses that `repo` already holds.
    """
    problems = [f"{p}: named in the layout, not in the repository" for p in sorted(layout) if p and not (repo / p).exists()]
    for entry in LISTS_FILES:
        present, waiting = names_in(layout[entry])
        problems += [f"{entry}{n}: named in the layout, not in the repository"
                     for n in sorted(present) if not (repo / entry / n).is_file()]
        problems += [f"{entry}{n}: in the repository, but the layout says it waits"
                     for n in sorted(waiting) if (repo / entry / n).exists()]
    return problems


def packages(directory: Path) -> set[str]:
    """The modules and packages directly under `directory`, by name."""
    return ({p.stem for p in directory.glob("*.py") if p.stem != "__init__"}
            | {p.name for p in directory.iterdir() if (p / "__init__.py").is_file()})


@pytest.fixture(scope="module")
def layout() -> dict[str, str]:
    return entries((REPO / "AGENTS.md").read_text(encoding="utf-8"))


def test_every_directory_and_file_the_layout_names_exists(layout):
    assert {ROOT, "surgical_core/", "evalkit/tools/", "docs/"} <= set(layout), "the parser lost an entry"
    assert all(names_in(layout[e])[0] for e in LISTS_FILES), "an entry that lists files lists none"
    problems = missing_paths(layout, REPO)
    assert not problems, "AGENTS.md's layout disagrees with the repository:\n" + "\n".join(problems)


def test_the_layout_names_exactly_the_tools_on_disk(layout):
    modules = packages(TOOLS)
    assert modules, f"no module found under {TOOLS}; the scan reads nothing"
    problems = drift(tools_entry(layout), modules)
    assert not problems, "AGENTS.md's layout disagrees with evalkit/tools/:\n" + "\n".join(problems)


def test_the_layout_names_exactly_the_packages_of_surgical_core(layout):
    on_disk = packages(REPO / "surgical_core")
    assert on_disk, "no module found under surgical_core/; the scan reads nothing"
    problems = drift(layout["surgical_core/"], on_disk, "surgical_core/")
    assert not problems, "AGENTS.md's layout disagrees with surgical_core/:\n" + "\n".join(problems)


PAGE = (f"{HEADING}\n\n```\nrepo/\n├── LICENSE  README.md  (CITATION.cff: waits on the release)\n├── evalkit/\n"
        "│   └── tools/   what only reads scores: paired_stats\n"
        "│                compare_eval\n"
        "│                (kmerge: which side waits on docs/porting.md)\n"
        "├── pipeline/   pilot_check\n"
        "├── docs/       evaluation.md, (data_contract.md: waits on the export stage)\n"
        "└── .github/workflows/ci.yml\n```\n")


def test_the_parser_reads_each_entry_with_its_path():
    assert set(entries(PAGE)) == {ROOT, "evalkit/", "evalkit/tools/", "pipeline/", "docs/", ".github/workflows/ci.yml"}
    assert names_in(entries(PAGE)[ROOT]) == ({"LICENSE", "README.md"}, {"CITATION.cff"})


def test_a_file_inside_a_directory_is_read_as_its_path_and_leaves_the_root_entry_alone():
    page = f"{HEADING}\n\n```\n├── LICENSE  README.md\n├── viewer/\n│   └── index.html  the page\n```\n"
    assert entries(page) == {ROOT: "LICENSE README.md", "viewer/": "", "viewer/index.html": "the page"}


@pytest.mark.parametrize("body, refusal", [
    ("├── pipeline/   stages\n├──recon3d_wrapper/   3D\n", "a branch that is not an entry"),
    ("├── docs/   a.md\n│   │   └── figures/\n", "an entry deeper than its parent"),
    ("├── LICENSE\n├── README.md\n", "two entries for one path"),
])
def test_a_line_the_parser_cannot_place_is_refused(body, refusal):
    with pytest.raises(ValueError, match=refusal):
        entries(f"{HEADING}\n\n```\n{body}```\n")


def test_a_missing_path_a_missing_listed_file_and_an_arrived_waiting_file_are_reported(tmp_path):
    for d in ("evalkit/tools", "docs", ".github/workflows"):
        (tmp_path / d).mkdir(parents=True)
    for f in ("LICENSE", "CITATION.cff", "docs/data_contract.md", ".github/workflows/ci.yml"):
        (tmp_path / f).write_text("")
    assert missing_paths(entries(PAGE), tmp_path) == [
        "pipeline/: named in the layout, not in the repository",
        "README.md: named in the layout, not in the repository",
        "CITATION.cff: in the repository, but the layout says it waits",
        "docs/evaluation.md: named in the layout, not in the repository",
        "docs/data_contract.md: in the repository, but the layout says it waits",
    ]


def test_a_tool_left_out_is_reported_and_the_description_does_not_count():
    assert drift(tools_entry(entries(PAGE)), {"paired_stats", "compare_eval", "scores", "pilot_check"}) == [
        "pilot_check: under evalkit/tools/, not named in the layout",
        "scores: under evalkit/tools/, not named in the layout",
    ]


def test_a_name_kept_for_a_removed_tool_is_reported():
    # The words after the colon in the parenthesis are not names, so they are not reported as gone.
    assert drift(tools_entry(entries(PAGE)), {"paired_stats"}) == [
        "compare_eval: named in the layout, not under evalkit/tools/",
    ]


def test_a_tool_the_layout_says_waits_is_reported_once_it_arrives():
    assert drift(tools_entry(entries(PAGE)), {"paired_stats", "compare_eval", "kmerge"}) == [
        "kmerge: under evalkit/tools/, but the layout says it waits",
    ]


def test_a_page_without_the_layout_is_refused():
    with pytest.raises(ValueError, match="no fenced layout"):
        entries("# AGENTS\n\nno layout\n")
    with pytest.raises(ValueError, match="no tools/ entry"):
        tools_entry(entries(f"{HEADING}\n\n```\n├── evalkit/\n```\n"))
