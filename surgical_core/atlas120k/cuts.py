"""Read the cuts judged inside the ATLAS-120k clips, `atlas120k_meta/cuts.csv`, and name a clip's pieces.

A cut is a scene change inside a video's mp4. `load_cuts` returns the rows
judged a cut: per clip, the mp4 frames after which its cuts fall; the frame
after each, `cut_after`, starts the new scene. It refuses a table it cannot
read for certain: other columns, a judgement other than `yes` or `no`, a
cut whose two frames are missing or not adjacent, a row that is not a cut
yet names a cut's frames, a cut listed twice, and a clip name that is not a
clip's. The extraction stage checks that each cut falls inside its clip.
A clip split at its cuts gives pieces named `<clip>s<j>`; `whole_clip`
returns the clip a piece came from.
"""

import csv
import os
import re

COLUMNS = ("clip", "spot_first", "spot_last", "d_pix", "boundary_before", "boundary_after", "cut", "cut_before",
           "cut_after", "rechecked")

# `<procedure>__<video>__gt_<n>`, the name of a clip of the population before any is split.
CLIP_NAME = re.compile(r"[a-z_]+__[A-Za-z0-9_-]+__gt_\d{4}")
# A clip of the population: a whole clip, or the `j`th piece of one split at its cuts.
PIECE_NAME = re.compile(rf"({CLIP_NAME.pattern})(?:s\d+)?")


def whole_clip(name: str) -> str:
    """Return the clip before any split that the population clip `name` is, or is a piece of.

    Raises:
        ValueError: `name` is neither `<procedure>__<video>__gt_<n>` nor that name followed by `s<j>`.
    """
    m = PIECE_NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"{name!r} is not a clip of the population, <procedure>__<video>__gt_<n>[s<j>]")
    return m.group(1)


def _frame(value: str, where: str, column: str) -> int:
    """Return a frame number of the table, or refuse a value that is not one."""
    if not value.isdigit():
        raise ValueError(f"{where}: {column} is {value!r}, not a frame number")
    return int(value)


def load_cuts(path: str) -> dict[str, tuple[int, ...]]:
    """Read the table and return, per clip with a cut, the mp4 frames its cuts fall after, ascending.

    Raises:
        FileNotFoundError: the table is missing. Without it no clip would be
            split, and that is reported here, not one clip at a time.
        ValueError: the table is refused, for a reason the module's header
            lists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no cut table at {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: the columns are {reader.fieldnames}, not {list(COLUMNS)}")
        rows = list(reader)

    # Check each row, and collect the cuts.
    cuts: dict[str, set[int]] = {}
    for line, row in enumerate(rows, start=2):
        where = f"{path}:{line}"
        clip = row["clip"]
        if not CLIP_NAME.fullmatch(clip):
            raise ValueError(f"{where}: {clip!r} is not a clip name, <procedure>__<video>__gt_<n>")
        if row["cut"] == "no":
            if row["cut_before"] or row["cut_after"]:
                raise ValueError(f"{where}: a candidate that is not a cut names a cut's frames")
            continue
        if row["cut"] != "yes":
            raise ValueError(f"{where}: cut is {row['cut']!r}, not 'yes' or 'no'")
        before = _frame(row["cut_before"], where, "cut_before")
        after = _frame(row["cut_after"], where, "cut_after")
        if after != before + 1:
            raise ValueError(f"{where}: a cut falls between two adjacent frames, not between {before} and {after}")
        if before in cuts.get(clip, ()):
            raise ValueError(f"{where}: the cut after frame {before} of {clip} is listed twice")
        cuts.setdefault(clip, set()).add(before)
    return {clip: tuple(sorted(c)) for clip, c in sorted(cuts.items())}
