"""Compare the class table's mapping of ATLAS-120k's ids onto the benchmark's classes with ATLAS-bench's own.

The tool reads `mapping`, the dict from each original id to its benchmark class, out of ATLAS-bench's
`datasets/class_mapping.py`, without running the file. It compares that dict with the mapping of
`evalkit/class_tables/atlas120k.json` on every id either side holds, and prints each id that one side maps and
the other does not, or that the two sides map to different classes. It exits non-zero on any such id.

Usage:
    python -m evalkit.tools.check_benchmark_mapping /path/to/ATLAS-bench/datasets/class_mapping.py
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Mapping
from pathlib import Path

from evalkit.classes import load_table

# The name ATLAS-bench gives the dict, in `datasets/class_mapping.py`.
UPSTREAM_NAME = "mapping"


def read_upstream_mapping(path: str | Path) -> dict[int, int]:
    """Return the dict that `path` assigns to `UPSTREAM_NAME` at its top level, read as a literal.

    The file is parsed, never run: it is another repository's code.

    Raises:
        ValueError: the file is not Python, assigns the name other than once, or assigns it something other than
            a dict literal of integer keys and values.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        raise ValueError(f"{path} is not Python: {e}") from None
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == UPSTREAM_NAME for t in node.targets)]
    if len(values) != 1:
        raise ValueError(f"{path} assigns `{UPSTREAM_NAME}` {len(values)} times at its top level, not once")
    try:
        mapping = ast.literal_eval(values[0])
    except (TypeError, ValueError):
        raise ValueError(f"{path}: `{UPSTREAM_NAME}` is not a literal") from None
    if not isinstance(mapping, dict) or not all(type(k) is int and type(v) is int for k, v in mapping.items()):
        raise ValueError(f"{path}: `{UPSTREAM_NAME}` is not a dict of integer ids to integer classes")
    return mapping


def disagreements(upstream: Mapping[int, int], ours: Mapping[int, int]) -> list[str]:
    """Return one line per id that the two mappings do not map alike, in id order; empty when they agree."""
    out = []
    for i in sorted(set(upstream) | set(ours)):
        if i not in ours:
            out.append(f"id {i}: ATLAS-bench maps it to {upstream[i]}, and the class table has no such id")
        elif i not in upstream:
            out.append(f"id {i}: the class table maps it to {ours[i]}, and ATLAS-bench has no such id")
        elif upstream[i] != ours[i]:
            out.append(f"id {i}: ATLAS-bench maps it to {upstream[i]}, the class table to {ours[i]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("class_mapping", help="ATLAS-bench's datasets/class_mapping.py.")
    args = ap.parse_args()
    try:
        upstream = read_upstream_mapping(args.class_mapping)
    except (OSError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    ours = load_table("atlas120k", "benchmark").mask_id_to_class
    problems = disagreements(upstream, ours)
    print("\n".join(problems) or f"all {len(ours)} ids map alike")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
