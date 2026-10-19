"""Read the rows of camera trajectory score files, and refuse rows that cannot be compared.

A row is one clip under one condition, as `trajectory_eval.score.score`
writes it. Rows are compared only when they record one
`trajectory_code_sha` and one `clip_list_sha`: two of either mean two
versions of the scorer, or clips cut two ways. The reader refuses:

- a file that is not a list of rows;
- a row without `clip` or `cond`;
- a row whose `trajectory_code_sha` or `clip_list_sha` is missing or is not a sha256 hex digest;
- rows that record more than one `trajectory_code_sha`, or more than one `clip_list_sha`;
- a clip scored twice under one condition.
"""

import json
import re
from pathlib import Path

HASH_KEY = "trajectory_code_sha"
# The hash of the clip list a row was scored on: which frames, ground truth rows and times make each clip.
LIST_KEY = "clip_list_sha"
# What two values of each recorded hash mean.
RECORDED = {HASH_KEY: "versions of the scorer", LIST_KEY: "clip lists"}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def read_rows(paths: list[Path]) -> list[dict]:
    """Return the rows of every file, in order.

    Raises:
        ValueError: a file is not a list of rows, a row names no clip or condition, a row records no hash of the
            scorer or of the clip list, the rows record more than one of either, or a clip is scored twice under
            one condition. The message names the files.
    """
    rows, by_value, seen = [], {key: {} for key in RECORDED}, {}
    for path in map(Path, paths):
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError(f"{path} is not a list of rows")
        for i, row in enumerate(data):
            if "clip" not in row or "cond" not in row:
                raise ValueError(f"{path}: row {i} names no clip or no condition")
            for key in RECORDED:
                if not (isinstance(row.get(key), str) and HASH_PATTERN.fullmatch(row[key])):
                    raise ValueError(f"{path}: row {i} records no {key}, so what made it is not known")
                by_value[key].setdefault(row[key], set()).add(str(path))
            key = (row["clip"], row["cond"])
            if key in seen:
                raise ValueError(f"{row['clip']} is scored twice under {row['cond']}, in {seen[key]} and {path}")
            seen[key] = str(path)
        rows += data
    for key, what in RECORDED.items():
        if len(by_value[key]) > 1:
            raise ValueError(f"the rows record {len(by_value[key])} {what}, which are not compared: "
                             + "; ".join(f"{h[:12]} in {sorted(files)}" for h, files in sorted(by_value[key].items())))
    return rows
