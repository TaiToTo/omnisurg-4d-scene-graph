"""Read the rows of camera trajectory score files, and refuse rows that cannot be compared.

A row is one clip under one condition, as `trajectory_eval.score.score`
writes it. Rows are compared only when they record one
`trajectory_code_sha`: two hashes mean two versions of the scorer. The
reader refuses:

- a file that is not a list of rows;
- a row without `trajectory_code_sha`;
- rows that record more than one `trajectory_code_sha`;
- a clip scored twice under one condition.
"""

import json
from pathlib import Path

HASH_KEY = "trajectory_code_sha"


def read_rows(paths: list[Path]) -> list[dict]:
    """Return the rows of every file, in order.

    Raises:
        ValueError: a file is not a list of rows, a row records no hash, the rows record more than one hash, or a
            clip is scored twice under one condition. The message names the files.
    """
    rows, by_hash, seen = [], {}, {}
    for path in map(Path, paths):
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError(f"{path} is not a list of rows")
        for i, row in enumerate(data):
            if HASH_KEY not in row:
                raise ValueError(f"{path}: row {i} records no {HASH_KEY}, so the scorer that made it is not known")
            by_hash.setdefault(row[HASH_KEY], set()).add(str(path))
            key = (row["clip"], row["cond"])
            if key in seen:
                raise ValueError(f"{row['clip']} is scored twice under {row['cond']}, in {seen[key]} and {path}")
            seen[key] = str(path)
        rows += data
    if len(by_hash) > 1:
        raise ValueError(f"the rows record {len(by_hash)} versions of the scorer, which are not compared: "
                         + "; ".join(f"{h[:12]} in {sorted(files)}" for h, files in sorted(by_hash.items())))
    return rows
