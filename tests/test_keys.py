"""The keys of `evalkit.keys` against the specification, the frame driver and the tools.

The metrics table of `docs/evaluation.md` is read, not copied, so a metric
added to one and not the other, or a direction changed in one, fails here.
The frame driver's keys and the keys the tools expect are compared on a
scored frame, in both directions.
"""
import dataclasses
import typing
from pathlib import Path

import numpy as np
import pytest

import scenes as S
from evalkit.classes import VIEWS, load_table
from evalkit.frame import KEYS, ViewScores, score_frame
from evalkit.keys import CLIP_METRICS, FRAME_METRICS, SIGNS
from evalkit.tools.scores import EVALUATOR_FIELDS, metric_key, metric_keys

SPEC = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"
HEADING = "## Metrics"

# The words of the table's `better` column, as the signs of `SIGNS`.
BETTER = {"higher": +1, "lower": -1, "reference only": 0}


def spec_table(text: str) -> list[tuple[str, int]]:
    """The keys of the table under `HEADING`, in its order, each with the sign its `better` column gives.

    Raises:
        ValueError: The page has no such heading or no table under it; the
            table has no `key` or no `better` column; or a `better` cell is
            not one of `BETTER`.
    """
    lines = text.split("\n")
    if HEADING not in lines:
        raise ValueError(f"no heading {HEADING!r}")
    table = []
    for line in lines[lines.index(HEADING) + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(table) < 3:
        raise ValueError(f"no metrics table under {HEADING!r}")
    header, rows = table[0], table[2:]
    if "key" not in header or "better" not in header:
        raise ValueError(f"the metrics table has no `key` or no `better` column: {header}")
    key_at, better_at = header.index("key"), header.index("better")
    out = []
    for cells in rows:
        key, better = cells[key_at].strip("`"), cells[better_at]
        if better not in BETTER:
            raise ValueError(f"{key}: the table says better is {better!r}, not one of {list(BETTER)}")
        out.append((key, BETTER[better]))
    return out


def disagreements(
    rows: list[tuple[str, int]], frame_metrics=FRAME_METRICS, clip_metrics=CLIP_METRICS, signs=SIGNS,
) -> list[str]:
    """Where `evalkit.keys` and the specification's table differ; empty when they agree."""
    spec = [key for key, _ in rows]
    ours = list(frame_metrics) + list(clip_metrics)
    out = []
    if sorted(spec) != sorted(ours):
        out.append(f"the table names {sorted(spec)}, evalkit.keys {sorted(ours)}")
    if [key for key in spec if key in frame_metrics] != list(frame_metrics):
        out.append(f"the per-frame metrics are in another order than the table's: {list(frame_metrics)}")
    if set(signs) != set(ours):
        out.append(f"SIGNS covers {sorted(signs)}, not every metric")
    for key, sign in rows:
        if key in signs and signs[key] != sign:
            out.append(f"{key}: the table says {sign:+d}, SIGNS {signs[key]:+d}")
    return out


def unkeyed_metric_fields(cls: type, keys) -> set[str]:
    """The fields of `cls` typed as a metric value that no key reads, plus the keys that read no field."""
    hints = typing.get_type_hints(cls)
    metric_fields = {f.name for f in dataclasses.fields(cls) if hints[f.name] == float | None}
    return metric_fields ^ set(keys.values())


def test_the_keys_are_the_specification_s_table():
    assert not disagreements(spec_table(SPEC.read_text(encoding="utf-8")))


def test_a_missing_row_a_swapped_row_and_a_flipped_direction_are_reported():
    rows = spec_table(SPEC.read_text(encoding="utf-8"))
    without_sq = [r for r in rows if r[0] != "SQ"]
    assert any("the table names" in d for d in disagreements(without_sq))
    swapped = [rows[1], rows[0], *rows[2:]]
    assert any("another order" in d for d in disagreements(swapped))
    flipped = [(key, -sign if key == "mIoU" else sign) for key, sign in rows]
    assert any(d.startswith("mIoU:") for d in disagreements(flipped))
    unmarked = [(key, +1 if key == "time_IoU" else sign) for key, sign in rows]
    assert any(d.startswith("time_IoU:") for d in disagreements(unmarked))


def test_a_metric_without_a_direction_is_reported():
    rows = spec_table(SPEC.read_text(encoding="utf-8"))
    signs = {key: sign for key, sign in SIGNS.items() if key != "time_IoU"}
    assert any("SIGNS covers" in d for d in disagreements(rows, signs=signs))


def test_a_page_without_a_readable_table_is_refused():
    with pytest.raises(ValueError, match="no heading"):
        spec_table("# evaluation\n\nno table here\n")
    with pytest.raises(ValueError, match="no metrics table"):
        spec_table(f"{HEADING}\n\ntext, and no table\n\n## next\n")
    with pytest.raises(ValueError, match="no `better` column"):
        spec_table(f"{HEADING}\n\n| key | pilot |\n|---|---|\n| `SQ` | `SQ` |\n")
    with pytest.raises(ValueError, match="not one of"):
        spec_table(f"{HEADING}\n\n| key | better |\n|---|---|\n| `SQ` | larger |\n")


def test_every_metric_field_of_a_view_is_read_by_one_key():
    assert not unkeyed_metric_fields(ViewScores, KEYS)


def test_a_metric_field_no_key_reads_is_reported():
    @dataclasses.dataclass
    class Planted:
        f1_50: float | None
        extra: float | None

    assert unkeyed_metric_fields(Planted, {"F1_50": "f1_50"}) == {"extra"}
    assert unkeyed_metric_fields(Planted, {"F1_50": "f1_50", "extra": "extra", "SQ": "sq"}) == {"sq"}


def test_the_tools_expect_exactly_the_keys_the_frame_driver_writes():
    table = load_table("atlas120k")
    ids = np.array([0, 12, 14], dtype=np.int32)[S.exact().gt]
    frame = score_frame(ids, S.exact().lab, np.ones((S.H, S.W), dtype=np.float32), table)
    written = [metric_key(key, view) for view in VIEWS for key in frame.views[view].metrics()]
    summary = {field: None for field in EVALUATOR_FIELDS} | {"views": list(VIEWS)}
    assert metric_keys(summary) == written + list(CLIP_METRICS)
