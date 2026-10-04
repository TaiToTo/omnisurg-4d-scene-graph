"""The keys of `evalkit.keys` against the specification, the frame driver and the tools.

The metrics table of `docs/evaluation.md` is read, not copied, so a metric
added to one and not the other fails here. The frame driver's keys and the
keys the tools expect are compared on a scored frame, in both directions.
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


def spec_table(text: str) -> list[tuple[str, bool]]:
    """The keys of the table under `HEADING`, in its order, each with whether its row says "reference only".

    Raises:
        ValueError: The page has no such heading, or no table under it.
    """
    lines = text.split("\n")
    rows = []
    for line in lines[lines.index(HEADING) + 1:]:
        if line.startswith("#"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 1 and cells[1].startswith("`"):
            rows.append((cells[1].strip("`"), "reference only" in cells[0]))
    if not rows:
        raise ValueError(f"no metrics table under {HEADING!r}")
    return rows


def disagreements(rows: list[tuple[str, bool]]) -> list[str]:
    """Where `evalkit.keys` and the specification's table differ; empty when they agree."""
    spec = [key for key, _ in rows]
    ours = list(FRAME_METRICS) + list(CLIP_METRICS)
    out = []
    if sorted(spec) != sorted(ours):
        out.append(f"the table names {sorted(spec)}, evalkit.keys {sorted(ours)}")
    if [key for key in spec if key in FRAME_METRICS] != list(FRAME_METRICS):
        out.append(f"the per-frame metrics are in another order than the table's: {list(FRAME_METRICS)}")
    if set(SIGNS) != set(ours):
        out.append(f"SIGNS covers {sorted(SIGNS)}, not every metric")
    for key, reference in rows:
        if key in SIGNS and (SIGNS[key] == 0) != reference:
            out.append(f"{key}: the table says reference only = {reference}, its sign is {SIGNS[key]}")
    return out


def unkeyed_metric_fields(cls: type, keys) -> set[str]:
    """The fields of `cls` typed as a metric value that no key reads, plus the keys that read no field."""
    hints = typing.get_type_hints(cls)
    metric_fields = {f.name for f in dataclasses.fields(cls) if hints[f.name] == float | None}
    return metric_fields ^ set(keys.values())


def test_the_keys_are_the_specification_s_table():
    assert not disagreements(spec_table(SPEC.read_text(encoding="utf-8")))


def test_a_missing_row_a_swapped_row_and_a_lost_reference_mark_are_reported():
    rows = spec_table(SPEC.read_text(encoding="utf-8"))
    without_sq = [r for r in rows if r[0] != "SQ"]
    assert any("the table names" in d for d in disagreements(without_sq))
    swapped = [rows[1], rows[0], *rows[2:]]
    assert any("another order" in d for d in disagreements(swapped))
    unmarked = [(key, False) for key, _ in rows]
    assert any(d.startswith("time_IoU:") for d in disagreements(unmarked))


def test_a_page_without_the_table_is_refused():
    with pytest.raises(ValueError):
        spec_table("# evaluation\n\nno table here\n")
    with pytest.raises(ValueError, match="no metrics table"):
        spec_table(f"{HEADING}\n\ntext, and no table\n\n## next\n")


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
