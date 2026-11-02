"""Hold the rules of `paper/ipcai2027/graph/` that would give a plausible wrong number without failing.

Each test plants the fault: a clip short of the span, a point on the node's own region read as another node's, an
instrument lost over its own class read as its own class, a draw with nothing to divide by, a seed component under
the tracking stage's minimum area, and a component two nodes claim.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evalkit.classes import load_table

GRAPH = Path(__file__).resolve().parent.parent / "paper" / "ipcai2027" / "graph"


@pytest.fixture
def area(monkeypatch):
    """Import the area's modules by name, as running a script by path does, and forget them afterwards."""
    monkeypatch.syspath_prepend(str(GRAPH))
    yield importlib.import_module
    for name, mod in list(sys.modules.items()):
        if str(GRAPH) in str(getattr(mod, "__file__", "") or ""):
            del sys.modules[name]


def test_a_clip_short_of_the_span_is_left_out_and_later_frames_are_cut(area):
    cs = area("count_statements")
    times = np.array([0.0, 0.53, 1.06, 5.3, 5.83])
    assert cs.frames_in_span([0, 1, 2, 3], times, 5.4) is None
    assert cs.frames_in_span([0, 1, 2, 3, 4], times, 5.4) == [0, 1, 2, 3]


def test_a_point_on_the_nodes_own_region_is_not_another_node(area):
    ev = area("exit_events")
    # Node 2's region id is 3; region id 2 is node 1, another node.
    assert ev.node_kind({"node": "2", "region_under": "3"}) == "own_node"
    assert ev.node_kind({"node": "2", "region_under": "2"}) == "another_node"
    assert ev.node_kind({"node": "2", "region_under": "-1"}) == "no_node"


def test_an_instrument_lost_over_its_own_class_counts_as_an_instrument(area):
    ev = area("exit_events")
    table = load_table("atlas120k")
    tool = next(c for c in table.entries if str(table.type_of(c)) == "tool")
    tissue = next(c for c in table.entries if str(table.type_of(c)) == "tissue")
    mask_id = {c: m for m, c in table.mask_id_to_class.items()}
    assert ev.point_kind(table, tool, str(mask_id[tool])) == "instrument"
    assert ev.point_kind(table, tissue, str(mask_id[tissue])) == "own_class"
    assert ev.point_kind(table, tissue, "") == "no_reference"


def test_a_draw_with_nothing_to_divide_by_is_refused(area):
    vb = area("video_bootstrap")
    with pytest.raises(ValueError, match="no finite statistic"):
        vb.interval(np.array([0.5, np.nan, 0.7]))
    with pytest.raises(ValueError, match="two or more videos"):
        vb.resampled_sums(np.array([[1.0, 2.0]]), 10)
    draws = vb.resampled_sums(np.array([[1.0, 2.0], [3.0, 4.0]]), 50)
    assert draws.shape == (50, 2) and set(draws[:, 0]) <= {2.0, 4.0, 6.0}


def _frame(classes: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(classes=classes, mask=classes > 0)


def test_a_seed_component_under_the_minimum_area_is_no_node(area):
    cs = area("component_seeds")
    c = np.zeros((10, 20), dtype=int)
    c[1:5, 1:5] = 2   # 16 px
    c[1:3, 10:12] = 2  # 4 px, under the minimum
    classes, areas, tracks = cs.component_tracks({0: _frame(c)}, c.shape, min_area=10)
    assert classes == [2] and areas == [16]
    assert set(np.unique(tracks[0])) == {-1, 0}


def test_a_component_two_nodes_claim_goes_to_the_larger_overlap(area):
    cs = area("component_seeds")
    seed = np.zeros((10, 30), dtype=int)
    seed[2:8, 2:8] = 4
    seed[2:8, 10:12] = 4
    later = np.zeros_like(seed)
    later[2:8, 3:12] = 4  # one component that overlaps both seed components
    classes, _, tracks = cs.component_tracks({0: _frame(seed), 1: _frame(later)}, seed.shape, min_area=10)
    assert classes == [4, 4]
    assert set(np.unique(tracks[1])) == {-1, 0}
