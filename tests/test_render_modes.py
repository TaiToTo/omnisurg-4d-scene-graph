"""`sam_input_image` answers every mode in `SAM_INPUT_MODES` and refuses the rest.

The module needs the `render` extra (matplotlib); without it the tests
skip, and CI runs them in the job that installs the extra.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from surgical_core.geometry import render  # noqa: E402


def _inputs(h=24, w=32):
    yy, xx = np.mgrid[0:h, 0:w]
    depth = 0.3 + 0.002 * xx + 0.001 * yy
    depth[10:14, 12:18] += 0.05                     # a step, so the edge modes have an edge
    K = np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]])
    gray01 = (depth - depth.min()) / (depth.max() - depth.min())
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return depth, K, gray01, rgb


@pytest.mark.parametrize("mode", render.SAM_INPUT_MODES)
def test_every_listed_mode_renders(mode):
    depth, K, gray01, rgb = _inputs()
    img = render.sam_input_image(mode, depth, K, gray01, rgb)
    assert img.shape == (*depth.shape, 3) and img.dtype == np.uint8, mode


def test_unknown_mode_is_refused():
    """A misspelt mode used to come back as colormapped depth, and a whole
    condition could run as depth under another name."""
    depth, K, gray01, rgb = _inputs()
    with pytest.raises(ValueError, match="unknown input mode 'depht'"):
        render.sam_input_image("depht", depth, K, gray01, rgb)


def test_a_mode_burns_edges_in_exactly_when_its_gain_changes_it():
    depth, K, gray01, rgb = _inputs()
    for mode in render.SAM_INPUT_MODES:
        off = render.sam_input_image(mode, depth, K, gray01, rgb, edge_gain=0.0)
        on = render.sam_input_image(mode, depth, K, gray01, rgb, edge_gain=0.85)
        assert render.uses_geom_edge(mode) == (not np.array_equal(off, on)), mode


def test_an_unknown_mode_has_no_answer_on_edges():
    with pytest.raises(ValueError, match="unknown input mode 'depht'"):
        render.uses_geom_edge("depht")


@pytest.mark.parametrize("mode", render.SAM_INPUT_MODES)
def test_each_mode_computes_the_normals_at_most_once(mode, monkeypatch):
    from surgical_core.geometry import normals
    calls = []
    real = normals.camera_normals

    def counting(depth, K):
        calls.append(mode)
        return real(depth, K)

    monkeypatch.setattr(normals, "camera_normals", counting)
    depth, K, gray01, rgb = _inputs()
    render.sam_input_image(mode, depth, K, gray01, rgb)
    assert len(calls) <= 1, f"{mode} computed the normals {len(calls)} times"
