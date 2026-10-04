"""`sam_input_image` answers every mode in `SAM_INPUT_MODES` and refuses the rest.

The module needs the `render` extra (matplotlib, scipy); without it the tests
skip, and CI runs them in the job that installs the extra.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

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
