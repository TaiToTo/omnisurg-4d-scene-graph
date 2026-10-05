"""The geometry and the evaluator agree on which depth values count.

The evaluator imports no other package, so it keeps its own `DEPTH_MIN`.
This test is what makes the two copies safe: a pixel the geometry treats as
having no depth is one the evaluator refuses, and the reverse.
"""
import numpy as np
import pytest

from evalkit.scored import DEPTH_MIN as EVALUATOR_DEPTH_MIN
from evalkit.scored import valid_depth
from surgical_core.geometry import DEPTH_MIN, valid_depth_mask


def test_the_two_thresholds_are_equal():
    assert DEPTH_MIN == EVALUATOR_DEPTH_MIN


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, 5e-7, 1e-6, 2e-6, 1.0, np.nan, np.inf, -np.inf, -1.0])
def test_a_depth_the_geometry_rejects_is_one_the_evaluator_refuses(value, dtype):
    depth = np.full((2, 2), value, dtype=dtype)
    if valid_depth_mask(depth).all():
        assert valid_depth(depth).all()
    else:
        with pytest.raises(ValueError):
            valid_depth(depth)
