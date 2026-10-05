"""Which depth values count as a measurement: finite and above `DEPTH_MIN`.

The one definition for the geometry. The evaluator keeps an equal
`DEPTH_MIN` of its own, since it imports no other package, and
`tests/test_depth_min_agrees.py` keeps the two in step. numpy only.
"""

import numpy as np

# The smallest depth that counts. Depth 0 and NaN mark a pixel with no depth.
DEPTH_MIN = 1e-6


def valid_depth_mask(depth):
    """The bool mask of the pixels whose depth is finite and above `DEPTH_MIN`."""
    return np.isfinite(depth) & (depth > DEPTH_MIN)
