"""Which depth values count as a measurement: finite and above `DEPTH_MIN`.

The one definition for the geometry. The evaluator has an equal
`DEPTH_MIN` of its own, and `tests/test_depth_min_agrees.py` fails if the
two differ. numpy only.
"""

import numpy as np

# The smallest depth that counts. Depth 0 and NaN mark a pixel with no depth.
DEPTH_MIN = 1e-6


def valid_depth_mask(depth):
    """The bool mask of the pixels whose depth is finite and above `DEPTH_MIN`."""
    return np.isfinite(depth) & (depth > DEPTH_MIN)
