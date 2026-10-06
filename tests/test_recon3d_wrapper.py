"""Check that the reconstruction interface refuses a malformed sequence and that DA3 refuses a late GPU choice.

Neither test needs torch or weights.
"""

import sys

import numpy as np
import pytest

from recon3d_wrapper import Reconstruction
from recon3d_wrapper.da3 import DA3


@pytest.mark.parametrize("field, value", [
    ("conf", np.ones((2, 6, 9))), ("intrinsics", np.ones((2, 4, 4))), ("extrinsics", np.ones((2, 4, 4))),
    ("depth", np.ones((2, 6))),
])
def test_a_reconstruction_that_is_not_one_sequence_is_refused(field, value):
    parts = {"depth": np.ones((2, 6, 8)), "conf": np.ones((2, 6, 8)), "intrinsics": np.ones((2, 3, 3)),
             "extrinsics": np.ones((2, 3, 4))}
    parts[field] = value
    with pytest.raises(ValueError, match="not one sequence"):
        Reconstruction(**parts)


def test_a_gpu_chosen_after_torch_is_imported_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="already imported"):
        DA3(gpu=0)
