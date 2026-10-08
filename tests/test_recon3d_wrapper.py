"""Check the reconstruction interface and the DA3 wrapper's own logic, without weights.

The interface refuses shapes that are not one sequence. `select_gpu`
writes the GPU into the environment. The DA3 wrapper cuts homogeneous
extrinsics and picks the device.

The torch test skips without the `recon3d` extra; the install job runs it.
"""

import os
import sys

import numpy as np
import pytest

from recon3d_wrapper import Reconstruction, select_gpu
from recon3d_wrapper.da3 import DA3, world_to_camera


def parts(n=2, h=6, w=8):
    return {"depth": np.ones((n, h, w)), "conf": np.ones((n, h, w)), "intrinsics": np.ones((n, 3, 3)),
            "extrinsics": np.ones((n, 3, 4))}


@pytest.mark.parametrize("wrong, match", [
    ({"conf": np.ones((2, 6, 9))}, r"conf \(2, 6, 9\), not \(2, 6, 8\)"),
    ({"intrinsics": np.ones((2, 4, 4))}, r"intrinsics \(2, 4, 4\), not \(2, 3, 3\)"),
    ({"extrinsics": np.ones((2, 4, 4))}, r"extrinsics \(2, 4, 4\), not \(2, 3, 4\)"),
    ({"depth": np.ones((2, 6))}, r"depth \(2, 6\), not \(N, H, W\)"),
    ({"depth": np.ones((2, 6)), "conf": np.ones((2, 6))}, r"depth \(2, 6\), not \(N, H, W\)"),
    ({"depth": np.ones(())}, r"depth \(\), not \(N, H, W\)"),
    ({"depth": np.ones((0, 6, 8)), "conf": np.ones((0, 6, 8)), "intrinsics": np.ones((0, 3, 3)),
      "extrinsics": np.ones((0, 3, 4))}, "N above 0"),
])
def test_a_reconstruction_that_is_not_one_sequence_is_refused_and_the_message_says_why(wrong, match):
    with pytest.raises(ValueError, match=match):
        Reconstruction(**{**parts(), **wrong})


def test_a_well_formed_reconstruction_is_kept():
    rec = Reconstruction(**parts())
    assert rec.depth.shape == (2, 6, 8)


def test_homogeneous_extrinsics_are_cut_to_their_top_three_rows():
    E = np.tile(np.eye(4), (3, 1, 1))
    E[:, :3, 3] = np.arange(3)[:, None]
    assert np.array_equal(world_to_camera(E), E[:, :3, :])
    assert world_to_camera(E).shape == (3, 3, 4)


def test_extrinsics_already_three_by_four_pass_as_they_are():
    E = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert world_to_camera(E) is E


def test_extrinsics_that_are_neither_shape_or_not_homogeneous_are_refused():
    with pytest.raises(ValueError, match=r"\(2, 2, 4\), not"):
        world_to_camera(np.ones((2, 2, 4)))
    with pytest.raises(ValueError, match="not"):
        world_to_camera(np.ones((3, 4)))
    bad = np.tile(np.eye(4), (2, 1, 1))
    bad[1, 3, 0] = 0.5
    with pytest.raises(ValueError, match="not homogeneous"):
        world_to_camera(bad)


def test_a_gpu_chosen_before_torch_is_imported_is_written_to_the_environment(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    select_gpu(3)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_no_gpu_choice_leaves_the_environment_as_it_is(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    select_gpu(None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    select_gpu(None)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_gpu_chosen_after_torch_is_imported_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="already imported"):
        DA3(gpu=0)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_auto_takes_the_device_torch_reports_and_puts_the_model_on_it(monkeypatch):
    torch = pytest.importorskip("torch")
    api = pytest.importorskip("depth_anything_3.api")

    class Weights:
        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(api.DepthAnything3, "from_pretrained", staticmethod(lambda model_id: Weights()))
    model = DA3(device="auto", process_res=252)
    assert model.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert model.model.device == model.device
    assert model.process_res == 252
