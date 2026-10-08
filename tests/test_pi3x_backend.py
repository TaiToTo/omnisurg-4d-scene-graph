"""Check the parts of the Pi3X backend that need no model: the pose inversion, the frame loader and the precision.

None of these tests needs torch or the weights.
"""

import sys

import numpy as np
import pytest
from PIL import Image

from recon3d_wrapper.pi3x import PATCH, Pi3X, autocast_dtype, c2w_to_extrinsics, load_images, target_size
from surgical_core.geometry.camera import cam_to_world


def _random_c2w(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        poses[i, :3, :3] = q
        poses[i, :3, 3] = rng.normal(scale=2.0, size=3)
    return poses


# ------------------------------------------------------------- c2w_to_extrinsics


def test_the_extrinsics_undo_the_camera_to_world_pose():
    poses = _random_c2w(6, seed=1)
    E = c2w_to_extrinsics(poses)
    assert E.shape == (6, 3, 4) and E.dtype == np.float32
    cam = np.random.default_rng(2).normal(size=(100, 3)) + [0.0, 0.0, 5.0]
    for i, pose in enumerate(poses):
        world = cam @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(cam_to_world(cam, E[i][:3, :3], E[i][:3, 3]), world, atol=1e-5)
        np.testing.assert_allclose(-E[i][:3, :3].T @ E[i][:3, 3], pose[:3, 3], atol=1e-5)


def test_the_uninverted_reading_does_not_round_trip():
    # Guards the test above: a cam_to_world symmetric in R would pass it for a wrong inversion too.
    poses = _random_c2w(3, seed=3)
    E = c2w_to_extrinsics(poses)
    cam = np.random.default_rng(4).normal(size=(50, 3)) + [0.0, 0.0, 5.0]
    world = cam @ poses[0, :3, :3].T + poses[0, :3, 3]
    assert np.abs(cam @ E[0][:3, :3].T + E[0][:3, 3] - world).max() > 1.0


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros((2, 3, 4)), np.zeros((2, 4, 3))])
def test_poses_of_another_shape_are_refused(bad):
    with pytest.raises(ValueError, match=r"\(N, 4, 4\)"):
        c2w_to_extrinsics(bad)


# ------------------------------------------------------------- the frame loader


def _frames(tmp_path, n, size):
    paths = []
    for i in range(n):
        arr = np.zeros((size[1], size[0], 3), np.uint8)
        arr[..., 0] = i * 10
        paths.append(tmp_path / f"{i:06d}.png")
        Image.fromarray(arr).save(paths[-1])
    return paths


def test_the_loader_keeps_the_order_given(tmp_path):
    # Upstream's loader sorts a directory again; a strided or reordered list would pair frames with other moments.
    paths = _frames(tmp_path, 4, (56, 42))
    out = load_images([paths[2], paths[0], paths[3], paths[1]], pixel_limit=10_000)
    assert [round(float(out[k, ..., 0].mean()) * 255 / 10) for k in range(4)] == [2, 0, 3, 1]
    assert out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.parametrize("size, limit", [((640, 480), 100_000), ((854, 480), 255_000), ((1280, 720), 255_000),
                                         ((320, 240), 255_000)])
def test_the_target_size_is_on_the_patch_grid_and_under_the_budget(size, limit):
    w, h = target_size(*size, limit)
    assert w % PATCH == 0 and h % PATCH == 0 and w * h <= limit
    assert abs(w / h - size[0] / size[1]) < PATCH / min(w, h), "the aspect ratio is kept to within one patch"


def test_no_frame_is_refused():
    with pytest.raises(ValueError, match="empty"):
        load_images([])


def test_a_gpu_chosen_after_torch_is_imported_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="already imported"):
        Pi3X(gpu=0)


@pytest.mark.parametrize("device", ["mps", "cuda:1", ""])
def test_a_device_without_a_precision_is_refused_before_torch_is_imported(monkeypatch, device):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(ValueError, match="is not one of auto, cpu, cuda"):
        Pi3X(device=device)
    assert "torch" not in sys.modules


def test_the_cpu_runs_float32_and_cuda_the_workbenchs_reduced_precision():
    # On the CPU, bfloat16 fails in upstream's camera head: "lu_cpu" is not implemented for BFloat16.
    assert autocast_dtype("cpu") is None
    assert autocast_dtype("cuda", (8, 0)) == "bfloat16"
    assert autocast_dtype("cuda", (7, 5)) == "float16"
    for device, capability in (("mps", None), ("cuda", None)):
        with pytest.raises(ValueError, match="no precision"):
            autocast_dtype(device, capability)
