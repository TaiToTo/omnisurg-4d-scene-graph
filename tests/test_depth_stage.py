"""Run `pipeline.depth` through a stand-in model, and check the files it writes and the clips it refuses.

The stand-in returns fixed depth, so these tests need no torch and no
weights. `pipeline.byte_check` checks the real model against the
workbench's stage. Writing the images and the point clouds needs the
`render` extra; those tests skip without it.
"""

import json

import numpy as np
import pytest
from PIL import Image

from pipeline.depth import check_cropped, run_depth
from recon3d_wrapper import Reconstruction


class StandIn:
    """Return a fixed depth ramp at (H, W), confidence 1, and a camera that moves along x."""

    model_id = "stand-in"

    def __init__(self, shape=(6, 8), zero_frame=None, n_out=None):
        self.shape, self.zero_frame, self.n_out = shape, zero_frame, n_out
        self.seen = []

    def reconstruct(self, image_paths):
        self.seen = [p.name for p in image_paths]
        n, (h, w) = len(image_paths) if self.n_out is None else self.n_out, self.shape
        depth = np.stack([np.full((h, w), 1.0 + i) + np.linspace(0, 1, w) for i in range(n)]).astype(np.float32)
        if self.zero_frame is not None:
            depth[self.zero_frame] = 0.0
        K = np.tile(np.array([[10.0, 0, w / 2], [0, 10.0, h / 2], [0, 0, 1]], np.float32), (n, 1, 1))
        E = np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1))
        E[:, 0, 3] = np.arange(n)
        return Reconstruction(depth=depth, conf=np.ones_like(depth), intrinsics=K, extrinsics=E)


def make_clip(root, name="VID01_s15_80_crop", n=3, dataset=None, cropped=True):
    c = root / name
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((12, 16, 3), 40 * i, np.uint8)).save(c / "input_images" / f"{i:06d}.png")
    manifest = {"frames": [{"native_frame": i} for i in range(n)]}
    if dataset:
        manifest = {"dataset": dataset, **manifest}
    (c / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))
    if cropped:
        (c / "crop_info.json").write_text(json.dumps({"y0": 0, "y1": 12, "x0": 0, "x1": 16}))
    return c


def test_the_stage_writes_every_file_and_the_manifest_keys_in_order(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path)
    model = StandIn()
    run_depth(clip, model, process_res=504)
    assert model.seen == ["000000.png", "000001.png", "000002.png"]
    assert sorted(p.name for p in (clip / "depth_raw").iterdir()) == [f"depth_{i:06d}.npy" for i in range(3)]
    assert sorted(p.name for p in (clip / "depth_vis").iterdir()) == [f"{i:04d}.jpg" for i in range(3)]
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == [f"frame_{i:04d}.glb" for i in range(3)]
    with np.load(clip / "exports" / "mini_npz" / "results.npz") as z:
        assert z.files == ["depth", "conf", "extrinsics", "intrinsics"]
        assert all(z[k].dtype == np.float32 for k in z.files)
        assert z["depth"].shape == (3, 6, 8)
    info = json.loads((clip / "frame_manifest.json").read_text())["depth_info"]
    assert list(info) == ["model", "process_res", "depth_shape", "depth_range", "conf_range", "border_inpaint"]
    assert info["model"] == "stand-in" and info["process_res"] == 504 and info["depth_shape"] == [3, 6, 8]
    assert info["border_inpaint"] is False
    assert np.array_equal(np.load(clip / "depth_raw" / "depth_000001.npy"), model.reconstruct([clip] * 2).depth[1])


def test_a_frame_with_no_usable_depth_gets_no_point_cloud_and_the_rest_do(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path)
    run_depth(clip, StandIn(zero_frame=1), process_res=504)
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == ["frame_0000.glb", "frame_0002.glb"]
    assert len(list((clip / "depth_raw").iterdir())) == 3


def test_without_point_clouds_nothing_is_written_under_pc_vis(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path)
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    assert not (clip / "pc_vis").exists()
    assert (clip / "exports" / "mini_npz" / "results.npz").is_file()


def test_a_cholecseg8k_clip_not_cut_to_the_view_is_refused(tmp_path):
    clip = make_clip(tmp_path, cropped=False)
    with pytest.raises(ValueError, match="crop_info.json"):
        check_cropped(clip, json.loads((clip / "frame_manifest.json").read_text()))
    with pytest.raises(ValueError, match="crop_info.json"):
        run_depth(clip, StandIn(), process_res=504)
    assert not (clip / "depth_raw").exists(), "refused before anything is written"


def test_an_atlas120k_clip_needs_no_crop_info(tmp_path):
    clip = make_clip(tmp_path, name="adrenalectomy__x__gt_0001", dataset="atlas120k", cropped=False)
    check_cropped(clip, json.loads((clip / "frame_manifest.json").read_text()))


def test_a_clip_without_manifest_or_images_is_refused(tmp_path):
    clip = make_clip(tmp_path)
    for p in (clip / "input_images").iterdir():
        p.unlink()
    with pytest.raises(FileNotFoundError, match="input_images"):
        run_depth(clip, StandIn(), process_res=504)
    (clip / "frame_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="frame_manifest.json"):
        run_depth(clip, StandIn(), process_res=504)


@pytest.mark.parametrize("n_out", [2, 4])
def test_a_model_that_returns_another_number_of_frames_is_refused(tmp_path, n_out):
    clip = make_clip(tmp_path)
    with pytest.raises(ValueError, match=f"returned {n_out} of 3 frames"):
        run_depth(clip, StandIn(n_out=n_out), process_res=504)
    assert not (clip / "depth_raw").exists(), "refused before anything is written"
