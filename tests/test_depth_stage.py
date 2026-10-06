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
    """Return a fixed depth ramp at (H, W), confidence 1, and a camera that moves along x.

    `returns` makes it answer with that many frames, whatever the number of images.
    """

    model_id = "stand-in"

    def __init__(self, shape=(6, 8), zero_frame=None, returns=None):
        self.shape, self.zero_frame, self.returns = shape, zero_frame, returns
        self.seen = []

    def reconstruct(self, image_paths):
        self.seen = [p.name for p in image_paths]
        n, (h, w) = (self.returns if self.returns is not None else len(image_paths)), self.shape
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


@pytest.mark.parametrize("returns", [2, 4])
def test_a_model_that_returns_other_than_one_frame_per_image_is_refused_before_anything_is_written(tmp_path, returns):
    clip = make_clip(tmp_path, n=3)
    with pytest.raises(ValueError, match=f"returned {returns} frames for 3 images"):
        run_depth(clip, StandIn(returns=returns), process_res=504, write_glb=False)
    assert not (clip / "depth_raw").exists()
    assert "depth_info" not in json.loads((clip / "frame_manifest.json").read_text())


def test_a_clip_that_already_holds_the_output_is_refused_and_the_message_names_what_is_there(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path)
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    before = sorted(p.relative_to(clip) for p in clip.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="already holds .*depth_raw.*results.npz with keys .*depth_info with keys"):
        run_depth(clip, StandIn(), process_res=504, write_glb=False)
    assert sorted(p.relative_to(clip) for p in clip.rglob("*") if p.is_file()) == before


def test_another_version_of_the_stage_left_its_mark_and_the_refusal_shows_it(tmp_path):
    clip = make_clip(tmp_path)
    manifest = json.loads((clip / "frame_manifest.json").read_text())
    manifest["depth_info"] = {"model": "x", "backproject_mode": "ray", "ray_map_available": True}
    (clip / "frame_manifest.json").write_text(json.dumps(manifest))
    (clip / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(clip / "exports" / "mini_npz" / "results.npz", depth=np.ones((3, 6, 8)), ray_map=np.ones((3, 6, 8, 3)))
    with pytest.raises(ValueError, match="'ray_map'.*'backproject_mode'"):
        run_depth(clip, StandIn(), process_res=504, write_glb=False)


def test_overwrite_removes_the_earlier_run_so_that_no_stale_frame_stays(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path, n=3)
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    (clip / "pc_vis").mkdir()
    (clip / "pc_vis" / "frame_0002.glb").write_bytes(b"stale")
    (clip / "input_images" / "000002.png").unlink()
    (clip / "input_images" / "000001.png").unlink()
    run_depth(clip, StandIn(), process_res=504, write_glb=False, overwrite=True)
    assert [p.name for p in (clip / "depth_raw").iterdir()] == ["depth_000000.npy"]
    assert [p.name for p in (clip / "depth_vis").iterdir()] == ["0000.jpg"]
    assert not (clip / "pc_vis").exists()
    with np.load(clip / "exports" / "mini_npz" / "results.npz") as z:
        assert z["depth"].shape == (1, 6, 8)
    assert json.loads((clip / "frame_manifest.json").read_text())["depth_info"]["depth_shape"] == [1, 6, 8]


def test_a_run_that_fails_after_overwrite_was_asked_leaves_the_earlier_output(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path, n=3)
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    before = sorted(p.relative_to(clip) for p in clip.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="returned 2 frames"):
        run_depth(clip, StandIn(returns=2), process_res=504, write_glb=False, overwrite=True)
    assert sorted(p.relative_to(clip) for p in clip.rglob("*") if p.is_file()) == before
