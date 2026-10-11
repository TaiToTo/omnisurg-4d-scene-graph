"""Run `pipeline.depth` through a stand-in model, and check the files it writes and the clips it refuses.

The stand-in returns fixed depth, so these tests need no torch and no
weights. `pipeline.byte_check` checks the real model against the
workbench's stage. Writing the images and the point clouds needs the
`render` extra; those tests skip without it.
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

import pipeline.depth
from pipeline.depth import check_cropped, existing_output, main, run_depth
from recon3d_wrapper import Reconstruction


class StandIn:
    """Return a fixed depth ramp at (H, W), confidence 1, and a camera that moves along x.

    `returns` makes it return that many frames, whatever the number of images.
    """

    model_id = "stand-in"
    device = "cpu"

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


def test_a_stereomis_clip_needs_no_crop_info(tmp_path):
    clip = make_clip(tmp_path, name="P1__clip_0001", dataset="stereomis", cropped=False)
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
    assert list((clip / "pc_vis").iterdir()) == []
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


OTHERS = ("depth_vis/{:04d}__pi3x.jpg", "pc_vis/frame_{:04d}__pi3x.glb", "pc_vis/seg_frame_{:04d}__sam3d.glb",
          "pc_vis/graph_frame_{:04d}.json", "pc_vis/hierarchy_index.json", "exports/mini_npz/results__pi3x.npz")


def plant_others(clip, n=3):
    """Write the files that another depth source and the later stages write into the depth stage's directories.

    They carry a `__<source>` suffix, or names of their own with no suffix.
    """
    for i in range(n):
        for rel in OTHERS:
            path = clip / rel.format(i)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{rel} {i}".encode())
    manifest = json.loads((clip / "frame_manifest.json").read_text())
    manifest["geometry_sources"] = {"pi3x": {"model": "pi3x"}}
    (clip / "frame_manifest.json").write_text(json.dumps(manifest))
    return others_in(clip)


def others_in(clip):
    own = {"depth_raw", "frame_manifest.json", "crop_info.json", "input_images"}
    return sorted((p.relative_to(clip), p.read_bytes()) for p in clip.rglob("*")
                  if p.is_file() and p.parts[len(clip.parts)] not in own
                  and not (p.parent.name == "depth_vis" and p.stem.isdigit())
                  and not (p.parent.name == "pc_vis" and p.name.startswith("frame_") and p.stem[6:].isdigit())
                  and p.name != "results.npz")


def test_a_clip_that_holds_only_other_stages_files_is_not_refused_and_they_stay(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path)
    others = plant_others(clip)
    assert len(others) == 14
    assert existing_output(clip, json.loads((clip / "frame_manifest.json").read_text())) == []
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    assert others_in(clip) == others
    manifest = json.loads((clip / "frame_manifest.json").read_text())
    assert manifest["geometry_sources"] == {"pi3x": {"model": "pi3x"}} and "depth_info" in manifest


def test_overwrite_removes_only_the_stage_s_own_files(tmp_path):
    pytest.importorskip("matplotlib")
    clip = make_clip(tmp_path)
    run_depth(clip, StandIn(), process_res=504, write_glb=False)
    (clip / "pc_vis").mkdir()
    (clip / "pc_vis" / "frame_0000.glb").write_bytes(b"da3 cloud")
    others = plant_others(clip)
    assert existing_output(clip, json.loads((clip / "frame_manifest.json").read_text())) == [
        "depth_raw (3 files)", "depth_vis (3 files)", "pc_vis (1 file)",
        "exports/mini_npz/results.npz with keys ['depth', 'conf', 'extrinsics', 'intrinsics']",
        "depth_info with keys ['model', 'process_res', 'depth_shape', 'depth_range', 'conf_range', 'border_inpaint']"]
    run_depth(clip, StandIn(), process_res=504, write_glb=False, overwrite=True)
    assert others_in(clip) == others
    assert not (clip / "pc_vis" / "frame_0000.glb").exists()


def run_main(monkeypatch, root, *args, model=None):
    """Run the command on `root` with a stand-in in place of DA3, or with a DA3 that must not be built."""
    def build(**kwargs):
        if model is None:
            raise AssertionError("the model was built")
        return model
    monkeypatch.setattr(pipeline.depth, "DA3", build)
    monkeypatch.setattr(sys, "argv", ["pipeline.depth", "--input-dir", str(root), "--no-glb", *args])
    main()


def test_a_missing_input_dir_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no such directory"):
        run_main(monkeypatch, tmp_path / "nowhere")


def test_a_mistyped_clip_name_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    make_clip(tmp_path)
    with pytest.raises(SystemExit, match=r"no clip \(a directory with frame_manifest.json\) at: VID02_s15_80_crop"):
        run_main(monkeypatch, tmp_path, "--clips", "VID01_s15_80_crop", "VID02_s15_80_crop")
    assert not (tmp_path / "VID01_s15_80_crop" / "depth_raw").exists()


def test_a_directory_without_clips_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no clip under"):
        run_main(monkeypatch, tmp_path)


def test_a_clip_that_fails_does_not_stop_the_others_and_the_run_exits_non_zero(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    make_clip(tmp_path, name="VID01_s15_80_crop")
    make_clip(tmp_path, name="VID02_s15_80_crop", cropped=False)
    make_clip(tmp_path, name="VID03_s15_80_crop")
    with pytest.raises(SystemExit, match=r"1 of 3 clip\(s\) failed: VID02_s15_80_crop"):
        run_main(monkeypatch, tmp_path, model=StandIn())
    assert "failed: ValueError" in capsys.readouterr().out
    for name in ("VID01_s15_80_crop", "VID03_s15_80_crop"):
        assert len(list((tmp_path / name / "depth_raw").iterdir())) == 3
    assert not (tmp_path / "VID02_s15_80_crop" / "depth_raw").exists()


def test_every_clip_runs_and_the_command_exits_clean(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    make_clip(tmp_path, name="VID01_s15_80_crop")
    make_clip(tmp_path, name="VID02_s15_80_crop")
    run_main(monkeypatch, tmp_path, model=StandIn())
    for name in ("VID01_s15_80_crop", "VID02_s15_80_crop"):
        assert len(list((tmp_path / name / "depth_raw").iterdir())) == 3
