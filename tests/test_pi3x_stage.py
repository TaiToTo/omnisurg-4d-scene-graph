"""Check the Pi3X stage: its round-trip check, its manifest merge, and runs through a stand-in model.

The round-trip check is tested on an exact pinhole camera, where the right
answer is known, and on the same data with the pose inverted. A run's point
clouds are read back and compared with the stand-in's world points. Each
refusal is planted, and the refused clip must be left as it was. Writing the
images and the point clouds needs the `render` extra; those tests skip
without it.
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

import pipeline.pi3x
from pipeline.pi3x import ROUNDTRIP_MIN_RATIO, main, read_manifest, run_pi3x, update_manifest, verify_roundtrip
from recon3d_wrapper import Reconstruction
from recon3d_wrapper.pi3x import c2w_to_extrinsics

# glTF's axes are the world's with y and z flipped.
GLTF = np.array([1.0, -1.0, -1.0])


def pinhole(n=3, h=24, w=32, seed=0):
    """Return a reconstruction from an exact pinhole camera that turns and moves, and its camera-to-world poses."""
    rng = np.random.default_rng(seed)
    K = np.array([[40.0, 0, (w - 1) / 2], [0, 40.0, (h - 1) / 2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    poses = np.tile(np.eye(4), (n, 1, 1))
    depth = np.zeros((n, h, w), np.float32)
    points = np.zeros((n, h, w, 3), np.float32)
    for i in range(n):
        a = 0.3 * i
        poses[i, :3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        poses[i, :3, 3] = [0.4 * i, 0.1 * i, 0.2 * i]
        z = 5.0 + 2.0 * np.sin(u / 5.0) + rng.normal(scale=0.05, size=(h, w))
        cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)
        depth[i] = z
        points[i] = cam @ poses[i, :3, :3].T + poses[i, :3, 3]
    rec = Reconstruction(depth=depth, conf=np.ones((n, h, w), np.float32),
                         intrinsics=np.repeat(K[None].astype(np.float32), n, axis=0),
                         extrinsics=c2w_to_extrinsics(poses), points=points)
    return rec, poses


def inverted(rec, poses):
    """Return the same reconstruction with the camera-to-world poses taken as extrinsics, uninverted."""
    return Reconstruction(depth=rec.depth, conf=rec.conf, intrinsics=rec.intrinsics,
                          extrinsics=poses[:, :3, :].astype(np.float32), points=rec.points)


# ------------------------------------------------------------- the round-trip check


def test_an_exact_pinhole_round_trips_and_the_inverted_control_is_far():
    out = verify_roundtrip(pinhole()[0])
    assert out["max_rel"] < 1e-6 and out["p999_rel"] < 1e-6 and out["frames_checked"] == 3
    assert out["p999_rel_if_pose_inverted"] > 1e-2


def test_an_inverted_pose_fails_the_ratio():
    out = verify_roundtrip(inverted(*pinhole()))
    assert out["p999_rel"] > 1e-2
    assert out["p999_rel_if_pose_inverted"] / out["p999_rel"] < ROUNDTRIP_MIN_RATIO


def test_the_control_is_reduced_like_the_error():
    # The gate is a ratio, so both sides must be the same statistic of the same norm.
    rec, poses = pinhole(n=4, h=40, w=52, seed=11)
    assert verify_roundtrip(inverted(rec, poses))["p999_rel"] == pytest.approx(
        verify_roundtrip(rec)["p999_rel_if_pose_inverted"], rel=1e-6)


def test_unusable_depth_is_left_out_and_none_at_all_is_refused():
    rec, _ = pinhole()
    rec.depth[0, :2, :] = np.nan
    rec.depth[1, 3, 4] = -1.0
    rec.points[0, :2, :] = np.nan
    out = verify_roundtrip(rec)
    assert np.isfinite(out["max_rel"]) and out["max_rel"] < 1e-6
    rec.depth[:] = 0.0
    with pytest.raises(ValueError, match="no pixel"):
        verify_roundtrip(rec)


@pytest.mark.parametrize("part, index", [("points", (1, 5, 5)), ("extrinsics", 2), ("intrinsics", 0)])
def test_an_error_that_is_not_finite_is_refused(part, index):
    # NaN percentiles would pass both of the stage's comparisons, and the manifest would hold NaN.
    rec, _ = pinhole()
    getattr(rec, part)[index] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        verify_roundtrip(rec)


def test_a_reconstruction_without_world_points_is_refused():
    rec, _ = pinhole()
    with pytest.raises(ValueError, match="no world points"):
        verify_roundtrip(Reconstruction(depth=rec.depth, conf=rec.conf, intrinsics=rec.intrinsics,
                                        extrinsics=rec.extrinsics))


# ------------------------------------------------------------- the manifest merge


PROV = {"model": "pi3x", "vo": False}


def _manifest(tmp_path, n):
    (tmp_path / "frame_manifest.json").write_text(json.dumps({
        "video_id": "VID01_test", "n_frames": n,
        "frames": [{"seq_idx": i, "native_frame": 100 + i, "glb_centroid": [float(i), 0.0, 0.0]} for i in range(n)]}))
    return tmp_path


def _geometry(indices):
    return {i: {"glb_centroid": [9.0, float(i), 0.0], "n_vertices": 10} for i in indices}


def test_the_merge_nests_under_the_source_and_leaves_da3_keys_alone(tmp_path):
    clip = _manifest(tmp_path, 3)
    before = json.loads((clip / "frame_manifest.json").read_text())
    update_manifest(clip, read_manifest(clip, 3), "pi3x", _geometry(range(3)), PROV)
    after = json.loads((clip / "frame_manifest.json").read_text())
    assert after["geometry_sources"]["pi3x"] == PROV
    for old, new in zip(before["frames"], after["frames"]):
        assert {k: new[k] for k in old} == old
        assert new["geometry_sources"]["pi3x"]["glb_centroid"] != old["glb_centroid"]
    update_manifest(clip, read_manifest(clip, 3), "other", _geometry(range(3)), {"model": "other"})
    assert set(json.loads((clip / "frame_manifest.json").read_text())["geometry_sources"]) == {"pi3x", "other"}


# ------------------------------------------------------------- a run


class StandIn:
    """Return the pinhole reconstruction for as many frames as given, with confidence that rises along x.

    Args:
        invert: return the poses uninverted.
        zero_frame: the frame whose depth is emptied.
        hole: the number of columns emptied at the left of frame 0.
        returns: the number of frames returned, whatever the number of images.
        nan_point: put NaN in one world point where the depth is usable.
    """

    model_id = "stand-in"
    device = "cpu"

    def __init__(self, invert=False, zero_frame=None, hole=0, returns=None, nan_point=False):
        self.invert, self.zero_frame, self.hole, self.returns = invert, zero_frame, hole, returns
        self.nan_point = nan_point
        self.calls = 0

    def reconstruct(self, image_paths):
        self.calls += 1
        rec, poses = pinhole(n=self.returns if self.returns is not None else len(image_paths))
        rec.conf[:] = np.linspace(0.05, 0.95, rec.conf.shape[2], dtype=np.float32)
        if self.zero_frame is not None:
            rec.depth[self.zero_frame] = 0.0
        rec.depth[0, :, :self.hole] = 0.0
        if self.nan_point:
            rec.points[1, 5, 5] = np.nan
        return inverted(rec, poses) if self.invert else rec


def colour(i):
    """Return frame i's colour: its three channels differ, and so do any two frames'."""
    return (10 + i, 100 + i, 200 + i)


def _clip(tmp_path, n=3, earlier_run=False, name="clip"):
    """Make a clip with DA3's cloud of frame 0, and with what an earlier run of this stage left, if asked."""
    c = tmp_path / name
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((12, 16, 3), colour(i), np.uint8)).save(c / "input_images" / f"{i:06d}.png")
    (c / "frame_manifest.json").write_text(json.dumps(
        {"n_frames": n, "frames": [{"seq_idx": i, "glb_centroid": [0.0, 0.0, 0.0]} for i in range(n)]}, indent=2))
    (c / "pc_vis").mkdir()
    (c / "pc_vis" / "frame_0000.glb").write_bytes(b"da3")
    if earlier_run:
        (c / "pc_vis" / "frame_0007__pi3x.glb").write_bytes(b"stale")
        m = json.loads((c / "frame_manifest.json").read_text())
        m["geometry_sources"] = {"pi3x": {"model": "pi3x", "vo": True}}
        m["frames"][1]["geometry_sources"] = {"pi3x": {"glb_centroid": [9.0, 9.0, 9.0]}}
        (c / "frame_manifest.json").write_text(json.dumps(m, indent=2))
    return c


def read_cloud(path):
    """Read a GLB point cloud back: its vertices and its RGB colours."""
    trimesh = pytest.importorskip("trimesh")
    cloud = trimesh.load(path, force="scene").to_geometry()
    return np.asarray(cloud.vertices), np.asarray(cloud.colors)[:, :3]


def test_a_run_writes_the_source_files_and_the_manifest_records(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path)
    run_pi3x(clip, StandIn(), pixel_limit=255_000)
    assert (clip / "exports" / "mini_npz" / "results__pi3x.npz").is_file()
    assert sorted(p.name for p in (clip / "depth_vis").iterdir()) == [f"{i:04d}__pi3x.jpg" for i in range(3)]
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == ["frame_0000.glb"] + [
        f"frame_{i:04d}__pi3x.glb" for i in range(3)], "DA3's cloud stays"
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert list(m["geometry_sources"]["pi3x"]) == [
        "model", "vo", "vo_fixed_scale", "pixel_limit", "conf_thre", "resolution", "n_frames", "median_vertices",
        "runtime_sec", "conf_coverage", "depth_range", "roundtrip"]
    assert list(m["frames"][1]["geometry_sources"]["pi3x"]) == [
        "glb_centroid", "n_vertices", "camera_pos_glb", "camera_forward_glb", "camera_up_glb"]
    assert m["frames"][0]["glb_centroid"] == [0.0, 0.0, 0.0]
    assert m["n_frames"] == 3


def test_a_cloud_holds_the_frames_world_points_in_gltf_and_its_images_colours(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path)
    run_pi3x(clip, StandIn(hole=4), pixel_limit=255_000)
    rec, poses = pinhole()
    m = json.loads((clip / "frame_manifest.json").read_text())
    for i in range(3):
        # The expected points are the stand-in's world points, less the hole. The pinhole computed them, not the
        # stage's functions.
        usable = np.ones((24, 32), bool)
        if i == 0:
            usable[:, :4] = False
        expected = rec.points[i][usable] * GLTF
        vertices, colours = read_cloud(clip / "pc_vis" / f"frame_{i:04d}__pi3x.glb")
        record = m["frames"][i]["geometry_sources"]["pi3x"]
        assert record["glb_centroid"] == pytest.approx(expected.mean(0), abs=1e-4)
        assert record["n_vertices"] == len(vertices) == len(expected)
        assert vertices + record["glb_centroid"] == pytest.approx(expected, abs=1e-4)
        assert (colours == colour(i)).all(), "the frame's own image, as RGB"
        # The camera is at the pose's translation and looks along the pose's z.
        assert record["camera_pos_glb"] == pytest.approx(poses[i, :3, 3] * GLTF, abs=1e-5)
        assert record["camera_forward_glb"] == pytest.approx(poses[i, :3, 2] * GLTF, abs=1e-5)


def test_the_summaries_cover_every_pixel_with_usable_depth_and_conf_thre_drops_from_the_clouds(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path)
    run_pi3x(clip, StandIn(hole=4), pixel_limit=255_000, conf_thre=0.5)
    rec, _ = pinhole()
    conf = np.linspace(0.05, 0.95, 32)
    usable = np.ones((3, 24, 32), bool)
    usable[0, :, :4] = False
    m = json.loads((clip / "frame_manifest.json").read_text())
    run = m["geometry_sources"]["pi3x"]
    assert run["conf_coverage"]["0.5"] == pytest.approx((np.broadcast_to(conf, usable.shape)[usable] > 0.5).mean())
    assert run["depth_range"] == pytest.approx([rec.depth[usable].min(), rec.depth[usable].max()])
    for i in range(3):
        vertices, _ = read_cloud(clip / "pc_vis" / f"frame_{i:04d}__pi3x.glb")
        assert len(vertices) == 24 * (conf > 0.5).sum(), "the clouds keep the confident pixels only"


def test_max_points_samples_each_cloud_the_same_way_every_run(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clips = [_clip(tmp_path, name=name) for name in ("a", "b")]
    for clip in clips:
        run_pi3x(clip, StandIn(), pixel_limit=255_000, max_points=40)
    rec, _ = pinhole()
    m = json.loads((clips[0] / "frame_manifest.json").read_text())
    assert m["geometry_sources"]["pi3x"]["median_vertices"] == 40
    for i in range(3):
        vertices, _ = read_cloud(clips[0] / "pc_vis" / f"frame_{i:04d}__pi3x.glb")
        assert len(vertices) == m["frames"][i]["geometry_sources"]["pi3x"]["n_vertices"] == 40
        absolute = vertices + m["frames"][i]["geometry_sources"]["pi3x"]["glb_centroid"]
        distance = np.linalg.norm(absolute[:, None] - (rec.points[i].reshape(-1, 3) * GLTF)[None], axis=-1)
        assert distance.min(axis=1).max() < 1e-4, "every point is one of the frame's"
        name = f"frame_{i:04d}__pi3x.glb"
        assert (clips[0] / "pc_vis" / name).read_bytes() == (clips[1] / "pc_vis" / name).read_bytes()


def test_a_clip_that_holds_this_stages_output_is_refused_before_the_model_runs(tmp_path):
    clip = _clip(tmp_path, earlier_run=True)
    model = StandIn()
    with pytest.raises(ValueError, match=r"already holds .*pc_vis \(1 file\).*vo=True.*on 1 frame"):
        run_pi3x(clip, model, pixel_limit=255_000)
    assert model.calls == 0
    assert (clip / "pc_vis" / "frame_0007__pi3x.glb").read_bytes() == b"stale"


def test_a_clip_that_holds_only_da3s_output_is_not_refused(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path)
    (clip / "depth_vis").mkdir()
    (clip / "depth_vis" / "0000.jpg").write_bytes(b"da3")
    run_pi3x(clip, StandIn(), pixel_limit=255_000)
    assert (clip / "depth_vis" / "0000.jpg").read_bytes() == b"da3"
    assert (clip / "pc_vis" / "frame_0000.glb").read_bytes() == b"da3"


def test_overwrite_removes_the_earlier_runs_clouds_and_records(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path, earlier_run=True)
    run_pi3x(clip, StandIn(zero_frame=1), pixel_limit=255_000, overwrite=True)
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == [
        "frame_0000.glb", "frame_0000__pi3x.glb", "frame_0002__pi3x.glb"], "DA3's cloud stays, the stale one goes"
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert m["geometry_sources"]["pi3x"]["vo"] is False
    assert "geometry_sources" not in m["frames"][1], "a frame with no cloud this run keeps no record of the last"


# ------------------------------------------------------------- what the stage refuses


def edit_manifest(clip, edit):
    p = clip / "frame_manifest.json"
    m = json.loads(p.read_text())
    edit(m)
    p.write_text(json.dumps(m, indent=2))


def without_images(clip):
    for p in (clip / "input_images").iterdir():
        p.unlink()


def without_a_manifest(clip):
    (clip / "frame_manifest.json").unlink()


def with_an_image_gained(clip):
    Image.fromarray(np.zeros((12, 16, 3), np.uint8)).save(clip / "input_images" / "000003.png")


def with_a_duplicate_seq_idx(clip):
    # The set of values still has three members; only the list shows the duplicate.
    edit_manifest(clip, lambda m: (m["frames"].insert(1, {"seq_idx": 0}), m.update(n_frames=4)))


def with_a_seq_idx_skipped(clip):
    # The count still fits; only the values show that frame 2 has no entry.
    edit_manifest(clip, lambda m: m["frames"][2].update(seq_idx=5))


def with_unreadable_image(clip):
    (clip / "input_images" / "000001.png").write_bytes(b"not a png")


# Each case: what to plant in the clip, the stand-in's settings, the run's settings, the refusal.
BEFORE_THE_MODEL = [
    (without_images, {}, {}, FileNotFoundError, "no input_images"),
    (without_a_manifest, {}, {}, FileNotFoundError, "frame_manifest.json"),
    (lambda clip: edit_manifest(clip, lambda m: m["frames"][1].pop("seq_idx")), {}, {}, ValueError, "no seq_idx"),
    (lambda clip: edit_manifest(clip, lambda m: m.update(frames=[])), {}, {}, ValueError, "0 seq_idx values"),
    (lambda clip: edit_manifest(clip, lambda m: m["frames"].pop()), {}, {}, ValueError, "2 seq_idx values"),
    (with_an_image_gained, {}, {}, ValueError, "holds 4 frames, but the manifest's 3"),
    (with_a_duplicate_seq_idx, {}, {}, ValueError, r"not 0 to 2 once each: \[0, 0, 1, 2\]"),
    (with_a_seq_idx_skipped, {}, {}, ValueError, r"not 0 to 2 once each: \[0, 1, 5\]"),
    (lambda clip: edit_manifest(clip, lambda m: m.update(n_frames=4)), {}, {}, ValueError, "n_frames is 4"),
    (lambda clip: edit_manifest(clip, lambda m: m.pop("n_frames")), {}, {}, ValueError, "n_frames is None"),
]

AFTER_THE_MODEL = [
    (None, {"returns": 2}, {}, ValueError, "returned 2 of 3 frames"),
    (None, {"invert": True}, {}, ValueError, "inverted"),
    (None, {}, {"roundtrip_tol": 1e-12}, ValueError, "exceeds 1.0e-12"),
    (None, {"nan_point": True}, {}, ValueError, "not finite"),
    (with_unreadable_image, {}, {}, ValueError, "cannot read .*000001.png"),
    (None, {}, {"conf_thre": 0.99}, ValueError, "no frame keeps a pixel"),
]

REFUSALS = [pytest.param(*case, 0, id=case[-1]) for case in BEFORE_THE_MODEL] + [
    pytest.param(*case, 1, id=case[-1]) for case in AFTER_THE_MODEL]


def snapshot(clip):
    return {str(p.relative_to(clip)): p.read_bytes() for p in sorted(clip.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("plant, model_kwargs, run_kwargs, exc, match, model_calls", REFUSALS)
def test_a_refused_clip_is_left_as_it_was_and_keeps_the_earlier_runs_output(
        tmp_path, plant, model_kwargs, run_kwargs, exc, match, model_calls):
    clip = _clip(tmp_path, earlier_run=True)
    if plant is not None:
        plant(clip)
    model = StandIn(**model_kwargs)
    before = snapshot(clip)
    with pytest.raises(exc, match=match):
        run_pi3x(clip, model, pixel_limit=255_000, overwrite=True, **run_kwargs)
    assert snapshot(clip) == before
    assert model.calls == model_calls


# ------------------------------------------------------------- the command


def run_main(monkeypatch, root, *args, model=None):
    """Run the command on `root` with a stand-in in place of Pi3X, or with a Pi3X that must not be built."""
    def build(**kwargs):
        if model is None:
            raise AssertionError("the model was built")
        return model
    monkeypatch.setattr(pipeline.pi3x, "Pi3X", build)
    monkeypatch.setattr(sys, "argv", ["pipeline.pi3x", "--input-dir", str(root), *args])
    main()


def test_a_missing_input_dir_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no such directory"):
        run_main(monkeypatch, tmp_path / "nowhere")


def test_a_mistyped_clip_name_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    _clip(tmp_path)
    with pytest.raises(SystemExit, match=r"no clip \(a directory with frame_manifest.json\) at: clpi"):
        run_main(monkeypatch, tmp_path, "--clips", "clip", "clpi")
    assert not (tmp_path / "clip" / "exports").exists()


def test_a_directory_without_clips_stops_the_run_before_the_model_loads(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no clip under"):
        run_main(monkeypatch, tmp_path)


def test_a_device_without_a_precision_stops_the_run_before_the_model_loads(tmp_path, monkeypatch, capsys):
    _clip(tmp_path)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--device", "mps")
    assert "invalid choice: 'mps'" in capsys.readouterr().err


class OutOfMemory(StandIn):
    """Raise as a GPU that runs out of memory does, on the clip named `on`."""

    def __init__(self, on):
        super().__init__()
        self.on = on

    def reconstruct(self, image_paths):
        if image_paths[0].parent.parent.name == self.on:
            raise RuntimeError("CUDA out of memory")
        return super().reconstruct(image_paths)


def test_a_clip_that_fails_does_not_stop_the_others_and_the_run_exits_non_zero(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    for name in ("clip_a", "clip_b", "clip_c"):
        _clip(tmp_path, name=name)
    # Running out of memory is neither an OSError nor a ValueError; the run still goes on to the next clip.
    with pytest.raises(SystemExit, match=r"1 of 3 clip\(s\) failed: clip_b"):
        run_main(monkeypatch, tmp_path, model=OutOfMemory("clip_b"))
    assert "failed: RuntimeError: CUDA out of memory" in capsys.readouterr().out
    for name in ("clip_a", "clip_c"):
        assert (tmp_path / name / "exports" / "mini_npz" / "results__pi3x.npz").is_file()
    assert not (tmp_path / "clip_b" / "exports").exists()
