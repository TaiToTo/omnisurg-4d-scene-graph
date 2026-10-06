"""Check the Pi3X stage: its round-trip check, its manifest merge, and a run through a stand-in model.

The round-trip check is tested on an exact pinhole camera, where the right
answer is known, and on the same data with the pose inverted. The merge is
tested in both directions in which `input_images/` and the manifest can
disagree. The run needs the `render` extra; it skips without it.
"""

import json

import numpy as np
import pytest
from PIL import Image

from pipeline.pi3x import ROUNDTRIP_MIN_RATIO, run_pi3x, update_manifest, verify_roundtrip
from recon3d_wrapper import Reconstruction
from recon3d_wrapper.pi3x import c2w_to_extrinsics


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
    """Return the same reconstruction with the camera-to-world poses passed through uninverted."""
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
    update_manifest(clip, "pi3x", _geometry(range(3)), 3, PROV)
    after = json.loads((clip / "frame_manifest.json").read_text())
    assert after["geometry_sources"]["pi3x"] == PROV
    for old, new in zip(before["frames"], after["frames"]):
        assert {k: new[k] for k in old} == old
        assert new["geometry_sources"]["pi3x"]["glb_centroid"] != old["glb_centroid"]
    update_manifest(clip, "other", _geometry(range(3)), 3, {"model": "other"})
    assert set(json.loads((clip / "frame_manifest.json").read_text())["geometry_sources"]) == {"pi3x", "other"}


@pytest.mark.parametrize("n_images", [4, 6])
def test_a_frame_lost_or_gained_in_input_images_is_refused(tmp_path, n_images):
    # A lost frame leaves the reconstructed keys a subset of the manifest's; only the count catches it.
    clip = _manifest(tmp_path, 5)
    with pytest.raises(ValueError, match="different frame lists"):
        update_manifest(clip, "pi3x", _geometry(range(n_images)), n_images, PROV)


def test_a_manifest_without_seq_idx_or_none_at_all_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="frame_manifest.json"):
        update_manifest(tmp_path, "pi3x", _geometry(range(2)), 2, PROV)
    (tmp_path / "frame_manifest.json").write_text(json.dumps({"frames": [{"native_frame": 100}]}))
    with pytest.raises(ValueError, match="seq_idx"):
        update_manifest(tmp_path, "pi3x", _geometry(range(1)), 1, PROV)


# ------------------------------------------------------------- a run


class StandIn:
    """Return the pinhole reconstruction for as many frames as given."""

    model_id = "stand-in"

    def __init__(self, invert=False):
        self.invert = invert
        self.calls = 0

    def reconstruct(self, image_paths):
        self.calls += 1
        rec, poses = pinhole(n=len(image_paths))
        return inverted(rec, poses) if self.invert else rec


def _clip(tmp_path, n=3):
    c = tmp_path / "clip"
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((12, 16, 3), 40 * i, np.uint8)).save(c / "input_images" / f"{i:06d}.png")
    (c / "frame_manifest.json").write_text(json.dumps(
        {"n_frames": n, "frames": [{"seq_idx": i, "glb_centroid": [0.0, 0.0, 0.0]} for i in range(n)]}, indent=2))
    (c / "pc_vis").mkdir()
    (c / "pc_vis" / "frame_0007__pi3x.glb").write_bytes(b"stale")
    (c / "pc_vis" / "frame_0000.glb").write_bytes(b"da3")
    return c


def test_a_run_writes_the_source_files_and_the_manifest_records(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("trimesh")
    clip = _clip(tmp_path)
    run_pi3x(clip, StandIn(), pixel_limit=255_000)
    assert (clip / "exports" / "mini_npz" / "results__pi3x.npz").is_file()
    assert sorted(p.name for p in (clip / "depth_vis").iterdir()) == [f"{i:04d}__pi3x.jpg" for i in range(3)]
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == ["frame_0000.glb"] + [
        f"frame_{i:04d}__pi3x.glb" for i in range(3)], "a stale cloud of this source goes, DA3's stays"
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert list(m["geometry_sources"]["pi3x"]) == [
        "model", "vo", "vo_fixed_scale", "pixel_limit", "conf_thre", "resolution", "n_frames", "median_vertices",
        "runtime_sec", "conf_coverage", "depth_range", "roundtrip"]
    assert list(m["frames"][1]["geometry_sources"]["pi3x"]) == [
        "glb_centroid", "n_vertices", "camera_pos_glb", "camera_forward_glb", "camera_up_glb"]
    assert m["frames"][0]["glb_centroid"] == [0.0, 0.0, 0.0]


def test_a_run_with_inverted_poses_is_refused_before_anything_is_written(tmp_path):
    clip = _clip(tmp_path)
    with pytest.raises(ValueError, match="inverted"):
        run_pi3x(clip, StandIn(invert=True), pixel_limit=255_000)
    assert not (clip / "exports").exists()


@pytest.mark.parametrize("fault", ["no manifest", "a frame gained"])
def test_a_clip_whose_manifest_does_not_fit_is_refused_before_the_model_runs(tmp_path, fault):
    clip = _clip(tmp_path)
    if fault == "no manifest":
        (clip / "frame_manifest.json").unlink()
    else:
        Image.fromarray(np.zeros((12, 16, 3), np.uint8)).save(clip / "input_images" / "000003.png")
    model = StandIn()
    with pytest.raises((FileNotFoundError, ValueError), match="frame_manifest.json|different frame lists"):
        run_pi3x(clip, model, pixel_limit=255_000)
    assert model.calls == 0
    assert not (clip / "exports").exists() and not (clip / "depth_vis").exists()
