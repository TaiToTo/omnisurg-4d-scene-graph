"""Run `pipeline.point_clouds` on a synthetic depth bundle, and check what it writes and what it refuses.

Writing the point clouds needs trimesh, which the `render` extra installs;
those tests skip without it.
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from pipeline.point_clouds import main, run_point_clouds

# A quarter turn about y: its transpose is another rotation, so a stage that uses R where it should use R.T shows.
QUARTER_TURN_Y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)


def make_clip(tmp_path, n=3, zero_frame=None, seq_idx=True, turn_frame=None, hole=None, name="clip"):
    """A clip of `n` frames: images whose three channels differ, flat depth, and a camera that moves along x.

    `turn_frame` gets the quarter turn as its rotation; `hole` names a (frame, row, column) pixel without depth.
    """
    c = tmp_path / name
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((12, 16, 3), (10 + i, 100 + i, 200 + i), np.uint8)).save(
            c / "input_images" / f"{i:06d}.png")
    frames = [{"seq_idx": i, "native_frame": 100 + i} if seq_idx else {"native_frame": 100 + i} for i in range(n)]
    (c / "frame_manifest.json").write_text(json.dumps({"n_frames": n, "frames": frames}, indent=2))
    depth = np.stack([np.full((6, 8), 1.0 + i, np.float32) for i in range(n)])
    if zero_frame is not None:
        depth[zero_frame] = 0.0
    if hole is not None:
        depth[hole] = 0.0
    K = np.tile(np.array([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]], np.float32), (n, 1, 1))
    E = np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1))
    E[:, 0, 3] = np.arange(n)
    if turn_frame is not None:
        E[turn_frame, :3, :3] = QUARTER_TURN_Y
    (c / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(c / "exports" / "mini_npz" / "results.npz", depth=depth, conf=np.ones_like(depth), extrinsics=E,
             intrinsics=K)
    return c


def expected_gltf_points(depth, K, E):
    """The frame's points in glTF's frame, from the bundle's arrays by the formulas, without the stage's functions."""
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    cam = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1).reshape(-1, 3)
    world = (E[:3, :3].T @ (cam - E[:3, 3]).T).T
    return (world * [1, -1, -1])[depth.reshape(-1) > 0]


def test_each_frame_gets_a_cloud_and_its_placement_in_the_manifest(tmp_path):
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path)
    assert run_point_clouds(clip) == 3
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == [f"frame_{i:04d}.glb" for i in range(3)]
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert m["n_frames"] == 3
    assert list(m["frames"][1]) == ["seq_idx", "native_frame", "glb_centroid", "camera_pos_glb",
                                    "camera_forward_glb", "camera_up_glb"]
    # The camera moves along x with an identity rotation, so it is at -t.
    assert m["frames"][2]["camera_pos_glb"] == pytest.approx([-2.0, 0.0, 0.0])


def test_a_cloud_holds_the_frames_points_in_gltf_and_its_images_colours(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    clip = make_clip(tmp_path, turn_frame=2, hole=(0, 0, 0))
    bundle = dict(np.load(clip / "exports" / "mini_npz" / "results.npz"))
    run_point_clouds(clip)
    m = json.loads((clip / "frame_manifest.json").read_text())
    for i in range(3):
        gltf = expected_gltf_points(bundle["depth"][i], bundle["intrinsics"][i], bundle["extrinsics"][i])
        cloud = trimesh.load(clip / "pc_vis" / f"frame_{i:04d}.glb", force="scene").to_geometry()
        centroid = np.array(m["frames"][i]["glb_centroid"])
        assert centroid == pytest.approx(gltf.mean(0), abs=1e-5)
        # The pixel without depth is not in the cloud, and every other pixel is, centred on the centroid.
        assert len(cloud.vertices) == len(gltf)
        assert np.sort(cloud.vertices + centroid, axis=0) == pytest.approx(np.sort(gltf, axis=0), abs=1e-5)
        # The colours are the frame's own image, as RGB.
        assert (np.asarray(cloud.colors)[:, :3] == (10 + i, 100 + i, 200 + i)).all()


def test_a_frame_with_no_usable_depth_gets_no_cloud_and_the_rest_do(tmp_path):
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path, zero_frame=1)
    assert run_point_clouds(clip) == 2
    assert not (clip / "pc_vis" / "frame_0001.glb").exists()
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert "glb_centroid" not in m["frames"][1] and "glb_centroid" in m["frames"][2]


def without_bundle(clip):
    (clip / "exports" / "mini_npz" / "results.npz").unlink()


def with_fewer_extrinsics(clip):
    p = clip / "exports" / "mini_npz" / "results.npz"
    z = dict(np.load(p))
    np.savez(p, **{**z, "extrinsics": z["extrinsics"][:2]})


def without_an_image(clip):
    (clip / "input_images" / "000002.png").unlink()


def with_an_unreadable_image(clip):
    (clip / "input_images" / "000001.png").write_bytes(b"not a png")


def without_a_manifest(clip):
    (clip / "frame_manifest.json").unlink()


def edit_frames(clip, edit):
    p = clip / "frame_manifest.json"
    m = json.loads(p.read_text())
    edit(m["frames"])
    p.write_text(json.dumps(m, indent=2))


def with_a_duplicate_seq_idx(clip):
    edit_frames(clip, lambda frames: frames[1].update(seq_idx=0))


def with_an_extra_frame(clip):
    edit_frames(clip, lambda frames: frames.append({"seq_idx": 3, "native_frame": 103}))


def with_a_frame_short(clip):
    edit_frames(clip, lambda frames: frames.pop())


def with_a_wrong_n_frames(clip):
    p = clip / "frame_manifest.json"
    p.write_text(p.read_text().replace('"n_frames": 3', '"n_frames": 4'))


REFUSALS = [
    (without_bundle, FileNotFoundError, "run the depth stage first"),
    (with_fewer_extrinsics, ValueError, "2 extrinsics"),
    (without_an_image, FileNotFoundError, "1 of them have no image: 000002.png"),
    (with_an_unreadable_image, ValueError, "cannot read .*000001.png"),
    (without_a_manifest, FileNotFoundError, "frame_manifest.json"),
    (lambda clip: edit_frames(clip, lambda frames: frames[1].pop("seq_idx")), ValueError, "no seq_idx"),
    (with_a_duplicate_seq_idx, ValueError, "not 0 to 2 once each"),
    (with_an_extra_frame, ValueError, "4 seq_idx values are not 0 to 2"),
    (with_a_frame_short, ValueError, "2 seq_idx values are not 0 to 2"),
    (with_a_wrong_n_frames, ValueError, "n_frames is 4, but the manifest holds 3"),
]


@pytest.mark.parametrize("plant, exc, match", REFUSALS, ids=[f.__name__ for f, _, _ in REFUSALS])
def test_a_refused_clip_is_left_as_it_was(tmp_path, plant, exc, match):
    clip = make_clip(tmp_path)
    plant(clip)
    manifest_path = clip / "frame_manifest.json"
    manifest_before = manifest_path.read_text() if manifest_path.exists() else None
    with pytest.raises(exc, match=match):
        run_point_clouds(clip)
    assert not (clip / "pc_vis").exists(), "refused before any cloud is written"
    assert (manifest_path.read_text() if manifest_path.exists() else None) == manifest_before


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["point_clouds", *argv])
    main()


def test_main_refuses_a_mistyped_clip_and_an_empty_directory(tmp_path, monkeypatch):
    make_clip(tmp_path)
    with pytest.raises(SystemExit, match="no clip .* at: clpi"):
        run_main(monkeypatch, "--input-dir", str(tmp_path), "--clips", "clpi")
    assert not (tmp_path / "clip" / "pc_vis").exists(), "the run stops before any clip runs"
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no clip under"):
        run_main(monkeypatch, "--input-dir", str(tmp_path / "empty"))


def test_main_runs_every_clip_after_one_fails_and_names_the_error(tmp_path, monkeypatch, capsys):
    pytest.importorskip("trimesh")
    broken = make_clip(tmp_path, name="clip_a")
    good = make_clip(tmp_path, name="clip_b")
    # A bundle cut short is not an OSError or a ValueError; the run still goes on to the next clip.
    npz = broken / "exports" / "mini_npz" / "results.npz"
    npz.write_bytes(npz.read_bytes()[:100])
    with pytest.raises(SystemExit, match="1 of 2 clip\\(s\\) failed: clip_a"):
        run_main(monkeypatch, "--input-dir", str(tmp_path))
    assert "failed: BadZipFile" in capsys.readouterr().out
    assert (good / "pc_vis" / "frame_0002.glb").exists()
