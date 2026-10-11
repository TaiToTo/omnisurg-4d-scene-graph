"""Test the D4D reader on files made up for the test; D4D's loader itself is replaced by a stand-in package.

Each refusal has a test that plants its fault:

- no `d4d` on the path, a `d4d` whose loader differs from the pinned commit's, and one that lacks a file;
- a dataset root that holds no clip;
- a clip's masks with one missing in the middle, or named by other times than its frames;
- a mask that is not an image, and a scan with no finite point.
"""

import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import surgical_core.d4d as d4d

FAKE_LOADER = "class D4D:\n    def __init__(self, root):\n        self.root = root\n    def __iter__(self):\n        return iter([])\n"


def fake_package(tmp_path, loader_source: str = FAKE_LOADER, init_source: str = "") -> object:
    """Write a `d4d` package under `tmp_path` and return its directory."""
    package = tmp_path / "d4d"
    package.mkdir()
    (package / "__init__.py").write_text(init_source)
    (package / "loader.py").write_text(loader_source)
    return package


@pytest.fixture
def on_path(tmp_path, monkeypatch):
    """Put `tmp_path` first on the path, and forget any `d4d` imported before or during the test."""
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("d4d", "d4d.loader"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield tmp_path
    for name in ("d4d", "d4d.loader"):
        sys.modules.pop(name, None)


def clip(tmp_path, frames: list[str], **kw) -> SimpleNamespace:
    """Return a stand-in for a loader's clip whose rectified left frames are named by `frames`."""
    fields = dict(name="Clip_1", path=tmp_path, session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(
        name="specimen_1")), left_img_paths=[str(tmp_path / "left_images_rect" / f"{f}.png") for f in frames])
    return SimpleNamespace(**(fields | kw))


def test_the_blob_hash_is_the_one_git_prints(tmp_path):
    path = tmp_path / "loader.py"
    path.write_bytes(b"print('hi')\r\n\x00")
    git = subprocess.run(["git", "hash-object", str(path)], check=True, capture_output=True, text=True).stdout.strip()
    assert d4d.blob_hash(path) == git


def test_the_pinned_blobs_name_both_files_of_the_package():
    assert set(d4d.LOADER_BLOBS) == {"__init__.py", "loader.py"}
    assert all(len(h) == 40 for h in d4d.LOADER_BLOBS.values())


def test_no_d4d_on_the_path_is_refused(on_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(on_path)])
    with pytest.raises(ImportError, match="clone https://github.com/reubendocea/d4d at commit efe12d26"):
        d4d.loader()


def test_a_d4d_whose_loader_differs_is_refused_before_any_of_it_runs(on_path, monkeypatch):
    # The stand-in's `__init__.py` is the pinned one, and would write a file if it were imported.
    ran = on_path / "ran"
    package = fake_package(on_path, init_source=f"open({str(ran)!r}, 'w').close()\n")
    monkeypatch.setitem(d4d.LOADER_BLOBS, "__init__.py", d4d.blob_hash(package / "__init__.py"))
    with pytest.raises(RuntimeError, match=r"loader.py: blob [0-9a-f]{40}, where commit efe12d26"):
        d4d.loader()
    assert not ran.exists()


def test_a_d4d_that_lacks_a_file_is_refused(on_path, monkeypatch):
    package = fake_package(on_path)
    (package / "loader.py").unlink()
    monkeypatch.setitem(d4d.LOADER_BLOBS, "__init__.py", d4d.blob_hash(package / "__init__.py"))
    with pytest.raises(RuntimeError, match="loader.py: blob no file"):
        d4d.loader()


def test_a_d4d_at_the_pinned_files_is_imported(on_path, monkeypatch):
    package = fake_package(on_path)
    for name in d4d.LOADER_BLOBS:
        monkeypatch.setitem(d4d.LOADER_BLOBS, name, d4d.blob_hash(package / name))
    assert d4d.loader()("root").root == "root"


def test_a_root_that_holds_no_clip_is_refused(on_path, monkeypatch):
    package = fake_package(on_path)
    for name in d4d.LOADER_BLOBS:
        monkeypatch.setitem(d4d.LOADER_BLOBS, name, d4d.blob_hash(package / name))
    with pytest.raises(ValueError, match="holds no clip"):
        d4d.clips(str(on_path))


def test_times_come_from_the_names_and_the_nearest_frame_is_found(tmp_path):
    c = clip(tmp_path, ["100_000000000", "100_500000000", "101_000000000"])
    times = d4d.frame_times(c)
    assert times.tolist() == [100.0, 100.5, 101.0]
    assert d4d.clip_key(c) == "specimen_1/s_a/Clip_1"
    assert d4d.nearest_frame(times, 95.0) == (0, 5.0)
    assert d4d.nearest_frame(times, 100.6) == (1, 100.5 - 100.6)


def test_files_named_by_the_frames_are_returned_in_their_order(tmp_path):
    frames = ["100_000000000", "100_500000000", "101_000000000"]
    (tmp_path / "masks").mkdir()
    for f in reversed(frames):
        (tmp_path / "masks" / f"{f}.png").write_bytes(b"")
    assert [p.stem for p in d4d.frame_files(clip(tmp_path, frames), "masks", ".png")] == frames


@pytest.mark.parametrize("files", [
    ["100_000000000", "101_000000000"],
    ["100_000000000", "100_500000001", "101_000000000"],
])
def test_files_with_one_missing_or_named_by_another_time_are_refused(tmp_path, files):
    (tmp_path / "masks").mkdir()
    for f in files:
        (tmp_path / "masks" / f"{f}.png").write_bytes(b"")
    with pytest.raises(ValueError, match="files in masks/ for 3 frames"):
        d4d.frame_files(clip(tmp_path, ["100_000000000", "100_500000000", "101_000000000"]), "masks", ".png")


def test_a_mask_is_dilated_by_a_square_of_mask_dilate_pixels(tmp_path):
    raw = np.zeros((20, 30), np.uint8)
    raw[10, 15] = 255
    cv2.imwrite(str(tmp_path / "m.png"), raw)
    mask = d4d.instrument_mask(tmp_path / "m.png", (30, 20))
    half = d4d.MASK_DILATE // 2
    assert mask.sum() == d4d.MASK_DILATE ** 2
    assert mask[10 - half:11 + half, 15 - half:16 + half].all()


def test_a_mask_of_another_size_is_resized_with_the_nearest_neighbour(tmp_path):
    raw = np.zeros((714, 894), np.uint8)
    raw[300:303, 400:520] = 255
    cv2.imwrite(str(tmp_path / "m.png"), raw)
    kernel = np.ones((d4d.MASK_DILATE, d4d.MASK_DILATE), np.uint8)
    for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
        resized = cv2.resize(raw, (640, 512), interpolation=interpolation)
        want = cv2.dilate((resized > 0).astype(np.uint8), kernel) > 0
        got = d4d.instrument_mask(tmp_path / "m.png", (640, 512))
        # A bilinear resize spreads the mask's edge, so only the nearest neighbour gives the same mask.
        assert np.array_equal(got, want) == (interpolation == cv2.INTER_NEAREST)


def test_a_mask_that_is_not_an_image_is_refused(tmp_path):
    (tmp_path / "m.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not an image"):
        d4d.instrument_mask(tmp_path / "m.png", (640, 512))


def test_a_scan_keeps_its_finite_points_and_an_empty_one_is_refused(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    # Values a 32-bit float holds exactly, since the PLY writer may store them so.
    points = np.array([[0.0, 0.0, 0.5], [np.nan, 0.0, 0.5], [0.25, 0.125, 0.75]])
    trimesh.PointCloud(points).export(tmp_path / "scan.ply")
    assert np.array_equal(d4d.read_scan(str(tmp_path / "scan.ply")), points[[0, 2]])
    trimesh.PointCloud(points[[1]]).export(tmp_path / "empty.ply")
    with pytest.raises(ValueError, match="no finite point"):
        d4d.read_scan(str(tmp_path / "empty.ply"))


def test_pixel_hits_counts_the_points_inside_the_view_and_in_front_of_the_camera():
    params = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 1.0, "width": 4, "height": 3}
    # Camera at the origin looking down +z; a world point (x, y, z) lands at (100 x / z + 2, 100 y / z + 1).
    points = np.array([
        [0.0, 0.0, 1.0],      # pixel (u 2, v 1)
        [0.005, 0.0, 1.0],    # u 2.5, the same pixel
        [-0.02, 0.01, 1.0],   # u 0, v 2: the corner
        [0.02, 0.0, 1.0],     # u 4, one past the last column
        [0.0, 0.0, -1.0],     # behind the camera
    ])
    hits = d4d.pixel_hits(points, np.eye(4), params)
    want = np.zeros((3, 4), np.int32)
    want[1, 2], want[2, 0] = 2, 1
    assert np.array_equal(hits, want)


def test_pixel_hits_moves_the_points_into_the_camera_by_the_inverse_pose():
    params = {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 1.0, "width": 4, "height": 3}
    pose = np.eye(4)
    pose[:3, 3] = [0.01, 0.0, 0.0]  # the camera sits 1 cm along +x in the world
    hits = d4d.pixel_hits(np.array([[0.01, 0.0, 1.0]]), pose, params)
    assert hits[1, 2] == 1 and hits.sum() == 1
