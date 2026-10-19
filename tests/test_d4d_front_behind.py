"""Test `evalkit.tools.d4d_front_behind` on regions with known depths, and on a made-up clip.

The made-up clip's view is 64 × 48 pixels, cut into four quadrants of 32 × 24. The scan puts one point in
each pixel, at 0.1, 0.2, 0.3 and 0.4 m in the four quadrants. Each refusal has a test that plants its fault:

- a side without a scan, a pose or a calibration;
- a side's depth that is missing or was made for another frame;
- seed regions that are missing, whose record names another depth, or whose shape is not the depth's;
- stereo depth that is not the frame's, and a frame that cannot be read;
- an answer or regions that are unknown, and a population side whose clip the dataset does not hold.
"""

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import surgical_core.d4d as d4d
from evalkit.tools import d4d_front_behind as fb

# The regions of the known-depth cases: three columns of 34, 33 and 33 pixels, 100 rows. The scan reaches the
# top ten rows, at 0.1, 0.2 and 0.3 m.
H, W = 100, 100
LABELS = np.zeros((H, W), np.int32)
LABELS[:, 34:67] = 1
LABELS[:, 67:] = 2
GT = np.full((H, W), np.nan, np.float32)
GT[:10] = np.array([0.10] * 34 + [0.20] * 33 + [0.30] * 33, np.float32)
RIGHT = np.tile(np.array([1.0] * 34 + [2.0] * 33 + [3.0] * 33, np.float32), (H, 1))
VALID = np.ones((H, W), bool)


def acc(answer_depth, answer="mono", pixels="all", labels=LABELS, gt=GT):
    return fb.score_pairs(fb.node_stats(labels, VALID, gt, answer_depth, pixels), answer)


def test_an_answer_in_the_scans_order_is_right_and_the_reverse_is_wrong():
    assert acc(RIGHT)["acc"] == 1.0
    assert acc(4.0 - RIGHT)["acc"] == 0.0


def test_the_area_floor_names_the_larger_region_and_leaves_out_a_tie_in_area():
    area = acc(None, "area")
    assert (area["n_pairs"], area["acc"], area["pairs"]) == (2, 1.0, [[0, 1], [0, 2]])


def test_a_region_the_scan_reaches_on_too_few_pixels_is_not_scored():
    thin = GT.copy()
    thin[:, 67:] = np.nan
    thin[:2, 67:] = 0.30
    assert sorted(fb.node_stats(LABELS, VALID, thin, RIGHT)) == [0, 1]


def test_the_answer_pixels_choose_where_the_answers_median_is_taken():
    mixed = 4.0 - RIGHT
    mixed[:10] = RIGHT[:10]
    assert acc(mixed, pixels="all")["acc"] == 0.0
    assert acc(mixed, pixels="gt")["acc"] == 1.0


def test_each_pair_is_recorded_with_its_gap_and_whether_it_was_right():
    r = acc(RIGHT)
    assert r["pairs"] == [[0, 1], [0, 2], [1, 2]] and r["ok"] == [1, 1, 1]
    assert r["gap_mm"] == [100.0, 200.0, 100.0] and r["gap_median_mm"] == 100.0


def test_no_region_is_not_scored():
    labels = LABELS.copy()
    labels[LABELS == 2] = -1
    assert sorted(fb.node_stats(labels, VALID, GT, RIGHT)) == [0, 1]


def test_the_scans_depth_keeps_the_nearest_point_of_each_pixel():
    params = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "width": 4, "height": 3}
    points = np.array([[0.005, 0.005, 1.0], [0.01, 0.01, 2.0], [0.0025, 0.0125, 0.5], [0.0, 0.0, -1.0]])
    depth = d4d.scan_depth(points, np.eye(4), params)
    assert depth.dtype == np.float32 and depth[0, 0] == 1.0 and depth[2, 0] == 0.5
    assert np.isnan(depth).sum() == 10


# The made-up clip.
CW, CH = 64, 48
PARAMS = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "width": CW, "height": CH}
FRAMES = ["100_000000000", "100_500000000", "101_000000000"]
KEY = "specimen_1/s_a/Clip_1"
QUADRANT_DEPTH = (0.1, 0.2, 0.3, 0.4)


def quadrant(r, c, h=CH, w=CW):
    return 2 * (r >= h // 2) + (c >= w // 2)


QUADRANT_MAP = np.array([[QUADRANT_DEPTH[quadrant(r, c)] for c in range(CW)] for r in range(CH)], np.float32)


def scan_points() -> np.ndarray:
    return np.array([[(c + 0.5) / 100 * z, (r + 0.5) / 100 * z, z] for r in range(CH) for c in range(CW)
                     for z in [QUADRANT_DEPTH[quadrant(r, c)]]])


def make_clip(tmp_path, **changes) -> SimpleNamespace:
    """Write the clip's frames, empty instrument masks and stereo depth, and return the clip."""
    for folder in ("left_images_rect", "left_images_rect_masks", "stereo_depth"):
        (tmp_path / folder).mkdir(exist_ok=True)
    for f in FRAMES:
        cv2.imwrite(str(tmp_path / "left_images_rect" / f"{f}.png"), np.full((CH, CW, 3), 128, np.uint8))
        cv2.imwrite(str(tmp_path / "left_images_rect_masks" / f"{f}.png"), np.zeros((CH, CW), np.uint8))
        np.save(tmp_path / "stereo_depth" / f"{f}.npy", QUADRANT_MAP)
    clip = SimpleNamespace(name="Clip_1", path=tmp_path, pointclouds={"end": str(tmp_path / "101_200000000.ply")},
                           poses={"end": np.eye(4)}, endoscope_params=PARAMS,
                           session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(name="specimen_1")),
                           left_img_paths=[str(tmp_path / "left_images_rect" / f"{f}.png") for f in FRAMES],
                           stereo_depth_paths=[str(tmp_path / "stereo_depth" / f"{f}.npy") for f in FRAMES])
    for k, v in changes.items():
        setattr(clip, k, v)
    return clip


def write_depth_and_seeds(tmp_path, frame_index=2, seed_shape=(CH // 2, CW // 2), swap=False) -> tuple:
    """Write the side's depth, at half resolution, and its seed regions, the quadrants, with their record.

    The depth orders the quadrants as the scan does, or with the first two swapped.
    """
    depth_dir, seed_dir = tmp_path / "depth", tmp_path / "seed"
    depth_dir.mkdir(exist_ok=True)
    seed_dir.mkdir(exist_ok=True)
    h, w = CH // 2, CW // 2
    order = (2.0, 1.0, 3.0, 4.0) if swap else (1.0, 2.0, 3.0, 4.0)
    depth = np.array([[order[quadrant(r, c, h, w)] for c in range(w)] for r in range(h)], np.float32)
    stem = d4d.side_stem(KEY, "end")
    np.savez_compressed(depth_dir / f"{stem}.npz", depth=depth, conf=depth, intrinsics=np.eye(3, dtype=np.float32),
                        extrinsics=np.eye(4, dtype=np.float32)[:3], frame_index=np.int32(frame_index),
                        frame_minus_gt_s=np.float32(-0.2), window=np.int32(3), process_res=np.int32(504),
                        image_path=np.str_("frame.png"))
    sh, sw = seed_shape
    np.save(seed_dir / f"{stem}.npy", np.array([[quadrant(r, c, sh, sw) for c in range(sw)] for r in range(sh)],
                                               np.int32))
    digest = hashlib.sha256((depth_dir / f"{stem}.npz").read_bytes()).hexdigest()
    (seed_dir / f"{stem}.json").write_text(json.dumps({
        "depth_file": f"{stem}.npz", "depth_sha256": digest, "window": 3, "process_res": 504, "frame_index": 2,
        "sam_input": "normal_edge", "points_per_side": 24, "edge_gain": 1.0, "smooth": False,
        "edge_ring_masked": True, "sam_ckpt": "sam_vit_h_4b8939.pth", "n_regions": 4}))
    return depth_dir, seed_dir


@pytest.fixture
def scan(monkeypatch):
    monkeypatch.setattr(d4d, "read_scan", lambda path: scan_points())


def side(clip, answer, regions, depth_dir=None, seed_dir=None, pixels="all"):
    return fb.score_side((clip, "end", answer, regions, pixels, depth_dir, seed_dir))


def test_a_side_is_scored_on_the_frame_nearest_its_scan(tmp_path, scan):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path, swap=True)
    r = side(make_clip(tmp_path), "mono", "ours", depth_dir, seed_dir)
    # Six pairs of quadrants; the swapped depth answers the pair of quadrants 0 and 1 wrong.
    assert (r["n_nodes"], r["n_pairs"], r["n_correct"]) == (4, 6, 5)
    assert (r["key"], r["side"], r["frame_index"], r["depth_window"], r["session"]) == (
        KEY, "end", 2, 3, "specimen_1/s_a")
    assert r["frame_minus_gt_s"] == pytest.approx(-0.2)


def test_an_instrument_leaves_its_pixels_out_of_a_region(tmp_path, scan):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path)
    clip = make_clip(tmp_path)
    mask = np.zeros((CH, CW), np.uint8)
    mask[:20, :28] = 255
    cv2.imwrite(str(tmp_path / "left_images_rect_masks" / f"{FRAMES[2]}.png"), mask)
    # Dilated, the instrument covers quadrant 0 but for 4 x 4 pixels, too few to score it.
    assert side(clip, "mono", "ours", depth_dir, seed_dir)["n_nodes"] == 3


@pytest.mark.parametrize("answer, n_pairs, n_correct", [
    # The quadrants tie in area but for quadrant 3, which an instrument shrinks; it is named the farthest.
    ("area", 3, 3),
    # The distance from the principal point puts quadrant 2 before quadrant 1.
    ("screen", 6, 5),
    ("gt_half", 6, 6),
    ("stereo", 6, 6),
])
def test_each_answer_scores_the_quadrants_as_its_depth_orders_them(tmp_path, scan, answer, n_pairs, n_correct):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path)
    clip = make_clip(tmp_path)
    mask = np.zeros((CH, CW), np.uint8)
    mask[40:, 56:] = 255
    cv2.imwrite(str(tmp_path / "left_images_rect_masks" / f"{FRAMES[2]}.png"), mask)
    r = side(clip, answer, "ours", depth_dir, seed_dir)
    assert (r["n_nodes"], r["n_pairs"], r["n_correct"]) == (4, n_pairs, n_correct)


def test_the_fixed_cells_are_the_same_on_every_frame():
    assert np.array_equal(fb.voronoi_labels(CH, CW), fb.voronoi_labels(CH, CW))
    assert len(np.unique(fb.voronoi_labels(CH, CW))) == fb.VORONOI_K


@pytest.mark.parametrize("changes", [{"pointclouds": {}}, {"poses": {}}, {"endoscope_params": None}])
def test_a_side_without_a_scan_a_pose_or_a_calibration_is_refused(tmp_path, scan, changes):
    with pytest.raises(ValueError, match="no scan, pose or calibration"):
        side(make_clip(tmp_path, **changes), "area", "voronoi")


def test_a_missing_depth_is_refused(tmp_path, scan):
    with pytest.raises(FileNotFoundError, match="run pipeline.d4d_depth first"):
        side(make_clip(tmp_path), "mono", "voronoi", tmp_path / "nowhere")


def test_a_depth_made_for_another_frame_is_refused(tmp_path, scan):
    depth_dir, _ = write_depth_and_seeds(tmp_path, frame_index=1)
    with pytest.raises(ValueError, match="made for frame 1, not 2"):
        side(make_clip(tmp_path), "mono", "voronoi", depth_dir)


def test_missing_seed_regions_are_refused(tmp_path, scan):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path)
    (seed_dir / f"{d4d.side_stem(KEY, 'end')}.json").unlink()
    with pytest.raises(FileNotFoundError, match="run pipeline.d4d_seed first"):
        side(make_clip(tmp_path), "area", "ours", depth_dir, seed_dir)


def test_seed_regions_made_from_another_depth_are_refused(tmp_path, scan):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path)
    (tmp_path / "other").mkdir()
    other_depth, _ = write_depth_and_seeds(tmp_path / "other", swap=True)
    stem = d4d.side_stem(KEY, "end")
    (depth_dir / f"{stem}.npz").write_bytes((other_depth / f"{stem}.npz").read_bytes())
    with pytest.raises(ValueError, match="made from another depth"):
        side(make_clip(tmp_path), "area", "ours", depth_dir, seed_dir)


def test_seed_regions_of_another_shape_than_the_depth_are_refused(tmp_path, scan):
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path, seed_shape=(CH, CW))
    with pytest.raises(ValueError, match=r"regions \(48, 64\), depth \(24, 32\)"):
        side(make_clip(tmp_path), "area", "ours", depth_dir, seed_dir)


def test_stereo_depth_that_is_not_the_frames_is_refused(tmp_path, scan):
    clip = make_clip(tmp_path)
    clip.stereo_depth_paths = clip.stereo_depth_paths[:2]
    with pytest.raises(ValueError, match="stereo depth at position 2 is not frame 101_000000000"):
        side(clip, "stereo", "voronoi")


def test_a_frame_that_cannot_be_read_is_refused(tmp_path, scan):
    clip = make_clip(tmp_path)
    (tmp_path / "left_images_rect" / f"{FRAMES[2]}.png").write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError, match="cannot read the frame"):
        side(clip, "area", "kmeans")


def test_an_unknown_answer_or_regions_is_refused(tmp_path, scan):
    with pytest.raises(ValueError, match="unknown answer"):
        side(make_clip(tmp_path), "lidar", "voronoi")
    with pytest.raises(ValueError, match="unknown regions"):
        side(make_clip(tmp_path), "area", "slic")


def test_a_population_side_whose_clip_the_dataset_does_not_hold_is_refused(tmp_path):
    population = {"sides": [{"key": KEY, "side": "end"}, {"key": "specimen_9/s/Clip_1", "side": "start"}]}
    with pytest.raises(ValueError, match="1 side.* name a clip the dataset does not hold.*specimen_9"):
        fb.population_tasks([make_clip(tmp_path)], population, "area", "voronoi", "all", None, None)


def test_the_bootstrap_resamples_units_and_pools_their_pairs():
    recs = [{"key": "a/s/c1", "session": "a/s", "specimen": "a", "n_pairs": 4, "n_correct": 3},
            {"key": "a/s/c2", "session": "a/s", "specimen": "a", "n_pairs": 0, "n_correct": 0},
            {"key": "b/t/c1", "session": "b/t", "specimen": "b", "n_pairs": 6, "n_correct": 3}]
    clip = fb.bootstrap(recs, "clip")
    assert clip["n_units"] == 2 and clip["acc"] == 0.6 and 0.5 <= clip["lo"] <= clip["hi"] <= 0.75
    assert fb.bootstrap(recs[1:2], "clip") is None


def test_main_writes_every_side_with_its_totals(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    clip = make_clip(tmp_path)
    trimesh.PointCloud(scan_points()).export(clip.pointclouds["end"])
    depth_dir, seed_dir = write_depth_and_seeds(tmp_path, swap=True)
    population = tmp_path / "population.json"
    population.write_text(json.dumps({"sides": [{"key": KEY, "side": "end"}]}))
    monkeypatch.setattr(d4d, "clips", lambda root: [clip])
    fb.main(["--root", str(tmp_path), "--population", str(population), "--answer", "mono", "--regions", "ours",
             "--depth-dir", str(depth_dir), "--seed-dir", str(seed_dir), "--out", str(tmp_path / "out.json"),
             "--jobs", "1"])
    out = json.loads((tmp_path / "out.json").read_text())
    assert (out["n_frames"], out["n_pairs"], out["acc"]) == (1, 6, 5 / 6)
    assert out["front_behind_code_sha"] == fb.front_behind_code_sha()
    assert out["seed_config"] == [{"edge_gain": 1.0, "edge_ring_masked": True, "points_per_side": 24,
                                   "process_res": 504, "sam_ckpt": "sam_vit_h_4b8939.pth",
                                   "sam_input": "normal_edge", "smooth": False, "window": 3, "n_sides": 1}]
    assert out["bootstrap"]["clip"]["acc"] == 5 / 6


@pytest.mark.parametrize("args, message", [
    (["--answer", "mono", "--regions", "voronoi"], "--answer mono and --regions ours need --depth-dir"),
    (["--answer", "area", "--regions", "ours", "--depth-dir", "d"], "--regions ours needs --seed-dir"),
])
def test_main_refuses_an_answer_or_regions_without_the_stages_files(tmp_path, capsys, args, message):
    with pytest.raises(SystemExit):
        fb.main(["--root", str(tmp_path), "--population", "p.json", "--out", "o.json", *args])
    assert message in capsys.readouterr().err
