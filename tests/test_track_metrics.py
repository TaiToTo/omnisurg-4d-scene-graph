"""The measures of identity, on frames one pixel high, each value worked out by hand, and the refusals of the tool.

A frame is drawn as two strings of ten pixels: its GT, one letter per class
(`L` liver, `G` gallbladder, `T` grasper), and its regions, one digit per
region id. Every pixel has depth. The comment beside a frame gives the
IoUs and the hits the evaluator's pairing gives them. The tests that
compute IDF1 need scipy, the `tools` extra.
"""
import json

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import VIEWS
from evalkit.inputs import DEPTH_FILE
from evalkit.tools.scores import ruler
from evalkit.tools.track_metrics import (
    FrameTracks, frame_tracks, hold, id_true_positives, identity, main, measure_condition, reentry,
)

LIVER, GALLBLADDER, GRASPER = C.LIVER, C.GALLBLADDER, C.GRASPER
CODES = {"L": LIVER, "G": GALLBLADDER, "T": GRASPER}
# ATLAS-120k's Liver, Gallbladder and the marker that excludes a frame.
ATLAS_LIVER, ATLAS_GALLBLADDER, ATLAS_EXCLUDED = 12, 14, 42


def gt_map(s: str, codes=CODES) -> np.ndarray:
    return np.array([[codes[ch] for ch in s]], dtype=np.int32)


def region_map(s: str) -> np.ndarray:
    return np.array([[-1 if ch == "." else int(ch) for ch in s]], dtype=np.int32)


def frames_of(pairs, times=None, view="all") -> list[FrameTracks]:
    times = [0.6 * k for k in range(len(pairs))] if times is None else times
    return [frame_tracks(gt_map(g), region_map(r), np.ones((1, len(g)), dtype=bool), C.TABLE, view, t)
            for (g, r), t in zip(pairs, times)]


def needs_scipy():
    pytest.importorskip("scipy", reason="IDF1's assignment needs the `tools` extra")


# The scene of the hold tests, seed on frame 2. The liver is picked there, the gallbladder is not.
HOLD_SCENE = [
    ("LLLLLGGGGG", "0000111111"),  # (L,0) 4/5 = 0.8, (G,1) 5/6: hits L-0, G-1
    ("LLLLLGGGGG", "1111111111"),  # (L,1) = (G,1) = 0.5: the higher class id is taken first, so G-1
    ("LLLLLGGGGG", "0000011223"),  # seed: L-0 at 1.0; G's best region is 0.4, so G is not picked
    ("LLLLLLLGGG", "0000000111"),  # (L,0) 7/7, (G,1) 3/3
    ("LLLLLLLLLL", "0000000000"),  # (L,0) 10/10; G absent
]
HOLD_TIMES = [0.0, 0.6, 1.2, 1.8, 2.4]


def test_a_frame_pairs_classes_and_regions_as_the_evaluator_does():
    f = frames_of([("LLLLLGGGGG", "0000011122")])[0]
    assert f.classes == (LIVER, GALLBLADDER) and f.n_regions == 3
    assert dict(f.hits) == {LIVER: 0, GALLBLADDER: 1}
    assert dict(f.ious) == pytest.approx({(LIVER, 0): 1.0, (GALLBLADDER, 1): 0.6, (GALLBLADDER, 2): 0.4})


def test_a_view_removes_the_classes_it_leaves_out_and_the_regions_on_them_only():
    pairs = [("LLLLLTTTTT", "0000011111")]
    assert frames_of(pairs)[0].classes == (LIVER, GRASPER)
    geometric = frames_of(pairs, view="geometric")[0]
    assert geometric.classes == (LIVER,) and geometric.n_regions == 1 and dict(geometric.hits) == {LIVER: 0}


def test_hold_follows_the_region_picked_on_the_seed_frame():
    h = hold(frames_of(HOLD_SCENE, HOLD_TIMES), seed=2)
    # The liver's samples: 0.8, 0 (region 0 is gone), 1.0, 1.0. The gallbladder's: 0 on frames 0, 1 and 3.
    assert h["hold_mean"] == pytest.approx(2.8 / 4)
    assert h["hold_all_mean"] == pytest.approx(2.8 / 7)
    assert h["picked_share"] == 0.5
    assert (h["n_seed_classes"], h["n_picked"], h["n_hold_samples"], h["n_hold_all_samples"]) == (2, 1, 4, 7)


def test_the_hold_curve_bins_the_picked_samples_by_seconds_from_the_seed():
    curve = hold(frames_of(HOLD_SCENE, HOLD_TIMES), seed=2)["hold_curve"]
    assert [(b["above_sec"], b["to_sec"], b["n"]) for b in curve] == [
        (0.0, 0.5, 0), (0.5, 1.0, 2), (1.0, 2.0, 2), (2.0, 4.0, 0), (4.0, 8.0, 0), (8.0, None, 0)]
    # 0.6 s away: frames 1 (0) and 3 (1.0). 1.2 s away: frames 0 (0.8) and 4 (1.0).
    assert [b["mean"] for b in curve] == pytest.approx([None, 0.5, 0.9, None, None, None])


def test_hold_is_not_defined_when_the_seed_frame_has_no_class_in_the_view():
    h = hold(frames_of([("TTTTTTTTTT", "0000000000")] * 3, view="geometric"), seed=1)
    assert (h["hold_mean"], h["hold_all_mean"], h["picked_share"]) == (None, None, None)


def test_id_switches_and_fragmentations_walk_each_class_over_the_frames_it_is_present_on():
    needs_scipy()
    frames = frames_of([
        ("LLLLLGGGGG", "0000012345"),  # L-0; G's regions are 0.2 each, no hit, and none before: nothing
        ("LLLLLGGGGG", "0000011111"),  # L-0, G-1
        ("LLLLLGGGGG", "0000022222"),  # L-0, G-2: a switch
        ("LLLLLGGGGG", "0000012345"),  # L-0; G not hit after a hit: a fragmentation
        ("LLLLLGGGGG", "0000012345"),  # L-0; G not hit again: nothing more
        ("LLLLLGGGGG", "0000022222"),  # L-0, G-2: the region of G's last hit, no switch
        ("LLLLLLLLLL", "0000000000"),  # L-0; G absent
    ])
    ident = identity(frames)
    assert (ident["ID_switches"], ident["fragmentations"]) == (1, 1)
    # Together: L-0 on 7 frames, G-1 on 1, G-2 on 2; the best one region per class is L-0 and G-2, 9 frames.
    # GT objects 6 x 2 + 1 = 13; predicted objects 6 + 2 + 2 + 6 + 6 + 2 + 1 = 25.
    assert ident["id_true_positives"] == 9 and (ident["n_gt_objects"], ident["n_pred_objects"]) == (13, 25)
    assert ident["IDF1"] == pytest.approx(18 / 38)


def test_idf1_takes_the_best_region_for_each_class_over_the_clip_not_the_first_best():
    needs_scipy()
    # A-0 together on 3 frames, A-1 on 2, B-0 on 2. Greedy takes A-0 and leaves B nothing: 3. The best is 4.
    a, b = 1, 2
    frames = [FrameTracks(0.0, (a,), {a: 0}, 1, {})] * 3 + [FrameTracks(0.0, (a, b), {a: 1, b: 0}, 2, {})] * 2
    assert id_true_positives(frames) == 4


def test_idf1_is_not_defined_without_a_gt_object():
    ident = identity(frames_of([("TTTTTTTTTT", "0000000000")] * 2, view="geometric"))
    assert ident["IDF1"] is None and ident["n_pred_objects"] == 0


def test_reentry_compares_the_regions_on_either_side_of_a_gap():
    frames = frames_of([
        ("LLLLLGGGGG", "0000011111"),  # L-0
        ("GGGGGGGGGG", "1111111111"),  # L absent
        ("LLLLLGGGGG", "0000011111"),  # L-0: the gap is recovered
        ("GGGGGGGGGG", "1111111111"),
        ("LLLLLGGGGG", "2222211111"),  # L-2: not recovered
        ("GGGGGGGGGG", "1111111111"),
        ("LLLLLGGGGG", "3456711111"),  # L's regions are 0.2 each, no hit: the gap is unmatched
    ])
    assert reentry(frames) == {"reentry": 0.5, "n_gaps": 2, "n_gaps_recovered": 1, "n_gaps_unmatched": 1}


# What the tracker records for the hold scene: a seed on the centre of five frames, carried both ways.
BOTH_WAYS = {"seed_source": "sam", "seed_frame": 2, "bidir": True, "frames": [0, 1, 2, 3, 4]}
FORWARD = {"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2, 3]}


def write_hold_scene(tmp_path, clip="VID01_a", tag="c", seed_info=BOTH_WAYS):
    C.write_clip(tmp_path / "data", clip, [gt_map(g) for g, _ in HOLD_SCENE], times=HOLD_TIMES)
    C.write_labels(tmp_path / "tracks", clip, tag, {k: region_map(r) for k, (_, r) in enumerate(HOLD_SCENE)},
                   seed_info=seed_info)


def measure(tmp_path, clips=("VID01_a",), tag="c", dataset="cholecseg8k"):
    return measure_condition(dataset, None, list(clips), tmp_path / "data", tmp_path / "tracks", tag)


def test_a_clip_on_disk_gives_the_hand_derived_values_in_every_view(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    summary = measure(tmp_path)
    assert summary["propagation"] == "both_ways_from_centre" and summary["views"] == list(VIEWS)
    row = summary["per_clip"][0]
    assert (row["seed_frame"], row["n_gt_frames"], row["n_excluded_frames"]) == (2, 5, 0)
    for view in VIEWS:  # liver and gallbladder are tissue, so every view scores the same pixels
        assert row[f"hold_mean/{view}"] == pytest.approx(0.7)
        assert row[f"hold_all_mean/{view}"] == pytest.approx(0.4)
        assert row[f"picked_share/{view}"] == 0.5
        # L: hit by 0 on every frame but 1, a fragmentation. G: hit by 1 on 0, 1 and 3, a fragmentation on 2.
        assert (row[f"ID_switches/{view}"], row[f"fragmentations/{view}"]) == (0, 2)
        # L-0 on 4 frames and G-1 on 3: 2 x 7 / (9 GT objects + 10 predicted objects).
        assert row[f"IDF1/{view}"] == pytest.approx(14 / 19)
        assert row[f"reentry/{view}"] is None and row["counts"][view]["n_gaps"] == 0


def test_the_frames_are_walked_in_time_order_not_in_index_order(tmp_path):
    needs_scipy()
    # Frame 1 comes last in time. In time order the liver is hit by 0, 0, 1: one switch; by index, two.
    gt = [gt_map("LLLLLLLLLL")] * 3
    C.write_clip(tmp_path / "data", "VID01_a", gt, times=[0.0, 1.2, 0.6])
    C.write_labels(tmp_path / "tracks", "VID01_a", "c",
                   {0: region_map("0" * 10), 1: region_map("1" * 10), 2: region_map("0" * 10)},
                   seed_info={"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2]})
    row = measure(tmp_path)["per_clip"][0]
    assert row["ID_switches/all"] == 1
    # Hold: 1.0 at 0.6 s (frame 2), 0 at 1.2 s (frame 1).
    assert [b["mean"] for b in row["hold_curve"]["all"][1:3]] == [1.0, 0.0]


def write_atlas_clip(tmp_path, gt, gt_frames, regions, seed_info):
    C.write_clip(tmp_path / "data", "VID01_a", gt, gt_frames=gt_frames, dataset="atlas120k")
    C.write_labels(tmp_path / "tracks", "VID01_a", "c", regions, seed_info=seed_info)


def test_a_frame_without_gt_and_an_excluded_frame_leave_no_gap(tmp_path):
    needs_scipy()
    codes = {"L": ATLAS_LIVER, "G": ATLAS_GALLBLADDER, "X": ATLAS_EXCLUDED}
    gt = [gt_map(s, codes) for s in ("LLLLLGGGGG", "LLLLLGGGGG", "XLLLLGGGGG", "LLLLLGGGGG")]
    regions = {k: region_map(r) for k, r in enumerate(("0000011111", "0000011111", "0000011111", "2222211111"))}
    write_atlas_clip(tmp_path, gt, [0, 2, 3], regions, FORWARD)
    row = measure(tmp_path, dataset="atlas120k")["per_clip"][0]
    assert (row["n_gt_frames"], row["n_excluded_frames"]) == (2, 1)
    # Frames 0 and 3 are consecutive GT frames: the liver's move from 0 to 2 is a switch, and no gap.
    assert row["ID_switches/all"] == 1
    assert (row["counts"]["all"]["n_gaps"], row["counts"]["all"]["n_gaps_unmatched"]) == (0, 0)


def test_a_seed_frame_without_gt_is_refused(tmp_path):
    gt = [gt_map("LLLLLGGGGG")] * 3
    C.write_clip(tmp_path / "data", "VID01_a", gt, gt_frames=[0, 2])
    C.write_labels(tmp_path / "tracks", "VID01_a", "c", {k: region_map("0000011111") for k in range(3)},
                   seed_info={"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": [0, 1, 2]})
    with pytest.raises(ValueError, match="the seed frame 1 is not a GT frame"):
        measure(tmp_path)


def test_an_excluded_seed_frame_is_refused(tmp_path):
    codes = {"L": ATLAS_LIVER, "X": ATLAS_EXCLUDED}
    write_atlas_clip(tmp_path, [gt_map("XLLLLLLLLL", codes)] + [gt_map("LLLLLLLLLL", codes)] * 3, None,
                     {k: region_map("0" * 10) for k in range(4)}, FORWARD)
    with pytest.raises(ValueError, match="the seed frame 0 is an excluded frame"):
        measure(tmp_path, dataset="atlas120k")


def test_an_excluded_frame_with_faulty_depth_is_refused(tmp_path):
    codes = {"L": ATLAS_LIVER, "X": ATLAS_EXCLUDED}
    write_atlas_clip(tmp_path, [gt_map("LLLLLLLLLL", codes)] * 3 + [gt_map("XLLLLLLLLL", codes)], None,
                     {k: region_map("0" * 10) for k in range(4)}, FORWARD)
    path = tmp_path / "data" / "VID01_a" / DEPTH_FILE
    depth = np.load(path)["depth"]
    depth[3, 0, 5] = np.nan
    np.savez(path, depth=depth)
    with pytest.raises(ValueError, match="no valid depth"):
        measure(tmp_path, dataset="atlas120k")


@pytest.mark.parametrize("info, why", [
    ({"seed_source": "per_frame", "frames": "all"}, "segmented frame by frame"),
    ({"seed_source": "gt", "seed_frame": 2, "bidir": True, "frames": [0, 1, 2, 3, 4]},
     "holds no propagation rule: .*not from the tracker's own masks"),
])
def test_a_condition_without_a_seed_frame_to_hold_from_is_refused(tmp_path, info, why):
    write_hold_scene(tmp_path, seed_info=info)
    with pytest.raises(ValueError, match=why):
        measure(tmp_path)


def test_labels_without_a_record_are_refused(tmp_path):
    write_hold_scene(tmp_path, seed_info=None)
    with pytest.raises(FileNotFoundError):
        measure(tmp_path)


def test_two_rules_in_one_condition_are_refused(tmp_path):
    write_hold_scene(tmp_path, clip="VID01_a")
    write_hold_scene(tmp_path, clip="VID02_b", seed_info={**FORWARD, "frames": [0, 1, 2, 3, 4]})
    with pytest.raises(ValueError, match="a condition has one rule"):
        measure(tmp_path, clips=("VID01_a", "VID02_b"))


def test_the_score_tools_refuse_its_json_as_a_score(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    with pytest.raises(ValueError, match="so this one is neither"):
        ruler(measure(tmp_path))


def test_the_command_line_writes_the_json(tmp_path, capsys):
    needs_scipy()
    write_hold_scene(tmp_path)
    (tmp_path / "clips.txt").write_text("VID01_a\n")
    out = tmp_path / "c.json"
    main(["--dataset", "cholecseg8k", "--clips", str(tmp_path / "clips.txt"), "--data-root", str(tmp_path / "data"),
          "--tracks-root", str(tmp_path / "tracks"), "--tag", "c", "--out", str(out)])
    assert json.loads(out.read_text())["per_clip"][0]["hold_mean/all"] == pytest.approx(0.7)
    assert "1 clips measured" in capsys.readouterr().out
