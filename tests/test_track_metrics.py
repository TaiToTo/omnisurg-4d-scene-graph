"""The measures of identity, on frames one pixel high, each value worked out by hand, and the refusals of the tool.

A frame is drawn as two strings of ten pixels: its GT, one letter per class
(`L` liver, `G` gallbladder, `T` grasper), and its regions, one digit per
region id. Every pixel has depth. The comment beside a frame gives the
IoUs and the hits the evaluator's pairing gives them. The tests that
compute IDF1 need scipy, the `tools` extra. The tool is called with the
component sizes 0 and 4, which a frame of ten pixels can tell apart.
"""
import json

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import VIEWS
from evalkit.inputs import DEPTH_FILE, read_clip
from evalkit.objects import gt_objects
from evalkit.scored import scored_pixels
from evalkit.tools import component_scores, track_metrics
from evalkit.tools.component_scores import component_objects, tool_sha
from evalkit.tools.scores import ruler
from evalkit.tools.track_metrics import (
    FrameTracks, hold, id_true_positives, identity, link_pieces, main, measure_condition, reentry, tracks_on_frame,
)

LIVER, GALLBLADDER, GRASPER = C.LIVER, C.GALLBLADDER, C.GRASPER
CODES = {"L": LIVER, "G": GALLBLADDER, "T": GRASPER}
# ATLAS-120k's Liver, Gallbladder and the marker that excludes a frame.
ATLAS_LIVER, ATLAS_GALLBLADDER, ATLAS_EXCLUDED = 12, 14, 42
SIZES = (0, 4)


def gt_map(s: str, codes=CODES) -> np.ndarray:
    return np.array([[codes[ch] for ch in s]], dtype=np.int32)


def region_map(s: str) -> np.ndarray:
    return np.array([[-1 if ch == "." else int(ch) for ch in s]], dtype=np.int32)


def class_frames(pairs, times=None, view="all") -> list[FrameTracks]:
    """Each frame's classes paired with its regions, each class its own track, as the tool builds them."""
    times = [0.6 * k for k in range(len(pairs))] if times is None else times
    out = []
    for (g, r), t in zip(pairs, times):
        s = scored_pixels(gt_map(g), C.TABLE, view, np.ones((1, len(g)), dtype=bool))
        objs = gt_objects(s.classes, s.mask)
        out.append(tracks_on_frame(objs, objs.ids, region_map(r), 0, t))
    return out


def pieces_of(gts: list[str]):
    return [component_objects(gt_map(g), np.ones((1, len(g)), dtype=bool), 0) for g in gts]


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
    f = class_frames([("LLLLLGGGGG", "0000011122")])[0]
    assert f.tracks == (LIVER, GALLBLADDER) and f.n_regions == 3
    assert dict(f.hits) == {LIVER: 0, GALLBLADDER: 1}
    assert dict(f.ious) == pytest.approx({(LIVER, 0): 1.0, (GALLBLADDER, 1): 0.6, (GALLBLADDER, 2): 0.4})


def test_a_view_removes_its_left_out_classes_and_their_regions_only():
    pairs = [("LLLLLTTTTT", "0000011111")]
    assert class_frames(pairs)[0].tracks == (LIVER, GRASPER)
    geometric = class_frames(pairs, view="geometric")[0]
    assert geometric.tracks == (LIVER,) and geometric.n_regions == 1 and dict(geometric.hits) == {LIVER: 0}


def test_hold_follows_the_region_picked_on_the_seed_frame():
    h = hold(class_frames(HOLD_SCENE, HOLD_TIMES), seed=2)
    # The liver's samples: 0.8, 0 (region 0 is gone), 1.0, 1.0. The gallbladder's: 0 on frames 0, 1 and 3.
    assert h["hold_mean"] == pytest.approx(2.8 / 4)
    assert h["hold_all_mean"] == pytest.approx(2.8 / 7)
    assert h["picked_share"] == 0.5
    assert (h["n_seed_tracks"], h["n_picked"], h["n_hold_samples"], h["n_hold_all_samples"]) == (2, 1, 4, 7)


def test_a_track_absent_on_the_seed_frame_gives_no_sample():
    # The grasper enters after the seed frame, so only the liver is followed.
    frames = class_frames([("LLLLLLLLLL", "0000000000"), ("LLLLLTTTTT", "0000011111"), ("LLLLLTTTTT", "0000011111")])
    h = hold(frames, seed=0)
    assert (h["hold_all_mean"], h["n_hold_all_samples"], h["n_seed_tracks"]) == (1.0, 2, 1)
    # The share is over the seed frame's tracks, not the clip's.
    assert h["picked_share"] == 1.0


def test_the_hold_curve_bins_the_picked_samples_by_seconds_from_the_seed():
    curve = hold(class_frames(HOLD_SCENE, HOLD_TIMES), seed=2)["hold_curve"]
    assert [(b["above_sec"], b["to_sec"], b["n"]) for b in curve] == [
        (0.0, 0.5, 0), (0.5, 1.0, 2), (1.0, 2.0, 2), (2.0, 4.0, 0), (4.0, 8.0, 0), (8.0, None, 0)]
    # 0.6 s away: frames 1 (0) and 3 (1.0). 1.2 s away: frames 0 (0.8) and 4 (1.0).
    assert [b["mean"] for b in curve] == pytest.approx([None, 0.5, 0.9, None, None, None])


def test_the_hold_curve_reads_seconds_not_frames():
    # Two later frames, 0.3 s and 3.0 s from the seed: the first and the fourth bin, not the second and the third.
    frames = class_frames([("LLLLLLLLLL", "0000000000")] * 3, times=[0.0, 0.3, 3.0])
    assert [b["n"] for b in hold(frames, seed=0)["hold_curve"]] == [1, 0, 0, 1, 0, 0]


def test_a_sample_on_a_bin_edge_goes_to_the_bin_below_it():
    frames = class_frames([("LLLLLLLLLL", "0000000000")] * 2, times=[0.0, 1.0])
    assert [b["n"] for b in hold(frames, seed=0)["hold_curve"]] == [0, 1, 0, 0, 0, 0]


def test_hold_is_not_defined_when_the_seed_frame_has_no_class_in_the_view():
    h = hold(class_frames([("TTTTTTTTTT", "0000000000")] * 3, view="geometric"), seed=1)
    assert (h["hold_mean"], h["hold_all_mean"], h["picked_share"]) == (None, None, None)


def test_switches_and_fragmentations_walk_each_track_where_it_is_present():
    needs_scipy()
    frames = class_frames([
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
    # Together: L-0 on 7 frames, G-1 on 1, G-2 on 2; the best one region per track is L-0 and G-2, 9 frames.
    # GT objects 6 x 2 + 1 = 13; predicted objects 6 + 2 + 2 + 6 + 6 + 2 + 1 = 25.
    assert ident["id_true_positives"] == 9 and (ident["n_gt_objects"], ident["n_pred_objects"]) == (13, 25)
    assert ident["IDF1"] == pytest.approx(18 / 38)


def test_idf1_gives_each_track_its_best_region_over_the_clip_not_the_first():
    needs_scipy()
    # A-0 together on 3 frames, A-1 on 2, B-0 on 2. Greedy takes A-0 and leaves B nothing: 3. The best is 4.
    a, b = 1, 2
    frames = [FrameTracks(0.0, (a,), {a: 0}, 1, {})] * 3 + [FrameTracks(0.0, (a, b), {a: 1, b: 0}, 2, {})] * 2
    assert id_true_positives(frames) == 4


def test_a_hit_after_a_miss_by_another_region_is_a_switch():
    needs_scipy()
    frames = class_frames([
        ("LLLLLLLLLL", "0000000000"),  # L-0
        ("LLLLLLLLLL", "0123456789"),  # L's regions are 0.1 each, no hit: a fragmentation
        ("LLLLLLLLLL", "1111111111"),  # L-1: another region than the last hit, a switch
    ])
    ident = identity(frames)
    assert (ident["ID_switches"], ident["fragmentations"]) == (1, 1)


def test_idf1_is_not_defined_without_a_gt_object():
    ident = identity(class_frames([("TTTTTTTTTT", "0000000000")] * 2, view="geometric"))
    assert ident["IDF1"] is None and ident["n_pred_objects"] == 0


def test_reentry_compares_the_regions_on_either_side_of_a_gap():
    frames = class_frames([
        ("LLLLLGGGGG", "0000011111"),  # L-0
        ("GGGGGGGGGG", "1111111111"),  # L absent
        ("LLLLLGGGGG", "0000011111"),  # L-0: the gap is recovered
        ("GGGGGGGGGG", "1111111111"),
        ("LLLLLGGGGG", "2222211111"),  # L-2: not recovered
        ("GGGGGGGGGG", "1111111111"),
        ("LLLLLGGGGG", "3456711111"),  # L's regions are 0.2 each, no hit: the gap is unmatched
    ])
    assert reentry(frames) == {"reentry": 0.5, "n_gaps": 2, "n_gaps_recovered": 1, "n_gaps_unmatched": 1}


def test_a_piece_is_linked_to_the_track_whose_last_piece_it_overlaps():
    # Liver 0-3 then 2-5: IoU 2/6. The gallbladder's 6-9 overlaps its 4-9 at 4/6; its new piece 0-1 starts a track.
    assert link_pieces(pieces_of(["LLLLGGGGGG", "GGLLLLGGGG"])) == [(0, 1), (0, 2, 1)]


def test_a_track_is_compared_by_its_last_piece_not_its_first():
    # The liver drifts 0-3, 2-5, 4-7: IoU 2/6 with the piece before, 0 with the first.
    assert link_pieces(pieces_of(["LLLLGGGGGG", "GGLLLLGGGG", "GGGGLLLLGG"])) == [(0, 1), (0, 2, 1), (0, 2, 1)]


def test_a_piece_at_the_link_iou_exactly_is_linked():
    # Liver 0-9 then 0-2: IoU 3/10, the threshold itself.
    assert link_pieces(pieces_of(["LLLLLLLLLL", "LLLGGGGGGG"])) == [(0,), (0, 1)]


def test_a_piece_below_the_link_iou_starts_a_track():
    # Liver 0-3 then 3-6: IoU 1/7. The gallbladder's 7-9 overlaps its 4-9 at 3/6.
    assert link_pieces(pieces_of(["LLLLGGGGGG", "GGGLLLLGGG"])) == [(0, 1), (2, 3, 1)]


def test_a_track_waits_three_frames_without_a_piece_and_not_four():
    back_after_three = ["LLLLLGGGGG"] + ["GGGGGGGGGG"] * 3 + ["LLLLLGGGGG"]
    assert link_pieces(pieces_of(back_after_three))[-1] == (0, 1)
    back_after_four = ["LLLLLGGGGG"] + ["GGGGGGGGGG"] * 4 + ["LLLLLGGGGG"]
    assert link_pieces(pieces_of(back_after_four))[-1] == (2, 1)


def test_pieces_of_two_classes_never_share_a_track():
    # The liver moves onto the gallbladder's place: IoU 1 with the gallbladder's track, 0 with its own.
    assert link_pieces(pieces_of(["LLLLLGGGGG", "GGGGGLLLLL"])) == [(0, 1), (2, 3)]


def test_the_piece_with_the_higher_iou_takes_the_track():
    # Liver 0-5, then pieces 0-1 (IoU 2/6) and 3-6 (IoU 3/7): the second keeps the track, the first starts one.
    assert link_pieces(pieces_of(["LLLLLLGGGG", "LLGLLLLGGG"])) == [(0, 1), (2, 0, 3, 1)]


def test_equal_ious_keep_the_order_of_the_pieces_then_of_the_tracks():
    # Liver 0-5, then pieces 0-1 and 4-5 at IoU 2/6 each: the first piece keeps the track.
    assert link_pieces(pieces_of(["LLLLLLGGGG", "LLGGLLGGGG"])) == [(0, 1), (0, 2, 3, 1)]
    # Liver 0-2 and 7-9, then 0-9 at IoU 3/10 with each: the earlier track takes it.
    assert link_pieces(pieces_of(["LLLGGGGLLL", "LLLLLLLLLL"])) == [(0, 1, 2), (0,)]


# What the tracker records for the hold scene: a seed on the centre of five frames, carried both ways.
BOTH_WAYS = {"seed_source": "sam", "seed_frame": 2, "bidir": True, "frames": [0, 1, 2, 3, 4]}
FORWARD = {"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2, 3]}


def write_hold_scene(tmp_path, clip="VID01_a", tag="c", seed_info=BOTH_WAYS):
    C.write_clip(tmp_path / "data", clip, [gt_map(g) for g, _ in HOLD_SCENE], times=HOLD_TIMES)
    C.write_labels(tmp_path / "tracks", clip, tag, {k: region_map(r) for k, (_, r) in enumerate(HOLD_SCENE)},
                   seed_info=seed_info)


def write_scene(tmp_path, scene, seed_info, gt_frames=None, times=None):
    C.write_clip(tmp_path / "data", "VID01_a", [gt_map(g) for g, _ in scene], gt_frames=gt_frames, times=times)
    C.write_labels(tmp_path / "tracks", "VID01_a", "c", {k: region_map(r) for k, (_, r) in enumerate(scene)},
                   seed_info=seed_info)


def measure(tmp_path, clips=("VID01_a",), tag="c", dataset="cholecseg8k"):
    return measure_condition(dataset, None, list(clips), tmp_path / "data", tmp_path / "tracks", tag, SIZES)


def test_a_clip_on_disk_gives_the_hand_derived_values_in_every_view(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    summary = measure(tmp_path)
    assert summary["propagation"] == "both_ways_from_centre" and summary["views"] == list(VIEWS)
    row = summary["per_clip"][0]
    assert (row["seed_frame"], row["n_scored_frames"], row["n_excluded_frames"]) == (2, 5, 0)
    for view in VIEWS:  # liver and gallbladder are tissue, so every view scores the same pixels
        assert row[f"hold_mean/{view}"] == pytest.approx(0.7)
        assert row[f"hold_all_mean/{view}"] == pytest.approx(0.4)
        assert row[f"picked_share/{view}"] == 0.5
        # L: hit by 0 on every frame but 1, a fragmentation. G: hit by 1 on 0, 1 and 3, a fragmentation on 2.
        assert (row[f"ID_switches/{view}"], row[f"fragmentations/{view}"]) == (0, 2)
        # L-0 on 4 frames and G-1 on 3: 2 x 7 / (9 GT objects + 10 predicted objects).
        assert row[f"IDF1/{view}"] == pytest.approx(14 / 19)
        assert row[f"reentry/{view}"] is None and row["counts"][view]["n_gaps"] == 0
        # Each class is one piece on each frame, linked into one track, so at size 0 the values are the class's.
        at0 = row["by_min_px"]["0"]
        assert {k: at0[f"{k}/{view}"] for k in track_metrics.MEASURES} == pytest.approx(
            {k: row[f"{k}/{view}"] for k in track_metrics.MEASURES})


def test_a_liver_cut_in_two_is_two_component_tracks_and_one_class_track(tmp_path):
    needs_scipy()
    write_scene(tmp_path, [
        ("LLLTTTLLLL", "0001112222"),  # seed. Geometric: the liver's 7 pixels; region 2 hits it at 4/7
        ("LLLTTTLLLL", "0001112222"),  # pieces 0-2 and 6-9 each hit by its own region at 1.0
        ("LLLTTTLLLL", "3331114222"),  # region 0 is gone; piece 0-2 is hit by region 3, piece 6-9 by region 2 at 3/4
    ], {**FORWARD, "frames": [0, 1, 2]})
    row = measure(tmp_path)["per_clip"][0]
    # The whole liver: region 2 at 4/7, then 3/7 and no hit, a fragmentation.
    assert row["hold_mean/geometric"] == pytest.approx(0.5) and row["fragmentations/geometric"] == 1
    at0, at4 = row["by_min_px"]["0"], row["by_min_px"]["4"]
    # Piece 0-2: 1.0, then 0. Piece 6-9: 1.0, then 3/4. The first piece's move from region 0 to 3 is a switch.
    assert at0["hold_mean/geometric"] == pytest.approx(2.75 / 4) and at0["ID_switches/geometric"] == 1
    # At 4 pixels the 3-pixel piece goes, and so do regions 0, 3 and 4, and region 2 on the last frame. Hold still
    # reads region 2's overlap there, 3/4; the pairing has no region left, a fragmentation.
    assert at4["hold_mean/geometric"] == pytest.approx(1.75 / 2) and at4["fragmentations/geometric"] == 1
    # IDF1: piece 6-9 with region 2 on two frames, 3 GT objects, 2 predicted objects.
    assert at4["IDF1/geometric"] == pytest.approx(4 / 5)
    assert (at0["counts"]["geometric"]["n_tracks"], at4["counts"]["geometric"]["n_tracks"]) == (2, 1)


def test_hold_starts_on_the_seed_frames_place_among_the_gt_frames(tmp_path):
    needs_scipy()
    # Frames 1 and 5 have no GT, so the seed frame 3 is the third GT frame. Region 0 and 1 cover the liver and the
    # gallbladder on every GT frame but 4, where regions 2 and 3 do.
    scene = [("LLLLLGGGGG", "0000011111")] * 4 + [("LLLLLGGGGG", "2222233333")] + [("LLLLLGGGGG", "0000011111")] * 2
    write_scene(tmp_path, scene, {"seed_source": "sam", "seed_frame": 3, "bidir": True, "frames": list(range(7))},
                gt_frames=[0, 2, 3, 4, 6])
    row = measure(tmp_path)["per_clip"][0]
    # Picked on frame 3: regions 0 and 1. Samples on frames 0, 2, 4 and 6: 1, 1, 0, 1 for each class.
    assert row["hold_mean/all"] == pytest.approx(6 / 8)


def test_the_frames_are_walked_in_time_order_not_in_index_order(tmp_path):
    needs_scipy()
    # Frame 1 comes last in time. In time order the liver is hit by 0, 0, 1: one switch; by index, two.
    write_scene(tmp_path, [("LLLLLLLLLL", "0000000000"), ("LLLLLLLLLL", "1111111111"), ("LLLLLLLLLL", "0000000000")],
                {"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2]}, times=[0.0, 1.2, 0.6])
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
    assert (row["n_scored_frames"], row["n_excluded_frames"]) == (2, 1)
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
     "holds no propagation rule: the seed came from 'gt'"),
])
def test_a_condition_without_a_seed_frame_to_hold_from_is_refused(tmp_path, info, why):
    write_hold_scene(tmp_path, seed_info=info)
    with pytest.raises(ValueError, match=why):
        measure(tmp_path)


def test_seeds_made_outside_the_tracking_stage_are_refused_since_the_json_records_no_seed(tmp_path):
    write_hold_scene(tmp_path, seed_info={**BOTH_WAYS, "seed_source": "external"})
    with pytest.raises(ValueError, match="made outside the tracking stage"):
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


def test_the_json_records_the_hashes_of_the_tool_and_of_component_scores(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    summary = measure(tmp_path)
    assert summary["tool_sha"] == tool_sha(track_metrics.__file__, component_scores.__file__)
    assert summary["min_px"] == [0, 4] and set(summary["per_clip"][0]["by_min_px"]) == {"0", "4"}


def test_the_json_records_each_clips_input_hashes_and_its_frame_times(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    shas = measure(tmp_path)["input_shas"]["VID01_a"]
    inputs = read_clip(tmp_path / "data", tmp_path / "tracks", "c", "VID01_a", C.TABLE)
    assert {k: shas[k] for k in inputs.shas} == dict(inputs.shas)
    # The same frames at other times: the hold curve bins by seconds, so the times are hashed too.
    write_scene(tmp_path / "later", HOLD_SCENE, BOTH_WAYS, times=[2 * t for t in HOLD_TIMES])
    later = measure(tmp_path / "later")["input_shas"]["VID01_a"]
    assert {k: later[k] for k in inputs.shas} == dict(inputs.shas) and later["times"] != shas["times"]


def test_the_hashes_are_taken_before_any_clip_is_measured(tmp_path, monkeypatch):
    needs_scipy()
    # A source file edited during a run must not relabel the values of the code in memory.
    write_hold_scene(tmp_path)
    calls = []
    monkeypatch.setattr(track_metrics, "eval_code_sha", lambda: calls.append("eval") or "e")
    monkeypatch.setattr(track_metrics, "tool_sha", lambda *paths: calls.append("tool") or {})
    monkeypatch.setattr(track_metrics, "read_clip", lambda *a: calls.append("clip") or read_clip(*a))
    summary = measure(tmp_path)
    assert calls.index("clip") > max(calls.index("eval"), calls.index("tool"))
    assert (summary["eval_code_sha"], summary["tool_sha"]) == ("e", {})


def test_the_score_tools_refuse_its_json_as_a_score(tmp_path):
    needs_scipy()
    write_hold_scene(tmp_path)
    with pytest.raises(ValueError, match="so this one is neither"):
        ruler(measure(tmp_path))


def test_the_command_line_writes_the_json_at_every_fixed_size(tmp_path, capsys):
    needs_scipy()
    write_hold_scene(tmp_path)
    (tmp_path / "clips.txt").write_text("VID01_a\n")
    out = tmp_path / "c.identity.json"
    main(["--dataset", "cholecseg8k", "--clips", str(tmp_path / "clips.txt"), "--data-root", str(tmp_path / "data"),
          "--tracks-root", str(tmp_path / "tracks"), "--tag", "c", "--out", str(out)])
    written = json.loads(out.read_text())
    assert written["per_clip"][0]["hold_mean/all"] == pytest.approx(0.7)
    assert written["min_px"] == [0, 100, 300, 400]
    # At 100 pixels nothing of a ten-pixel frame is left: no track, so hold is not defined.
    assert written["per_clip"][0]["by_min_px"]["100"]["hold_mean/all"] is None
    assert "1 clips measured" in capsys.readouterr().out
