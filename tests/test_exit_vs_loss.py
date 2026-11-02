"""Exit against loss: the frustum test, the sorting rules, the events, the matching and the controls."""
import json

import numpy as np
import pytest

from evalkit.tools.exit_vs_loss import (
    gt_events_of, read_gt_tracks,
    Event, classify, events_of, inside, inverted, known_position, match_events, shuffled_poses, static_poses,
)

K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
SIZE = (100, 80)
IDENTITY = np.hstack([np.eye(3), np.zeros((3, 1))])


def pose(tx: float = 0.0) -> np.ndarray:
    """A world-to-camera pose that moves the camera by `-tx` along x: a world point appears shifted by `tx`."""
    return np.hstack([np.eye(3), np.array([[tx], [0.0], [0.0]])])


def test_a_point_ahead_and_inside_the_frame_is_inside_and_one_behind_or_beside_is_not():
    assert inside(np.array([0.0, 0.0, 2.0]), K, IDENTITY, SIZE, 0.05)
    assert not inside(np.array([0.0, 0.0, -2.0]), K, IDENTITY, SIZE, 0.05)
    # At z = 2 the frame's right edge is x = 1.0; x = 1.2 projects to u = 110.
    assert not inside(np.array([1.2, 0.0, 2.0]), K, IDENTITY, SIZE, 0.0)


def test_the_margin_takes_a_band_off_each_side():
    p = np.array([0.9, 0.0, 2.0])  # u = 95: inside the frame, inside a 5 % band of the right edge
    assert inside(p, K, IDENTITY, SIZE, 0.0)
    assert not inside(p, K, IDENTITY, SIZE, 0.1)


def test_the_test_does_not_change_when_the_scene_and_the_translations_are_scaled_together():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.normal(size=3) * 2
        E = np.hstack([np.linalg.qr(rng.normal(size=(3, 3)))[0], rng.normal(size=(3, 1))])
        for s in (0.01, 7.0, 1e3):
            Es = E.copy()
            Es[:, 3] *= s
            assert inside(p, K, E, SIZE, 0.05) == inside(p * s, K, Es, SIZE, 0.05)


def test_an_exit_whose_point_stays_in_view_is_a_loss_and_one_out_of_view_on_its_frame_a_true_exit():
    frames = list(range(6))
    Kn = np.repeat(K[None], 6, axis=0)
    still = np.repeat(IDENTITY[None], 6, axis=0)
    assert classify("exit", np.array([0.0, 0.0, 2.0]), 2, frames, Kn, still, SIZE, 0.05, 3) == ("lost_in_view", False)
    assert classify("exit", np.array([0.0, 0.0, 2.0]), 4, frames, Kn, still, SIZE, 0.05, 3) == ("lost_in_view", True)
    # From frame 2 on the camera has turned away: the point projects far outside.
    away = np.stack([IDENTITY, IDENTITY] + [pose(5.0)] * 4)
    assert classify("exit", np.array([0.0, 0.0, 2.0]), 2, frames, Kn, away, SIZE, 0.05, 3) == ("left_view", False)
    # In view on frame 2 and 3, out on frame 4: out within the window.
    later = np.stack([IDENTITY] * 4 + [pose(5.0)] * 2)
    assert classify("exit", np.array([0.0, 0.0, 2.0]), 2, frames, Kn, later, SIZE, 0.05, 3) == ("leaving", False)


def test_an_enter_reads_the_window_before_it_and_the_frame_before_decides():
    frames = list(range(6))
    Kn = np.repeat(K[None], 6, axis=0)
    still = np.repeat(IDENTITY[None], 6, axis=0)
    p = np.array([0.0, 0.0, 2.0])
    assert classify("enter", p, 4, frames, Kn, still, SIZE, 0.05, 3) == ("reacquired_in_view", False)
    assert classify("enter", p, 2, frames, Kn, still, SIZE, 0.05, 3) == ("reacquired_in_view", True)
    before_out = np.stack([pose(5.0)] * 4 + [IDENTITY] * 2)
    assert classify("enter", p, 4, frames, Kn, before_out, SIZE, 0.05, 3) == ("entered_view", False)
    mixed = np.stack([pose(5.0), IDENTITY, IDENTITY, IDENTITY, IDENTITY, IDENTITY])
    assert classify("enter", p, 3, frames, Kn, mixed, SIZE, 0.05, 3) == ("entering", False)


def test_a_window_whose_deciding_frame_is_not_a_frame_is_refused():
    with pytest.raises(ValueError, match="decides"):
        classify("enter", np.zeros(3), 0, [0, 1, 2], np.repeat(K[None], 3, 0), np.repeat(IDENTITY[None], 3, 0),
                 SIZE, 0.0, 3)


def test_events_are_each_change_of_presence_and_a_node_never_present_has_none():
    assert [(e.kind, e.t) for e in events_of({0, 1, 2, 5, 6}, range(8))] == [("exit", 3), ("enter", 5), ("exit", 7)]
    assert events_of(set(), range(8)) == []
    assert events_of({0, 1, 2}, range(3)) == []


def test_the_known_position_is_the_mean_of_the_last_k_present_frames_before_an_exit_and_the_first_k_from_an_enter():
    cen = {t: np.array([float(t), 0.0, 0.0]) for t in (0, 1, 2, 3, 7, 8)}
    assert known_position(cen, 4, 3, "exit").tolist() == [2.0, 0.0, 0.0]
    assert known_position(cen, 7, 3, "enter").tolist() == [7.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="no present frame"):
        known_position(cen, 0, 3, "exit")


def test_events_match_within_one_frame_closest_first_and_each_once():
    pred = [Event("exit", 0, 5), Event("exit", 0, 6), Event("enter", 0, 5), Event("exit", 1, 5)]
    gt = [Event("exit", 0, 6), Event("exit", 0, 8)]
    assert match_events(pred, gt) == {1: 0}
    assert match_events(pred, gt, tol=3) == {1: 0, 0: 1}


def test_the_still_camera_repeats_the_seed_pose_and_a_shuffle_keeps_the_poses():
    E = np.stack([pose(float(t)) for t in range(5)])
    assert (static_poses(E, 2) == E[2]).all()
    S = shuffled_poses(E, np.random.default_rng(0))
    assert sorted(S[:, 0, 3].tolist()) == sorted(E[:, 0, 3].tolist()) and not (S == E).all()


def test_the_inverted_reading_is_the_pose_s_inverse():
    rng = np.random.default_rng(1)
    E = np.hstack([np.linalg.qr(rng.normal(size=(3, 3)))[0], rng.normal(size=(3, 1))])[None]
    full = np.vstack([E[0], [0, 0, 0, 1]])
    assert np.allclose(np.vstack([inverted(E)[0], [0, 0, 0, 1]]) @ full, np.eye(4))


def test_read_gt_tracks_gives_each_node_its_frames_and_refuses_a_stray_index(tmp_path):
    """Two nodes on three GT frames: node 1 leaves on frame 2; a map with index 2 is refused."""
    d = tmp_path / "clip"
    d.mkdir()
    (d / "nodes.json").write_text(json.dumps({"class_ids": [3, 7]}))
    for t in range(3):
        ids = np.full((8, 8), -1, dtype=np.int32)
        ids[:4] = 0
        if t < 2:
            ids[4:] = 1
        np.save(d / f"label_{t:04d}.npy", ids)
    classes, frames, masks = read_gt_tracks(d, (8, 8), 1)
    assert classes == [3, 7] and frames == [0, 1, 2]
    assert sorted(masks[0]) == [0, 1, 2] and sorted(masks[1]) == [0, 1]
    assert [(e.kind, e.t) for e in gt_events_of(set(masks[1]), frames, 0)] == [("exit", 2)]
    bad = np.full((8, 8), 2, dtype=np.int32)
    np.save(d / "label_0003.npy", bad)
    with pytest.raises(ValueError, match="outside -1..1"):
        read_gt_tracks(d, (8, 8), 1)
