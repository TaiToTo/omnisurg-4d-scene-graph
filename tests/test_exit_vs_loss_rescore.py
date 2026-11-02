"""The pose-free reference: border contact, the reference label of an exit and an enter, the baseline, the metrics."""
import numpy as np

from evalkit.tools.exit_vs_loss_rescore import boundary2d, box, metrics, reference, touches

SHAPE = (10, 20)


def mask(y0, y1, x0, x1):
    m = np.zeros(SHAPE, dtype=bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def test_box_and_touches_read_the_border_within_b():
    assert box(mask(2, 4, 3, 7)) == (2, 4, 3, 7)
    assert box(np.zeros(SHAPE, dtype=bool)) is None
    assert not touches((2, 4, 3, 7), SHAPE, 1.0)
    assert touches((2, 4, 1, 7), SHAPE, 1.0)       # x0 within 1 px of the left edge
    assert touches((2, 4, 3, 18), SHAPE, 1.0)      # x1 within 1 px of the right edge
    assert touches((2, 8, 3, 7), SHAPE, 1.0)       # y1 within 1 px of the bottom edge


def test_reference_sorts_an_exit_by_presence_then_by_the_last_mask():
    gt_frames = set(range(6))
    inside = {0: mask(3, 5, 8, 12), 1: mask(3, 5, 8, 12)}
    assert reference("exit", 1, inside, gt_frames, SHAPE, 0.4) == ("gt_present", 1)
    assert reference("exit", 2, inside, gt_frames, SHAPE, 0.4) == ("gt_exit_interior", 1)
    at_border = {0: mask(3, 5, 0, 4), 1: mask(3, 5, 0, 4)}
    assert reference("exit", 3, at_border, gt_frames, SHAPE, 0.4) == ("gt_exit_border", 1)
    assert reference("exit", 9, at_border, gt_frames, SHAPE, 0.4) == ("gt_unverifiable", None)
    assert reference("exit", 2, {}, gt_frames, SHAPE, 0.4) == ("gt_unverifiable", None)


def test_reference_sorts_an_enter_backwards():
    gt_frames = set(range(6))
    present = {4: mask(3, 5, 16, 19), 5: mask(3, 5, 16, 19)}
    assert reference("enter", 5, present, gt_frames, SHAPE, 0.4) == ("gt_present", 4)
    assert reference("enter", 3, present, gt_frames, SHAPE, 0.4) == ("gt_enter_border", 4)
    assert reference("enter", 3, {4: mask(3, 5, 8, 12)}, gt_frames, SHAPE, 0.4) == ("gt_enter_interior", 4)


def test_boundary2d_reads_the_tracked_mask_on_its_last_or_first_frame():
    tracked = {0: mask(3, 5, 8, 12), 1: mask(3, 5, 0, 4)}
    assert boundary2d("exit", 2, tracked, SHAPE, 0.4) == "left_view"
    assert boundary2d("exit", 1, tracked, SHAPE, 0.4) == "lost_in_view"
    assert boundary2d("enter", 1, tracked, SHAPE, 0.4) == "entered_view"


def test_metrics_count_precision_recall_and_the_false_exits_called_in_view():
    """Six events of three videos: four true, two false; the condition keeps four, three of them true."""
    rng = np.random.default_rng(0)
    tp = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    present = ~tp
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    hit = [("a", "c1", 0, 3), ("a", "c1", 1, 4), ("b", "c2", 0, 2), ("b", "c2", 0, 2), None, None]
    before = metrics(np.ones(6, dtype=bool), tp, present, np.zeros(6, dtype=bool), hit, 3, groups, None, rng)
    assert (before["precision"], before["recall"], before["gt_true_picked"]) == (4 / 6, 1.0, 3)
    kept = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    in_view = ~kept
    after = metrics(kept, tp, present, in_view, hit, 3, groups, before, rng)
    assert (after["precision"], after["tp"], after["fp"]) == (3 / 4, 3, 1)
    assert after["recall"] == 1.0 and after["false_exit_in_view_share"] == 0.5
    assert after["tp_retention"] == 3 / 4 and abs(after["precision_diff_from_before"] - (3 / 4 - 4 / 6)) < 1e-12
    lo, hi = after["precision_diff_ci95_video"]
    assert lo <= after["precision_diff_from_before"] <= hi
    assert after["precision_diff_mark"] == ""  # three videos give no mark
