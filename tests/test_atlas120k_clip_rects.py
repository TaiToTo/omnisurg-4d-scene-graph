"""Reading the per-clip crop rectangles a person confirmed.

Whether a rectangle is right can only be judged by eye, so what is tested here
is the reading: which verdicts are taken and which fall through, that a
degenerate rectangle stops the load, and how the per-video fallback works.
The last test reads the committed file and pins that every clip the paper
measures has a confirmed rectangle in it.
"""

import json
from pathlib import Path

import pytest

from surgical_core.atlas120k.clip_rects import MIN_SIDE, load_clip_rects, rect_for

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"
DEFAULT = (0, 0, 100, 100)


def _write(tmp_path, rows):
    p = tmp_path / "verdicts.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)


def _row(**kw):
    base = dict(procedure="proc", video="vid", clip="clip_0001",
                rect=[10, 20, 300, 200], verdict="ng", src_size=[1920, 1080])
    base.update(kw)
    return base


def test_ok_and_ng_are_both_adopted(tmp_path):
    """Accepted and redrawn alike, `rect` is the rectangle to use."""
    p = _write(tmp_path, [_row(verdict="ok"), _row(clip="clip_0002", verdict="ng")])
    t = load_clip_rects(p)
    assert t[("proc", "vid", "clip_0001")] == (10, 20, 300, 200)
    assert t[("proc", "vid", "clip_0002")] == (10, 20, 300, 200)


def test_skip_and_unjudged_fall_through(tmp_path):
    """A skipped or unjudged clip is absent, so the caller falls back to the video's rectangle."""
    p = _write(tmp_path, [_row(verdict="skip"), _row(clip="clip_0002", verdict="")])
    assert load_clip_rects(p) == {}


def test_last_wins(tmp_path):
    """The tool appends, so a correction comes after what it corrects."""
    p = _write(tmp_path, [_row(rect=[0, 0, 500, 500]), _row(rect=[1, 2, 300, 400])])
    assert load_clip_rects(p)[("proc", "vid", "clip_0001")] == (1, 2, 300, 400)


def test_a_later_skip_withdraws_the_rectangle(tmp_path):
    """The later entry wins for `skip` too: a clip withdrawn after it was
    accepted falls back to the video's rectangle."""
    p = _write(tmp_path, [_row(verdict="ok"), _row(verdict="skip")])
    assert load_clip_rects(p) == {}


def test_rect_is_clamped_into_the_frame(tmp_path):
    """A drag past the frame edge is clamped, not rejected."""
    p = _write(tmp_path, [_row(rect=[1800, 900, 500, 500], src_size=[1920, 1080])])
    assert load_clip_rects(p)[("proc", "vid", "clip_0001")] == (1800, 900, 120, 180)


def test_rect_past_the_left_and_top_is_cut_not_moved(tmp_path):
    """Past the left or top edge, what is kept is the part inside the frame.
    Moving the origin to 0 and keeping the size would crop a region nobody
    drew."""
    p = _write(tmp_path, [_row(rect=[-100, -50, 300, 200], src_size=[640, 480])])
    assert load_clip_rects(p)[("proc", "vid", "clip_0001")] == (0, 0, 200, 150)


def test_rect_without_src_size_is_refused(tmp_path):
    """Without the frame size the rectangle cannot be checked, and a negative
    origin would wrap around in a numpy slice instead of failing."""
    row = _row(rect=[-100, -50, 300, 200])
    del row["src_size"]
    with pytest.raises(ValueError, match="no src_size"):
        load_clip_rects(_write(tmp_path, [row]))


def test_clamping_into_a_degenerate_rect_raises(tmp_path):
    """If clamping leaves a sliver, that is not a crop that "fit"."""
    p = _write(tmp_path, [_row(rect=[1900, 1000, 500, 500], src_size=[1920, 1080])])
    with pytest.raises(ValueError, match="degenerate"):
        load_clip_rects(p)


def test_degenerate_rect_raises(tmp_path):
    """One rectangle drawn wrong stops the load; it is not used as it is."""
    p = _write(tmp_path, [_row(rect=[0, 0, MIN_SIDE - 1, 500])])
    with pytest.raises(ValueError, match="degenerate"):
        load_clip_rects(p)


def test_floats_are_rounded(tmp_path):
    """The tool writes floats; pixels are integers."""
    p = _write(tmp_path, [_row(rect=[10.4, 20.6, 300.5, 200.4])])
    assert load_clip_rects(p)[("proc", "vid", "clip_0001")] == (10, 21, 300, 200)


def test_missing_file_raises(tmp_path):
    """An empty table is not returned for a missing file: a path that has no
    effect is the hardest fault to notice."""
    with pytest.raises(FileNotFoundError):
        load_clip_rects(str(tmp_path / "no_such.json"))


def test_rect_for_reports_its_source(tmp_path):
    t = load_clip_rects(_write(tmp_path, [_row()]))
    assert rect_for(t, "proc", "vid", "clip_0001", DEFAULT) == ((10, 20, 300, 200), "confirmed")
    assert rect_for(t, "proc", "vid", "clip_0009", DEFAULT) == (DEFAULT, "video")
    assert rect_for({}, "proc", "vid", "clip_0001", DEFAULT) == (DEFAULT, "video")


def test_every_measured_clip_has_a_confirmed_rectangle_in_the_committed_file():
    """The committed judgement covers the whole population: no clip the paper
    measures falls back to a per-video rectangle."""
    table = load_clip_rects(str(META / "crop_rects.json"))
    assert len(table) == 494
    for line in (META / "clips.txt").read_text(encoding="utf-8").split():
        procedure, rest = line.split("__", 1)
        video, n = rest.rsplit("__gt_", 1)
        _, source = rect_for(table, procedure, video, f"clip_{n}", DEFAULT)
        assert source == "confirmed", line
