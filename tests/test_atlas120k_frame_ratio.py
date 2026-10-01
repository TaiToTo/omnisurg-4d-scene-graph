"""The clip-index to mp4 frame ratio: reading the table and refusing what is not in it.

Whether a ratio is right can only be checked against pixels, so what is tested
here is that the table is used safely: an unmeasured video is refused rather
than given a ratio of 1, a malformed table is refused as a whole, and the
committed table covers every video the paper measures.
"""

import json
from pathlib import Path

import pytest

from surgical_core.atlas120k.frame_ratio import FrameRatios

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"


@pytest.fixture(scope="module")
def table():
    return FrameRatios.load(str(META / "frame_ratio.json"))


def _write(tmp_path, videos, match_tol=8.0):
    p = tmp_path / "frame_ratio.json"
    p.write_text(json.dumps({"match_tol": match_tol, "videos": videos}), encoding="utf-8")
    return str(p)


def test_measured_ratios_are_looked_up(table):
    assert table.ratio("appendectomy", "41RKDh3INiU") == 1
    assert table.mp4_index("appendectomy", "41RKDh3INiU", 1234) == 1234
    assert table.ratio("cholecystectomy", "1ud3syYKD3A") == 4
    assert table.mp4_index("cholecystectomy", "1ud3syYKD3A", 30) == 120
    assert table.mp4_index("liver_resection", "wAnvytUuDkA", 1152) == 2304


def test_frame_numbers_may_be_strings(table):
    """`clip_index.json` sometimes writes the number as a string."""
    assert table.mp4_index("cholecystectomy", "Va3QomCEaTE", "30") == 120


def test_unmeasured_video_is_refused(table):
    """A video not in the table is not given ratio 1. That silence is how a
    frame from another moment gets in."""
    with pytest.raises(KeyError, match="not been measured"):
        table.ratio("newprocedure", "newvideo")
    with pytest.raises(KeyError, match="not been measured"):
        table.mp4_index("newprocedure", "newvideo", 0)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameRatios.load(str(tmp_path / "no_such.json"))


def test_malformed_ratio_is_refused(tmp_path):
    for bad in (0, -1, 2.5, "2"):
        p = _write(tmp_path, [{"procedure": "p", "video": "v", "ratio": bad}])
        with pytest.raises(ValueError, match="positive integer"):
            FrameRatios.load(p)


def test_video_measured_twice_is_refused(tmp_path):
    p = _write(tmp_path, [{"procedure": "p", "video": "v", "ratio": 1},
                          {"procedure": "p", "video": "v", "ratio": 2}])
    with pytest.raises(ValueError, match="twice"):
        FrameRatios.load(p)


def test_committed_table_covers_the_population(table):
    """Every video the paper measures was measured, and the ratios are the
    three that were observed."""
    assert len(table) == 97
    videos = set()
    for line in (META / "clips.txt").read_text(encoding="utf-8").split():
        procedure, rest = line.split("__", 1)
        videos.add((procedure, rest.rsplit("__gt_", 1)[0]))
    assert all(v in table for v in videos)
    ratios = {table.ratio(*v) for v in videos}
    assert ratios <= {1, 2, 3, 4}


def test_committed_table_matches_within_tolerance():
    """Each listed match is a real match: well inside the tolerance that
    separates a match from a miss."""
    data = json.loads((META / "frame_ratio.json").read_text(encoding="utf-8"))
    assert data["n_videos"] == len(data["videos"]) == 97
    for row in data["videos"]:
        assert row["mp4_frame"] == row["native_frame"] * row["ratio"], row
        assert row["diff"] < data["match_tol"] / 4, row
