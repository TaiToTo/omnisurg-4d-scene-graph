"""Regions per frame: every id counted on every pixel, averaged over the frames and then over the clips."""
import json

import numpy as np
import pytest

import clip_dirs as C
from evalkit.tools import region_counts
from evalkit.tools.component_scores import tool_sha
from evalkit.tools.region_counts import count_clip, count_condition, frame_count, main


def labels(s: str) -> np.ndarray:
    return np.array([[-1 if ch == "." else int(ch) for ch in s]], dtype=np.int32)


def test_a_frame_counts_each_region_id_once_and_no_region_not_at_all():
    assert frame_count(labels("0011..2223")) == 4
    assert frame_count(labels("..........")) == 0


@pytest.mark.parametrize("bad, why", [
    (np.zeros((2, 2), dtype=np.float32), "integer array"),
    (np.zeros((2, 2, 1), dtype=np.int32), "integer array"),
    (np.full((2, 2), -2, dtype=np.int32), "-1 for no region"),
])
def test_a_map_that_is_no_label_map_is_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        frame_count(bad)


def test_a_clip_averages_over_its_frames_and_a_condition_over_its_clips(tmp_path):
    C.write_labels(tmp_path, "VID01_a", "c", {0: labels("0011223344"), 1: labels("00........")})
    C.write_labels(tmp_path, "VID02_b", "c", {0: labels("0000000000")})
    assert count_clip(tmp_path, "c", "VID01_a") == {"clip": "VID01_a", "regions_per_frame": 3.0, "n_frames": 2}
    summary = count_condition(["VID01_a", "VID02_b"], tmp_path, "c")
    assert summary["regions_per_frame"] == 2.0 and summary["track_dir_name"] == "c"
    assert summary["tool_sha"] == tool_sha(region_counts.__file__)


def test_a_clip_without_predictions_is_refused(tmp_path):
    (tmp_path / "VID01_a" / "c").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no prediction"):
        count_clip(tmp_path, "c", "VID01_a")


def test_a_file_not_named_as_a_label_is_refused(tmp_path):
    C.write_labels(tmp_path, "VID01_a", "c", {0: labels("0011223344")})
    np.save(tmp_path / "VID01_a" / "c" / "label_x.npy", labels("0000000000"))
    with pytest.raises(ValueError, match="label_x.npy"):
        count_clip(tmp_path, "c", "VID01_a")


def test_the_command_line_writes_the_json(tmp_path, capsys):
    C.write_labels(tmp_path / "tracks", "VID01_a", "c", {0: labels("0011223344")})
    (tmp_path / "clips.txt").write_text("VID01_a\n")
    out = tmp_path / "c.regions.json"
    main(["--clips", str(tmp_path / "clips.txt"), "--tracks-root", str(tmp_path / "tracks"), "--tag", "c",
          "--out", str(out)])
    assert json.loads(out.read_text())["per_clip"][0]["regions_per_frame"] == 5.0
    assert "5.00 regions per frame over 1 clips" in capsys.readouterr().out
