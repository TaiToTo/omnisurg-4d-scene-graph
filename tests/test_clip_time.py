"""`surgical_core.clip_time.frame_times`: a pure function, no GPU, no video.

The function is the one place that turns a manifest into seconds, so what
these tests pin is the rule itself: which key wins, that the ratio is applied,
and that every case where seconds cannot be made is refused rather than
filled in with a default.
"""
import importlib.abc
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from surgical_core import clip_time
from surgical_core.clip_time import frame_times


def _clip(tmp_path: Path, frames: list[dict], **meta) -> tuple[str, str]:
    """Write a clip that holds nothing but a `frame_manifest.json`."""
    d = tmp_path / "clip_0001"
    d.mkdir()
    with open(d / "frame_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dict(frames=frames, **meta), f)
    return str(tmp_path), "clip_0001"


def _plant_table(monkeypatch, ratio: int | None) -> None:
    """Stand in for the table of measured ratios, or remove it.

    `None` makes the import fail, whether or not the real module is installed:
    a `None` entry in `sys.modules` is how Python marks a module that cannot
    be imported.
    """
    if ratio is None:
        monkeypatch.setitem(sys.modules, clip_time.MEASURED_RATIOS, None)
        return
    mod = types.ModuleType(clip_time.MEASURED_RATIOS)
    mod.frame_ratio = lambda procedure, youtube_id: ratio
    monkeypatch.setitem(sys.modules, clip_time.MEASURED_RATIOS, mod)


def _plant_broken_table(monkeypatch, missing: str | None) -> None:
    """Stand in for a table that is found but fails to import something.

    The stand-in goes through the real import machinery (a finder on
    `sys.meta_path`), so what `frame_times` sees is exactly what a table with
    a bad import would raise: a `ModuleNotFoundError` naming `missing`, or
    one raised without a name when `missing` is `None`.
    """
    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            if missing is None:
                raise ModuleNotFoundError("raised without a name")
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == clip_time.MEASURED_RATIOS:
                return importlib.machinery.ModuleSpec(name, Loader())
            return None

    # The finder answers for the table only; its parent packages need to
    # exist for the dotted import to reach it.
    parent = clip_time.MEASURED_RATIOS.rpartition(".")[0]
    pkg = types.ModuleType(parent)
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, parent, pkg)
    monkeypatch.delitem(sys.modules, clip_time.MEASURED_RATIOS, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder()] + sys.meta_path)


def test_cholec_uses_timestamp_sec(tmp_path):
    root, clip = _clip(tmp_path, [{"timestamp_sec": 1.5}, {"timestamp_sec": 2.0}])
    assert frame_times(root, clip).tolist() == [1.5, 2.0]


def test_atlas_divides_native_frame_by_fps(tmp_path):
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0)
    assert frame_times(root, clip).tolist() == [1.0, 2.0]


def test_frame_ratio_is_applied(tmp_path):
    """ATLAS-120k's `native_frame` is the clip index's number: without the ratio
    the clip's time shrinks."""
    root, clip = _clip(tmp_path, [{"native_frame": 0}, {"native_frame": 10}],
                       fps_native=59.94, frame_ratio=3)
    t = frame_times(root, clip)
    assert t.tolist() == pytest.approx([0.0, 3 * 10 / 59.94])


def test_no_ratio_and_no_video_means_ratio_1(tmp_path, monkeypatch):
    """A manifest that names no video has no ratio to look up, so it passes at
    1 whether or not a table is there. That keeps the populations without a
    ratio (CholecSeg8k) as they are; an ATLAS-120k clip extracted before the
    ratio was recorded is a different thing, and the next tests refuse it."""
    _plant_table(monkeypatch, None)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0)
    assert frame_times(root, clip).tolist() == [1.0, 2.0]


def test_atlas_manifest_without_ratio_is_refused_when_table_says_otherwise(
        tmp_path, monkeypatch):
    """The failing case must fail: no silent default of 1.

    Such a manifest has neither `frame_ratio` nor `gt_step_sec_actual`, so
    the step check cannot catch it; asking the table is the only way to know.
    """
    _plant_table(monkeypatch, 3)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=60.0, procedure="rarp", youtube_id="AZw_lOLChmM")
    with pytest.raises(RuntimeError, match="measured ratio of this video is 3"):
        frame_times(root, clip)


def test_atlas_manifest_without_ratio_passes_when_table_says_1(tmp_path, monkeypatch):
    """Only a ratio known to differ from 1 is refused."""
    _plant_table(monkeypatch, 1)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0, procedure="adrenalectomy",
                       youtube_id="16GPCUPkXYQ")
    assert frame_times(root, clip).tolist() == [1.0, 2.0]


def test_atlas_manifest_without_ratio_is_refused_without_a_table(tmp_path, monkeypatch):
    """No table to ask means the ratio is unknown, and unknown is refused."""
    _plant_table(monkeypatch, None)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0, procedure="adrenalectomy",
                       youtube_id="16GPCUPkXYQ")
    with pytest.raises(RuntimeError, match="is not available"):
        frame_times(root, clip)


@pytest.mark.parametrize("missing", ["cv2", "surgical_core.atlas", None])
def test_a_dependency_missing_inside_the_table_is_not_called_no_table(
        tmp_path, monkeypatch, missing):
    """A table that is there but cannot import something of its own is a
    broken install, not a missing table, and the error must say which.

    `surgical_core.atlas` is the workbench's old name for the package, the
    import most likely to be left inside the table by mistake, and a string
    prefix of the table's own name: a prefix comparison would call it "no
    table". An error raised without a name names nothing, so it is passed on
    rather than guessed about.
    """
    _plant_broken_table(monkeypatch, missing)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0, procedure="adrenalectomy",
                       youtube_id="16GPCUPkXYQ")
    with pytest.raises(ModuleNotFoundError, match=missing or "without a name"):
        frame_times(root, clip)


def test_a_package_above_the_table_missing_is_no_table(tmp_path, monkeypatch):
    """Until `atlas120k-meta/readers` lands, the package the table lives in
    does not exist either; that is the ordinary "no table" case."""
    parent = clip_time.MEASURED_RATIOS.rpartition(".")[0]
    monkeypatch.setitem(sys.modules, parent, None)
    monkeypatch.delitem(sys.modules, clip_time.MEASURED_RATIOS, raising=False)
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0, procedure="adrenalectomy",
                       youtube_id="16GPCUPkXYQ")
    with pytest.raises(RuntimeError, match="is not available"):
        frame_times(root, clip)


@pytest.mark.parametrize("ratio", [0, -2])
def test_a_ratio_that_is_not_positive_is_refused(tmp_path, ratio):
    """A ratio of 0 makes every time 0 and, without `gt_step_sec_actual`,
    nothing would notice."""
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}],
                       fps_native=30.0, frame_ratio=ratio)
    with pytest.raises(RuntimeError, match="cannot be a ratio"):
        frame_times(root, clip)


def test_step_that_disagrees_with_gt_step_sec_actual_is_refused(tmp_path):
    """The failing case must fail: a clip of ratio 3 claiming ratio 1 has a
    step a third of what the manifest says."""
    root, clip = _clip(tmp_path, [{"native_frame": 0}, {"native_frame": 10}],
                       fps_native=59.94, frame_ratio=1, gt_step_sec_actual=0.500501)
    with pytest.raises(RuntimeError, match="frame step"):
        frame_times(root, clip)


def test_right_ratio_passes_the_step_check(tmp_path):
    root, clip = _clip(tmp_path, [{"native_frame": 0}, {"native_frame": 10}],
                       fps_native=59.94, frame_ratio=3, gt_step_sec_actual=0.500501)
    assert frame_times(root, clip)[-1] == pytest.approx(0.500501, abs=1e-4)


def test_reversed_times_are_kept(tmp_path):
    """In 11 CholecSeg8k clips the extracted frames are not in chronological
    order, while their recorded times are right. They are not reordered: a
    caller pairs frames by these times, not by index."""
    root, clip = _clip(tmp_path, [{"timestamp_sec": 2.0}, {"timestamp_sec": 1.0}])
    assert frame_times(root, clip).tolist() == [2.0, 1.0]


def test_timestamps_on_only_some_frames_are_refused(tmp_path):
    """A clip is timed by one rule for every frame. Falling through to
    `native_frame` would drop the timestamps the extractor did write."""
    root, clip = _clip(tmp_path, [{"timestamp_sec": 1.0, "native_frame": 30},
                                  {"native_frame": 60}], fps_native=30.0)
    with pytest.raises(RuntimeError, match="1 of 2 frames carry"):
        frame_times(root, clip)


def test_no_way_to_make_seconds_is_refused(tmp_path):
    """No default: a clip whose seconds cannot be made is not made evenly spaced."""
    root, clip = _clip(tmp_path, [{"seq_idx": 0}, {"seq_idx": 1}])
    with pytest.raises(RuntimeError, match="cannot make frame times"):
        frame_times(root, clip)


def test_missing_fps_is_refused(tmp_path):
    root, clip = _clip(tmp_path, [{"native_frame": 30}, {"native_frame": 60}])
    with pytest.raises(RuntimeError):
        frame_times(root, clip)


def test_returns_float_array(tmp_path):
    root, clip = _clip(tmp_path, [{"native_frame": 1}, {"native_frame": 2}],
                       fps_native=30.0)
    t = frame_times(root, clip)
    assert isinstance(t, np.ndarray) and t.dtype == float and t.shape == (2,)
