"""Run `pipeline.d4d_depth` on a made-up clip through a stand-in model, and check what it writes and refuses.

The stand-in returns, for each frame it is given, depth equal to the frame's position in the clip, so a file
shows which frame it kept. Each refusal has a test that plants its fault:

- a clip of `--clips` with no side in the population, and a population clip the dataset does not hold;
- a side's file made with another frame, window or resolution;
- a side with no scan, and a model that returns another number of frames than it was given.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import surgical_core.d4d as d4d
from pipeline import d4d_depth
from recon3d_wrapper import Reconstruction

# Forty frames, 0.1 s apart from t = 100 s. The start scan is 4 s before the first frame, the end scan 3 s after
# the last.
FRAMES = [f"{100 + i // 10}_{(i % 10) * 100_000_000:09d}" for i in range(40)]
KEY = "specimen_1/s_a/Clip_1"


def make_clip(scans: dict | None = None) -> SimpleNamespace:
    if scans is None:
        scans = {"start": "/scans/96_000000000.ply", "end": "/scans/106_900000000.ply"}
    return SimpleNamespace(name="Clip_1", pointclouds=scans,
                           session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(name="specimen_1")),
                           left_img_paths=[f"/clip/left_images_rect/{f}.png" for f in FRAMES])


class StandInModel:
    """Return depth equal to each frame's position in the clip, read from its name; `extra` adds frames."""

    def __init__(self, extra: int = 0):
        self.calls = []
        self.extra = extra

    def reconstruct(self, paths):
        self.calls.append(list(paths))
        n = len(paths) + self.extra
        pos = [FRAMES.index(Path(p).stem) for p in paths] + [0] * self.extra
        depth = np.stack([np.full((4, 5), float(i)) for i in pos])
        return Reconstruction(depth=depth, conf=depth + 1, intrinsics=np.tile(np.eye(3), (n, 1, 1)),
                              extrinsics=np.tile(np.eye(4)[:3], (n, 1, 1)))


@pytest.mark.parametrize("n, index, expected", [
    (40, 20, (5, 35)),   # centred
    (40, 0, (0, 30)),    # moved inside at the start
    (40, 39, (10, 40)),  # moved inside at the end
    (21, 20, (0, 21)),   # a clip shorter than the window is taken whole
])
def test_the_window_is_centred_on_the_frame_and_kept_inside_the_clip(n, index, expected):
    assert d4d_depth.window(n, index, 30) == expected


def test_a_side_keeps_the_depth_of_the_frame_nearest_its_scan(tmp_path):
    model = StandInModel()
    dst = tmp_path / "side.npz"
    d4d_depth.run_side(make_clip(), "end", model, 30, 504, dst)
    with np.load(dst) as z:
        assert z.files == ["depth", "conf", "intrinsics", "extrinsics", "frame_index", "frame_minus_gt_s", "window",
                           "process_res", "image_path"]
        assert (z["depth"] == 39).all() and z["depth"].dtype == np.float32
        assert int(z["frame_index"]) == 39 and int(z["window"]) == 30 and int(z["process_res"]) == 504
        assert float(z["frame_minus_gt_s"]) == pytest.approx(-3.0, abs=1e-5)
        assert str(z["image_path"]) == f"/clip/left_images_rect/{FRAMES[39]}.png"
    assert model.calls == [[f"/clip/left_images_rect/{f}.png" for f in FRAMES[10:40]]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["side.npz"]


def test_a_side_without_a_scan_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no end scan"):
        d4d_depth.run_side(make_clip({"start": "/scans/96_000000000.ply"}), "end", StandInModel(), 30, 504,
                           tmp_path / "side.npz")


def test_a_model_that_returns_other_frames_than_the_window_is_refused(tmp_path):
    with pytest.raises(ValueError, match="returned 31 frames for 30"):
        d4d_depth.run_side(make_clip(), "start", StandInModel(extra=1), 30, 504, tmp_path / "side.npz")
    assert not any(tmp_path.iterdir())


def test_the_sides_of_the_clips_given_are_run_and_an_unknown_clip_is_refused():
    population = {"sides": [{"key": "b/s/c", "side": "start"}, {"key": KEY, "side": "end"},
                            {"key": KEY, "side": "start"}]}
    assert d4d_depth.population_sides(population, None) == [("b/s/c", "start"), (KEY, "end"), (KEY, "start")]
    assert d4d_depth.population_sides(population, ["b/s/c"]) == [("b/s/c", "start")]
    with pytest.raises(ValueError, match=r"1 clip\(s\) with no side in the population: \['x/y/z'\]"):
        d4d_depth.population_sides(population, ["x/y/z"])


def run_main(tmp_path, monkeypatch, sides=("start", "end"), clips=None, extra=()):
    """Run `main` on the made-up clip with the stand-in model, and return the model."""
    model = StandInModel()
    monkeypatch.setattr(d4d, "clips", lambda root: [make_clip()] if clips is None else clips)
    monkeypatch.setattr(d4d_depth, "DA3", lambda gpu, process_res: model)
    population = tmp_path / "population.json"
    population.write_text(json.dumps({"sides": [{"key": KEY, "side": s} for s in sides]}))
    d4d_depth.main(["--root", str(tmp_path), "--population", str(population), "--out", str(tmp_path / "out"),
                    *extra])
    return model


def test_main_writes_each_side_and_skips_a_side_already_made(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "specimen_1__s_a__Clip_1__end.npz", "specimen_1__s_a__Clip_1__start.npz"]
    assert run_main(tmp_path, monkeypatch).calls == []


def test_main_refuses_a_side_made_with_another_window(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, sides=("end",))
    with pytest.raises(ValueError, match="'window': 30.*where this run makes.*'window': 20"):
        run_main(tmp_path, monkeypatch, sides=("end",), extra=["--window", "20"])


def test_main_refuses_a_side_made_from_another_frame(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, sides=("end",))
    path = tmp_path / "out" / "specimen_1__s_a__Clip_1__end.npz"
    with np.load(path) as z:
        fields = {k: z[k] for k in z.files}
    fields["frame_index"] = np.int32(38)
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match="'frame_index': 38"):
        run_main(tmp_path, monkeypatch, sides=("end",))


def test_main_refuses_a_population_clip_the_dataset_does_not_hold(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"1 clip\(s\) of the population are not in"):
        run_main(tmp_path, monkeypatch, clips=[])


def test_main_refuses_a_shard_out_of_range(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, extra=["--shard", "2", "--n-shards", "2"])
