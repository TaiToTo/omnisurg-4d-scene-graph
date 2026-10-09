"""Test `check_provenance` on records as the workbench's stages wrote them and as this repository's stages write them.

The workbench's records are written from the keys and values of real
ones, cut short where a list of frames is long. The records of
`pipeline.track` and `pipeline.per_frame` come from running the stages
with a stand-in segmenter and tracker, which needs the `render` extra.
Each problem the tool reports has a test that plants it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools.check_provenance import ABSENT, read_provenance, report

REPO = Path(__file__).resolve().parents[1]
CLIPS = ["clip0", "clip1", "clip2"]

# A record of `op_edge_center` as the workbench's tracker wrote it, before it recorded `depth_source`.
TRACKED = {"clip": "", "seed_frame": 1, "seed_source": "sam", "seed_labels": "", "sam_input": "normal_edge",
           "track_base": "rgb", "bidir": True, "stride": 1, "seed_min_area": 400,
           "seed_input": {"produced_by": "sam", "cache_path": None, "points_per_side": 24,
                          "seed_sam_kwargs": {"pred_iou_thresh": 0.8, "stability_score_thresh": 0.8,
                                              "min_mask_region_area": 100},
                          "seed_edge_gain": 1.0, "seed_smooth": False, "edge_ring_masked": True},
           "n_seed_regions": 7, "instrument_seed": "off", "frames": [0, 1, 2]}
# A record of `t12_gtseed`, seeded from GT masks: `sam_input` is what was passed, not what made the seed.
GT_SEEDED = {**TRACKED, "seed_source": "gt", "sam_input": "normal",
             "seed_input": {**TRACKED["seed_input"], "produced_by": "gt", "points_per_side": None,
                            "seed_sam_kwargs": None, "seed_edge_gain": None, "seed_smooth": None,
                            "edge_ring_masked": None}}
# A record of a per-frame condition, as the workbench's dispatcher wrote it before it recorded `point_grids`.
PER_FRAME = {"clip": "", "tag": "op_edge_perframe", "seed_source": "per_frame", "track_base": "rgb",
             "sam_input": "normal_edge", "frames": "all", "depth_source": "da3", "seed_min_area": 400, "seed_topk": 0,
             "seed_input": TRACKED["seed_input"]}
# A record of the workbench's GT control `none_ceil`, which no segmenter made: it has no `sam_input`.
CONTROL = {"clip": "", "seed_frame": 1, "seed_source": "gt", "track_base": "none", "control": "ceil",
           "n_seed_regions": 35, "frames": [0, 1, 2]}


def write(root: Path, condition: str, records: dict) -> None:
    """Write each clip's record into `<root>/<clip>/<condition>/seed_info.json`; a str is written as it is."""
    for clip, record in records.items():
        d = root / clip / condition
        d.mkdir(parents=True, exist_ok=True)
        text = record if isinstance(record, str) else json.dumps({**record, "clip": clip})
        (d / "seed_info.json").write_text(text, encoding="utf-8")


def test_a_condition_whose_records_hold_the_stated_settings_has_no_problem(tmp_path):
    write(tmp_path, "track_rgb_op_edge_center", {c: TRACKED for c in CLIPS})
    prov = read_provenance(tmp_path, "track_rgb_op_edge_center", CLIPS, "sam", "rgb")
    assert prov.problems() == []
    assert report(prov, "sam", "rgb") == [
        "track_rgb_op_edge_center: 3 clips, stated seed_source sam, track_base rgb",
        "  seed_source sam, sam_input normal_edge, track_base rgb: 3 clip(s)",
    ]


@pytest.mark.parametrize("record, source, base", [(GT_SEEDED, "gt", "rgb"), (PER_FRAME, "per_frame", "rgb"),
                                                  ({**TRACKED, "seed_source": "external"}, "external", "rgb")])
def test_each_seed_source_the_records_hold_is_read(tmp_path, record, source, base):
    write(tmp_path, "track_rgb_t", {c: record for c in CLIPS})
    assert read_provenance(tmp_path, "track_rgb_t", CLIPS, source, base).problems() == []


def test_a_seed_source_other_than_stated_is_a_problem(tmp_path):
    # A condition named for GT seeds whose tracker cut its own seeds: its labels were complete all the same.
    write(tmp_path, "track_rgb_t12_gtseed", {c: TRACKED for c in CLIPS})
    prov = read_provenance(tmp_path, "track_rgb_t12_gtseed", CLIPS, "gt", "rgb")
    assert prov.problems() == ["3 setting(s) other than stated: clip0: seed_source 'sam', not 'gt'; "
                               "clip1: seed_source 'sam', not 'gt'; clip2: seed_source 'sam', not 'gt'"]


def test_a_tracker_input_other_than_stated_is_a_problem(tmp_path):
    write(tmp_path, "track_rgb_t", {"clip0": TRACKED, "clip1": {**TRACKED, "track_base": "normal"}})
    prov = read_provenance(tmp_path, "track_rgb_t", ["clip0", "clip1"], "sam", "rgb")
    assert prov.problems() == [
        "the clips hold 2 combinations of seed_source, sam_input, track_base, so the population mixes conditions",
        "1 setting(s) other than stated: clip1: track_base 'normal', not 'rgb'",
    ]


def test_a_population_whose_records_hold_two_segmenter_inputs_is_a_problem(tmp_path):
    write(tmp_path, "track_rgb_t", {"clip0": TRACKED, "clip1": TRACKED, "clip2": {**TRACKED, "sam_input": "normal"}})
    prov = read_provenance(tmp_path, "track_rgb_t", CLIPS, "sam", "rgb")
    assert prov.problems() == [
        "the clips hold 2 combinations of seed_source, sam_input, track_base, so the population mixes conditions"]
    assert report(prov, "sam", "rgb")[1:3] == ["  seed_source sam, sam_input normal_edge, track_base rgb: 2 clip(s)",
                                                "  seed_source sam, sam_input normal, track_base rgb: 1 clip(s)"]


def test_a_clip_without_a_record_is_a_problem(tmp_path):
    write(tmp_path, "track_rgb_t", {"clip0": TRACKED})
    (tmp_path / "clip1" / "track_rgb_t").mkdir(parents=True)
    prov = read_provenance(tmp_path, "track_rgb_t", CLIPS, "sam", "rgb")
    assert prov.problems() == ["2 clip(s) without seed_info.json: ['clip1', 'clip2']"]


@pytest.mark.parametrize("text, why", [('{"seed_source": "sa', "not JSON"), ("[1, 2]", "a JSON list, not an object"),
                                       (b"\xff\xfe", "not JSON")])
def test_a_record_that_cannot_be_read_is_a_problem(tmp_path, text, why):
    write(tmp_path, "track_rgb_t", {"clip0": TRACKED, "clip1": TRACKED})
    path = tmp_path / "clip1" / "track_rgb_t" / "seed_info.json"
    path.write_bytes(text) if isinstance(text, bytes) else path.write_text(text)
    prov = read_provenance(tmp_path, "track_rgb_t", ["clip0", "clip1"], "sam", "rgb")
    assert len(prov.problems()) == 1
    assert prov.problems()[0].startswith(f"1 clip(s) whose seed_info.json cannot be read: clip1 ({why}")


def test_a_record_without_a_setting_holds_no_such_key_in_its_place(tmp_path):
    # The workbench read a missing key as None. It shows here, and a missing stated setting is never the one stated.
    write(tmp_path, "track_none_ceil", {c: CONTROL for c in CLIPS})
    prov = read_provenance(tmp_path, "track_none_ceil", CLIPS, "gt", "none")
    assert prov.problems() == []
    assert report(prov, "gt", "none")[1] == "  seed_source gt, sam_input (no such key), track_base none: 3 clip(s)"
    write(tmp_path, "track_rgb_t", {"clip0": {k: v for k, v in TRACKED.items() if k != "seed_source"}})
    prov = read_provenance(tmp_path, "track_rgb_t", ["clip0"], "sam", "rgb")
    assert prov.mismatched == [("clip0", "seed_source", ABSENT, "sam")]
    assert prov.problems() == ["1 setting(s) other than stated: clip0: seed_source (no such key), not 'sam'"]


def test_a_record_holding_null_differs_from_one_without_the_key(tmp_path):
    write(tmp_path, "track_rgb_t", {"clip0": {**PER_FRAME, "sam_input": None},
                                    "clip1": {k: v for k, v in PER_FRAME.items() if k != "sam_input"}})
    prov = read_provenance(tmp_path, "track_rgb_t", ["clip0", "clip1"], "per_frame", "rgb")
    assert sorted(map(repr, prov.settings)) == ["('per_frame', (no such key), 'rgb')", "('per_frame', None, 'rgb')"]


def test_an_unknown_seed_source_or_a_root_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"'auto' is no seed source; one of \('sam', 'external', 'gt', 'per_frame'\)"):
        read_provenance(tmp_path, "track_rgb_t", CLIPS, "auto", "rgb")
    with pytest.raises(ValueError, match="no such directory"):
        read_provenance(tmp_path / "nowhere", "track_rgb_t", CLIPS, "sam", "rgb")


def run_stage(tmp_path, stage: str, **settings) -> None:
    """Run `pipeline.track` or `pipeline.per_frame` on one clip with a stand-in segmenter and tracker."""
    pytest.importorskip("matplotlib")
    from PIL import Image

    from pipeline.per_frame import run_per_frame
    from pipeline.track import run_track
    from sam3_wrapper import FrameResult

    clip = tmp_path / "clips" / "clip0"
    (clip / "input_images").mkdir(parents=True)
    (clip / "exports" / "mini_npz").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((8, 10, 3), 50 * i, np.uint8)).save(clip / "input_images" / f"{i:06d}.png")
    depth = np.stack([np.full((8, 10), 0.3 + 0.01 * i, np.float32) for i in range(3)])
    K = np.tile(np.array([[10.0, 0, 5], [0, 10.0, 4], [0, 0, 1]]), (3, 1, 1))
    np.savez(clip / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K)

    class Segmenter:
        def label_map(self, image, points_per_side):
            return np.zeros(image.shape[:2], int)

    class Tracker:
        def init_video(self, frames):
            self.n = len(frames)

        def add_masks(self, frame_idx, masks, obj_ids):
            self.masks, self.ids = masks, obj_ids

        def propagate(self, start, reverse=False):
            for f in (range(start, -1, -1) if reverse else range(start, self.n)):
                yield FrameResult(f, list(self.ids), np.stack(self.masks), np.ones(len(self.ids)))

    if stage == "track":
        run_track(clip, tmp_path / "tracks", "t", "both_ways_from_centre", Segmenter(), Tracker(), seed_min_area=1,
                  **settings)
    else:
        run_per_frame(clip, tmp_path / "tracks", "t", Segmenter(), seed_min_area=1, **settings)


@pytest.mark.parametrize("stage, settings, condition, source, base", [
    ("track", dict(sam_input="normal_edge", track_base="rgb"), "track_rgb_t", "sam", "rgb"),
    ("track", dict(sam_input="rgb", track_base="normal"), "track_normal_t", "sam", "normal"),
    ("per_frame", dict(sam_input="depth"), "track_rgb_t", "per_frame", "rgb"),
])
def test_the_records_this_repository_s_stages_write_are_read(tmp_path, stage, settings, condition, source, base):
    run_stage(tmp_path, stage, **settings)
    prov = read_provenance(tmp_path / "tracks", condition, ["clip0"], source, base)
    assert prov.problems() == []
    assert list(prov.settings) == [(source, settings["sam_input"], base)]
    other = "sam" if source == "per_frame" else "per_frame"
    assert read_provenance(tmp_path / "tracks", condition, ["clip0"], other, base).mismatched == [
        ("clip0", "seed_source", source, other)]


def test_the_command_exits_non_zero_on_a_problem_and_zero_otherwise(tmp_path):
    write(tmp_path / "tracks", "track_rgb_op_edge_center", {c: TRACKED for c in CLIPS})
    population = tmp_path / "clips.txt"
    population.write_text("\n".join(CLIPS) + "\n")
    cmd = [sys.executable, "-m", "evalkit.tools.check_provenance", "--tracks-root", str(tmp_path / "tracks"),
           "--tag", "op_edge_center", "--clips", str(population)]
    ok = subprocess.run([*cmd, "--seed-source", "sam"], cwd=REPO, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[1] == "  seed_source sam, sam_input normal_edge, track_base rgb: 3 clip(s)"
    bad = subprocess.run([*cmd, "--seed-source", "gt"], cwd=REPO, capture_output=True, text=True)
    assert bad.returncode == 1 and "!! 3 setting(s) other than stated" in bad.stdout
    refused = subprocess.run([*cmd, "--seed-source", "sam", "--track-base", "normal", "--tracks-root",
                              str(tmp_path / "nowhere")], cwd=REPO, capture_output=True, text=True)
    assert refused.returncode == 1 and "no such directory" in refused.stderr
