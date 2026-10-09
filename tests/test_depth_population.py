"""Test the population driver with a stand-in for the depth command.

The stand-in writes the stage's bundle and `depth_info` for every clip it is
given, with one depth map per image, and records which GPU got which clips.
A clip named `broken` makes it exit non-zero without writing; one named
`flaky` makes it exit non-zero after writing; one named `slow` makes it
wait. Each refusal has a test that plants its fault.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pipeline.depth_population as driver
from pipeline.depth_population import check_depth, main, run_population, shards

STAND_IN = '''
import json, sys, time
from pathlib import Path
import numpy as np
args = sys.argv[1:]
root = Path(args[args.index("--input-dir") + 1])
gpu = args[args.index("--gpu") + 1]
clips = args[args.index("--clips") + 1:]
assert "--no-glb" in args and args[args.index("--device") + 1] == "cuda"
(root / "_calls").mkdir(exist_ok=True)
(root / "_calls" / f"gpu{gpu}.json").write_text(json.dumps(clips))
if "broken" in clips:
    sys.exit(1)
if "slow" in clips:
    time.sleep(60)
for clip in clips:
    n = len(list((root / clip / "input_images").glob("*.png")))
    (root / clip / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    np.savez(root / clip / "exports" / "mini_npz" / "results.npz", depth=np.ones((n, 2, 2), np.float32),
             conf=np.ones((n, 2, 2), np.float32), extrinsics=np.zeros((n, 4, 4)), intrinsics=np.zeros((n, 3, 3)))
    manifest = json.loads((root / clip / "frame_manifest.json").read_text())
    manifest["depth_info"] = {"model": "stand-in", "process_res": 504}
    (root / clip / "frame_manifest.json").write_text(json.dumps(manifest))
if "flaky" in clips:
    sys.exit(1)
'''


def make_clip(root: Path, name: str, n: int = 3) -> Path:
    clip = root / name
    (clip / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(clip / "input_images" / f"{i:06d}.png")
    (clip / "frame_manifest.json").write_text(json.dumps({"frames": [{"seq_idx": i} for i in range(n)]}))
    return clip


def write_depth(root: Path, name: str, n: int = 3, keys=driver.BUNDLE_KEYS, model: str = "stand-in",
                process_res: int = 504, depth_info: bool = True) -> None:
    """Write what the depth stage leaves in a clip: the bundle with `keys`, and `depth_info` in the manifest."""
    (root / name / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    np.savez(root / name / driver.BUNDLE, **{k: np.ones((n, 2, 2), np.float32) for k in keys})
    if depth_info:
        manifest = json.loads((root / name / "frame_manifest.json").read_text())
        manifest["depth_info"] = {"model": model, "process_res": process_res}
        (root / name / "frame_manifest.json").write_text(json.dumps(manifest))


def calls(root: Path) -> dict[str, list[str]]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((root / "_calls").glob("gpu*.json"))}


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    monkeypatch.setattr(driver, "DEPTH_COMMAND", [sys.executable, str(script)])
    root = tmp_path / "clips"
    root.mkdir()
    return root


def test_clips_are_dealt_out_in_turn_and_a_gpu_with_none_is_left_out():
    assert shards(["a", "b", "c", "d", "e"], [0, 1]) == {0: ["a", "c", "e"], 1: ["b", "d"]}
    assert shards(["a"], [3, 5]) == {3: ["a"]}
    assert shards([], [0]) == {}


def test_a_gpu_listed_twice_is_refused_before_anything_runs(stand_in):
    make_clip(stand_in, "clip0")
    with pytest.raises(ValueError, match=r"a GPU is listed twice in \[0, 0\]"):
        run_population(stand_in, ["clip0"], [0, 0])
    assert not (stand_in / "_calls").exists()


def test_every_clip_without_depth_runs_on_its_gpu_and_a_done_clip_is_skipped(stand_in):
    names = [f"clip{i}" for i in range(5)]
    for name in names:
        make_clip(stand_in, name)
    write_depth(stand_in, "clip1")
    run_population(stand_in, names, [0, 1])
    assert calls(stand_in) == {"gpu0": ["clip0", "clip3"], "gpu1": ["clip2", "clip4"]}
    for name in names:
        assert (stand_in / name / driver.BUNDLE).is_file()
        assert "depth_info" in json.loads((stand_in / name / "frame_manifest.json").read_text())
    assert (stand_in / "_logs" / "depth_gpu0.log").is_file() and (stand_in / "_logs" / "depth_gpu1.log").is_file()


def test_a_clip_with_a_bundle_but_no_depth_info_is_run_again(stand_in):
    # The stage writes the bundle before the manifest, so a clip stopped between the two holds the bundle alone.
    make_clip(stand_in, "clip0")
    write_depth(stand_in, "clip0", depth_info=False)
    run_population(stand_in, ["clip0"], [0])
    assert calls(stand_in) == {"gpu0": ["clip0"]}


def test_a_clip_of_the_population_that_is_not_there_is_refused_before_anything_runs(stand_in):
    make_clip(stand_in, "clip0")
    with pytest.raises(FileNotFoundError, match=r"1 clip\(s\) of the population are not under .*\['clip9'\]"):
        run_population(stand_in, ["clip0", "clip9"], [0])
    assert not (stand_in / "_calls").exists()


def test_a_process_that_fails_is_reported_with_its_log_and_the_clips_it_left(stand_in):
    # GPU 0 gets clip0 and clip2, GPU 1 gets broken.
    for name in ("clip0", "broken", "clip2"):
        make_clip(stand_in, name)
    with pytest.raises(RuntimeError, match=r"GPU 1 exited 1, see .*depth_gpu1.log; 1 clip\(s\) without depth: "
                                            r"\['broken'\]$"):
        run_population(stand_in, ["clip0", "broken", "clip2"], [0, 1])
    for name in ("clip0", "clip2"):
        assert (stand_in / name / driver.BUNDLE).is_file()


def test_a_bundle_with_another_number_of_depth_maps_than_images_is_refused(stand_in):
    make_clip(stand_in, "clip0", n=3)
    write_depth(stand_in, "clip0", n=2)
    with pytest.raises(ValueError, match=r"clip0 \(2 depth maps for 3 images\)"):
        check_depth(stand_in, ["clip0"])


def test_a_bundle_of_another_version_of_the_stage_is_refused(stand_in):
    make_clip(stand_in, "clip0")
    write_depth(stand_in, "clip0", keys=driver.BUNDLE_KEYS | {"ray_map"})
    with pytest.raises(ValueError, match=r"1 with a bundle of another version of the depth stage: "
                                          r"\[\"clip0 \(keys \['conf', 'depth', 'extrinsics', 'intrinsics', "
                                          r"'ray_map'\]\)\"\]"):
        check_depth(stand_in, ["clip0"])


def test_a_bundle_that_cannot_be_read_is_refused(stand_in):
    make_clip(stand_in, "clip0")
    write_depth(stand_in, "clip0")
    (stand_in / "clip0" / driver.BUNDLE).write_bytes(b"half a bundle")
    with pytest.raises(ValueError, match=r"1 with a bundle that cannot be read: \['clip0 \("):
        check_depth(stand_in, ["clip0"])


def test_depth_made_at_two_settings_is_refused(stand_in):
    for name in ("clip0", "clip1", "clip2"):
        make_clip(stand_in, name)
    write_depth(stand_in, "clip0")
    write_depth(stand_in, "clip1", process_res=252)
    write_depth(stand_in, "clip2", model="other")
    with pytest.raises(ValueError, match=r"depth made at more than one setting: other at 504 \(1 clip\(s\), clip2 "
                                          r"among them\), stand-in at 252 \(1 clip\(s\), clip1 among them\), "
                                          r"stand-in at 504 \(1 clip\(s\), clip0 among them\)"):
        check_depth(stand_in, ["clip0", "clip1", "clip2"])
    check_depth(stand_in, ["clip0"])


def test_a_failed_process_is_reported_even_when_every_clip_has_its_depth(stand_in):
    make_clip(stand_in, "flaky")
    with pytest.raises(RuntimeError, match=r"GPU 0 exited 1, see .*depth_gpu0.log$"):
        run_population(stand_in, ["flaky"], [0])
    assert (stand_in / "flaky" / driver.BUNDLE).is_file()


def test_a_stopped_driver_stops_its_processes(stand_in, monkeypatch):
    make_clip(stand_in, "slow")
    started = []
    real_popen = subprocess.Popen

    class Interrupted(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(driver.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_population(stand_in, ["slow"], [0])
    assert len(started) == 1 and real_popen.wait(started[0], timeout=10) != 0


def test_the_command_runs_the_population_file_and_exits_non_zero_on_a_refusal(stand_in, monkeypatch):
    make_clip(stand_in, "clip0")
    population = stand_in.parent / "clips.txt"
    population.write_text("clip0\n")
    monkeypatch.setattr(sys, "argv", ["pipeline.depth_population", "--input-dir", str(stand_in),
                                      "--clips", str(population), "--gpus", "2"])
    main()
    assert calls(stand_in) == {"gpu2": ["clip0"]}
    population.write_text("clip0\nclip9\n")
    with pytest.raises(SystemExit, match="FileNotFoundError"):
        main()
    monkeypatch.setattr(sys, "argv", ["pipeline.depth_population", "--input-dir", str(stand_in / "nowhere"),
                                      "--clips", str(population)])
    with pytest.raises(SystemExit, match="no such directory"):
        main()
