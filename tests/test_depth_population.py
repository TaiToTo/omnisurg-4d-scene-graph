"""Test the population driver with a stand-in for the depth command.

The stand-in writes a bundle for every clip it is given, with one depth map
per image, and records which GPU got which clips. A clip named `broken`
makes it exit non-zero without writing; one named `flaky` makes it exit
non-zero after writing. Each refusal has a test that plants its fault.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pipeline.depth_population as driver
from pipeline.depth_population import check_depth, main, run_population, shards

STAND_IN = '''
import json, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]
root = Path(args[args.index("--input-dir") + 1])
gpu = args[args.index("--gpu") + 1]
clips = args[args.index("--clips") + 1:]
assert "--no-glb" in args
(root / "_calls").mkdir(exist_ok=True)
(root / "_calls" / f"gpu{gpu}.json").write_text(json.dumps(clips))
if "broken" in clips:
    sys.exit(1)
for clip in clips:
    n = len(list((root / clip / "input_images").glob("*.png")))
    (root / clip / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(root / clip / "exports" / "mini_npz" / "results.npz", depth=np.ones((n, 2, 2), np.float32))
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


def test_every_clip_without_depth_runs_on_its_gpu_and_a_done_clip_is_skipped(stand_in):
    names = [f"clip{i}" for i in range(5)]
    for name in names:
        make_clip(stand_in, name)
    (stand_in / "clip1" / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(stand_in / "clip1" / "exports" / "mini_npz" / "results.npz", depth=np.ones((3, 2, 2), np.float32))
    run_population(stand_in, names, [0, 1])
    assert calls(stand_in) == {"gpu0": ["clip0", "clip3"], "gpu1": ["clip2", "clip4"]}
    for name in names:
        assert (stand_in / name / "exports" / "mini_npz" / "results.npz").is_file()
    assert (stand_in / "_logs" / "depth_gpu0.log").is_file() and (stand_in / "_logs" / "depth_gpu1.log").is_file()


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
                                            r"\['broken'\]"):
        run_population(stand_in, ["clip0", "broken", "clip2"], [0, 1])
    for name in ("clip0", "clip2"):
        assert (stand_in / name / "exports" / "mini_npz" / "results.npz").is_file()


def test_a_bundle_with_another_number_of_depth_maps_than_images_is_refused(stand_in):
    make_clip(stand_in, "clip0", n=3)
    (stand_in / "clip0" / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(stand_in / "clip0" / "exports" / "mini_npz" / "results.npz", depth=np.ones((2, 2, 2), np.float32))
    with pytest.raises(ValueError, match=r"clip0 \(2 depth maps for 3 images\)"):
        check_depth(stand_in, ["clip0"])


def test_a_failed_process_is_reported_even_when_every_clip_has_its_depth(stand_in):
    make_clip(stand_in, "flaky")
    with pytest.raises(RuntimeError, match=r"GPU 0 exited 1, see .*depth_gpu0.log$"):
        run_population(stand_in, ["flaky"], [0])
    assert (stand_in / "flaky" / "exports" / "mini_npz" / "results.npz").is_file()


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
