"""Test the condition driver through the stages' own command lines, with stand-in models.

Each clip runs a stand-in command: it records the arguments it was given and
its process id, then runs `pipeline.track.main` or `pipeline.per_frame.main`
with a segmenter and a tracker that need no torch and no weights. The labels
and the record are therefore the ones the stages write. A clip named `broken`
makes the command exit non-zero, one named `slow` makes it wait, and one
named `drifted` makes it rewrite its record with another seed edge gain.
Each refusal has a test that plants its fault. The stages need the `render`
extra.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

import pipeline.condition_population as driver  # noqa: E402
from pipeline.condition_population import (build_parser, check_population, expected_record, main,  # noqa: E402
                                           run_population, stage_args)

REPO = Path(__file__).resolve().parents[1]
H, W, N = 12, 16, 5

STAND_IN = '''
import json, os, sys, time
from pathlib import Path
import numpy as np
import pipeline.per_frame as per_frame
import pipeline.track as track
from sam3_wrapper import FrameResult

stage, args = sys.argv[1], sys.argv[2:]
tracks_root = Path(args[args.index("--tracks-root") + 1])
clip = args[args.index("--clips") + 1]
(tracks_root / "_calls").mkdir(parents=True, exist_ok=True)
(tracks_root / "_calls" / f"{clip}.json").write_text(json.dumps(dict(pid=os.getpid(), args=args)))
if clip == "broken":
    sys.exit(1)
if clip == "slow":
    time.sleep(60)


class Segmenter:
    """Cut every image into a left half and a right half."""

    device = "cpu"

    def label_map(self, image, points_per_side):
        labels = np.zeros(image.shape[:2], int)
        labels[:, image.shape[1] // 2:] = 1
        return labels


class Tracker:
    """Return the seed masks on every frame from the start frame on, in the direction asked."""

    device = "cpu"

    def init_video(self, frames):
        self.n = len(frames)

    def add_masks(self, frame_idx, masks, obj_ids):
        self.masks, self.ids = masks, obj_ids

    def propagate(self, start, reverse=False):
        for f in (range(start, -1, -1) if reverse else range(start, self.n)):
            yield FrameResult(f, list(self.ids), np.stack(self.masks), np.ones(len(self.ids)))


track.Sam3VideoInstanceSession = lambda model_id, device, gpu: Tracker()
track.SeedSegmenter = per_frame.SeedSegmenter = lambda checkpoint, device: Segmenter()
sys.argv = [stage, *args]
(track if stage == "track" else per_frame).main()
if clip == "drifted":
    record = tracks_root / clip / f"track_rgb_{args[args.index('--tag') + 1]}" / "seed_info.json"
    info = json.loads(record.read_text())
    info["seed_input"]["seed_edge_gain"] = 0.5
    record.write_text(json.dumps(info))
'''

TRACK = ["--rule", "both_ways_from_centre", "--sam-input", "normal_edge", "--track-base", "rgb",
         "--seed-edge-gain", "1.0", "--seed-no-smooth", "--seed-min-area", "1"]
PER_FRAME = ["--sam-input", "rgb", "--points-per-side", "8", "--seed-min-area", "1"]


def make_clip(root: Path, name: str, n: int = N, pi3x: bool = False) -> None:
    """Write a clip of `n` frames with DA3's bundle, and Pi3X's when asked."""
    clip = root / name
    (clip / "input_images").mkdir(parents=True)
    (clip / "exports" / "mini_npz").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((H, W, 3), 40 * i, np.uint8)).save(clip / "input_images" / f"{i:06d}.png")
    yy, xx = np.mgrid[0:H, 0:W]
    depth = np.stack([0.3 + 0.01 * xx + 0.005 * yy + 0.001 * i for i in range(n)]).astype(np.float32)
    K = np.tile(np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]), (n, 1, 1))
    np.savez(clip / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K, extrinsics=np.zeros((n, 3, 4)))
    if pi3x:
        np.savez(clip / "exports" / "mini_npz" / "results__pi3x.npz", depth=2 * depth, intrinsics=K)


def settings(stage: str, *options: str, tag: str = "t", sam_ckpt: str | None = None) -> object:
    """Parse a condition's settings as the command line does."""
    ckpt = ["--sam-ckpt", sam_ckpt] if sam_ckpt else []
    return build_parser().parse_args([stage, "--input-dir", "-", "--clips", "-", "--tracks-root", "-", "--tag", tag,
                                      "--gpus", "0", *ckpt, *options])


def calls(tracks_root: Path) -> dict[str, list[str]]:
    return {p.stem: json.loads(p.read_text())["args"] for p in sorted((tracks_root / "_calls").glob("*.json"))}


def wait_for(condition, seconds: float = 10) -> None:
    """Wait until `condition()` holds, or fail."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Return the clips' root, the tracks root and the SAM weights, with the stand-in as both stages' command."""
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    monkeypatch.setattr(driver, "STAGE_COMMANDS", {s: [sys.executable, str(script), s] for s in driver.STAGE_COMMANDS})
    monkeypatch.setattr(driver, "POLL_SEC", 0.02)
    # The stand-in imports the stages from this checkout, installed or not.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])]))
    ckpt = tmp_path / "sam.pth"
    ckpt.write_bytes(b"")
    (tmp_path / "clips").mkdir()
    return tmp_path / "clips", tmp_path / "tracks", str(ckpt)


def test_each_clip_runs_on_a_gpu_of_its_own_with_every_setting_given(run):
    root, tracks, ckpt = run
    names = ["clip0", "clip1", "clip2"]
    for name in names:
        make_clip(root, name)
    s = settings("track", *TRACK, sam_ckpt=ckpt)
    run_population("track", s, root, tracks, names, [3, 5])
    got = calls(tracks)
    assert sorted(got) == names
    assert {a[a.index("--gpu") + 1] for a in got.values()} == {"3", "5"}
    for name, args in got.items():
        assert args == ["--input-dir", str(root), "--tracks-root", str(tracks), "--tag", "t", "--clips", name,
                        "--device", "cuda", "--gpu", args[args.index("--gpu") + 1], *stage_args("track", s)]
        assert len(list((tracks / name / "track_rgb_t").glob("label_*.npy"))) == N
        log = (tracks / "_logs" / "track_rgb_t" / f"{name}.log").read_text()
        assert log.splitlines()[0].endswith("--sam-ckpt " + ckpt)


@pytest.mark.parametrize("stage, options, pi3x", [
    ("track", TRACK, False),
    ("track", ["--rule", "forward_from_first", "--sam-input", "normal", "--track-base", "rgb", "--seed-min-area", "1",
               "--keep-edge-ring", "--depth-source", "pi3", "--points-per-side", "16"], True),
    ("track", ["--rule", "both_ways_from_centre", "--sam-input", "rgb_edge", "--track-base", "rgb_edge",
               "--seed-min-area", "1", "--keep-edge-ring"], False),
    ("per_frame", PER_FRAME, False),
    ("per_frame", ["--sam-input", "normal_edge", "--depth-source", "pi3", "--edge-gain", "0.5", "--seed-min-area", "1",
                   "--keep-edge-ring"], True),
])
def test_the_record_each_stage_writes_holds_the_settings_the_driver_passes(run, stage, options, pi3x):
    # The stage parses the driver's arguments and writes its own record; the driver's check reads it after the run.
    root, tracks, ckpt = run
    make_clip(root, "clip0", pi3x=pi3x)
    make_clip(root, "clip1", n=4, pi3x=pi3x)
    s = settings(stage, *options, sam_ckpt=ckpt)
    run_population(stage, s, root, tracks, ["clip0", "clip1"], [0])
    info = json.loads((tracks / "clip1" / driver.condition_dir(stage, s) / "seed_info.json").read_text())
    assert {k: info[k] for k in expected_record(stage, s)} == json.loads(json.dumps(expected_record(stage, s)))


@pytest.mark.parametrize("in_tracks", [False, True])
def test_seed_labels_made_outside_reach_the_stage(run, in_tracks):
    # The seeds are in a directory per clip, or, with {clip} in the path, in a condition kmerge wrote beside the tracks.
    root, tracks, _ = run
    seeds = str(tracks / "{clip}" / "track_rgb_k10") if in_tracks else str(root.parent / "seeds")
    for name, n in (("clip0", 5), ("clip1", 4)):
        make_clip(root, name, n=n)
        seed_dir = tracks / name / "track_rgb_k10" if in_tracks else root.parent / "seeds" / name
        seed_dir.mkdir(parents=True)
        labels = np.zeros((H, W), int)
        labels[H // 2:] = 7
        np.save(seed_dir / f"label_{n // 2:04d}.npy", labels)
    s = settings("track", "--rule", "both_ways_from_centre", "--sam-input", "rgb", "--track-base", "rgb",
                 "--seed-min-area", "1", "--seed-labels", seeds)
    run_population("track", s, root, tracks, ["clip0", "clip1"], [0, 1])
    assert all(a[a.index("--seed-labels") + 1] == seeds and "--sam-ckpt" not in a for a in calls(tracks).values())
    info = json.loads((tracks / "clip1" / "track_rgb_t" / "seed_info.json").read_text())
    assert (info["seed_source"], info["seed_frame"], info["seed_input"]["cache_path"]) == ("external", 2, seeds)


def test_a_clip_whose_labels_exist_is_skipped(run):
    root, tracks, ckpt = run
    for name in ("clip0", "clip1"):
        make_clip(root, name)
    s = settings("per_frame", *PER_FRAME, sam_ckpt=ckpt)
    run_population("per_frame", s, root, tracks, ["clip0"], [0])
    (tracks / "_calls" / "clip0.json").unlink()
    run_population("per_frame", s, root, tracks, ["clip0", "clip1"], [0])
    assert sorted(calls(tracks)) == ["clip1"]


def test_labels_made_at_another_setting_are_refused_by_name_before_anything_runs(run):
    root, tracks, ckpt = run
    for name in ("clip0", "clip1", "clip2"):
        make_clip(root, name)
    run_population("track", settings("track", *TRACK, sam_ckpt=ckpt), root, tracks, ["clip0", "clip1"], [0])
    for p in (tracks / "_calls").iterdir():
        p.unlink()
    other = settings("track", *[o for o in TRACK if o != "--seed-no-smooth"], sam_ckpt=ckpt)
    with pytest.raises(ValueError, match=r"^clip0's seed_info.json records other settings than this run passes: "
                                          r"seed_input .*'seed_smooth': False.*, not .*'seed_smooth': True.*; "
                                          r"clip1's seed_info.json records other settings"):
        run_population("track", other, root, tracks, ["clip0", "clip1", "clip2"], [0])
    assert calls(tracks) == {}


def test_a_record_of_other_settings_after_a_clip_ran_stops_the_run(run, monkeypatch):
    root, tracks, ckpt = run
    for name in ("drifted", "slow", "clip2"):
        make_clip(root, name)
    started = {}
    real_popen = subprocess.Popen

    class Recorded(real_popen):
        def __init__(self, cmd, *args, **kwargs):
            super().__init__(cmd, *args, **kwargs)
            started[cmd[cmd.index("--clips") + 1]] = self

    monkeypatch.setattr(driver.subprocess, "Popen", Recorded)
    with pytest.raises(ValueError, match=r"^drifted's seed_info.json records other settings than this run passes: "
                                          r"seed_input .*'seed_edge_gain': 0.5"):
        run_population("track", settings("track", *TRACK, sam_ckpt=ckpt), root, tracks, ["drifted", "slow", "clip2"],
                       [0, 1])
    # The slow clip's process is stopped, and the clip after the drifted one never starts.
    assert sorted(started) == ["drifted", "slow"]
    assert real_popen.wait(started["slow"], timeout=10) == -signal.SIGTERM


@pytest.mark.parametrize("fault, match", [
    (lambda d: (d / "seed_info.json").unlink(), r"clip0's track_rgb_t has no seed_info.json"),
    (lambda d: (d / "seed_info.json").write_text('{"clip": "cl'), r"clip0's seed_info.json cannot be read"),
    (lambda d: (d / "label_0003.npy").unlink(), r"clip0 has 4 label maps in track_rgb_t for 5 images"),
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({**json.loads((d / "seed_info.json").read_text()), "clip": "clip9"})),
     r"clip0's seed_info.json records the clip 'clip9'"),
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({**json.loads((d / "seed_info.json").read_text()), "seed_frame": 0})),
     r"clip0's seed_info.json records seed frame 0 of 5 frames labelled; both_ways_from_centre seeds frame 2 of 5"),
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({**json.loads((d / "seed_info.json").read_text()), "frames": [0, 1, 2, 3]})),
     r"clip0's seed_info.json records seed frame 2 of 4 frames labelled; both_ways_from_centre seeds frame 2 of 5"),
    # A key the driver does not know, as a later version of the stage would write it.
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({**json.loads((d / "seed_info.json").read_text()), "track_edge_ring_masked": True})),
     r"clip0's seed_info.json records other settings than this run passes: "
     r"track_edge_ring_masked True, not \(no such key\)"),
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({k: v for k, v in json.loads((d / "seed_info.json").read_text()).items() if k != "depth_source"})),
     r"clip0's seed_info.json records other settings than this run passes: depth_source \(no such key\), not 'da3'"),
    (lambda d: (d / "seed_info.json").write_text(
        json.dumps({k: v for k, v in json.loads((d / "seed_info.json").read_text()).items() if k != "frames"})),
     r"clip0's seed_info.json records other settings than this run passes: frames \(no such key\)"),
])
def test_labels_that_do_not_hold_the_condition_are_refused_by_name_before_anything_runs(run, fault, match):
    root, tracks, ckpt = run
    for name in ("clip0", "clip1"):
        make_clip(root, name)
    s = settings("track", *TRACK, sam_ckpt=ckpt)
    run_population("track", s, root, tracks, ["clip0"], [0])
    fault(tracks / "clip0" / "track_rgb_t")
    for p in (tracks / "_calls").iterdir():
        p.unlink()
    with pytest.raises(ValueError, match=match):
        run_population("track", s, root, tracks, ["clip0", "clip1"], [0])
    assert calls(tracks) == {}
    with pytest.raises(ValueError, match=match):
        check_population("track", s, root, tracks, ["clip0"])


def test_a_clip_that_fails_is_reported_with_its_log_and_the_others_run(run):
    root, tracks, ckpt = run
    for name in ("clip0", "broken", "clip2"):
        make_clip(root, name)
    with pytest.raises(RuntimeError, match=r"^broken exited 1, see .*_logs/track_rgb_t/broken.log; "
                                            r"1 clip\(s\) without track_rgb_t: \['broken'\]$"):
        run_population("per_frame", settings("per_frame", *PER_FRAME, sam_ckpt=ckpt), root, tracks,
                       ["clip0", "broken", "clip2"], [0])
    assert sorted(calls(tracks)) == ["broken", "clip0", "clip2"]
    assert (tracks / "clip2" / "track_rgb_t" / "seed_info.json").is_file()


@pytest.mark.parametrize("stage, options, gpus, match", [
    ("track", TRACK, [0, 0], r"a GPU is listed twice in \[0, 0\]"),
    ("track", TRACK + ["--sam-ckpt", "nowhere.pth"], [0], r"no SAM weights at nowhere.pth"),
    ("track", TRACK, [0], r"--sam-ckpt is needed to cut the frames, unless --seed-labels gives the seed regions"),
    ("track", TRACK + ["--seed-labels", "nowhere"], [0], r"2 clip\(s\) have no seed labels under nowhere: "
                                                         r"\['clip0', 'clip1'\]"),
    ("per_frame", PER_FRAME + ["--depth-source", "pi3", "--sam-ckpt", "{ckpt}"], [0],
     r"2 clip\(s\) of the population lack the depth under .*: \['clip0 \(exports/mini_npz/results__pi3x.npz\)', "
     r"'clip1 \(exports/mini_npz/results__pi3x.npz\)'\]"),
    # The stage records the seed input's ring only, so a tracker's input that burns edges would keep no record of it.
    ("track", ["--rule", "both_ways_from_centre", "--sam-input", "rgb", "--track-base", "rgb_edge", "--sam-ckpt",
               "{ckpt}"], [0], r"--track-base rgb_edge burns edges and --sam-input rgb records no edge ring"),
    # The seed's input would burn edges, so the seed labels alone leave the ring unrecorded.
    ("track", ["--rule", "both_ways_from_centre", "--sam-input", "normal_edge", "--track-base", "normal_edge",
               "--seed-labels", "nowhere"], [0], r"--track-base normal_edge burns edges and --seed-labels records no"),
])
def test_a_run_that_cannot_make_the_condition_is_refused_before_anything_runs(run, stage, options, gpus, match):
    root, tracks, ckpt = run
    for name in ("clip0", "clip1"):
        make_clip(root, name)
    with pytest.raises(ValueError, match=match):
        run_population(stage, settings(stage, *[o.replace("{ckpt}", ckpt) for o in options]), root, tracks,
                       ["clip0", "clip1"], gpus)
    assert not (tracks / "_calls").exists()


def test_a_tracker_input_that_burns_edges_runs_when_the_seed_s_input_records_the_ring(run):
    root, tracks, ckpt = run
    make_clip(root, "clip0")
    s = settings("track", "--rule", "both_ways_from_centre", "--sam-input", "normal_edge", "--track-base", "rgb_edge",
                 "--seed-min-area", "1", sam_ckpt=ckpt)
    run_population("track", s, root, tracks, ["clip0"], [0])
    assert (tracks / "clip0" / "track_rgb_edge_t" / "seed_info.json").is_file()


def test_a_montage_left_without_its_labels_is_refused_before_anything_runs(run):
    # Labels removed by hand without their montage: the stage would refuse the clip on every run.
    root, tracks, ckpt = run
    make_clip(root, "clip0")
    s = settings("track", *TRACK, sam_ckpt=ckpt)
    run_population("track", s, root, tracks, ["clip0"], [0])
    for p in (tracks / "clip0" / "track_rgb_t").iterdir():
        p.unlink()
    (tracks / "clip0" / "track_rgb_t").rmdir()
    (tracks / "_calls" / "clip0.json").unlink()
    with pytest.raises(ValueError, match=r"1 clip\(s\) hold viz/montage_track_rgb_t.png without track_rgb_t"):
        run_population("track", s, root, tracks, ["clip0"], [0])
    assert calls(tracks) == {}


def test_a_record_of_other_settings_after_a_failure_names_the_failure_too(run):
    root, tracks, ckpt = run
    for name in ("broken", "drifted"):
        make_clip(root, name)
    with pytest.raises(ValueError, match=r"^drifted's seed_info.json records other settings .*; broken exited 1"):
        run_population("track", settings("track", *TRACK, sam_ckpt=ckpt), root, tracks, ["broken", "drifted"], [0])


def test_each_clip_that_ends_is_reported_as_it_ends(run, capsys):
    root, tracks, ckpt = run
    for name in ("clip0", "broken"):
        make_clip(root, name)
    with pytest.raises(RuntimeError):
        run_population("per_frame", settings("per_frame", *PER_FRAME, sam_ckpt=ckpt), root, tracks,
                       ["clip0", "broken"], [0])
    out = capsys.readouterr().out
    assert "[ok] GPU 0: clip0 (1/2)" in out and "[failed] GPU 0: broken exited 1 (2/2)" in out


def test_the_gpus_are_named_on_the_command_line():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["per_frame", "--input-dir", "-", "--clips", "-", "--tracks-root", "-",
                                   "--tag", "t", "--sam-ckpt", "x", "--sam-input", "rgb"])


def test_a_signal_ignored_under_nohup_stays_ignored(handlers_restored):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    driver.install_stop_handlers()
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is driver.stop


def test_a_clip_without_depth_is_refused_before_anything_runs(run):
    root, tracks, ckpt = run
    make_clip(root, "clip0")
    with pytest.raises(ValueError, match=r"1 clip\(s\) of the population lack the depth under .*: "
                                          r"\['clip9 \(exports/mini_npz/results.npz\)'\]"):
        run_population("track", settings("track", *TRACK, sam_ckpt=ckpt), root, tracks, ["clip0", "clip9"], [0])
    assert not (tracks / "_calls").exists()


@pytest.mark.parametrize("stop_signal", driver.STOP_SIGNALS)
def test_a_driver_stopped_by_a_signal_stops_its_processes(run, stop_signal):
    # The driver runs as a process of its own, as `kill` finds it, with the stand-in as its stage command.
    root, tracks, ckpt = run
    make_clip(root, "slow")
    population = root.parent / "clips.txt"
    population.write_text("slow\n")
    argv = ["d", "per_frame", "--input-dir", str(root), "--clips", str(population), "--tracks-root", str(tracks),
            "--tag", "t", "--sam-ckpt", ckpt, "--gpus", "0", *PER_FRAME]
    script = (f"import sys; import pipeline.condition_population as d; d.STAGE_COMMANDS = {driver.STAGE_COMMANDS!r}; "
              f"sys.argv = {argv!r}; d.main()")
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=REPO, stdout=subprocess.DEVNULL)
    try:
        wait_for((tracks / "_calls" / "slow.json").is_file)
        child = json.loads((tracks / "_calls" / "slow.json").read_text())["pid"]
        assert alive(child)
        proc.send_signal(stop_signal)
        assert proc.wait(timeout=10) == 128 + stop_signal
        wait_for(lambda: not alive(child))
    finally:
        proc.kill()


def test_a_driver_stopped_while_starting_its_processes_stops_the_ones_started(run, monkeypatch):
    root, tracks, ckpt = run
    for name in ("slow", "clip1"):
        make_clip(root, name)
    started = []
    real_popen = subprocess.Popen

    class Interrupted(real_popen):
        def __init__(self, *args, **kwargs):
            if started:
                raise KeyboardInterrupt
            super().__init__(*args, **kwargs)
            started.append(self)

    monkeypatch.setattr(driver.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_population("per_frame", settings("per_frame", *PER_FRAME, sam_ckpt=ckpt), root, tracks,
                       ["slow", "clip1"], [0, 1])
    assert len(started) == 1 and real_popen.wait(started[0], timeout=10) != 0


@pytest.fixture
def handlers_restored():
    """Put back the signal handlers `main` replaces, so that the test process keeps its own."""
    old = {s: signal.getsignal(s) for s in driver.STOP_SIGNALS}
    yield
    for s, handler in old.items():
        signal.signal(s, handler)


def test_the_command_runs_the_population_file_and_exits_non_zero_on_a_refusal(run, monkeypatch, handlers_restored):
    root, tracks, ckpt = run
    make_clip(root, "clip0")
    population = root.parent / "clips.txt"
    population.write_text("clip0\n")
    argv = ["pipeline.condition_population", "per_frame", "--input-dir", str(root), "--clips", str(population),
            "--tracks-root", str(tracks), "--tag", "t", "--sam-ckpt", ckpt, "--gpus", "2", *PER_FRAME]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert calls(tracks)["clip0"][calls(tracks)["clip0"].index("--gpu") + 1] == "2"
    population.write_text("clip0\nclip9\n")
    with pytest.raises(SystemExit, match="ValueError: 1 clip"):
        main()
    monkeypatch.setattr(sys, "argv", [a if a != str(root) else str(root / "nowhere") for a in argv])
    with pytest.raises(SystemExit, match="no such directory"):
        main()
