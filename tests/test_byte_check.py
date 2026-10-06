"""`pipeline.byte_check` reports every difference it should, and refuses every pair it cannot vouch for.

The comparison is planted with faults a file at a time; the refusals with
toy repositories whose stage, run as a real subprocess, misbehaves in one
way each: it reads a third copy of `surgical_core`, imports the other
repository, writes into its inputs, exits without leaving a record, or
sits in a repository with uncommitted changes.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline import byte_check as bc

# A toy stage: it reads the working copy's images and manifest and writes what a depth stage would, one line
# replaced per test to plant a fault.
STAGE = '''\
import json
import os
import sys
import time
from pathlib import Path
PRELUDE
import surgical_core  # noqa: F401
root, clip = Path(sys.argv[1]), sys.argv[2]
c = root / clip
images = sorted((c / "input_images").iterdir())
(c / "depth_raw").mkdir()
for i, p in enumerate(images):
    (c / "depth_raw" / f"depth_{i:06d}.txt").write_text(p.name)
m = json.loads((c / "frame_manifest.json").read_text())
m["depth_info"] = {"n_frames": len(images), "runtime_sec": RUNTIME}
(c / "frame_manifest.json").write_text(json.dumps(m, indent=2))
EPILOGUE
'''

CMD = f"{shlex.quote(sys.executable)} stage.py {{root}} {{clip}}"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def make_repo(path: Path, prelude: str = "", runtime: str = "1.0", epilogue: str = "") -> Path:
    """A git repository at `path` holding a `surgical_core` package and the toy stage, committed."""
    (path / "surgical_core").mkdir(parents=True)
    (path / "surgical_core" / "__init__.py").write_text("")
    (path / "helper.py").write_text("")
    (path / "stage.py").write_text(STAGE.replace("PRELUDE", prelude).replace("RUNTIME", runtime)
                                   .replace("EPILOGUE", epilogue))
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "stage")
    return path


def make_clip(root: Path, name: str = "clip_0001", n: int = 3) -> Path:
    """A clip with `n` images, one mask, and a manifest carrying what a stage writes."""
    c = root / name
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        (c / "input_images" / f"{i:06d}.png").write_bytes(bytes([i]) * 8)
    (c / "seg_masks").mkdir()
    (c / "seg_masks" / "000000_class.png").write_bytes(b"mask")
    (c / "frame_manifest.json").write_text(json.dumps({
        "depth_info": {"model": "stale"},
        "frames": [{"native_frame": i, "glb_centroid": [0, 0, 0]} for i in range(n)]}, indent=2))
    return c


def run_check(tmp_path: Path, a: Path, b: Path | None = None, **kw) -> dict:
    clips = tmp_path / "clips"
    if not clips.exists():
        make_clip(clips)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return bc.check(clips, "clip_0001", bc.Side(a, CMD), bc.Side(b, CMD) if b else None,
                    Path(os.path.realpath(work)) / f"w{len(list(work.iterdir()))}", **kw)


# ------------------------------------------------------------- the runs


def test_one_stage_run_twice_writes_the_same_bytes(tmp_path):
    report = run_check(tmp_path, make_repo(tmp_path / "a"))
    r = report["result"]
    assert r["identical"] and r["n_differ"] == 0
    assert r["n_shared"] == 4, "the manifest and the three depth files"


def test_two_repositories_with_the_same_stage_agree(tmp_path):
    report = run_check(tmp_path, make_repo(tmp_path / "a"), make_repo(tmp_path / "b"))
    assert report["result"]["identical"]
    assert len(report["repos"]) == 2


def test_a_stage_that_writes_its_runtime_is_equal_with_that_file_counted_apart(tmp_path):
    r = run_check(tmp_path, make_repo(tmp_path / "a", runtime="time.time()"))["result"]
    assert r["identical"]
    assert r["volatile_only"] == ["clip_0001/frame_manifest.json"]


def test_a_stage_that_writes_different_bytes_differs(tmp_path):
    repo = make_repo(tmp_path / "a", epilogue='(c / "noise.bin").write_bytes(os.urandom(8))')
    r = run_check(tmp_path, repo)["result"]
    assert not r["identical"]
    assert r["differ"] == ["clip_0001/noise.bin"]


def test_a_stage_that_writes_a_time_stamp_differs(tmp_path):
    # Only `runtime_sec` is forgiven; the workbench's tool also forgave `timestamp`, which no stage writes.
    repo = make_repo(tmp_path / "a", epilogue='m["timestamp"] = time.time_ns()\n'
                                              '(c / "frame_manifest.json").write_text(json.dumps(m, indent=2))')
    r = run_check(tmp_path, repo)["result"]
    assert r["differ"] == ["clip_0001/frame_manifest.json"]


def test_frames_keeps_the_first_images_and_manifest_frames(tmp_path):
    r = run_check(tmp_path, make_repo(tmp_path / "a"), frames=2)["result"]
    assert r["identical"] and r["n_shared"] == 3


# ------------------------------------------------------------- the refusals


def test_a_run_that_imports_surgical_core_from_a_third_copy_is_refused(tmp_path):
    third = tmp_path / "sibling_clone"
    (third / "surgical_core").mkdir(parents=True)
    (third / "surgical_core" / "__init__.py").write_text("")
    repo = make_repo(tmp_path / "a", prelude=f"sys.path.insert(0, {str(third)!r})")
    with pytest.raises(ValueError, match="outside"):
        run_check(tmp_path, repo, make_repo(tmp_path / "b"))


def test_a_run_that_imports_from_the_other_repository_is_refused(tmp_path):
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", prelude=f"sys.path.insert(0, {str(a)!r})\nimport helper  # noqa: F401\nsys.path.pop(0)")
    with pytest.raises(ValueError, match="inside the other run's repository"):
        run_check(tmp_path, a, b)


def test_a_run_that_does_not_import_a_watched_package_is_refused(tmp_path):
    with pytest.raises(ValueError, match="absent_package was not imported"):
        run_check(tmp_path, make_repo(tmp_path / "a"), watched=("surgical_core", "absent_package"))


def test_a_run_that_leaves_no_record_of_its_imports_is_refused(tmp_path):
    # os._exit skips the exit handlers, so the hook never writes; nothing proves where the code came from.
    with pytest.raises(ValueError, match="no record"):
        run_check(tmp_path, make_repo(tmp_path / "a", epilogue="os._exit(0)"))


def test_a_repository_with_uncommitted_changes_is_refused(tmp_path):
    repo = make_repo(tmp_path / "a")
    (repo / "helper.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="uncommitted"):
        run_check(tmp_path, repo)


def test_a_run_that_writes_into_its_linked_inputs_is_refused(tmp_path):
    repo = make_repo(tmp_path / "a", epilogue='open(images[0], "ab").write(b"x")')
    with pytest.raises(ValueError, match="linked inputs"):
        run_check(tmp_path, repo)


def test_a_run_that_writes_nothing_is_refused(tmp_path):
    repo = make_repo(tmp_path / "a", prelude="import surgical_core\nsys.exit(0)")
    with pytest.raises(RuntimeError, match="wrote nothing"):
        run_check(tmp_path, repo)


def test_a_command_that_fails_is_reported_with_its_log(tmp_path):
    repo = make_repo(tmp_path / "a", prelude='raise SystemExit("the stage broke")')
    with pytest.raises(RuntimeError, match="the stage broke"):
        run_check(tmp_path, repo)


def test_runs_with_different_packages_are_refused():
    env = {"python": "3.12.3", "packages": {"numpy": "1.26.4", "torch": "2.10.0"}, "commits": {"pi3": "abc"}}
    bc.check_same_environment(env, json.loads(json.dumps(env)))
    for change in ({"packages": {"numpy": "2.0.0", "torch": "2.10.0"}}, {"commits": {"pi3": "def"}},
                   {"python": "3.12.4"}):
        with pytest.raises(ValueError, match="different environments"):
            bc.check_same_environment(env, {**env, **change})


def test_processes_of_one_run_with_different_packages_are_refused():
    rec = {"python": "3.12.3", "packages": {"numpy": "1.26.4"}, "commits": {}, "modules": {}}
    assert bc.environment([rec, dict(rec)])["packages"] == {"numpy": "1.26.4"}
    with pytest.raises(ValueError, match="different packages"):
        bc.environment([rec, {**rec, "packages": {"numpy": "2.0.0"}}])


# ------------------------------------------------------------- the working copy


def test_the_working_copy_holds_no_output_and_links_its_inputs(tmp_path):
    src = make_clip(tmp_path / "clips")
    (src / "depth_raw").mkdir()
    (src / "depth_raw" / "depth_000000.npy").write_bytes(b"x")
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst)
    # A leftover depth_raw/ would let a stage skip its work and look identical.
    assert not (dst / "depth_raw").exists()
    m = json.loads((dst / "frame_manifest.json").read_text())
    assert "depth_info" not in m and set(m["frames"][0]) == {"native_frame"}
    assert all(p.is_symlink() for p in (dst / "input_images").iterdir())
    assert set(bc.linked_inputs(tmp_path / "work")) == {
        "clip_0001/input_images/000000.png", "clip_0001/input_images/000001.png",
        "clip_0001/input_images/000002.png", "clip_0001/seg_masks/000000_class.png"}
    assert set(bc.collect(tmp_path / "work")) == {"clip_0001/frame_manifest.json"}


def test_the_working_copy_keeps_the_first_frames_only(tmp_path):
    src = make_clip(tmp_path / "clips", n=5)
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst, frames=2)
    assert sorted(p.name for p in (dst / "input_images").iterdir()) == ["000000.png", "000001.png"]
    assert [f["native_frame"] for f in json.loads((dst / "frame_manifest.json").read_text())["frames"]] == [0, 1]


@pytest.mark.parametrize("frames", [0, 6])
def test_frames_the_clip_does_not_have_are_refused(tmp_path, frames):
    src = make_clip(tmp_path / "clips", n=5)
    with pytest.raises(ValueError, match="cannot be kept"):
        bc.prepare_clip(src, tmp_path / "work" / "clip_0001", frames=frames)


def test_a_stage_input_the_clip_lacks_is_refused(tmp_path):
    src = make_clip(tmp_path / "clips")
    with pytest.raises(FileNotFoundError, match="results.npz"):
        bc.prepare_clip(src, tmp_path / "work" / "clip_0001", needs=("exports/mini_npz/results.npz",))


def test_collect_finds_a_file_no_list_names_and_one_beside_the_clip(tmp_path):
    root = tmp_path / "run"
    bc.prepare_clip(make_clip(tmp_path / "clips"), root / "clip_0001")
    (root / "clip_0001" / "surprise").mkdir()
    (root / "clip_0001" / "surprise" / "new.bin").write_bytes(b"x")
    (root / "_track" / "clip_0001").mkdir(parents=True)
    (root / "_track" / "clip_0001" / "label_000000.npy").write_bytes(b"l")
    found = bc.collect(root)
    assert {"clip_0001/surprise/new.bin", "_track/clip_0001/label_000000.npy"} <= set(found)
    assert not [k for k in found if "input_images" in k or "seg_masks" in k]


# ------------------------------------------------------------- the comparison


def _pair(tmp_path, a, b, name="depth_000000.npy"):
    """One file per run, from arrays (saved as `.npy`) or text; the two {path: file} mappings."""
    out = []
    for run, content in (("run0", a), ("run1", b)):
        d = tmp_path / run
        d.mkdir(parents=True)
        if isinstance(content, np.ndarray):
            np.save(d / name, content)
        else:
            (d / name).write_text(content)
        out.append({name: d / name})
    return out


def test_one_ulp_is_a_difference(tmp_path):
    a = np.full((4, 4), 1.0, dtype=np.float32)
    b = a.copy()
    b[0, 0] = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert bc.compare(*_pair(tmp_path, a, b))["n_differ"] == 1


def test_a_changed_value_is_reported_with_its_size(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = a.copy()
    b[1, 2] += 0.25
    r = bc.compare(*_pair(tmp_path, a, b))
    assert r["max_abs"] == pytest.approx(0.25) and r["max_rel"] == pytest.approx(0.25 / 6.0)


def test_a_missing_file_and_an_extra_one_are_reported(tmp_path):
    fa, fb = _pair(tmp_path, np.zeros(2), np.zeros(2))
    extra = tmp_path / "run1" / "depth_000001.npy"
    np.save(extra, np.zeros(2))
    fb = {"depth_000001.npy": extra}
    r = bc.compare(fa, fb)
    assert r["only_a"] == ["depth_000000.npy"] and r["only_b"] == ["depth_000001.npy"]


def test_a_shape_change_is_a_difference_with_no_gap(tmp_path):
    r = bc.compare(*_pair(tmp_path, np.zeros((2, 2), np.float32), np.zeros((3, 3), np.float32)))
    assert r["n_differ"] == 1 and r["max_abs"] == 0.0


def test_a_nan_neither_swallows_a_gap_nor_hides_when_it_moves(tmp_path):
    a = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    b = a.copy()
    b[1, 1] = 3.5
    assert bc.compare(*_pair(tmp_path / "x", a, b))["max_abs"] == pytest.approx(0.5)
    moved = bc.compare(*_pair(tmp_path / "y", np.array([np.nan, 1.0]), np.array([0.0, np.nan])))
    assert moved["n_differ"] == 1 and moved["max_abs"] == 0.0


def test_a_zero_reference_gives_no_relative_gap(tmp_path):
    r = bc.compare(*_pair(tmp_path, np.array([0.0, 4.0]), np.array([0.5, 4.0])))
    assert r["max_abs"] == pytest.approx(0.5) and r["max_rel"] == 0.0


M = "frame_manifest.json"


@pytest.mark.parametrize("a, b", [
    # A real field changed beside the runtime.
    ('{\n  "runtime_sec": 1.71,\n  "n_frames": 15\n}', '{\n  "runtime_sec": 2.04,\n  "n_frames": 14\n}'),
    # A time stamp, no longer forgiven.
    ('{\n  "timestamp": 1.0,\n  "n_frames": 14\n}', '{\n  "timestamp": 2.0,\n  "n_frames": 14\n}'),
    # 14 against 14.0, equal once parsed.
    ('{\n  "runtime_sec": 1.71,\n  "n_frames": 14\n}', '{\n  "runtime_sec": 2.04,\n  "n_frames": 14.0\n}'),
    # Keys reordered.
    ('{\n  "a": 1,\n  "b": 2,\n  "runtime_sec": 1.71\n}', '{\n  "b": 2,\n  "a": 1,\n  "runtime_sec": 2.04\n}'),
    # Reindented.
    ('{\n  "runtime_sec": 1.71,\n  "n_frames": 14\n}', '{\n    "runtime_sec": 2.04,\n    "n_frames": 14\n}'),
    # The runtime vanished.
    ('{\n  "runtime_sec": 1.71,\n  "n_frames": 14\n}', '{\n  "n_frames": 14\n}'),
    # Not one key per line, so nothing can be stripped without guessing.
    ('{"runtime_sec": 1.71, "n_frames": 14}', '{"runtime_sec": 2.04, "n_frames": 14}'),
])
def test_a_manifest_change_beyond_the_runtime_is_a_difference(tmp_path, a, b):
    r = bc.compare(*_pair(tmp_path, a, b, M))
    assert r["n_differ"] == 1 and r["volatile_only"] == []


def test_a_manifest_that_differs_in_its_runtime_alone_is_counted_apart(tmp_path):
    r = bc.compare(*_pair(tmp_path, '{\n  "runtime_sec": 1.71,\n  "n_frames": 15\n}',
                          '{\n  "runtime_sec": 2.04,\n  "n_frames": 15\n}', M))
    assert r["n_differ"] == 0 and r["volatile_only"] == [M]


# ------------------------------------------------------------- what is recorded


def test_a_weight_that_is_not_there_is_recorded_as_missing(tmp_path):
    assert bc.fingerprint(tmp_path / "nope.pth") is None
    (tmp_path / "w.pth").write_bytes(b"0123456789")
    assert bc.fingerprint(tmp_path / "w.pth")["bytes"] == 10
    (tmp_path / "cache" / "sub").mkdir(parents=True)
    (tmp_path / "cache" / "a.bin").write_bytes(b"xy")
    (tmp_path / "cache" / "sub" / "b.bin").write_bytes(b"z")
    assert bc.fingerprint(tmp_path / "cache")["files"] == 2


def test_a_directory_outside_git_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a git repository"):
        bc.git_state(tmp_path)


def test_the_command_line_writes_its_report_and_exits_zero_on_equal_runs(tmp_path):
    repo = make_repo(tmp_path / "a")
    make_clip(tmp_path / "clips")
    out = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, "-m", "pipeline.byte_check", "--root", str(tmp_path / "clips"),
                           "--clip", "clip_0001", "--a-repo", str(repo), "--a-cmd", CMD, "--json-out", str(out),
                           "--work", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "IDENTICAL" in proc.stdout
    assert json.loads(out.read_text())["result"]["identical"]
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("byte_check_")] == [], "the scratch is removed"
