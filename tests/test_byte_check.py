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


def make_repo(path: Path, prelude: str = "", runtime: str = "1.0", epilogue: str = "",
              files: dict[str, str] | None = None) -> Path:
    """A git repository at `path` holding a `surgical_core` package and the toy stage, committed."""
    (path / "surgical_core").mkdir(parents=True)
    (path / "surgical_core" / "__init__.py").write_text("")
    (path / "helper.py").write_text("")
    for name, text in (files or {}).items():
        (path / name).write_text(text)
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


def run_check(tmp_path: Path, a: Path, b: Path | None = None, a_watch: tuple = ("surgical_core",),
              b_watch: tuple = ("surgical_core",), a_alone: tuple = (), b_alone: tuple = (), **kw) -> dict:
    clips = tmp_path / "clips"
    if not clips.exists():
        make_clip(clips)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return bc.check(clips, "clip_0001", bc.Side(a, CMD, watch=a_watch, alone=a_alone),
                    bc.Side(b, CMD, watch=b_watch, alone=b_alone) if b else None,
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


def test_a_file_only_one_run_writes_breaks_identity(tmp_path):
    # Every shared file equal, one file extra: the verdict itself must see the one-sided file.
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", epilogue='(c / "extra.bin").write_bytes(b"x")')
    r = run_check(tmp_path, a, b)["result"]
    assert not r["identical"]
    assert r["only_b"] == ["clip_0001/extra.bin"] and r["n_differ"] == 0


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
    with pytest.raises(ValueError, match="does not track"):
        run_check(tmp_path, repo, make_repo(tmp_path / "b"))


def test_a_watched_module_from_an_untracked_file_is_refused(tmp_path):
    # The port's newest module is exactly the file most likely to be uncommitted, and an untracked file
    # leaves the repository clean.
    repo = make_repo(tmp_path / "a", prelude="import surgical_core.extra  # noqa: F401")
    (repo / "surgical_core" / "extra.py").write_text("")
    with pytest.raises(ValueError, match="does not track"):
        run_check(tmp_path, repo)


def test_a_stale_install_inside_the_repository_is_refused(tmp_path):
    # A `.venv` lies inside the repository's directory, but its files belong to no commit.
    repo = make_repo(tmp_path / "a", prelude='sys.path.insert(0, os.path.join(os.getcwd(), ".venv"))')
    (repo / ".venv" / "surgical_core").mkdir(parents=True)
    (repo / ".venv" / "surgical_core" / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="does not track"):
        run_check(tmp_path, repo)


def test_a_run_that_imports_from_the_other_repository_is_refused(tmp_path):
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", prelude=f"sys.path.insert(0, {str(a)!r})\nimport helper  # noqa: F401\nsys.path.pop(0)")
    with pytest.raises(ValueError, match="the other run's repository tracks"):
        run_check(tmp_path, a, b)


def test_a_shared_venv_inside_the_other_repository_is_not_refused(tmp_path):
    # Only the files the other repository tracks are its code; an untracked `.venv` under it is the machine's.
    b = make_repo(tmp_path / "b")
    (b / ".venv").mkdir()
    (b / ".venv" / "shared_dep.py").write_text("")
    a = make_repo(tmp_path / "a", prelude=f'sys.path.insert(0, {str(b / ".venv")!r})\n'
                                          'import shared_dep  # noqa: F401')
    assert run_check(tmp_path, a, b)["result"]["identical"]


def test_each_side_watches_its_own_packages(tmp_path):
    # The workbench imports its wrapper and the port never does, so the wrapper is watched on one side only.
    a = make_repo(tmp_path / "a", prelude="import wrapper  # noqa: F401", files={"wrapper.py": ""})
    b = make_repo(tmp_path / "b")
    assert run_check(tmp_path, a, b, a_watch=("surgical_core", "wrapper"))["result"]["identical"]
    with pytest.raises(ValueError, match="wrapper was not imported"):
        run_check(tmp_path, a, b, a_watch=("surgical_core", "wrapper"), b_watch=("surgical_core", "wrapper"))


def test_a_module_whose_file_is_a_relative_path_is_not_taken_for_a_file(tmp_path, monkeypatch):
    # torch.ops has `__file__ == "_ops.py"`; read from wherever the check runs, it would land in the other repository.
    a = make_repo(tmp_path / "a", prelude='import types\nsys.modules["torch_ops"] = types.ModuleType("torch_ops")\n'
                                          'sys.modules["torch_ops"].__file__ = "helper.py"')
    b = make_repo(tmp_path / "b")
    monkeypatch.chdir(b)
    assert run_check(tmp_path, a, b)["result"]["identical"]


def test_a_run_that_does_not_import_a_watched_package_is_refused(tmp_path):
    with pytest.raises(ValueError, match="absent_package was not imported"):
        run_check(tmp_path, make_repo(tmp_path / "a"), a_watch=("surgical_core", "absent_package"))


def test_a_run_that_leaves_no_record_of_its_imports_is_refused(tmp_path):
    # os._exit skips the exit handlers, so the hook never writes; nothing proves where the code came from.
    with pytest.raises(ValueError, match="no record"):
        run_check(tmp_path, make_repo(tmp_path / "a", epilogue="os._exit(0)"))


def test_a_repository_with_uncommitted_changes_is_refused(tmp_path):
    repo = make_repo(tmp_path / "a")
    (repo / "helper.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="uncommitted"):
        run_check(tmp_path, repo)


def test_a_run_that_writes_into_its_inputs_is_refused_and_the_clip_is_untouched(tmp_path):
    repo = make_repo(tmp_path / "a", epilogue='open(images[0], "ab").write(b"x")')
    with pytest.raises(ValueError, match="its inputs"):
        run_check(tmp_path, repo)
    # The working copy took the damage; the clip's own file did not.
    assert (tmp_path / "clips" / "clip_0001" / "input_images" / "000000.png").read_bytes() == bytes([0]) * 8


def test_a_run_that_only_overwrites_an_input_is_refused_for_that_and_not_as_idle(tmp_path):
    # The overwrite must be named, not hidden behind "wrote nothing".
    repo = make_repo(tmp_path / "a", prelude='import surgical_core\n'
                                             'c = Path(sys.argv[1]) / sys.argv[2]\n'
                                             '(c / "input_images" / "000000.png").write_bytes(b"CLOBBERED")\n'
                                             'sys.exit(0)')
    with pytest.raises(ValueError, match="its inputs"):
        run_check(tmp_path, repo)
    assert (tmp_path / "clips" / "clip_0001" / "input_images" / "000000.png").read_bytes() == bytes([0]) * 8


def test_a_run_that_writes_nothing_is_refused(tmp_path):
    repo = make_repo(tmp_path / "a", prelude="import surgical_core\nsys.exit(0)")
    with pytest.raises(RuntimeError, match="wrote nothing"):
        run_check(tmp_path, repo)


def test_a_stage_that_writes_only_into_the_manifest_is_not_taken_for_idle(tmp_path):
    # The manifest exists before the run, so only its hash can show it was written.
    repo = make_repo(tmp_path / "a", epilogue='import shutil\nshutil.rmtree(c / "depth_raw")')
    r = run_check(tmp_path, repo)["result"]
    assert r["identical"] and r["n_shared"] == 1


def test_a_command_that_fails_is_reported_with_its_log(tmp_path):
    repo = make_repo(tmp_path / "a", prelude='raise SystemExit("the stage broke")')
    with pytest.raises(RuntimeError, match="the stage broke"):
        run_check(tmp_path, repo)


def test_runs_with_different_packages_are_refused():
    env = {"python": "3.12.3", "blas": "openblas 0.3.27",
           "distributions": {"numpy": [{"name": "numpy", "version": "1.26.4", "commit": None}],
                             "pi3": [{"name": "pi3", "version": "0.1", "commit": "abc"}]}}
    bc.check_same_environment(env, json.loads(json.dumps(env)))
    for change in ({"python": "3.12.4"}, {"blas": "accelerate 2.0"},
                   {"distributions": {**env["distributions"],
                                      "pi3": [{"name": "pi3", "version": "0.1", "commit": "def"}]}}):
        with pytest.raises(ValueError, match="different environments"):
            bc.check_same_environment(env, {**env, **change})


def test_the_runs_own_code_is_no_environment_difference():
    # The two sides' own packages differ by design; their difference is the byte comparison's to judge.
    env = {"python": "3.12.3", "blas": None,
           "distributions": {"surgical_core": [{"name": "omnisurg", "version": "0.1", "commit": "abc"}]}}
    other = json.loads(json.dumps(env))
    other["distributions"]["surgical_core"][0]["commit"] = "def"
    bc.check_same_environment(env, other, own=frozenset({"surgical_core"}))
    with pytest.raises(ValueError, match="different environments"):
        bc.check_same_environment(env, other)


def test_runs_that_import_different_distributions_are_refused(tmp_path):
    # pytest is installed for the tests themselves, so one side importing it is a real difference, and no
    # fixed list of package names would have it.
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", prelude="import pytest  # noqa: F401")
    with pytest.raises(ValueError, match="different environments"):
        run_check(tmp_path, a, b)


def test_what_one_run_is_named_to_import_alone_is_recorded_and_not_compared(tmp_path):
    # The workbench's surgical_core imports trimesh, which no stage calls; named, it no longer refuses the pair.
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", prelude="import iniconfig  # noqa: F401")
    report = run_check(tmp_path, a, b, b_alone=("iniconfig",))
    assert report["result"]["identical"]
    assert report["alone"] == {"a": [], "b": ["iniconfig"]}
    assert report["environment"]["b"]["distributions"]["iniconfig"][0]["name"] == "iniconfig"
    assert "iniconfig" not in report["environment"]["a"]["distributions"]


def test_a_module_named_alone_that_its_run_did_not_import_is_refused(tmp_path):
    # A stale name would excuse a module the run never imported, and the list would no longer say what ran.
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", prelude="import iniconfig  # noqa: F401")
    with pytest.raises(ValueError, match="iniconfig, named as run a's alone, came from no distribution"):
        run_check(tmp_path, a, b, a_alone=("iniconfig",), b_alone=("iniconfig",))


def test_a_module_named_alone_that_the_other_run_imports_too_is_refused(tmp_path):
    # Both runs imported it, so its versions must agree, and naming it would skip that comparison.
    a = make_repo(tmp_path / "a", prelude="import iniconfig  # noqa: F401")
    b = make_repo(tmp_path / "b", prelude="import iniconfig  # noqa: F401")
    with pytest.raises(ValueError, match="iniconfig, named as run a's alone, was imported by the other run too"):
        run_check(tmp_path, a, b, a_alone=("iniconfig",))


def test_only_the_named_modules_are_left_out_of_the_comparison():
    env = {"python": "3.12.3", "blas": None,
           "distributions": {"numpy": [{"name": "numpy", "version": "1.26.4", "commit": None}],
                             "trimesh": [{"name": "trimesh", "version": "4.11.3", "commit": None}]}}
    other = {"python": "3.12.3", "blas": None,
             "distributions": {"numpy": [{"name": "numpy", "version": "1.26.4", "commit": None}]}}
    bc.check_same_environment(env, other, alone_a=("trimesh",))
    with pytest.raises(ValueError, match="different environments"):
        bc.check_same_environment(env, other)
    changed = json.loads(json.dumps(other))
    changed["distributions"]["numpy"][0]["version"] = "2.0.0"
    with pytest.raises(ValueError, match="numpy"):
        bc.check_same_environment(env, changed, alone_a=("trimesh",))


def test_one_command_run_twice_imports_nothing_alone(tmp_path):
    a = make_repo(tmp_path / "a")
    with pytest.raises(ValueError, match="one command runs twice"):
        run_check(tmp_path, a, a_alone=("iniconfig",))
    make_clip(tmp_path / "clips2")
    proc = subprocess.run([sys.executable, "-m", "pipeline.byte_check", "--root", str(tmp_path / "clips2"),
                           "--clip", "clip_0001", "--a-repo", str(a), "--a-cmd", CMD, "--a-alone", "iniconfig",
                           "--work", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "one command runs twice" in proc.stderr


def test_processes_of_one_run_with_different_packages_are_refused():
    rec = {"python": "3.12.3", "blas": None,
           "distributions": {"numpy": [{"name": "numpy", "version": "1.26.4", "commit": None}]}, "modules": {}}
    quiet = {"python": "3.12.3", "blas": None, "distributions": {}, "modules": {}}
    # A process that imported nothing says nothing about the run's distributions; the records merge.
    assert bc.environment([rec, quiet])["distributions"]["numpy"][0]["version"] == "1.26.4"
    changed = json.loads(json.dumps(rec))
    changed["distributions"]["numpy"][0]["version"] = "2.0.0"
    with pytest.raises(ValueError, match="different distributions"):
        bc.environment([rec, changed])


# ------------------------------------------------------------- the working copy


def test_the_working_copy_holds_no_output_and_copies_its_inputs(tmp_path):
    src = make_clip(tmp_path / "clips")
    (src / "depth_raw").mkdir()
    (src / "depth_raw" / "depth_000000.npy").write_bytes(b"x")
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst)
    # A leftover depth_raw/ would let a stage skip its work and look identical.
    assert not (dst / "depth_raw").exists()
    m = json.loads((dst / "frame_manifest.json").read_text())
    assert "depth_info" not in m and set(m["frames"][0]) == {"native_frame"}
    # Copies, not links: a stage that writes through a link would overwrite the clip's own files.
    assert not any(p.is_symlink() for p in (dst / "input_images").iterdir())
    assert set(bc.collect(tmp_path / "work")) == {
        "clip_0001/input_images/000000.png", "clip_0001/input_images/000001.png",
        "clip_0001/input_images/000002.png", "clip_0001/seg_masks/000000_class.png",
        "clip_0001/frame_manifest.json"}


def test_a_needed_directory_is_copied_so_a_stage_writing_beside_it_stays_in_the_copy(tmp_path):
    src = make_clip(tmp_path / "clips")
    (src / "exports" / "mini_npz").mkdir(parents=True)
    (src / "exports" / "mini_npz" / "results.npz").write_bytes(b"npz")
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst, needs=("exports/mini_npz",))
    assert not (dst / "exports" / "mini_npz").is_symlink()
    (dst / "exports" / "mini_npz" / "new.bin").write_bytes(b"x")
    assert not (src / "exports" / "mini_npz" / "new.bin").exists()


def test_the_working_copy_keeps_the_first_frames_only(tmp_path):
    src = make_clip(tmp_path / "clips", n=5)
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst, frames=2)
    assert sorted(p.name for p in (dst / "input_images").iterdir()) == ["000000.png", "000001.png"]
    assert [f["native_frame"] for f in json.loads((dst / "frame_manifest.json").read_text())["frames"]] == [0, 1]


def test_a_hidden_file_is_not_a_frame(tmp_path):
    # `.DS_Store` sorts before the images; counted as a frame, it would push a real frame out of the cut.
    src = make_clip(tmp_path / "clips", n=3)
    (src / "input_images" / ".DS_Store").write_bytes(b"junk")
    dst = tmp_path / "work" / "clip_0001"
    bc.prepare_clip(src, dst, frames=2)
    assert sorted(p.name for p in (dst / "input_images").iterdir()) == ["000000.png", "000001.png"]


def test_frames_cuts_n_frames_with_the_frames_it_counts(tmp_path):
    # A fail-closed stage counts the frames; a stale total would make it refuse the cut copy.
    src = make_clip(tmp_path / "clips", n=5)
    m = json.loads((src / "frame_manifest.json").read_text())
    m["n_frames"] = 5
    (src / "frame_manifest.json").write_text(json.dumps(m, indent=2))
    bc.prepare_clip(src, tmp_path / "work" / "clip_0001", frames=2)
    assert json.loads((tmp_path / "work" / "clip_0001" / "frame_manifest.json").read_text())["n_frames"] == 2


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


def test_a_moved_nan_gives_no_gap_at_all(tmp_path):
    # The gap is computed where both are finite; with the NaN moved, a gap of 1.0 would be a guess.
    fa, fb = _pair(tmp_path, np.array([np.nan, 1.0]), np.array([0.0, np.nan]))
    assert bc.numeric_spread(fa["depth_000000.npy"], fb["depth_000000.npy"]) is None


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
    # The runtime appears again under another key; stripping the top one and forgiving the file would guess.
    ('{\n  "runtime_sec": 1.71,\n  "inner": {"runtime_sec": 1}\n}',
     '{\n  "runtime_sec": 2.04,\n  "inner": {"runtime_sec": 1}\n}'),
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


def test_the_command_line_tells_a_difference_from_a_refusal_by_exit_code(tmp_path):
    repo = make_repo(tmp_path / "a", epilogue='(c / "noise.bin").write_bytes(os.urandom(8))')
    make_clip(tmp_path / "clips")
    base = [sys.executable, "-m", "pipeline.byte_check", "--root", str(tmp_path / "clips"), "--clip", "clip_0001",
            "--work", str(tmp_path), "--a-repo", str(repo), "--a-cmd", CMD]
    differs = subprocess.run(base, capture_output=True, text=True)
    assert differs.returncode == 1, differs.stderr
    (repo / "helper.py").write_text("changed = True\n")
    refused = subprocess.run(base, capture_output=True, text=True)
    assert refused.returncode == 2
    assert "refused" in refused.stderr
