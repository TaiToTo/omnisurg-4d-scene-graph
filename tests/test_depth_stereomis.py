"""Test the StereoMIS driver with stand-ins for the DA3 and Pi3X commands.

Each stand-in writes its stage's bundle and manifest record for every clip it
is given, at the setting the driver runs the stage at, and records which GPU
got which clips. A clip named `broken` makes the DA3 stand-in exit non-zero
without writing; one named `no_pi3x` makes the Pi3X stand-in do so. Each
refusal has a test that plants its fault.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("scipy")

import pipeline.depth_population as population  # noqa: E402
import pipeline.depth_stereomis as driver  # noqa: E402

MODEL, RES = population.SETTING

STAND_IN = '''
import json, sys
from pathlib import Path
import numpy as np
stage, args = sys.argv[1], sys.argv[2:]
root = Path(args[args.index("--input-dir") + 1])
gpu = args[args.index("--gpu") + 1]
clips = args[args.index("--clips") + 1:]
assert args[args.index("--device") + 1] == "cuda" and "--no-glb" not in args
if stage == "pi3x":
    assert args[args.index("--max-points") + 1] == "MAX_POINTS"
(root / "_calls").mkdir(exist_ok=True)
(root / "_calls" / f"{stage}_gpu{gpu}.json").write_text(json.dumps(clips))
if ("broken" in clips and stage == "da3") or ("no_pi3x" in clips and stage == "pi3x"):
    sys.exit(1)
for clip in clips:
    n = len(list((root / clip / "input_images").glob("*.png")))
    (root / clip / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    name = "results.npz" if stage == "da3" else "results__pi3x.npz"
    np.savez(root / clip / "exports" / "mini_npz" / name, depth=np.ones((n, 2, 2), np.float32),
             conf=np.ones((n, 2, 2), np.float32), extrinsics=np.zeros((n, 3, 4)), intrinsics=np.zeros((n, 3, 3)))
    manifest = json.loads((root / clip / "frame_manifest.json").read_text())
    if stage == "da3":
        manifest["depth_info"] = {"model": MODEL, "process_res": RES, "border_inpaint": False}
    else:
        manifest.setdefault("geometry_sources", {})["pi3x"] = dict(PI3X_SETTING, median_vertices=MAX_POINTS)
        for fr in manifest["frames"]:
            fr.setdefault("geometry_sources", {})["pi3x"] = {"n_vertices": MAX_POINTS}
    (root / clip / "frame_manifest.json").write_text(json.dumps(manifest))
'''.replace("MODEL", repr(MODEL)).replace("RES", repr(RES)).replace("PI3X_SETTING", repr(driver.PI3X_SETTING)) \
    .replace('"MAX_POINTS"', repr(str(driver.MAX_POINTS))).replace("MAX_POINTS", repr(driver.MAX_POINTS))


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    monkeypatch.setattr(driver, "DA3_COMMAND", [sys.executable, str(script), "da3"])
    monkeypatch.setattr(driver, "PI3X_COMMAND", [sys.executable, str(script), "pi3x"])
    root = tmp_path / "clips"
    root.mkdir()
    return root


def make_clip(root: Path, name: str, n: int = 3) -> None:
    (root / name / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(root / name / "input_images" / f"{i:06d}.png")
    manifest = {"dataset": "stereomis", "n_frames": n, "frames": [{"seq_idx": i} for i in range(n)]}
    (root / name / "frame_manifest.json").write_text(json.dumps(manifest))


def edit_manifest(root: Path, name: str, edit) -> None:
    path = root / name / "frame_manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def calls(root: Path) -> dict[str, list[str]]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((root / "_calls").glob("*.json"))}


def test_both_stages_run_on_every_clip_without_their_output_and_a_done_clip_is_skipped(stand_in):
    names = [f"P1__clip_000{i}" for i in range(4)]
    for name in names:
        make_clip(stand_in, name)
    driver.run_stereomis(stand_in, names[:3], [0, 1])
    assert calls(stand_in) == {"da3_gpu0": names[0:3:2], "da3_gpu1": [names[1]],
                               "pi3x_gpu0": names[0:3:2], "pi3x_gpu1": [names[1]]}
    for f in (stand_in / "_calls").iterdir():
        f.unlink()
    driver.run_stereomis(stand_in, names, [0, 1])
    assert calls(stand_in) == {"da3_gpu0": [names[3]], "pi3x_gpu0": [names[3]]}
    assert {p.name for p in (stand_in / "_logs").iterdir()} == {"depth_gpu0.log", "depth_gpu1.log",
                                                                 "pi3x_gpu0.log", "pi3x_gpu1.log"}


def test_the_population_is_the_usable_clips_of_every_sequence(monkeypatch):
    def clips(root, depth_root, seq):
        return ({"name": f"{seq}__clip_0000", "usable": seq != "P2_3"}, {"name": f"{seq}__clip_0001", "usable": False})

    monkeypatch.setattr(driver.stereomis, "clips", clips)
    assert driver.usable_clips(Path("."), Path(".")) == [f"{s}__clip_0000" for s in driver.stereomis.SEQUENCES
                                                         if s != "P2_3"]


def test_a_clip_that_is_not_there_is_refused_before_anything_runs(stand_in):
    make_clip(stand_in, "P1__clip_0000")
    with pytest.raises(FileNotFoundError, match=r"1 clip\(s\) of the population are not under .*P1__clip_0009"):
        driver.run_stereomis(stand_in, ["P1__clip_0000", "P1__clip_0009"], [0])
    assert not (stand_in / "_calls").exists()


def test_depth_at_another_setting_is_refused_before_anything_runs(stand_in):
    make_clip(stand_in, "P1__clip_0000")
    edit_manifest(stand_in, "P1__clip_0000", lambda m: m.update(
        depth_info={"model": MODEL, "process_res": 252, "border_inpaint": False}))
    with pytest.raises(ValueError, match=r"with depth made at another setting"):
        driver.run_stereomis(stand_in, ["P1__clip_0000"], [0])
    assert not (stand_in / "_calls").exists()


def unthinned(m: dict) -> None:
    """Record Pi3X output whose clouds were not thinned, as the workbench's `P1__clip_0001` holds."""
    m.setdefault("geometry_sources", {})["pi3x"] = dict(driver.PI3X_SETTING, median_vertices=250880)
    for fr in m["frames"]:
        fr.setdefault("geometry_sources", {})["pi3x"] = {"n_vertices": 250880}


def test_pi3x_output_at_another_setting_is_refused_before_anything_runs(stand_in):
    for name in ("P1__clip_0000", "P1__clip_0001", "P1__clip_0002"):
        make_clip(stand_in, name)
    edit_manifest(stand_in, "P1__clip_0001", unthinned)
    edit_manifest(stand_in, "P1__clip_0002", lambda m: m.setdefault("geometry_sources", {}).update(
        pi3x=dict(driver.PI3X_SETTING, pixel_limit=100_000)))
    with pytest.raises(ValueError, match=r"2 clip\(s\) with Pi3X output made at another setting .*"
                                         r"\['P1__clip_0001 \(clouds of up to 250880 points\)', "
                                         r"'P1__clip_0002 \(pixel_limit 100000\)'\]"):
        driver.run_stereomis(stand_in, ["P1__clip_0000", "P1__clip_0001", "P1__clip_0002"], [0])
    assert not (stand_in / "_calls").exists()


def test_a_failed_da3_process_is_reported_and_pi3x_does_not_start(stand_in):
    for name in ("P1__clip_0000", "broken"):
        make_clip(stand_in, name)
    with pytest.raises(RuntimeError, match=r"^DA3: GPU 1 exited 1, see .*depth_gpu1.log; 1 clip\(s\) without depth: "
                                           r"\['broken'\]$"):
        driver.run_stereomis(stand_in, ["P1__clip_0000", "broken"], [0, 1])
    assert set(calls(stand_in)) == {"da3_gpu0", "da3_gpu1"}


def test_a_clip_left_without_pi3x_output_is_reported(stand_in):
    for name in ("P1__clip_0000", "no_pi3x"):
        make_clip(stand_in, name)
    with pytest.raises(RuntimeError, match=r"^Pi3X: GPU 1 exited 1, see .*pi3x_gpu1.log; 1 clip\(s\) without Pi3X "
                                           r"output: \['no_pi3x'\]$"):
        driver.run_stereomis(stand_in, ["P1__clip_0000", "no_pi3x"], [0, 1])


def write_pi3x(root: Path, name: str, n: int, keys=population.BUNDLE_KEYS) -> None:
    (root / name / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    np.savez(root / name / driver.PI3X_BUNDLE, **{k: np.ones((n, 3, 4)) for k in keys})
    edit_manifest(root, name, lambda m: m.setdefault("geometry_sources", {}).update(pi3x=dict(driver.PI3X_SETTING)))


def test_pi3x_poses_that_are_not_one_per_image_are_refused(stand_in):
    make_clip(stand_in, "P1__clip_0000", n=3)
    write_pi3x(stand_in, "P1__clip_0000", n=2)
    with pytest.raises(ValueError, match=r"1 with another number of poses than images: "
                                         r"\['P1__clip_0000 \(2 poses for 3 images\)'\]"):
        driver.check_pi3x(stand_in, ["P1__clip_0000"])


def test_a_pi3x_bundle_of_another_version_or_unreadable_is_refused(stand_in):
    make_clip(stand_in, "P1__clip_0000")
    make_clip(stand_in, "P1__clip_0001")
    write_pi3x(stand_in, "P1__clip_0000", n=3, keys=population.BUNDLE_KEYS | {"points"})
    write_pi3x(stand_in, "P1__clip_0001", n=3)
    (stand_in / "P1__clip_0001" / driver.PI3X_BUNDLE).write_bytes(b"half a bundle")
    with pytest.raises(ValueError, match=r"1 with Pi3X output of another setting or version: \[\"P1__clip_0000 \(keys "
                                         r"\['conf', 'depth', 'extrinsics', 'intrinsics', 'points'\]\)\"\]; "
                                         r"1 with a Pi3X bundle that cannot be read: \['P1__clip_0001 \("):
        driver.check_pi3x(stand_in, ["P1__clip_0000", "P1__clip_0001"])


def test_the_command_takes_the_population_from_the_reader_and_exits_non_zero_on_a_refusal(stand_in, monkeypatch):
    make_clip(stand_in, "P1__clip_0000")
    monkeypatch.setattr(driver, "usable_clips", lambda root, depth_root: ["P1__clip_0000"])
    # `main` installs its stop handlers; the test process keeps its own.
    monkeypatch.setattr(driver.signal, "signal", lambda *a: None)
    argv = ["depth_stereomis", "--root", "x", "--depth-root", "y", "--input-dir", str(stand_in), "--gpus", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    assert calls(stand_in) == {"da3_gpu3": ["P1__clip_0000"], "pi3x_gpu3": ["P1__clip_0000"]}
    monkeypatch.setattr(driver, "usable_clips", lambda root, depth_root: ["P1__clip_0000", "P1__clip_0009"])
    with pytest.raises(SystemExit, match="FileNotFoundError"):
        driver.main()
