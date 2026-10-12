"""Test the D4D trajectory command on sessions made up for the test; D4D's loader is replaced by stand-in clips.

A made-up clip has `N` frames 0.5 s apart. Its session's tracker logs a pose every 0.05 s from a second
before the clip to a second after, along a curve that turns about z. A depth stage's bundle holds that truth.
Each refusal and each reason a clip is left out has a test that plants its fault:

- a population file that lists no clip or a clip twice, and a clip the dataset does not hold;
- a clip that exists already, as itself or as `.partial`, and an image that cannot be read or written;
- a missing bundle or manifest, a manifest of another clip or of other frames, and a session without its log;
- a census clip with no tracker log, a log that does not cover it, or too few tracked times;
- a scored clip with too few tracked frames, a jump or a turn faster than a camera moves, or no motion;
- a method named twice.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("scipy")

import surgical_core.d4d as d4d  # noqa: E402
import trajectory_eval.tools.score_d4d as cmd  # noqa: E402
from trajectory_eval.code_sha import trajectory_code_sha  # noqa: E402
from trajectory_eval.pose_metrics import gt_span  # noqa: E402

N = 40
T0 = 100.0
FRAME_DT = 0.5
LOG_DT = 0.05


def truth(t: np.ndarray) -> np.ndarray:
    """Return camera-to-tracker poses at times `t`, in metres: a curve of about 15 mm radius that turns about z."""
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, 3] = np.stack([0.02 * np.sin(t / 3), 0.01 * np.cos(t / 5), 0.003 * (t - T0)], 1)
    a = 0.05 * t
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    return T


def frame_times(n: int = N) -> np.ndarray:
    return T0 + FRAME_DT * np.arange(n)


def write_log(session: Path, times: np.ndarray, poses: np.ndarray | None = None) -> None:
    """Write a tracker log of `poses`, or of the truth, at `times`, as D4D names and holds it."""
    from scipy.spatial.transform import Rotation

    poses = truth(times) if poses is None else poses
    (session / "tf").mkdir(parents=True)
    for t, T in zip(times, poses):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        transform = {"translation": dict(zip("xyz", T[:3, 3])), "rotation": dict(zip("xyzw", q))}
        (session / "tf" / f"{d4d.TRACKER_PREFIXES[0]}_{t:.9f}.json").write_text(json.dumps({"transform": transform}))


def make_clip(root: Path, session: str = "s_a", name: str = "Clip_1", n: int = N, log: bool = True,
              log_times: np.ndarray | None = None) -> SimpleNamespace:
    """Write a clip's images and its session's tracker log, and return a stand-in for the loader's clip."""
    session_dir = root / "specimen_1" / session
    times = frame_times(n)
    images = session_dir / name / "left_images_rect"
    images.mkdir(parents=True)
    paths = []
    for i, t in enumerate(times):
        paths.append(str(images / f"{int(t)}_{round((t - int(t)) * 1e9):09d}.png"))
        cv2.imwrite(paths[-1], np.full((8, 10, 3), i, np.uint8))
    if log and not (session_dir / "tf").exists():
        write_log(session_dir, np.arange(times[0] - 1, times[-1] + 1 + LOG_DT / 2, LOG_DT)
                  if log_times is None else log_times)
    return SimpleNamespace(name=name, path=session_dir / name, left_img_paths=paths,
                           session=SimpleNamespace(name=session, path=session_dir,
                                                   specimen=SimpleNamespace(name="specimen_1")))


def write_bundles(clip_root: Path, clip: dict, methods=("da3", "pi3x"), poses: np.ndarray | None = None) -> None:
    """Write each method's bundle holding `poses`, or the truth, as world-to-camera extrinsics."""
    T = truth(clip["times"]) if poses is None else poses
    E = np.zeros((len(T), 3, 4))
    E[:, :, :3] = np.transpose(T[:, :3, :3], (0, 2, 1))
    E[:, :, 3] = -np.einsum("nij,nj->ni", E[:, :, :3], T[:, :3, 3])
    for m in methods:
        bundle = clip_root / clip["name"] / cmd.METHOD_BUNDLES[m]
        bundle.parent.mkdir(parents=True, exist_ok=True)
        np.savez(bundle, extrinsics=E)


def prepared(tmp_path, clips: list) -> list[dict]:
    """Write the trajectory clips of `clips` under `tmp_path / "clips"` with both bundles, and return them."""
    cut = [cmd.trajectory_clip(c) for c in clips]
    (tmp_path / "clips").mkdir()
    cmd.prepare(cut, tmp_path / "clips")
    for c in cut:
        write_bundles(tmp_path / "clips", c)
    return cut


def test_the_census_measures_the_tracked_camera_s_radius_and_path(tmp_path):
    clip = make_clip(tmp_path)
    rec = cmd.census([clip])[0]
    centres = truth(np.linspace(T0, frame_times()[-1], N))[:, :3, 3] * 1000
    span, path = rec.pop("span_mm"), rec.pop("path_mm")
    assert rec == dict(key="specimen_1/s_a/Clip_1", specimen="specimen_1", n_frames=N, duration_s=19.5,
                       tracker_prefix=d4d.TRACKER_PREFIXES[0], tracked_frac=1.0)
    assert span == pytest.approx(gt_span(centres), abs=1e-3) and span > 10
    assert path == pytest.approx(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum(), abs=1e-2)


def test_the_census_reads_each_session_s_log_once(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2")]
    listed = []
    real = d4d.tracker_log
    monkeypatch.setattr(d4d, "tracker_log", lambda s: listed.append(s) or real(s))
    assert [r["span_mm"] is not None for r in cmd.census(clips)] == [True, True]
    assert listed == [str(tmp_path / "specimen_1" / "s_a")]


@pytest.mark.parametrize("log, log_times, reason", [
    (False, None, "no_log"),
    (True, np.arange(T0 + 0.1, T0 + 21, LOG_DT), "log_not_covering"),
    (True, np.arange(T0 - 1, T0 + 21, LOG_DT)[::10], "few_tracked"),
])
def test_the_census_names_why_a_clip_is_not_measured(tmp_path, log, log_times, reason):
    rec = cmd.census([make_clip(tmp_path, log=log, log_times=log_times)])[0]
    assert rec["span_mm"] is None and rec["reason"] == reason
    assert cmd.selected([rec]) == []


def test_the_selection_keeps_the_clips_whose_span_reaches_the_least_in_the_census_order():
    records = [dict(key="a", span_mm=cmd.CLIP_MIN_SPAN_MM), dict(key="b", span_mm=0.999),
               dict(key="c", span_mm=None, reason="no_log"), dict(key="d", span_mm=50.0)]
    assert cmd.selected(records) == ["a", "d"]


@pytest.mark.parametrize("text, refusal", [("\n", "lists no clip"), ("a/b/c\nd/e/f\na/b/c\n", r"\['a/b/c'\]")])
def test_a_population_file_with_no_clip_or_a_clip_twice_is_refused(tmp_path, text, refusal):
    (tmp_path / "clips.txt").write_text(text)
    with pytest.raises(ValueError, match=refusal):
        cmd.read_population(tmp_path / "clips.txt")


def test_a_clip_the_dataset_does_not_hold_is_refused(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_2"), make_clip(tmp_path, name="Clip_1")]
    monkeypatch.setattr(d4d, "clips", lambda root: clips)
    assert cmd.loader_clips("root", ["specimen_1/s_a/Clip_1", "specimen_1/s_a/Clip_2"]) == clips
    with pytest.raises(ValueError, match=r"holds no clip \['specimen_1/s_a/Clip_9'\]"):
        cmd.loader_clips("root", ["specimen_1/s_a/Clip_1", "specimen_1/s_a/Clip_9"])


def test_a_clip_is_cut_to_clip_frames_spread_over_it_and_a_short_one_keeps_every_frame(tmp_path):
    long = cmd.trajectory_clip(make_clip(tmp_path, name="Clip_1", n=300))
    assert len(long["frames"]) == cmd.CLIP_FRAMES and long["frames"][0] == 0 and long["frames"][-1] == 299
    assert long["duration_s"] == 149.5 and long["interval_s"] == round(149.5 / (cmd.CLIP_FRAMES - 1), 4)
    short = cmd.trajectory_clip(make_clip(tmp_path, name="Clip_2"))
    assert short["frames"] == list(range(N)) and short["name"] == "specimen_1__s_a__Clip_2"
    assert np.array_equal(short["times"], frame_times())


def test_prepare_writes_each_clip_s_frames_and_manifest(tmp_path):
    clip = cmd.trajectory_clip(make_clip(tmp_path))
    cmd.prepare([clip], tmp_path)
    out = tmp_path / clip["name"]
    assert sorted(p.name for p in out.iterdir()) == ["frame_manifest.json", "input_images"]
    assert len(list((out / "input_images").iterdir())) == N
    assert np.array_equal(cv2.imread(str(out / "input_images" / "000007.png")), cv2.imread(clip["images"][7]))
    manifest = json.loads((out / "frame_manifest.json").read_text())
    assert manifest == dict(dataset="d4d", clip_key="specimen_1/s_a/Clip_1", n_frames=N, interval_sec=0.5,
                            duration_sec=19.5, frames=[dict(seq_idx=i, native_frame=i) for i in range(N)])


@pytest.mark.parametrize("held", ["specimen_1__s_a__Clip_2", "specimen_1__s_a__Clip_2.partial"])
def test_prepare_refuses_a_clip_that_exists_before_anything_is_written(tmp_path, held):
    clips = [cmd.trajectory_clip(make_clip(tmp_path, name=n)) for n in ("Clip_1", "Clip_2")]
    (tmp_path / "out" / held).mkdir(parents=True)
    with pytest.raises(FileExistsError, match=r"1 clip\(s\) .* exist already, or as \.partial: "
                                              r"\['specimen_1__s_a__Clip_2'\]"):
        cmd.prepare(clips, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [held]


def test_prepare_refuses_an_image_it_cannot_read_and_leaves_the_clip_partial(tmp_path):
    clip = cmd.trajectory_clip(make_clip(tmp_path))
    Path(clip["images"][3]).write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not an image"):
        cmd.prepare([clip], tmp_path / "out")
    assert [p.name for p in (tmp_path / "out").iterdir()] == [f"{clip['name']}.partial"]


def test_prepare_refuses_an_image_it_cannot_write(tmp_path, monkeypatch):
    clip = cmd.trajectory_clip(make_clip(tmp_path))
    monkeypatch.setattr(cmd.cv2, "imwrite", lambda *a: False)
    with pytest.raises(ValueError, match=r"000000\.png: not written"):
        cmd.prepare([clip], tmp_path / "out")


def test_every_missing_bundle_is_refused_before_anything_is_scored(tmp_path, monkeypatch):
    clips = prepared(tmp_path, [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2")])
    (tmp_path / "clips" / clips[0]["name"] / cmd.METHOD_BUNDLES["pi3x"]).unlink()
    (tmp_path / "clips" / clips[1]["name"] / cmd.METHOD_BUNDLES["da3"]).unlink()
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(FileNotFoundError, match=r"2 bundle\(s\) missing under .*: "
                                                r"\['specimen_1__s_a__Clip_1:pi3x', 'specimen_1__s_a__Clip_2:da3'\]"):
        cmd.score_clips(clips, tmp_path / "clips", ["da3", "pi3x"])


@pytest.mark.parametrize("edit", [
    dict(clip_key="specimen_1/s_a/Clip_9"),
    dict(frames=[dict(seq_idx=i, native_frame=i + 1) for i in range(N)]),
])
def test_a_clip_whose_manifest_names_another_clip_or_other_frames_is_refused(tmp_path, monkeypatch, edit):
    clip = prepared(tmp_path, [make_clip(tmp_path)])[0]
    path = tmp_path / "clips" / clip["name"] / "frame_manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | edit))
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(ValueError, match=r"1 clip\(s\) .* hold another clip or other frames .*: "
                                         r"\['specimen_1__s_a__Clip_1'\]"):
        cmd.score_clips([clip], tmp_path / "clips", ["da3"])


def test_a_clip_without_its_manifest_is_refused_before_anything_is_scored(tmp_path, monkeypatch):
    clip = prepared(tmp_path, [make_clip(tmp_path)])[0]
    (tmp_path / "clips" / clip["name"] / "frame_manifest.json").unlink()
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(FileNotFoundError, match=r"specimen_1__s_a__Clip_1/frame_manifest\.json"):
        cmd.score_clips([clip], tmp_path / "clips", ["da3"])


def test_a_clip_whose_session_has_lost_its_tracker_log_is_refused(tmp_path):
    clip = prepared(tmp_path, [make_clip(tmp_path)])[0]
    (tmp_path / "specimen_1" / "s_a" / "tf").rename(tmp_path / "specimen_1" / "s_a" / "tf_moved")
    with pytest.raises(ValueError, match="specimen_1/s_a/Clip_1: the session has no tracker log"):
        cmd.score_clips([clip], tmp_path / "clips", ["da3"])


def test_a_clip_is_scored_under_the_floors_then_each_method_on_one_population(tmp_path):
    clip = prepared(tmp_path, [make_clip(tmp_path)])[0]
    rows = cmd.score_clips([clip], tmp_path / "clips", ["pi3x", "da3"])
    assert [r["cond"] for r in rows] == ["floor_static", "floor_line", "pi3x", "da3"]
    by = {r["cond"]: r for r in rows}
    assert by["floor_static"]["ate_rel"] == 1.0
    assert by["da3"]["ate_rel"] < 1e-6 and 0 < by["floor_line"]["ate_rel"] < 1
    assert {(r["clip"], r["seq"], r["stratum"], r["n_tracked"], r["tracked_frac"]) for r in rows} == {
        ("specimen_1__s_a__Clip_1", "specimen_1", "moving", N, 1.0)}
    assert {r["trajectory_code_sha"] for r in rows} == {trajectory_code_sha()}


def test_the_frames_the_tracker_misses_leave_the_truth_and_every_estimate(tmp_path, monkeypatch):
    # The log stops for 1.7 s over the clip's first four frames, so those frames are not tracked.
    log = np.arange(T0 - 1, T0 + 21, LOG_DT)
    log = log[(log < T0 - 0.1) | (log > T0 + 1.6)]
    clip = cmd.trajectory_clip(make_clip(tmp_path, log_times=log))
    (tmp_path / "clips").mkdir()
    cmd.prepare([clip], tmp_path / "clips")
    poses = truth(clip["times"])
    poses[:4, :3, 3] += 5.0
    write_bundles(tmp_path / "clips", clip, methods=("da3",), poses=poses)
    seen = []
    real = cmd.score
    monkeypatch.setattr(cmd, "score", lambda c, *a: seen.append(c["times"]) or real(c, *a))
    rows = cmd.score_clips([clip], tmp_path / "clips", ["da3"])
    assert rows[-1]["ate_rel"] < 1e-6 and rows[-1]["n_tracked"] == N - 4
    assert rows[-1]["tracked_frac"] == round((N - 4) / N, 3)
    # A row's times run from the clip's first frame, which is not tracked.
    assert np.allclose(seen[0], np.arange(4, N) * FRAME_DT)


def jump(T: np.ndarray, at: int, mm: float = 0.0, deg: float = 0.0) -> np.ndarray:
    """Return `T` with every pose from frame `at` on moved `mm` along x and turned `deg` about x."""
    T = T.copy()
    T[at:, 0, 3] += mm
    a = np.radians(deg)
    turn = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T[at:, :3, :3] = T[at:, :3, :3] @ turn
    return T


def test_a_clip_is_left_out_for_the_first_reason_that_holds():
    times = frame_times(100)
    T = truth(times)
    T[:, :3, 3] *= 1000
    clip = dict(times=times)
    every = np.ones(100, bool)
    assert cmd.leave_out(clip, T, every) == ""
    # Fewer than `MIN_TRACKED_FRAC` of the frames, though more than `MIN_TRACKED_FRAMES`.
    assert cmd.leave_out(clip, T, np.arange(100) < 59) == "few_tracked"
    assert cmd.leave_out(clip, T, np.arange(100) < 60) == ""
    # Fewer than `MIN_TRACKED_FRAMES`, though more than `MIN_TRACKED_FRAC`.
    assert cmd.leave_out(dict(times=times[:40]), T[:40], np.arange(40) < 29) == "few_tracked"
    # A step faster than a camera moves or turns, between two tracked frames 0.5 s apart.
    assert cmd.leave_out(clip, jump(T, 50, mm=160.0), every) == "gt_jump"
    assert cmd.leave_out(clip, jump(T, 50, mm=140.0), every) == ""
    assert cmd.leave_out(clip, jump(T, 50, deg=160.0), every) == "gt_jump"
    assert cmd.leave_out(clip, jump(T, 50, deg=140.0), every) == ""
    # A camera whose RMS radius is below `CLIP_MIN_SPAN_MM`: a segment of length L has the radius L / sqrt(12).
    still = np.tile(np.eye(4), (100, 1, 1))
    still[:, 0, 3] = np.linspace(0, 4.0, 100)
    assert cmd.leave_out(clip, still, every) == ""
    still[:, 0, 3] = np.linspace(0, 3.0, 100)
    assert cmd.leave_out(clip, still, every) == "gt_static"


def test_a_clip_left_out_is_printed_and_gives_no_row(tmp_path, capsys):
    clips = prepared(tmp_path, [make_clip(tmp_path, log=False)])
    # The tracker's camera turns but never moves.
    log_times = np.arange(T0 - 1, T0 + 21, LOG_DT)
    still = truth(log_times)
    still[:, :3, 3] = 0.0
    write_log(tmp_path / "specimen_1" / "s_a", log_times, still)
    assert cmd.score_clips(clips, tmp_path / "clips", ["da3"]) == []
    assert "left out specimen_1__s_a__Clip_1: gt_static (40 of 40 frames tracked)" in capsys.readouterr().out


def test_the_three_steps_write_the_census_the_clips_and_the_rows(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2", log=False),
             make_clip(tmp_path, session="s_b", name="Clip_1", log=False)]
    monkeypatch.setattr(d4d, "clips", lambda root: clips)
    (tmp_path / "clips.txt").write_text("specimen_1/s_a/Clip_1\nspecimen_1/s_b/Clip_1\n")
    census, root = tmp_path / "census.json", ["--root", "unused"]
    cmd.main(["census", *root, "--clips", str(tmp_path / "clips.txt"), "--out", str(census)])
    assert [(r["key"], r.get("reason")) for r in json.loads(census.read_text())] == [
        ("specimen_1/s_a/Clip_1", None), ("specimen_1/s_b/Clip_1", "no_log")]
    cmd.main(["prepare", *root, "--census", str(census), "--out", str(tmp_path / "clips")])
    assert [p.name for p in (tmp_path / "clips").iterdir()] == ["specimen_1__s_a__Clip_1"]
    write_bundles(tmp_path / "clips", cmd.trajectory_clip(clips[0]), methods=("da3",))
    cmd.main(["score", *root, "--census", str(census), "--clip-root", str(tmp_path / "clips"), "--methods", "da3",
              "--out", str(tmp_path / "rows.json")])
    assert [r["cond"] for r in json.loads((tmp_path / "rows.json").read_text())] == [
        "floor_static", "floor_line", "da3"]
    with pytest.raises(SystemExit, match="FileNotFoundError: 1 bundle"):
        cmd.main(["score", *root, "--census", str(census), "--clip-root", str(tmp_path / "clips"), "--methods",
                  "pi3x", "--out", str(tmp_path / "rows.json")])


def test_a_method_named_twice_is_refused(capsys):
    with pytest.raises(SystemExit):
        cmd.main(["score", "--root", "r", "--census", "c", "--clip-root", "k", "--methods", "da3", "pi3x", "da3",
                  "--out", "o"])
    assert "--methods names a method more than once" in capsys.readouterr().err
