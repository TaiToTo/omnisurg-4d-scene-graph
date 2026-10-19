"""Test the D4D trajectory command on sessions made up for the test; D4D's loader is replaced by stand-in clips.

A made-up clip has `N` frames 0.5 s apart. Its session's tracker logs a pose every 0.05 s from a second
before the clip to a second after, along a curve that turns about z. A depth stage's bundle holds that truth.
Each refusal, each check of the clip list and each reason a clip is left out has a test that plants its fault.
The last test pins the committed clip list to its population and to the counts of `d4d_meta/README.md`.
"""

import hashlib
import json
from collections import Counter
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
from trajectory_eval.tools.scores import read_rows  # noqa: E402

META = Path(__file__).resolve().parent.parent / "d4d_meta"
N = 40
T0 = 100.0
FRAME_DT = 0.5
LOG_DT = 0.05


def truth(t: np.ndarray) -> np.ndarray:
    """Return camera-to-tracker poses at times `t`, in metres: a curve of about 15 mm radius that turns about z."""
    t = np.asarray(t, dtype=float)
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, 3] = np.stack([0.02 * np.sin(t / 3), 0.01 * np.cos(t / 5), 0.003 * (t - T0)], 1)
    a = 0.05 * t
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    return T


def frame_times(n: int = N) -> np.ndarray:
    return T0 + FRAME_DT * np.arange(n)


def default_log_times(n: int = N) -> np.ndarray:
    """Return the times of a log that runs from a second before a clip of `n` frames to a second after."""
    return np.arange(T0 - 1, frame_times(n)[-1] + 1 + LOG_DT / 2, LOG_DT)


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
        cv2.imwrite(paths[-1], np.full((8, 10, 3), i % 256, np.uint8))
    if log and not (session_dir / "tf").exists():
        write_log(session_dir, default_log_times(n) if log_times is None else log_times)
    return SimpleNamespace(name=name, path=session_dir / name, left_img_paths=paths,
                           session=SimpleNamespace(name=session, path=session_dir,
                                                   specimen=SimpleNamespace(name="specimen_1")))


def listed(clips: list, monkeypatch) -> list[dict]:
    """Return the clip list of `clips`, with the loader giving `clips`."""
    monkeypatch.setattr(d4d, "clips", lambda root: clips)
    return cmd.select([cmd.trajectory_clip(c) for c in clips])


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


def prepared(tmp_path, clips: list, monkeypatch) -> list[dict]:
    """Write the usable clips of `clips` under `tmp_path / "clips"` with both bundles, and return the clip list."""
    clip_list = listed(clips, monkeypatch)
    cmd.prepare(str(tmp_path), clip_list, tmp_path / "clips")
    for c in cmd.usable_clips(clip_list):
        write_bundles(tmp_path / "clips", c)
    return clip_list


def test_a_usable_clip_records_its_frames_times_and_tracked_frames(tmp_path, monkeypatch):
    entry = listed([make_clip(tmp_path)], monkeypatch)[0]
    centres = truth(frame_times())[:, :3, 3] * 1000
    assert entry.pop("span_mm") == pytest.approx(gt_span(centres), abs=1e-3)
    assert entry == dict(name="specimen_1__s_a__Clip_1", key="specimen_1/s_a/Clip_1", specimen="specimen_1",
                         usable=True, drop="", stratum="moving", n_tracked=N, frames=list(range(N)),
                         times=frame_times().tolist(), tracked=[True] * N)


def test_the_clip_list_reads_each_session_s_log_once(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2")]
    seen = []
    real = d4d.tracker_log
    monkeypatch.setattr(d4d, "tracker_log", lambda s: seen.append(s) or real(s))
    assert [e["usable"] for e in listed(clips, monkeypatch)] == [True, True]
    assert seen == [str(tmp_path / "specimen_1" / "s_a")]


@pytest.mark.parametrize("log, log_times, drop", [
    (False, None, "no_log"),
    (True, np.arange(T0 + 0.1, T0 + 21, LOG_DT), "log_not_covering"),
    (True, np.arange(T0 - 1, T0 + 19, LOG_DT), "log_not_covering"),
    (True, default_log_times()[::10], "few_tracked"),
])
def test_a_clip_the_tracker_cannot_score_records_why_and_no_frames(tmp_path, monkeypatch, log, log_times, drop):
    entry = listed([make_clip(tmp_path, log=log, log_times=log_times)], monkeypatch)[0]
    assert (entry["usable"], entry["drop"]) == (False, drop)
    assert entry["frames"] is entry["times"] is entry["tracked"] is None


def test_a_clip_whose_camera_turns_but_never_moves_is_static(tmp_path, monkeypatch):
    still = truth(default_log_times())
    still[:, :3, 3] = 0.0
    clip = make_clip(tmp_path, log=False)
    write_log(tmp_path / "specimen_1" / "s_a", default_log_times(), still)
    entry = listed([clip], monkeypatch)[0]
    assert (entry["drop"], entry["span_mm"], entry["stratum"], entry["n_tracked"]) == ("gt_static", 0.0, "slow", N)


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
    every = np.ones(100, bool)
    assert cmd.leave_out(times, T, every) == ""
    # Fewer than `MIN_TRACKED_FRAC` of the frames, though more than `MIN_TRACKED_FRAMES`.
    assert cmd.leave_out(times, T, np.arange(100) < 59) == "few_tracked"
    assert cmd.leave_out(times, T, np.arange(100) < 60) == ""
    # Fewer than `MIN_TRACKED_FRAMES`, though more than `MIN_TRACKED_FRAC`.
    assert cmd.leave_out(times[:40], T[:40], np.arange(40) < 29) == "few_tracked"
    # A step faster than a camera moves or turns, between two tracked frames 0.5 s apart.
    assert cmd.leave_out(times, jump(T, 50, mm=160.0), every) == "gt_jump"
    assert cmd.leave_out(times, jump(T, 50, mm=140.0), every) == ""
    assert cmd.leave_out(times, jump(T, 50, deg=160.0), every) == "gt_jump"
    assert cmd.leave_out(times, jump(T, 50, deg=140.0), every) == ""
    # A camera whose RMS radius is below `CLIP_MIN_SPAN_MM`: a segment of length L has the radius L / sqrt(12).
    still = np.tile(np.eye(4), (100, 1, 1))
    still[:, 0, 3] = np.linspace(0, 4.0, 100)
    assert cmd.leave_out(times, still, every) == ""
    still[:, 0, 3] = np.linspace(0, 3.0, 100)
    assert cmd.leave_out(times, still, every) == "gt_static"


def test_a_jump_is_a_rate_over_the_time_between_two_tracked_frames():
    times = frame_times(100)
    T = truth(times)
    T[:, :3, 3] *= 1000
    # 200 mm in 0.5 s is too fast, and in 1 s, across an untracked frame, it is not.
    assert cmd.leave_out(times, jump(T, 50, mm=200.0), np.ones(100, bool)) == "gt_jump"
    assert cmd.leave_out(times, jump(T, 50, mm=200.0), np.arange(100) != 49) == ""
    # 100 mm is not too fast between frames 0.5 s apart, and is between frames 0.25 s apart.
    assert cmd.leave_out(times, jump(T, 50, mm=100.0), np.ones(100, bool)) == ""
    assert cmd.leave_out(T0 + 0.25 * np.arange(100), jump(T, 50, mm=100.0), np.ones(100, bool)) == "gt_jump"


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
    short = cmd.trajectory_clip(make_clip(tmp_path, name="Clip_2"))
    assert short["frames"] == list(range(N)) and short["name"] == "specimen_1__s_a__Clip_2"
    assert np.array_equal(short["times"], frame_times())


def test_the_clip_list_is_written_one_clip_a_line_and_read_back(tmp_path, monkeypatch):
    clip_list = listed([make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, session="s_b", log=False)],
                       monkeypatch)
    cmd.write_clip_list(clip_list, tmp_path / "list.json")
    assert len((tmp_path / "list.json").read_text().splitlines()) == 4
    assert cmd.read_clip_list(tmp_path / "list.json") == clip_list


def edited(clip_list: list[dict], i: int, **edit) -> list[dict]:
    return [c | edit if k == i else c for k, c in enumerate(clip_list)]


@pytest.mark.parametrize("edit, refusal", [
    (lambda cl: {"a": 1}, "is not a list of clips"),
    (lambda cl: [{k: c[k] for k in reversed(cmd.CLIP_KEYS)} for c in cl], "keys .* in that order"),
    (lambda cl: [cl[0], cl[0]], "comes twice"),
    (lambda cl: edited(cl, 0, name="specimen_1__s_a__Clip_9"), "name or specimen is not its key's"),
    (lambda cl: edited(cl, 0, specimen="specimen_2"), "name or specimen is not its key's"),
    (lambda cl: edited(cl, 0, drop="gt_jump"), "usable True and the reason 'gt_jump'"),
    (lambda cl: edited(cl, 1, usable=True), "usable True and the reason 'no_log'"),
    (lambda cl: edited(cl, 1, drop="lost"), "a reason is one of"),
    (lambda cl: edited(cl, 0, times=cl[0]["times"][:-1]), "not lists of one length"),
    (lambda cl: edited(cl, 0, frames=list(range(cmd.CLIP_FRAMES + 1)), times=list(range(cmd.CLIP_FRAMES + 1)),
                       tracked=[True] * (cmd.CLIP_FRAMES + 1), n_tracked=cmd.CLIP_FRAMES + 1),
     "not lists of one length"),
    (lambda cl: edited(cl, 0, frames=cl[0]["frames"][::-1]), "frames or times do not increase"),
    (lambda cl: edited(cl, 0, times=[0.0] * N), "frames or times do not increase"),
    (lambda cl: edited(cl, 0, n_tracked=N - 1), f"n_tracked {N - 1}, but {N} tracked frames"),
    (lambda cl: edited(cl, 1, frames=[0], times=[1.0], tracked=[True]), "left out but records frames"),
])
def test_a_clip_list_the_scores_could_not_trust_is_refused(tmp_path, monkeypatch, edit, refusal):
    clip_list = listed([make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, session="s_b", log=False)],
                       monkeypatch)
    (tmp_path / "list.json").write_text(json.dumps(edit(clip_list)))
    with pytest.raises(ValueError, match=refusal):
        cmd.read_clip_list(tmp_path / "list.json")


def test_a_clip_list_the_reader_refuses_does_not_replace_the_old_one(tmp_path, monkeypatch):
    clip_list = listed([make_clip(tmp_path)], monkeypatch)
    (tmp_path / "list.json").write_text("old")
    with pytest.raises(ValueError, match="n_tracked"):
        cmd.write_clip_list(edited(clip_list, 0, n_tracked=0), tmp_path / "list.json")
    assert (tmp_path / "list.json").read_text() == "old"


def test_prepare_writes_each_usable_clip_s_frames_and_manifest(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, session="s_b", log=False)]
    clip_list = listed(clips, monkeypatch)
    cmd.prepare(str(tmp_path), clip_list, tmp_path / "out")
    out = tmp_path / "out" / "specimen_1__s_a__Clip_1"
    assert [p.name for p in (tmp_path / "out").iterdir()] == [out.name]
    assert sorted(p.name for p in out.iterdir()) == ["frame_manifest.json", "input_images"]
    assert len(list((out / "input_images").iterdir())) == N
    assert np.array_equal(cv2.imread(str(out / "input_images" / "000007.png")), cv2.imread(clips[0].left_img_paths[7]))
    manifest = json.loads((out / "frame_manifest.json").read_text())
    assert manifest == dict(dataset="d4d", clip_key="specimen_1/s_a/Clip_1", n_frames=N, interval_sec=0.5,
                            duration_sec=19.5, frames=[dict(seq_idx=i, native_frame=i) for i in range(N)])


@pytest.mark.parametrize("held", ["specimen_1__s_a__Clip_2", "specimen_1__s_a__Clip_2.partial"])
def test_prepare_refuses_a_clip_that_exists_before_anything_is_written(tmp_path, monkeypatch, held):
    clip_list = listed([make_clip(tmp_path, name=n) for n in ("Clip_1", "Clip_2")], monkeypatch)
    (tmp_path / "out" / held).mkdir(parents=True)
    with pytest.raises(FileExistsError, match=r"1 clip\(s\) .* exist already, or as \.partial: "
                                              r"\['specimen_1__s_a__Clip_2'\]"):
        cmd.prepare(str(tmp_path), clip_list, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [held]


def test_prepare_refuses_frame_times_the_dataset_does_not_give_before_anything_is_written(tmp_path, monkeypatch):
    clip_list = listed([make_clip(tmp_path)], monkeypatch)
    times = clip_list[0]["times"][:5] + [t + 1e-3 for t in clip_list[0]["times"][5:]]
    with pytest.raises(ValueError, match=r"other frame times than the clip list: \['specimen_1__s_a__Clip_1'\]"):
        cmd.prepare(str(tmp_path), edited(clip_list, 0, times=times), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_prepare_refuses_an_image_it_cannot_read_and_leaves_the_clip_partial(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)
    clip_list = listed([clip], monkeypatch)
    Path(clip.left_img_paths[3]).write_bytes(b"not a png")
    with pytest.raises(ValueError, match="not an image"):
        cmd.prepare(str(tmp_path), clip_list, tmp_path / "out")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["specimen_1__s_a__Clip_1.partial"]


def test_prepare_refuses_an_image_it_cannot_write(tmp_path, monkeypatch):
    clip_list = listed([make_clip(tmp_path)], monkeypatch)
    monkeypatch.setattr(cmd.cv2, "imwrite", lambda *a: False)
    with pytest.raises(ValueError, match=r"000000\.png: not written"):
        cmd.prepare(str(tmp_path), clip_list, tmp_path / "out")


def test_a_clip_list_with_no_usable_clip_is_refused_by_prepare_and_score(tmp_path, monkeypatch):
    clip_list = listed([make_clip(tmp_path, log=False)], monkeypatch)
    with pytest.raises(ValueError, match="marks no clip usable"):
        cmd.prepare(str(tmp_path), clip_list, tmp_path / "out")
    with pytest.raises(ValueError, match="marks no clip usable"):
        cmd.score_clips(str(tmp_path), clip_list, tmp_path / "out", ["da3"])


def test_every_missing_bundle_is_refused_before_anything_is_scored(tmp_path, monkeypatch):
    clip_list = prepared(tmp_path, [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2")],
                         monkeypatch)
    (tmp_path / "clips" / clip_list[0]["name"] / cmd.METHOD_BUNDLES["pi3x"]).unlink()
    (tmp_path / "clips" / clip_list[1]["name"] / cmd.METHOD_BUNDLES["da3"]).unlink()
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(FileNotFoundError, match=r"2 bundle\(s\) missing under .*: "
                                                r"\['specimen_1__s_a__Clip_1:pi3x', 'specimen_1__s_a__Clip_2:da3'\]"):
        cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["da3", "pi3x"])


@pytest.mark.parametrize("edit", [
    dict(clip_key="specimen_1/s_a/Clip_9"),
    dict(frames=[dict(seq_idx=i, native_frame=i + 1) for i in range(N)]),
])
def test_a_clip_whose_manifest_names_another_clip_or_other_frames_is_refused(tmp_path, monkeypatch, edit):
    clip_list = prepared(tmp_path, [make_clip(tmp_path)], monkeypatch)
    path = tmp_path / "clips" / clip_list[0]["name"] / "frame_manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | edit))
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(ValueError, match=r"1 clip\(s\) .* hold another clip or other frames .*: "
                                         r"\['specimen_1__s_a__Clip_1'\]"):
        cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["da3"])


def test_a_clip_without_its_manifest_is_refused_before_anything_is_scored(tmp_path, monkeypatch):
    clip_list = prepared(tmp_path, [make_clip(tmp_path)], monkeypatch)
    (tmp_path / "clips" / clip_list[0]["name"] / "frame_manifest.json").unlink()
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))
    with pytest.raises(FileNotFoundError, match=r"specimen_1__s_a__Clip_1/frame_manifest\.json"):
        cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["da3"])


def test_a_clip_whose_session_has_lost_its_tracker_log_is_refused(tmp_path, monkeypatch):
    clip_list = prepared(tmp_path, [make_clip(tmp_path)], monkeypatch)
    (tmp_path / "specimen_1" / "s_a" / "tf").rename(tmp_path / "specimen_1" / "s_a" / "tf_moved")
    with pytest.raises(ValueError, match="specimen_1/s_a/Clip_1: the session has no tracker log"):
        cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["da3"])


def test_a_log_that_marks_other_frames_tracked_than_the_clip_list_is_refused(tmp_path, monkeypatch):
    clip_list = prepared(tmp_path, [make_clip(tmp_path)], monkeypatch)
    tracked = [k != 5 for k in range(N)]
    with pytest.raises(ValueError, match="marks other frames tracked than the clip list"):
        cmd.score_clips(str(tmp_path), edited(clip_list, 0, tracked=tracked, n_tracked=N - 1), tmp_path / "clips",
                        ["da3"])


def test_a_clip_is_scored_under_the_floors_then_each_method(tmp_path, monkeypatch):
    clip_list = prepared(tmp_path, [make_clip(tmp_path)], monkeypatch)
    rows = cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["pi3x", "da3"])
    assert [r["cond"] for r in rows] == ["floor_static", "floor_line", "pi3x", "da3"]
    by = {r["cond"]: r for r in rows}
    assert by["floor_static"]["ate_rel"] == 1.0
    assert by["da3"]["ate_rel"] < 1e-6 and 0 < by["floor_line"]["ate_rel"] < 1
    assert {(r["clip"], r["seq"], r["stratum"], r["span_mm"], r["n_tracked"], r["tracked_frac"]) for r in rows} == {
        ("specimen_1__s_a__Clip_1", "specimen_1", "moving", clip_list[0]["span_mm"], N, 1.0)}
    assert {r["trajectory_code_sha"] for r in rows} == {trajectory_code_sha()}


def test_the_frames_the_tracker_misses_leave_the_truth_and_every_estimate(tmp_path, monkeypatch):
    # The log stops for 1.7 s over the clip's first four frames, so those frames are not tracked.
    log = default_log_times()
    clip = make_clip(tmp_path, log_times=log[(log < T0 - 0.1) | (log > T0 + 1.6)])
    clip_list = listed([clip], monkeypatch)
    assert clip_list[0]["tracked"] == [False] * 4 + [True] * (N - 4)
    cmd.prepare(str(tmp_path), clip_list, tmp_path / "clips")
    poses = truth(clip_list[0]["times"])
    poses[:4, :3, 3] += 5.0
    write_bundles(tmp_path / "clips", clip_list[0], methods=("da3",), poses=poses)
    seen = []
    real = cmd.score
    monkeypatch.setattr(cmd, "score", lambda c, *a: seen.append(c["times"]) or real(c, *a))
    rows = cmd.score_clips(str(tmp_path), clip_list, tmp_path / "clips", ["da3"])
    assert rows[-1]["ate_rel"] < 1e-6 and rows[-1]["n_tracked"] == N - 4
    assert rows[-1]["tracked_frac"] == round((N - 4) / N, 3)
    # A row's times run from the clip's first frame, which is not tracked.
    assert np.allclose(seen[0], np.arange(4, N) * FRAME_DT)


def test_the_three_steps_write_the_list_the_clips_and_rows_the_score_reader_reads(tmp_path, monkeypatch):
    clips = [make_clip(tmp_path, name="Clip_1"), make_clip(tmp_path, name="Clip_2", log=False),
             make_clip(tmp_path, session="s_b", name="Clip_1", log=False)]
    monkeypatch.setattr(d4d, "clips", lambda root: clips)
    (tmp_path / "clips.txt").write_text("specimen_1/s_a/Clip_1\nspecimen_1/s_b/Clip_1\n")
    clip_list, root = tmp_path / "trajectory_clips.json", ["--root", str(tmp_path)]
    cmd.main(["select", *root, "--population", str(tmp_path / "clips.txt"), "--out", str(clip_list)])
    assert [(c["key"], c["drop"]) for c in json.loads(clip_list.read_text())] == [
        ("specimen_1/s_a/Clip_1", ""), ("specimen_1/s_b/Clip_1", "no_log")]
    cmd.main(["prepare", *root, "--clips", str(clip_list), "--out", str(tmp_path / "clips")])
    assert [p.name for p in (tmp_path / "clips").iterdir()] == ["specimen_1__s_a__Clip_1"]
    write_bundles(tmp_path / "clips", json.loads(clip_list.read_text())[0], methods=("da3",))
    cmd.main(["score", *root, "--clips", str(clip_list), "--clip-root", str(tmp_path / "clips"), "--methods", "da3",
              "--out", str(tmp_path / "rows.json")])
    rows = read_rows([tmp_path / "rows.json"])
    assert [r["cond"] for r in rows] == ["floor_static", "floor_line", "da3"]
    assert {r["clip_list_sha"] for r in rows} == {hashlib.sha256(clip_list.read_bytes()).hexdigest()}
    with pytest.raises(SystemExit, match="FileNotFoundError: 1 bundle"):
        cmd.main(["score", *root, "--clips", str(clip_list), "--clip-root", str(tmp_path / "clips"), "--methods",
                  "pi3x", "--out", str(tmp_path / "rows.json")])


def test_a_method_named_twice_is_refused(capsys):
    with pytest.raises(SystemExit):
        cmd.main(["score", "--root", "r", "--clips", "c", "--clip-root", "k", "--methods", "da3", "pi3x", "da3",
                  "--out", "o"])
    assert "--methods names a method more than once" in capsys.readouterr().err


def test_the_committed_clip_list_holds_the_population_in_the_loader_s_order_with_the_readme_s_counts():
    clip_list = cmd.read_clip_list(META / "trajectory_clips.json")
    population = set((META / "clips.txt").read_text().split())
    assert [c["key"] for c in clip_list] == [k for k in (META / "census_clips.txt").read_text().split()
                                             if k in population]
    assert Counter(c["drop"] for c in clip_list) == {
        "": 9, "gt_static": 239, "few_tracked": 19, "log_not_covering": 3, "gt_jump": 1}
    usable = cmd.usable_clips(clip_list)
    assert len({c["specimen"] for c in usable}) == 4
    assert all(len(c["frames"]) == cmd.CLIP_FRAMES for c in usable)
    assert all(c["stratum"] == cmd.stratum_of(c["span_mm"]) for c in clip_list if c["span_mm"] is not None)
