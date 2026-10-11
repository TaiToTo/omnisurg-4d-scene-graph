"""Test the StereoMIS scoring command with a stand-in reader and a stand-in ceiling.

The stand-in sequence `P1` has two usable clips and one that is not; `P2_0`
has none. Each clip's truth moves along a curve; the ceiling returns the
truth itself, so it scores 0. Each clip under the clip root holds the truth
as a depth stage's bundle, and a manifest of its frames.
"""

import json
import re
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import surgical_core.stereomis as stereomis  # noqa: E402
import trajectory_eval.tools.pose_controls as pose_controls  # noqa: E402
import trajectory_eval.tools.score_stereomis as cmd  # noqa: E402
from trajectory_eval.code_sha import trajectory_code_sha  # noqa: E402
from trajectory_eval.score import method_trajectory, score  # noqa: E402

N = 20


def make_clip(k: int, usable: bool = True) -> dict:
    frames = list(range(1000 * k, 1000 * k + 12 * N, 12))
    return dict(name=f"P1__clip_{k:04d}", seq="P1", frames=frames, times=[f / 60.0 for f in frames], stride=12,
                span_mm=10.0 + k, depth_ok=1.0, usable=usable, drop="" if usable else "gt_jump", stratum="moving")


CLIPS = {"P1": (make_clip(0), make_clip(1, usable=False), make_clip(2)), "P2_0": ()}


def gt_pose(root, seq, frames) -> np.ndarray:
    """Return poses that move along a curve and turn, a different curve for each clip."""
    t = (np.asarray(frames) - frames[0]) / 60.0 + frames[0] / 1000.0
    T = np.tile(np.eye(4), (len(frames), 1, 1))
    T[:, :3, 3] = np.stack([20 * np.sin(t), 10 * np.cos(t / 2), 3 * t], 1)
    a = 0.05 * t
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    return T


def ceiling(root, depth_root, seq) -> dict:
    return {c["name"]: (gt_pose(root, seq, c["frames"])[:, :3, 3], gt_pose(root, seq, c["frames"])[:, :3, :3], 3)
            for c in CLIPS[seq] if c["usable"]}


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    monkeypatch.setattr(stereomis, "clips", lambda root, depth_root, seq: CLIPS[seq])
    monkeypatch.setattr(stereomis, "gt_pose", gt_pose)
    monkeypatch.setattr(pose_controls, "run_sequence", ceiling)
    return tmp_path


def write_clips(clip_root, methods=("da3", "pi3x"), skip=(), painted=None) -> None:
    """Write each clip's manifest and bundles, whose extrinsics are the truth's, world to camera, in metres.

    `painted` is each manifest's `instruments_masked`; None leaves the key out.
    """
    for c in CLIPS["P1"]:
        manifest = dict(sequence="P1", frames=[dict(seq_idx=i, native_frame=f) for i, f in enumerate(c["frames"])])
        if painted is not None:
            manifest["instruments_masked"] = painted
        (clip_root / c["name"]).mkdir(parents=True, exist_ok=True)
        (clip_root / c["name"] / "frame_manifest.json").write_text(json.dumps(manifest))
        T = gt_pose(None, "P1", c["frames"])
        E = np.zeros((N, 3, 4), np.float32)
        E[:, :, :3] = np.transpose(T[:, :3, :3], (0, 2, 1))
        E[:, :, 3] = -np.einsum("nij,nj->ni", E[:, :, :3], T[:, :3, 3] / 1000.0)
        for m in methods:
            if (c["name"], m) in skip:
                continue
            bundle = clip_root / c["name"] / cmd.METHOD_BUNDLES[m]
            bundle.parent.mkdir(parents=True, exist_ok=True)
            np.savez(bundle, extrinsics=E)


def edit_manifest(clip_root, name: str, **fields) -> None:
    path = clip_root / name / "frame_manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | fields))


def refuse_scoring(monkeypatch) -> None:
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the clips were checked"))


def test_the_controls_score_each_usable_clip_in_order_and_the_ceiling_records_its_failures(stand_in):
    rows = cmd.score_controls(stand_in, stand_in, ["P1", "P2_0"])
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, cond) for c in ("P1__clip_0000", "P1__clip_0002") for cond in ("floor_static", "floor_line", "ceil_stereo")]
    by = {(r["clip"], r["cond"]): r for r in rows}
    assert by["P1__clip_0000", "floor_static"]["ate_rel"] == 1.0
    assert by["P1__clip_0000", "ceil_stereo"]["ate_rel"] == 0.0
    assert by["P1__clip_0002", "floor_line"]["ate_rel"] < 1.0
    assert [r.get("vo_fail") for r in rows[:3]] == [None, None, 3]
    assert {r["trajectory_code_sha"] for r in rows} == {trajectory_code_sha()}


def test_a_usable_clip_the_ceiling_returns_no_trajectory_for_is_refused(stand_in, monkeypatch):
    def first_only(root, depth_root, seq):
        return {"P1__clip_0000": ceiling(root, depth_root, seq)["P1__clip_0000"]}

    monkeypatch.setattr(pose_controls, "run_sequence", first_only)
    with pytest.raises(ValueError, match=r"P1: the ceiling returned no trajectory for \['P1__clip_0002'\]"):
        cmd.score_controls(stand_in, stand_in, ["P1"])


def test_the_methods_score_each_usable_clip_and_the_shuffled_poses_keep_the_shape_and_lose_the_order(stand_in):
    write_clips(stand_in)
    rows = cmd.score_methods(stand_in, stand_in, ["P1", "P2_0"], stand_in, ["da3", "pi3x"], shuffled=True)
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, m) for c in ("P1__clip_0000", "P1__clip_0002") for m in ("da3", "da3_shuf", "pi3x", "pi3x_shuf")]
    assert rows[0]["ate_rel"] < 1e-4 and rows[1]["ate_rel"] > 0.5
    # The shuffled row scores the bundle's poses in the order the workbench's seed, 0, gives.
    c = CLIPS["P1"][0]
    T = gt_pose(None, "P1", c["frames"])
    order = np.random.default_rng(0).permutation(N)
    est = method_trajectory(stand_in / c["name"] / cmd.METHOD_BUNDLES["da3"], N)
    assert rows[1] == score(c, "da3_shuf", T, est[0][order], est[1][order])


def test_masked_clips_name_their_conditions_masked_even_when_one_clip_paints_no_image(stand_in):
    write_clips(stand_in, painted=N)
    # A masked clip with no tissue mask near its frames paints no image.
    edit_manifest(stand_in, "P1__clip_0002", instruments_masked=0)
    rows = cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["pi3x"], shuffled=True, masked=True)
    assert [r["cond"] for r in rows] == ["pi3x_masked", "pi3x_masked_shuf"] * 2


@pytest.mark.parametrize("painted, masked, refusal", [
    (None, True, r"--masked is given, but no clip under .* has a painted image"),
    (0, True, r"--masked is given, but no clip under .* has a painted image"),
    (N, False, r"40 images under .* are painted outside their tissue mask; pass --masked"),
])
def test_painted_images_that_contradict_masked_are_refused_before_anything_is_scored(stand_in, monkeypatch, painted,
                                                                                    masked, refusal):
    write_clips(stand_in, painted=painted)
    refuse_scoring(monkeypatch)
    with pytest.raises(ValueError, match=refusal):
        cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["da3"], masked=masked)


def test_a_clip_cut_from_other_frames_is_refused_before_anything_is_scored(stand_in, monkeypatch):
    write_clips(stand_in)
    # A changed frame offset cuts the clip one frame later; the bundle still holds one pose per frame.
    later = [dict(seq_idx=i, native_frame=f + 1) for i, f in enumerate(CLIPS["P1"][2]["frames"])]
    edit_manifest(stand_in, "P1__clip_0002", frames=later)
    refuse_scoring(monkeypatch)
    with pytest.raises(ValueError, match=r"1 clip\(s\) under .* hold other frames than the ground truth is read at: "
                                         r"\['P1__clip_0002'\]"):
        cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["da3"])


def test_a_clip_without_its_manifest_is_refused_before_anything_is_scored(stand_in, monkeypatch):
    write_clips(stand_in)
    (stand_in / "P1__clip_0002" / "frame_manifest.json").unlink()
    refuse_scoring(monkeypatch)
    with pytest.raises(FileNotFoundError, match=r"P1__clip_0002/frame_manifest\.json"):
        cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["da3"])


def test_every_missing_bundle_is_refused_before_anything_is_scored(stand_in, monkeypatch):
    write_clips(stand_in, skip={("P1__clip_0000", "pi3x"), ("P1__clip_0002", "da3")})
    refuse_scoring(monkeypatch)
    with pytest.raises(FileNotFoundError, match=r"2 bundle\(s\) missing under .*: "
                                                r"\['P1__clip_0000:pi3x', 'P1__clip_0002:da3'\]"):
        cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["da3", "pi3x"])


def test_the_command_writes_the_rows_and_a_refused_run_leaves_them(stand_in, monkeypatch):
    write_clips(stand_in, methods=("da3",))
    out = stand_in / "out" / "methods.json"
    base = ["score_stereomis", "--root", str(stand_in), "--depth-root", str(stand_in), "--sequences", "P1",
            "--clip-root", str(stand_in), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base + ["--methods", "da3"])
    cmd.main()
    written = out.read_text()
    assert [r["cond"] for r in json.loads(written)] == ["da3", "da3"]
    monkeypatch.setattr(sys, "argv", base + ["--methods", "pi3x"])
    with pytest.raises(SystemExit, match="FileNotFoundError: 2 bundle"):
        cmd.main()
    assert out.read_text() == written


@pytest.mark.parametrize("args, refusal", [
    (["--methods", "da3"], "--methods needs --clip-root"),
    (["--controls", "--clip-root", "clips"], "--clip-root, --shuffled and --masked apply to --methods only"),
    (["--controls", "--shuffled"], "--clip-root, --shuffled and --masked apply to --methods only"),
    (["--controls", "--masked"], "--clip-root, --shuffled and --masked apply to --methods only"),
    (["--controls", "--sequences", "P1", "P2_0", "P1"], "--sequences names ['P1'] more than once"),
    (["--methods", "da3", "pi3x", "da3", "--clip-root", "clips"], "--methods names ['da3'] more than once"),
])
def test_options_that_do_not_go_together_are_refused_before_any_clip_is_read(stand_in, monkeypatch, capsys, args,
                                                                             refusal):
    out = stand_in / "out.json"
    monkeypatch.setattr(sys, "argv", ["score_stereomis", "--root", str(stand_in), "--depth-root", str(stand_in),
                                      "--out", str(out), *args])
    monkeypatch.setattr(stereomis, "clips", lambda *a: pytest.fail("read the clips before the options were checked"))
    with pytest.raises(SystemExit) as exit_:
        cmd.main()
    assert exit_.value.code == 2
    assert re.search(re.escape(refusal), capsys.readouterr().err)
    assert not out.exists()
