"""Test the StereoMIS scoring command with a stand-in reader and a stand-in ceiling.

The stand-in sequence `P1` has two usable clips and one that is not; `P2_0`
has none. Each clip's truth moves along a curve; the ceiling returns the
truth itself, so it scores 0.
"""

import json
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import evalkit.tools.pose_controls as pose_controls  # noqa: E402
import surgical_core.stereomis as stereomis  # noqa: E402
import trajectory_eval.tools.score_stereomis as cmd  # noqa: E402
from trajectory_eval.code_sha import trajectory_code_sha  # noqa: E402
from trajectory_eval.score import METHOD_BUNDLES, score  # noqa: E402

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


def write_bundles(clip_root, methods=("da3", "pi3x"), skip=()) -> None:
    """Write each usable clip's bundles, whose extrinsics are the truth's, world to camera, in metres."""
    for c in CLIPS["P1"]:
        T = gt_pose(None, "P1", c["frames"])
        E = np.zeros((N, 3, 4), np.float32)
        E[:, :, :3] = np.transpose(T[:, :3, :3], (0, 2, 1))
        E[:, :, 3] = -np.einsum("nij,nj->ni", E[:, :, :3], T[:, :3, 3] / 1000.0)
        for m in methods:
            if (c["name"], m) in skip:
                continue
            (clip_root / c["name"] / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
            np.savez(clip_root / c["name"] / "exports" / "mini_npz" / METHOD_BUNDLES[m], extrinsics=E)


def test_the_controls_score_each_usable_clip_in_order_and_the_ceiling_records_its_failures(stand_in):
    rows = cmd.score_controls(stand_in, stand_in, ["P1", "P2_0"])
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, cond) for c in ("P1__clip_0000", "P1__clip_0002") for cond in cmd.CONTROLS]
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
    write_bundles(stand_in)
    rows = cmd.score_methods(stand_in, stand_in, ["P1", "P2_0"], stand_in, ["da3", "pi3x"], shuffled=True)
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, m) for c in ("P1__clip_0000", "P1__clip_0002") for m in ("da3", "da3_shuf", "pi3x", "pi3x_shuf")]
    assert rows[0]["ate_rel"] < 1e-4 and rows[1]["ate_rel"] > 0.5
    # The shuffled row scores the bundle's poses in the order of one fixed permutation.
    c = CLIPS["P1"][0]
    T = gt_pose(None, "P1", c["frames"])
    order = np.random.default_rng(cmd.SHUFFLE_SEED).permutation(N)
    est = cmd.method_trajectory(stand_in / c["name"], "da3", N)
    assert rows[1] == score(c, "da3_shuf", T, est[0][order], est[1][order])


def test_a_suffix_names_the_conditions_of_another_clip_root(stand_in):
    write_bundles(stand_in)
    rows = cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["pi3x"], suffix="_masked")
    assert [r["cond"] for r in rows] == ["pi3x_masked", "pi3x_masked"]


def test_every_missing_bundle_is_refused_before_anything_is_scored(stand_in, monkeypatch):
    write_bundles(stand_in, skip={("P1__clip_0000", "pi3x"), ("P1__clip_0002", "da3")})
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the bundles were checked"))
    with pytest.raises(FileNotFoundError, match=r"2 bundle\(s\) missing under .*: "
                                                r"\['P1__clip_0000:pi3x', 'P1__clip_0002:da3'\]"):
        cmd.score_methods(stand_in, stand_in, ["P1"], stand_in, ["da3", "pi3x"])


def test_the_command_writes_the_rows_and_exits_non_zero_on_a_refusal(stand_in, monkeypatch):
    write_bundles(stand_in, methods=("da3",))
    out = stand_in / "out" / "methods.json"
    base = ["score_stereomis", "--root", str(stand_in), "--depth-root", str(stand_in), "--sequences", "P1",
            "--clip-root", str(stand_in), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base + ["--methods", "da3"])
    cmd.main()
    assert [r["cond"] for r in json.loads(out.read_text())] == ["da3", "da3"]
    monkeypatch.setattr(sys, "argv", base + ["--methods", "pi3x"])
    with pytest.raises(SystemExit, match="FileNotFoundError: 2 bundle"):
        cmd.main()
    monkeypatch.setattr(sys, "argv", base[:7] + ["--out", str(out), "--methods", "da3"])
    with pytest.raises(SystemExit):
        cmd.main()
