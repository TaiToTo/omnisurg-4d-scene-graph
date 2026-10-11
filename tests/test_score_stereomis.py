"""Test the StereoMIS scoring command with a stand-in clip list, ground truth and ceiling.

The stand-in sequence `P1` has two usable clips and one that is not; `P2_0`
has none. Each clip's truth moves along a curve; the ceiling returns the
truth itself, so it scores 0.
"""

import hashlib
import json
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import surgical_core.stereomis as stereomis  # noqa: E402
import trajectory_eval.pose_controls as pose_controls  # noqa: E402
import trajectory_eval.tools.score_stereomis as cmd  # noqa: E402
from trajectory_eval.code_sha import trajectory_code_sha  # noqa: E402
from trajectory_eval.score import method_trajectory, score  # noqa: E402

N = stereomis.CLIP_FRAMES


def make_clip(k: int, usable: bool = True) -> dict:
    frames = list(range(1000 * k + 4, 1000 * k + 4 + 12 * N, 12))
    return dict(name=f"P1__clip_{k:04d}", seq="P1", usable=usable, drop="" if usable else "gt_jump",
                stratum="moving", span_mm=10.0 + k, depth_ok=1.0, stride=12, gt_row_offset=-3, frames=frames,
                times=[f * 0.0181 for f in frames], masks=[None] * N)


CLIP_LIST = (make_clip(0), make_clip(1, usable=False), make_clip(2))


def gt_pose(root, seq, rows) -> np.ndarray:
    """Return poses that move along a curve and turn, a different curve for each clip."""
    t = (np.asarray(rows) - rows[0]) / 60.0 + rows[0] / 1000.0
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, 3] = np.stack([20 * np.sin(t), 10 * np.cos(t / 2), 3 * t], 1)
    a = 0.05 * t
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    return T


def truth(c: dict) -> np.ndarray:
    return gt_pose(None, c["seq"], stereomis.gt_rows(c))


def ceiling(root, clips, seq) -> dict:
    return {c["name"]: (truth(c)[:, :3, 3], truth(c)[:, :3, :3], 3) for c in clips}


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    monkeypatch.setattr(stereomis, "gt_pose", gt_pose)
    monkeypatch.setattr(cmd, "ceiling", ceiling)
    return tmp_path


def write_bundles(clip_root, methods=("da3", "pi3x"), skip=()) -> None:
    """Write each usable clip's bundles, whose extrinsics are the truth's, world to camera, in metres."""
    for c in CLIP_LIST:
        T = truth(c)
        E = np.zeros((N, 3, 4), np.float32)
        E[:, :, :3] = np.transpose(T[:, :3, :3], (0, 2, 1))
        E[:, :, 3] = -np.einsum("nij,nj->ni", E[:, :, :3], T[:, :3, 3] / 1000.0)
        for m in methods:
            if (c["name"], m) in skip:
                continue
            bundle = clip_root / c["name"] / cmd.METHOD_BUNDLES[m]
            bundle.parent.mkdir(parents=True, exist_ok=True)
            np.savez(bundle, extrinsics=E)


def test_the_controls_score_each_usable_clip_in_order_and_the_ceiling_records_its_failures(stand_in):
    rows = cmd.score_controls(stand_in, CLIP_LIST, ["P1", "P2_0"])
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, cond) for c in ("P1__clip_0000", "P1__clip_0002") for cond in cmd.CONTROLS]
    by = {(r["clip"], r["cond"]): r for r in rows}
    assert by["P1__clip_0000", "floor_static"]["ate_rel"] == 1.0
    assert by["P1__clip_0000", "ceil_stereo"]["ate_rel"] == 0.0
    assert by["P1__clip_0002", "floor_line"]["ate_rel"] < 1.0
    assert [r.get("vo_fail") for r in rows[:3]] == [None, None, 3]
    assert {r["trajectory_code_sha"] for r in rows} == {trajectory_code_sha()}


def test_the_rows_and_times_are_the_clip_list_s(stand_in, monkeypatch):
    """The planted fault: rows and times the tables and the frame rate would not give."""
    asked = []
    monkeypatch.setattr(stereomis, "gt_pose", lambda root, seq, rows: asked.append(list(rows)) or gt_pose(
        root, seq, rows))
    clip = dict(CLIP_LIST[0], gt_row_offset=-9)
    rows = cmd.score_controls(stand_in, [clip], ["P1"])
    assert asked == [[f - 9 for f in clip["frames"]]]
    T = gt_pose(None, "P1", asked[0])
    line = pose_controls.line_floor(T[:, :3, 3], clip["times"])
    assert rows[1] == score(clip, "floor_line", T, *line)


def test_a_usable_clip_the_ceiling_returns_no_trajectory_for_is_refused(stand_in, monkeypatch):
    def first_only(root, clips, seq):
        return {"P1__clip_0000": ceiling(root, clips, seq)["P1__clip_0000"]}

    monkeypatch.setattr(cmd, "ceiling", first_only)
    with pytest.raises(ValueError, match=r"P1: the ceiling returned no trajectory for \['P1__clip_0002'\]"):
        cmd.score_controls(stand_in, CLIP_LIST, ["P1"])


def test_the_methods_score_each_usable_clip_and_the_shuffled_poses_keep_the_shape_and_lose_the_order(stand_in):
    write_bundles(stand_in)
    rows = cmd.score_methods(stand_in, CLIP_LIST, ["P1", "P2_0"], stand_in, ["da3", "pi3x"], shuffled=True)
    assert [(r["clip"], r["cond"]) for r in rows] == [
        (c, m) for c in ("P1__clip_0000", "P1__clip_0002") for m in ("da3", "da3_shuf", "pi3x", "pi3x_shuf")]
    assert rows[0]["ate_rel"] < 1e-4 and rows[1]["ate_rel"] > 0.5
    # The shuffled row scores the bundle's poses in the order of the controls' fixed shuffle.
    c = CLIP_LIST[0]
    est = method_trajectory(stand_in / c["name"] / cmd.METHOD_BUNDLES["da3"], N)
    assert rows[1] == score(c, "da3_shuf", truth(c), *pose_controls.shuffled(*est))


def test_a_suffix_names_the_conditions_of_another_clip_root(stand_in):
    write_bundles(stand_in)
    rows = cmd.score_methods(stand_in, CLIP_LIST, ["P1"], stand_in, ["pi3x"], suffix="_masked")
    assert [r["cond"] for r in rows] == ["pi3x_masked", "pi3x_masked"]


def test_every_missing_bundle_is_refused_before_anything_is_scored(stand_in, monkeypatch):
    write_bundles(stand_in, skip={("P1__clip_0000", "pi3x"), ("P1__clip_0002", "da3")})
    monkeypatch.setattr(cmd, "score", lambda *a: pytest.fail("scored before the bundles were checked"))
    with pytest.raises(FileNotFoundError, match=r"2 bundle\(s\) missing under .*: "
                                                r"\['P1__clip_0000:pi3x', 'P1__clip_0002:da3'\]"):
        cmd.score_methods(stand_in, CLIP_LIST, ["P1"], stand_in, ["da3", "pi3x"])


def test_the_command_reads_the_clip_list_records_its_hash_and_exits_non_zero_on_a_refusal(stand_in, monkeypatch):
    write_bundles(stand_in, methods=("da3",))
    clip_list = stand_in / "clips.json"
    stereomis.write_clip_list(CLIP_LIST, clip_list)
    out = stand_in / "out" / "methods.json"
    base = ["score_stereomis", "--root", str(stand_in), "--clips", str(clip_list), "--sequences", "P1",
            "--clip-root", str(stand_in), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base + ["--methods", "da3"])
    cmd.main()
    rows = json.loads(out.read_text())
    assert [r["cond"] for r in rows] == ["da3", "da3"]
    assert {r["clip_list_sha"] for r in rows} == {hashlib.sha256(clip_list.read_bytes()).hexdigest()}
    monkeypatch.setattr(sys, "argv", base + ["--methods", "pi3x"])
    with pytest.raises(SystemExit, match="FileNotFoundError: 2 bundle"):
        cmd.main()
    monkeypatch.setattr(sys, "argv", base[:7] + ["--out", str(out), "--methods", "da3"])
    with pytest.raises(SystemExit):
        cmd.main()


def test_the_command_refuses_a_clip_list_the_reader_refuses(stand_in, monkeypatch):
    clip_list = stand_in / "clips.json"
    stereomis.write_clip_list([dict(CLIP_LIST[0], usable=False)], clip_list)
    monkeypatch.setattr(sys, "argv", ["score_stereomis", "--root", str(stand_in), "--clips", str(clip_list),
                                      "--controls", "--out", str(stand_in / "controls.json")])
    with pytest.raises(SystemExit, match="ValueError: .* records usable False and the reason ''"):
        cmd.main()
    assert not (stand_in / "controls.json").exists()
