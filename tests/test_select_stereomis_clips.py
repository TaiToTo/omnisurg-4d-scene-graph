"""Test the step that writes the StereoMIS clip list, with the reader's `clips` replaced by a stand-in."""

import json

import pytest

pytest.importorskip("scipy")

import pipeline.select_stereomis_clips as step  # noqa: E402
import surgical_core.stereomis as stereomis  # noqa: E402

N = stereomis.CLIP_FRAMES


def make_clip(seq: str, k: int, usable: bool = True) -> dict:
    frames = list(range(4 + 1344 * k, 4 + 1344 * k + 12 * N, 12))
    return dict(name=f"{seq}__clip_{k:04d}", seq=seq, usable=usable, drop="" if usable else "gt_static",
                stratum="slow", span_mm=2.5, depth_ok=1.0, stride=12, gt_row_offset=-3, frames=frames,
                times=[f / 60.0 for f in frames], masks=[f + 1 for f in frames])


def stand_in(seq: str) -> tuple[dict, ...]:
    return (make_clip(seq, 0), make_clip(seq, 1, usable=seq != "P2_3"))


def test_every_sequence_s_clips_are_written_in_order_and_read_back(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(stereomis, "clips", lambda root, depth_root, seq: stand_in(seq))
    out = tmp_path / "clips.json"
    step.main(["--root", "r", "--depth-root", "d", "--out", str(out)])
    assert stereomis.read_clip_list(out) == tuple(c for s in stereomis.SEQUENCES for c in stand_in(s))
    assert [p.name for p in tmp_path.iterdir()] == ["clips.json"]
    assert capsys.readouterr().out.splitlines()[-1] == f"19 usable of 20 clips -> {out}"


def test_a_list_the_reader_refuses_is_not_written_and_the_old_one_stays(monkeypatch, tmp_path):
    """The planted fault: a clip one frame short."""
    def short(root, depth_root, seq):
        c = make_clip(seq, 0)
        return (dict(c, frames=c["frames"][:-1], times=c["times"][:-1], masks=c["masks"][:-1]),)

    monkeypatch.setattr(stereomis, "clips", short)
    out = tmp_path / "clips.json"
    out.write_text("[]\n")
    with pytest.raises(SystemExit, match="has 111 frames"):
        step.main(["--root", "r", "--depth-root", "d", "--out", str(out)])
    assert json.loads(out.read_text()) == [] and [p.name for p in tmp_path.iterdir()] == ["clips.json"]
