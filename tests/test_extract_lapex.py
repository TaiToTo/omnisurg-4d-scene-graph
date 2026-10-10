"""Test the LapEx extraction on a release made up for the test.

The made-up release has the real class table and two cases. Its masks are
JPEGs, as the release's are, drawn in flat blocks aligned to JPEG's 8-pixel
grid, so that every grey level survives the compression. Each refusal has a
test that plants its fault.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import pipeline.extract_lapex as extract_lapex
from pipeline.extract_lapex import CLASS_TABLE, extract_case, load_class_table, main

SIZE = (48, 32)  # width, height
CSV_ORDER = list(CLASS_TABLE)

# Each case's annotated frames: the time in milliseconds, as spelled in the file name, and the levels of the
# mask's blocks. Case 02 spells its times with six digits, as some cases of the release do.
CASES = {
    "01": {"0000040": (0, 18, 218), "0145160": (236, 163, 200)},
    "02": {"038920": (127, 36, 109)},
}


def _mask(levels: tuple[int, ...]) -> np.ndarray:
    """A mask of vertical blocks 16 pixels wide, one level each, repeating."""
    m = np.empty((SIZE[1], SIZE[0]), np.uint8)
    for i in range(SIZE[0] // 16):
        m[:, i * 16:(i + 1) * 16] = levels[i % len(levels)]
    return m


def _frame(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8)


def _write_jpeg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def _write_csv(root: Path, table: dict[int, str]) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    lines = [f"{table[level]},{level}" for level in table]
    (root / "metadata" / "segmented_entity.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def release(tmp_path):
    """Write the made-up release and return its root."""
    root = tmp_path / "LapEx_dataset"
    _write_csv(root, {level: CLASS_TABLE[level] for level in CSV_ORDER})
    for n, (case, frames) in enumerate(CASES.items()):
        for i, (ms, levels) in enumerate(frames.items()):
            _write_jpeg(root / case / "frames" / f"{ms}.jpg", _frame(10 * n + i))
            _write_jpeg(root / case / "seg" / f"{ms}_seg.jpg", _mask(levels))
            # The masks must come back as written, or the tests below test JPEG, not the stage.
            back = cv2.imread(str(root / case / "seg" / f"{ms}_seg.jpg"), cv2.IMREAD_GRAYSCALE)
            assert np.array_equal(back, _mask(levels))
    return root


@pytest.fixture
def out(tmp_path):
    """The directory the clips are written to, beside the release."""
    d = tmp_path / "clips"
    d.mkdir()
    return d


def test_each_annotated_frame_becomes_a_clip_of_one_frame(release, out):
    assert extract_case(release, "01", out) == ["01__gt_0000040", "01__gt_0145160"]
    clip = out / "01__gt_0145160"
    assert sorted(p.name for p in clip.iterdir()) == ["frame_manifest.json", "input_images", "seg_masks"]
    assert [p.name for p in (clip / "input_images").iterdir()] == ["000000.png"]
    assert [p.name for p in (clip / "seg_masks").iterdir()] == ["000000_class.png"]
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert (m["dataset"], m["case"], m["n_frames"]) == ("lapex", "01", 1)
    assert m["src_size"] == m["out_size"] == list(SIZE)
    assert m["frames"] == [dict(seq_idx=0, native_ms=145160, native_frame=145160 // 40, has_gt=True)]
    assert m["source"] == {"seg": str(release / "01" / "seg" / "0145160_seg.jpg"),
                           "frame": str(release / "01" / "frames" / "0145160.jpg")}


def test_a_time_spelled_with_six_digits_names_its_clip_as_spelled(release, out):
    assert extract_case(release, "02", out) == ["02__gt_038920"]
    m = json.loads((out / "02__gt_038920" / "frame_manifest.json").read_text())
    assert m["frames"][0]["native_ms"] == 38920


def test_the_frame_is_written_as_it_is_read(release, out):
    extract_case(release, "01", out)
    written = cv2.imread(str(out / "01__gt_0000040" / "input_images" / "000000.png"))
    assert np.array_equal(written, cv2.imread(str(release / "01" / "frames" / "0000040.jpg")))


def test_interstitial_space_moves_to_11_and_every_other_level_stays(release, out):
    extract_case(release, "01", out)
    written = cv2.imread(str(out / "01__gt_0000040" / "seg_masks" / "000000_class.png"), cv2.IMREAD_UNCHANGED)
    want = _mask((0, 18, 218))
    want[want == 0] = 11
    assert written.dtype == np.uint8
    assert np.array_equal(written, want)


def test_the_release_table_is_read_in_its_order(release):
    assert list(load_class_table(release)) == CSV_ORDER


def test_a_class_table_that_is_not_the_stages_is_refused(release):
    _write_csv(release, {**CLASS_TABLE, 99: "something_new"})
    with pytest.raises(ValueError, match="not the class table"):
        load_class_table(release)


def test_a_class_table_line_without_a_level_is_refused(release):
    (release / "metadata" / "segmented_entity.csv").write_text("stomach\n")
    with pytest.raises(ValueError, match="is not <name>,<grey level>"):
        load_class_table(release)


def test_a_level_listed_twice_is_refused(release):
    path = release / "metadata" / "segmented_entity.csv"
    path.write_text(path.read_text() + "stomach,218\n")
    with pytest.raises(ValueError, match="listed twice"):
        load_class_table(release)


def test_a_remap_onto_a_class_is_refused(release, monkeypatch):
    monkeypatch.setattr(extract_lapex, "REMAP", {0: 18})
    with pytest.raises(ValueError, match="which a class already uses"):
        load_class_table(release)


def test_a_release_without_a_class_table_is_refused(release):
    (release / "metadata" / "segmented_entity.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_class_table(release)


def test_a_case_without_masks_is_refused(release, out):
    for p in (release / "02" / "seg").iterdir():
        p.unlink()
    with pytest.raises(FileNotFoundError, match="no \\*_seg.jpg"):
        extract_case(release, "02", out)


def test_a_missing_case_is_refused(release, out):
    with pytest.raises(FileNotFoundError, match="no case directory"):
        extract_case(release, "03", out)


def test_a_mask_without_its_frame_is_refused(release, out):
    (release / "01" / "frames" / "0145160.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="has no frame"):
        extract_case(release, "01", out)


def test_a_mask_name_that_is_not_a_time_is_refused(release, out):
    _write_jpeg(release / "01" / "seg" / "a0040_seg.jpg", _mask((18,)))
    with pytest.raises(ValueError, match="is not named <ms>_seg.jpg"):
        extract_case(release, "01", out)


def test_a_time_between_two_frames_is_refused(release, out):
    _write_jpeg(release / "01" / "frames" / "0000050.jpg", _frame(99))
    _write_jpeg(release / "01" / "seg" / "0000050_seg.jpg", _mask((18,)))
    with pytest.raises(ValueError, match="not a whole frame"):
        extract_case(release, "01", out)


def test_a_mask_of_another_size_is_refused(release, out):
    _write_jpeg(release / "01" / "seg" / "0145160_seg.jpg", np.full((16, 16), 18, np.uint8))
    with pytest.raises(ValueError, match="its frame"):
        extract_case(release, "01", out)


def test_a_level_the_table_lacks_is_refused(release, out):
    _write_jpeg(release / "01" / "seg" / "0145160_seg.jpg", _mask((18, 64)))
    with pytest.raises(ValueError, match="levels the class table lacks: \\[64\\]"):
        extract_case(release, "01", out)


def test_a_file_that_cannot_be_read_is_refused(release, out):
    (release / "01" / "seg" / "0145160_seg.jpg").write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="cannot read"):
        extract_case(release, "01", out)


def test_a_refused_case_writes_nothing(release, out):
    # The second frame is refused; the first, already checked, must not have been written.
    _write_jpeg(release / "01" / "seg" / "0145160_seg.jpg", _mask((18, 64)))
    with pytest.raises(ValueError):
        extract_case(release, "01", out)
    assert list(out.iterdir()) == []


def test_an_earlier_clip_is_refused_unless_replaced(release, out):
    extract_case(release, "01", out)
    stale = out / "01__gt_0000040" / "input_images" / "000001.png"
    stale.write_bytes(b"left by an earlier run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        extract_case(release, "01", out)
    assert stale.exists()
    extract_case(release, "01", out, overwrite=True)
    assert not stale.exists()


def test_a_clip_a_later_stage_wrote_into_is_not_replaced(release, out):
    extract_case(release, "01", out)
    (out / "01__gt_0145160" / "depth_raw").mkdir()
    with pytest.raises(ValueError, match="depth_raw, which a later stage wrote"):
        extract_case(release, "01", out, overwrite=True)
    assert (out / "01__gt_0145160" / "depth_raw").is_dir()


def test_a_write_that_fails_leaves_no_partial_clip_and_the_earlier_clip_in_place(release, out, monkeypatch):
    extract_case(release, "01", out)
    before = (out / "01__gt_0000040" / "frame_manifest.json").read_bytes()
    monkeypatch.setattr(extract_lapex.cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="cannot write"):
        extract_case(release, "01", out, overwrite=True)
    assert sorted(p.name for p in out.iterdir()) == ["01__gt_0000040", "01__gt_0145160"]
    assert (out / "01__gt_0000040" / "frame_manifest.json").read_bytes() == before


def test_the_command_extracts_every_case_and_counts_its_clips(release, tmp_path, monkeypatch):
    monkeypatch.setattr(extract_lapex, "CASES", ("01", "02"))
    out = tmp_path / "clips"
    main(["--lapex-root", str(release), "--out", str(out)])
    summary = json.loads((out / "extraction_summary.json").read_text())
    assert summary["cases"] == {"01": 2, "02": 1}
    assert summary["n_clips"] == 3
    assert summary["remap"] == {"0": 11}


def test_a_run_over_some_cases_removes_the_summary_of_the_whole_release(release, tmp_path, monkeypatch):
    monkeypatch.setattr(extract_lapex, "CASES", ("01", "02"))
    out = tmp_path / "clips"
    main(["--lapex-root", str(release), "--out", str(out)])
    main(["--lapex-root", str(release), "--out", str(out), "--cases", "02", "--overwrite"])
    assert not (out / "extraction_summary.json").exists()
    assert sorted(p.name for p in out.iterdir()) == ["01__gt_0000040", "01__gt_0145160", "02__gt_038920"]


def test_the_command_refuses_a_missing_case_before_any_is_extracted(release, tmp_path):
    out = tmp_path / "clips"
    with pytest.raises(SystemExit, match="no case directory .* for: 03"):
        main(["--lapex-root", str(release), "--out", str(out), "--cases", "01", "03"])
    assert not out.exists()


def test_the_command_refuses_a_release_whose_table_differs(release, tmp_path):
    _write_csv(release, {**CLASS_TABLE, 99: "something_new"})
    with pytest.raises(SystemExit, match="not the class table"):
        main(["--lapex-root", str(release), "--out", str(tmp_path / "clips")])


def test_the_command_extracts_the_other_cases_after_one_fails_and_writes_no_summary(release, tmp_path):
    _write_jpeg(release / "01" / "seg" / "0145160_seg.jpg", _mask((18, 64)))
    out = tmp_path / "clips"
    with pytest.raises(SystemExit, match="1 of 2 case\\(s\\) failed: 01"):
        main(["--lapex-root", str(release), "--out", str(out), "--cases", "01", "02"])
    assert sorted(p.name for p in out.iterdir()) == ["02__gt_038920"]


def test_a_run_in_which_a_case_fails_removes_the_earlier_summary(release, tmp_path, monkeypatch):
    monkeypatch.setattr(extract_lapex, "CASES", ("01", "02"))
    out = tmp_path / "clips"
    main(["--lapex-root", str(release), "--out", str(out)])
    _write_jpeg(release / "01" / "seg" / "0145160_seg.jpg", _mask((18, 64)))
    with pytest.raises(SystemExit, match="1 of 2 case\\(s\\) failed: 01"):
        main(["--lapex-root", str(release), "--out", str(out), "--overwrite"])
    assert not (out / "extraction_summary.json").exists()
