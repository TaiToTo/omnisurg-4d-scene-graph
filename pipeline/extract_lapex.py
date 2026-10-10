"""Extract LapEx's annotated frames into the clip directories the pipeline reads, one clip per frame.

LapEx annotates single frames of each case's video, seconds apart. So each
clip holds one frame, the frame a mask was drawn on. The clip is named
`<case>__gt_<ms>`, after the frame's time in milliseconds, and holds
`input_images/000000.png`, `seg_masks/000000_class.png` and
`frame_manifest.json`. A mask stores each pixel's class as a grey level,
and the release's `metadata/segmented_entity.csv` names the levels. The
stage keeps every level but one: it moves interstitial space from 0 to 11.
A run that extracts every case of the release writes
`extraction_summary.json`, which counts the clips of each case. Any other
run removes the summary an earlier run wrote.

Usage:
    python -m pipeline.extract_lapex --lapex-root /path/to/LapEx_dataset --out /path/to/clips \\
        [--cases 01 02 ...] [--overwrite]
"""

import argparse
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The release's class table, grey level to class name, in the order of `metadata/segmented_entity.csv`. A release
# whose table differs is refused.
CLASS_TABLE = {
    236: "electrothermal_biforceps", 218: "stomach", 200: "liver", 163: "compress", 0: "interstitial_space",
    18: "adipose_tissue", 145: "liver_retractor", 182: "flat_grasper", 109: "abdominal_wall", 36: "diaphragm",
    127: "spleen",
}

# Interstitial space is anatomy, and LapEx leaves no pixel unlabelled. A tool that reads id 0 as background, as
# the workbench's evaluators did, would drop it. So 0 moves to 11, a level no class uses.
REMAP = {0: 11}

# The levels of the instruments and of the gauze, recorded in the manifest for a view without them.
INSTRUMENT_VALUES = (236, 145, 182)
GAUZE_VALUE = 163

# The release's frame rate. Its frames are named by their time in milliseconds, one every 40 ms.
FPS = 25
MS_PER_FRAME = 40

# The release's cases, one video each.
CASES = tuple(f"{i:02d}" for i in range(1, 31))

SEG_SUFFIX = "_seg.jpg"
SUMMARY = "extraction_summary.json"

# The files the stage writes into a clip. The depth stage and the stages after it write into the same directory.
OUTPUTS = ("input_images", "seg_masks", "frame_manifest.json")


@dataclass(frozen=True)
class Annotated:
    """One annotated frame of a case, read and checked.

    Attributes:
        ms: The frame's time in milliseconds, as the release spells it in the file names.
        seg: The mask's file.
        frame: The frame's file.
        image: The frame, BGR.
        levels: The mask's grey levels, with `REMAP` applied.
    """

    ms: str
    seg: Path
    frame: Path
    image: np.ndarray
    levels: np.ndarray


def load_class_table(lapex_root: Path) -> dict[int, str]:
    """Read the release's class table, refusing one that is not `CLASS_TABLE`.

    Raises:
        FileNotFoundError: the release has no `metadata/segmented_entity.csv`.
        ValueError: a line is not `<name>,<level>`, a level is listed twice, the table is not `CLASS_TABLE`, or
            `REMAP` moves a level onto a class's own level.
    """
    path = lapex_root / "metadata" / "segmented_entity.csv"
    table: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, level = line.rpartition(",")
        if not sep or not level.strip().isdigit():
            raise ValueError(f"{path}: {line!r} is not <name>,<grey level>")
        if int(level) in table:
            raise ValueError(f"{path}: level {int(level)} is listed twice")
        table[int(level)] = name
    if table != CLASS_TABLE:
        raise ValueError(f"{path} is not the class table the stage was written for:\n"
                         f"  release: {sorted(table.items())}\n  stage:   {sorted(CLASS_TABLE.items())}")
    taken = sorted(dst for dst in REMAP.values() if dst in table)
    if taken:
        raise ValueError(f"REMAP moves a level onto {taken}, which a class already uses")
    return table


def read_image(path: Path, flags: int) -> np.ndarray:
    """Read an image with OpenCV, refusing one that cannot be read."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def read_case(case_dir: Path) -> list[Annotated]:
    """Read and check every annotated frame of a case, in the order of the masks' names.

    Raises:
        FileNotFoundError: the case has no annotated frame, or a mask has no frame of its time.
        ValueError: a mask's name is not `<ms>_seg.jpg`, its time is not a whole frame at `FPS`, it is not its
            frame's size, or it holds a level the class table lacks.
        RuntimeError: a mask or a frame cannot be read.
    """
    seg_dir = case_dir / "seg"
    segs = sorted(p for p in seg_dir.glob(f"*{SEG_SUFFIX}")) if seg_dir.is_dir() else []
    if not segs:
        raise FileNotFoundError(f"{case_dir.name}: no *{SEG_SUFFIX} in {seg_dir}")
    out = []
    for seg in segs:
        # Find the frame of the mask's time; a time between two frames would give the wrong `native_frame`.
        ms = seg.name[: -len(SEG_SUFFIX)]
        if not re.fullmatch(r"\d+", ms):
            raise ValueError(f"{case_dir.name}: {seg.name} is not named <ms>{SEG_SUFFIX}")
        if int(ms) % MS_PER_FRAME:
            raise ValueError(f"{case_dir.name}: {seg.name} is at {int(ms)} ms, not a whole frame at {FPS} fps")
        frame = case_dir / "frames" / f"{ms}.jpg"
        if not frame.is_file():
            raise FileNotFoundError(f"{case_dir.name}: the mask at {ms} ms has no frame {frame}")

        # Read both, and refuse a size mismatch or a level the table lacks. The masks are JPEGs, so a level
        # the compression shifted would show up here.
        mask = read_image(seg, cv2.IMREAD_GRAYSCALE)
        image = read_image(frame, cv2.IMREAD_COLOR)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{case_dir.name}: the mask at {ms} ms is {mask.shape}, its frame {image.shape[:2]}")
        unknown = sorted(set(np.unique(mask).tolist()) - set(CLASS_TABLE))
        if unknown:
            raise ValueError(f"{case_dir.name}: the mask at {ms} ms holds levels the class table lacks: {unknown}")
        levels = mask.copy()
        for src, dst in REMAP.items():
            levels[mask == src] = dst
        out.append(Annotated(ms, seg, frame, image, levels))
    return out


def write_clip(clip_dir: Path, case: str, a: Annotated) -> None:
    """Write one clip's frame, mask and manifest into `clip_dir`, which must not exist.

    The manifest's keys and their order are the workbench's, so that a clip extracted here equals one extracted
    there, byte for byte.

    Raises:
        RuntimeError: OpenCV fails to write the frame or the mask; it reports a failure only by its return value.
    """
    (clip_dir / "input_images").mkdir(parents=True)
    (clip_dir / "seg_masks").mkdir()
    for path, img in ((clip_dir / "input_images" / "000000.png", a.image),
                      (clip_dir / "seg_masks" / "000000_class.png", a.levels)):
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"cannot write {path}")
    h, w = a.levels.shape
    manifest = dict(
        dataset="lapex", case=case, fps_native=FPS, src_size=[w, h], out_size=[w, h],
        clip_kind="design_A_seed_only", window=1,
        class_table={str(k): v for k, v in CLASS_TABLE.items()},
        remap={str(k): v for k, v in REMAP.items()},
        instrument_values=list(INSTRUMENT_VALUES), gauze_value=GAUZE_VALUE,
        frames=[dict(seq_idx=0, native_ms=int(a.ms), native_frame=int(a.ms) // MS_PER_FRAME, has_gt=True)],
        n_frames=1,
        source=dict(seg=str(a.seg), frame=str(a.frame)),
    )
    with open(clip_dir / "frame_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)


def extract_case(lapex_root: Path, case: str, out: Path, overwrite: bool = False) -> list[str]:
    """Extract every annotated frame of one case, and return the clips written.

    The stage writes and removes nothing until every frame of the case has been read and checked. Each clip is
    written under a temporary name and moved into place when it is complete.

    Raises:
        FileNotFoundError: the case has no directory or no annotated frame, or a mask has no frame.
        ValueError: `read_case` refuses a frame; a clip of the case already exists and `overwrite` is false; or
            a clip to be replaced holds a later stage's output.
        RuntimeError: a file cannot be read or written.
    """
    # Read and check every annotated frame of the case.
    case_dir = lapex_root / case
    if not case_dir.is_dir():
        raise FileNotFoundError(f"no case directory {case_dir}")
    frames = read_case(case_dir)
    names = [f"{case}__gt_{a.ms}" for a in frames]

    # Refuse clips an earlier run left, unless asked to replace them, and never replace a clip a later stage
    # wrote into: its depth and predictions would be removed with it.
    held = [out / n for n in names if (out / n).exists()]
    if held and not overwrite:
        raise ValueError(f"{case}: {len(held)} clip(s) exist, {held[0].name} first; pass --overwrite to replace them")
    for clip_dir in held:
        later = sorted(p.name for p in clip_dir.iterdir() if p.name not in OUTPUTS)
        if later:
            raise ValueError(f"{clip_dir} holds {', '.join(later)}, which a later stage wrote; remove {clip_dir} "
                             "by hand to extract it again")

    # Write each clip under a temporary name, then move it into place.
    for name, a in zip(names, frames):
        partial = out / f".{name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        try:
            write_clip(partial, case, a)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        if (out / name).exists():
            shutil.rmtree(out / name)
        partial.rename(out / name)
    return names


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lapex-root", required=True, help="The release's directory, holding metadata/ and the cases.")
    ap.add_argument("--out", required=True, help="The directory the clips are written to.")
    ap.add_argument("--cases", nargs="+", help="Cases to extract, such as 01. Default: all 30.")
    ap.add_argument("--overwrite", action="store_true", help="Replace the clips an earlier run left.")
    args = ap.parse_args(argv)

    # Check the class table, and find every case, before any frame is read.
    root, out = Path(args.lapex_root), Path(args.out)
    try:
        load_class_table(root)
    except (OSError, ValueError) as e:
        raise SystemExit(str(e)) from e
    cases = args.cases or list(CASES)
    missing = [c for c in cases if not (root / c).is_dir()]
    if missing:
        raise SystemExit(f"no case directory under {root} for: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)

    # Remove an earlier summary, which no longer counts the clips once this run replaces some.
    (out / SUMMARY).unlink(missing_ok=True)

    # Extract every case, even after one fails, and list the failures at the end.
    counts, failed = {}, []
    for case in cases:
        try:
            counts[case] = len(extract_case(root, case, out, args.overwrite))
            print(f"{case}: {counts[case]} clips")
        except Exception as e:
            print(f"{case}: failed: {type(e).__name__}: {e}")
            failed.append(case)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(cases)} case(s) failed: {', '.join(failed)}; no {SUMMARY} written")

    # Record the clips of each case, once every case of the release has been extracted.
    if sorted(cases) != sorted(CASES):
        print(f"{sum(counts.values())} clips -> {out}; no {SUMMARY} written: {len(cases)} of {len(CASES)} cases given")
        return
    summary = dict(cases=counts, n_clips=sum(counts.values()), remap={str(k): v for k, v in REMAP.items()},
                   design="A (seed = GT frame, 1 frame/clip)")
    with open(out / SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{summary['n_clips']} clips -> {out}")


if __name__ == "__main__":
    main()
