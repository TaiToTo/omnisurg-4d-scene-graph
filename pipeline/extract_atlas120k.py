"""Extract the annotated clips of ATLAS-120k videos into the clip directories the pipeline reads.

Each clip of a video's `clip_index.json` is split into runs of consecutive
annotated frames, and each run is thinned to one frame every `--step-sec`
seconds. A run left with at least `--min-frames` frames is written as
`<procedure>__<video>__gt_<n>`, holding `input_images/`, `seg_masks/` and
`frame_manifest.json`. The frames are the release's JPEGs, cut to the clip's
confirmed rectangle and resized to `--long-side`; the masks hold the class id
of each pixel. A run with the frames and rectangle of one already written is
not written again. Each video gets `<procedure>__<video>__extract_report.json`,
which says what became of every clip of its index. With `--population`, the
clips written must be the population's clips of those videos.

Usage:
    python -m pipeline.extract_atlas120k --atlas-root /path/to/ATLAS --out /path/to/clips \\
        --clip-rects atlas120k_meta/crop_rects.json --frame-ratios atlas120k_meta/frame_ratio.json \\
        [--videos <procedure>/<video> ...] [--population atlas120k_meta/clips.txt] [--overwrite]
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evalkit.classes import ClassTable, load_table
from surgical_core.atlas120k.clip_rects import Rect, load_clip_rects
from surgical_core.atlas120k.frame_ratio import MIN_NATIVE_FRAME, FrameRatios

SPLITS = ("train", "val", "test")

# The masks of a clip are under `masks/`; the release's README calls the directory `machine_masks/`.
MASK_DIRS = ("masks", "machine_masks")


def find_video(atlas_root: Path, procedure: str, video: str) -> tuple[str, Path, Path]:
    """Return the video's split, its directory in the release and its mp4.

    Raises:
        FileNotFoundError: the release has no such video in any split, or its mp4 is missing.
    """
    for split in SPLITS:
        gt_dir = atlas_root / "atlas120k" / split / procedure / video
        if gt_dir.is_dir():
            break
    else:
        raise FileNotFoundError(f"no atlas120k/{{{','.join(SPLITS)}}}/{procedure}/{video} under {atlas_root}")
    mp4 = atlas_root / "raw_data" / procedure / f"{video}.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(f"no mp4 at {mp4}")
    return split, gt_dir, mp4


def read_mp4(path: Path) -> tuple[float, tuple[int, int]]:
    """Return the mp4's frame rate and its frame size as (width, height).

    Raises:
        RuntimeError: the mp4 cannot be opened or reports no frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if not fps > 0:
        raise RuntimeError(f"{path} reports no frame rate")
    return fps, size


def stride_for(step_sec: float, fps: float, ratio: int) -> int:
    """Return the stride, in the clip index's frame numbers, nearest to one step of `step_sec` seconds.

    The clip index numbers frames at `fps / ratio`. Halves round up: Python's `round` rounds them to the even
    side, which would shift the stride by one at a step that falls on a half.
    """
    return max(1, int(math.floor(step_sec * fps / ratio + 0.5)))


def contiguous_runs(frames: list[int]) -> list[list[int]]:
    """Split ascending frame numbers into the runs whose numbers step by one.

    A clip of the index is not always continuous in time: seven clips skip frames, one of them by 2,266 frames
    (76 s). Thinning across the gap would put two frames 76 s apart next to each other.
    """
    runs = [[frames[0]]]
    for a, b in zip(frames, frames[1:]):
        if b - a == 1:
            runs[-1].append(b)
        else:
            runs.append([b])
    return runs


def clip_number(name: str) -> int:
    """Return the number in a clip's name, `clip_0051` -> 51.

    The output is named by this number, not by the clip's position in the index: the index skips numbers, and a
    name by position would not lead back to the clip it came from.

    Raises:
        ValueError: the name is not `clip_<n>`.
    """
    m = re.fullmatch(r"clip_(\d+)", name)
    if not m:
        raise ValueError(f"unexpected clip name {name!r}; expected clip_<n>")
    return int(m.group(1))


def target_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Return the (width, height) whose longer side is `long_side`. A smaller frame keeps its size."""
    if long_side <= 0 or max(w, h) <= long_side:
        return w, h
    scale = long_side / float(max(w, h))
    return max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)


def check_size(arr: np.ndarray, src_size: tuple[int, int], where: str) -> None:
    """Refuse an image whose size is not the mp4's.

    The rectangles are in the mp4's pixels. On a smaller image numpy shortens a slice without an error, and the
    frame and its mask would come from different regions.

    Raises:
        ValueError: the sizes differ.
    """
    if (arr.shape[1], arr.shape[0]) != tuple(src_size):
        raise ValueError(f"{where} is {arr.shape[1]}x{arr.shape[0]}, not the mp4's {src_size[0]}x{src_size[1]}")


def write_frame(dst: Path, jpg: Path, rect: Rect, size: tuple[int, int], src_size: tuple[int, int]) -> None:
    """Cut a frame to the rectangle, resize it with area averaging and write it as PNG."""
    bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {jpg}")
    check_size(bgr, src_size, str(jpg))
    x, y, w, h = rect
    img = bgr[y:y + h, x:x + w]
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)


def write_mask(dst: Path, png: Path, rect: Rect, size: tuple[int, int], src_size: tuple[int, int],
               table: ClassTable) -> None:
    """Read a mask as class ids, cut it to the rectangle, resize it with the nearest neighbour and write it.

    The release stores most masks as ids and some as colours, both within one video; the class table reads
    either, as the evaluator reads them. Nearest neighbour keeps every pixel a class the mask holds.
    """
    ids = table.mask_ids(Image.open(png)).astype(np.uint8)
    check_size(ids, src_size, str(png))
    x, y, w, h = rect
    ids = ids[y:y + h, x:x + w]
    if (ids.shape[1], ids.shape[0]) != size:
        ids = cv2.resize(ids, size, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(ids).save(dst)


def verify_ratio(ratios: FrameRatios, procedure: str, video: str, mp4: Path, gt_dir: Path,
                 clips: dict, digits: int) -> None:
    """Check the video's frame ratio against the pixels of one of its JPEGs.

    The ratio sets the stride, so a wrong one changes the step two to four times over without an error. The
    check reads the first frame numbered `MIN_NATIVE_FRAME` or later that has a JPEG, in clip order. An earlier
    frame cannot tell the ratios apart.

    Raises:
        RuntimeError: no frame of the video can be checked, or `verify_against_bundled` refuses the ratio.
    """
    for name in sorted(clips):
        for f in sorted(int(x) for x in clips[name]):
            jpg = gt_dir / name / "images" / f"frame_{f:0{digits}d}.jpg"
            if f >= MIN_NATIVE_FRAME and jpg.is_file():
                d = ratios.verify_against_bundled(str(mp4), str(jpg), procedure, video, f)
                print(f"  frame ratio {ratios.ratio(procedure, video)} holds on clip-index frame {f} "
                      f"(difference {d:.2f})")
                return
    raise RuntimeError(f"{procedure}/{video}: no JPEG of frame {MIN_NATIVE_FRAME} or later to check the frame "
                       "ratio on")


def existing_output(out: Path, procedure: str, video: str) -> list[Path]:
    """Return the clips and the report an earlier extraction of the video left in `out`."""
    found = sorted(p for p in out.glob(f"{procedure}__{video}__gt_*") if p.is_dir())
    report = out / f"{procedure}__{video}__extract_report.json"
    return found + ([report] if report.is_file() else [])


def extract_video(atlas_root: Path, procedure: str, video: str, out: Path, rects: dict, ratios: FrameRatios,
                  table: ClassTable, step_sec: float, min_frames: int, long_side: int, step_tol: float,
                  overwrite: bool = False) -> list[str]:
    """Extract one video's clips, write its report and return the names of the clips written.

    Raises:
        FileNotFoundError: the release has no such video or no mp4 of it.
        KeyError: the video's frame ratio was not measured, or a clip with masks has no confirmed rectangle.
        ValueError: the video already has output and `overwrite` is false, the step falls more than `step_tol`
            off `step_sec`, or a frame or mask is not the mp4's size.
        RuntimeError: a clip with masks has no JPEG for its first frame, or the frame ratio fails its check.
    """
    # Find the video, read its frame rate and its ratio, and check the ratio against the pixels.
    split, gt_dir, mp4 = find_video(atlas_root, procedure, video)
    index = json.loads((gt_dir / "clip_index.json").read_text())
    digits = index.get("frame_digits", 6)
    fps, src_size = read_mp4(mp4)
    ratio = ratios.ratio(procedure, video)
    verify_ratio(ratios, procedure, video, mp4, gt_dir, index["clips"], digits)

    # Set the stride, and refuse a step far from the one asked for: a wrong ratio puts it two to four times off.
    stride = stride_for(step_sec, fps, ratio)
    step = stride * ratio / fps
    if abs(step - step_sec) > step_tol * step_sec:
        raise ValueError(f"{procedure}/{video}: the step is {step:.4f} s, more than {step_tol:.0%} off "
                         f"{step_sec} s; suspect the frame ratio")
    print(f"  {src_size[0]}x{src_size[1]}, {fps:.3f} fps, ratio {ratio}, stride {stride} = {step:.4f} s")

    # Refuse output an earlier run left, unless asked to replace it. A frame left by a run that wrote more
    # frames would otherwise stay in the clip.
    held = existing_output(out, procedure, video)
    if held and not overwrite:
        raise ValueError(f"{procedure}/{video} already has output in {out} ({', '.join(p.name for p in held)}); "
                         "pass --overwrite to replace it")
    for p in held:
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    # Write each clip of the index, and record what became of it.
    fps_native = round(fps, 3)
    seen: dict[tuple, str] = {}
    rows = []
    for clip in sorted(index["clips"]):
        n = clip_number(clip)
        frames = sorted(int(x) for x in index["clips"][clip])
        row = dict(source_clip=clip, index=n, n_index=len(frames), n_usable=0, n_segments=0, reason="",
                   out_clips=[], n_picked=[], dropped_frames=0, duplicate_of=[])
        rows.append(row)

        # A clip without masks, or whose files leave no frame, is recorded and skipped. Whether the
        # release holds the clip's JPEGs is decided on its first frame; where it does, a frame needs both files.
        src = gt_dir / clip
        mask_dir = next((src / d for d in MASK_DIRS if (src / d).is_dir()), None)
        if mask_dir is None:
            row["reason"] = "no_mask_dir"
            continue
        img_dir = src / "images"
        has_jpegs = (img_dir / f"frame_{frames[0]:0{digits}d}.jpg").is_file()
        usable = [f for f in frames
                  if (mask_dir / f"frame_{f:0{digits}d}.png").is_file()
                  and (not has_jpegs or (img_dir / f"frame_{f:0{digits}d}.jpg").is_file())]
        row["n_usable"] = len(usable)
        if not usable:
            row["reason"] = "no_masks"
            continue

        # Thin each run of consecutive frames, and keep the runs long enough.
        runs = contiguous_runs(usable)
        row["n_segments"] = len(runs)
        keep = [(run, run[::stride]) for run in runs if len(run[::stride]) >= min_frames]
        row["dropped_frames"] = len(frames) - sum(len(run) for run, _ in keep)
        if not keep:
            row["reason"] = "too_short"
            continue
        row["reason"] = "kept" if len(runs) == 1 else "kept_after_split"

        # Write each kept run, unless one with the same frames and rectangle was written already.
        rect = rects.get((procedure, video, clip))
        if rect is None:
            raise KeyError(f"{procedure}/{video}/{clip} has no confirmed crop rectangle")
        for j, (_, picked) in enumerate(keep, start=1):
            name = f"{procedure}__{video}__gt_{n:04d}" + (f"s{j}" if len(keep) > 1 else "")
            key = (tuple(picked), tuple(rect))
            if key in seen:
                row["duplicate_of"].append(seen[key])
                continue
            if not has_jpegs:
                raise RuntimeError(f"{procedure}/{video}/{clip}: the release holds no JPEG of its first frame, "
                                   f"{frames[0]}, and frames are read from the release's JPEGs only")
            seen[key] = name
            row["out_clips"].append(name)
            row["n_picked"].append(len(picked))
            write_clip(out / name, picked, mask_dir, img_dir, digits, rect, long_side, src_size, table, dict(
                dataset="atlas120k", procedure=procedure, youtube_id=video, split=split,
                is_robot=bool(index.get("is_robot", False)), fps_native=fps_native, frame_ratio=ratio,
                src_size=list(src_size), out_size=None, clip_kind="gt", source_clip=clip, segment_index=j,
                n_segments_written=len(keep), pixel_source="jpeg", stride=stride, gt_stride=stride,
                gt_step_sec=step_sec, gt_step_sec_actual=round(stride * ratio / fps_native, 6),
                gt_min_frames=min_frames, gt_pad_factor=1.0))
            print(f"  {name}: {len(picked)} frames")
        if row["duplicate_of"] and not row["out_clips"]:
            row["reason"] = "duplicate"

    # Write the report last, so that it marks a video whose every clip was written.
    written = [c for r in rows for c in r["out_clips"]]
    report = dict(procedure=procedure, video=video, split=split, fps_native=fps_native, frame_ratio=ratio,
                  gt_step_sec=step_sec, gt_stride=stride, gt_step_sec_actual=round(step, 6),
                  gt_min_frames=min_frames, n_index_clips=len(rows), n_written=len(written), clips=rows)
    with open(out / f"{procedure}__{video}__extract_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return written


def write_clip(clip_dir: Path, picked: list[int], mask_dir: Path, img_dir: Path, digits: int, rect: Rect,
               long_side: int, src_size: tuple[int, int], table: ClassTable, meta: dict) -> None:
    """Write one clip's frames, masks and manifest. Every frame of the clip has ground truth and is an anchor.

    `meta` holds the manifest's keys before `crop`; its `out_size` is filled in here. The keys and their order
    are the workbench's, so that a clip extracted here equals one extracted there, byte for byte.
    """
    size = target_size(rect[2], rect[3], long_side)
    (clip_dir / "input_images").mkdir(parents=True)
    (clip_dir / "seg_masks").mkdir()
    for seq, f in enumerate(picked):
        stem = f"frame_{f:0{digits}d}"
        write_frame(clip_dir / "input_images" / f"{seq:06d}.png", img_dir / f"{stem}.jpg", rect, size, src_size)
        write_mask(clip_dir / "seg_masks" / f"{seq:06d}_class.png", mask_dir / f"{stem}.png", rect, size,
                   src_size, table)
    manifest = dict(meta, out_size=list(size), crop=dict(zip("xywh", rect)), crop_source="confirmed",
                    frames=[{"seq_idx": seq, "native_frame": f, "has_gt": True, "is_anchor": True}
                            for seq, f in enumerate(picked)])
    with open(clip_dir / "frame_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


def check_population(written: list[str], population: list[str], procedure: str, video: str) -> None:
    """Refuse a video whose clips written are not the population's clips of that video.

    Raises:
        ValueError: a clip of the population was not written, or a clip written is not in the population.
    """
    prefix = f"{procedure}__{video}__"
    want = {c for c in population if c.startswith(prefix)}
    missing, extra = sorted(want - set(written)), sorted(set(written) - want)
    if missing or extra:
        raise ValueError(f"{procedure}/{video}: the clips written are not the population's "
                         f"(not written: {missing}; not in the population: {extra})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas-root", required=True,
                    help="The directory that holds the release's atlas120k/ and the videos' raw_data/.")
    ap.add_argument("--out", required=True, help="The directory the clips are written to.")
    ap.add_argument("--clip-rects", required=True, help="The confirmed crop rectangles, crop_rects.json.")
    ap.add_argument("--frame-ratios", required=True, help="The measured frame ratios, frame_ratio.json.")
    ap.add_argument("--videos", nargs="*",
                    help="<procedure>/<video> to extract. Default: every video whose frame ratio was measured.")
    ap.add_argument("--population", help="A population file; the clips written must be its clips of the videos.")
    ap.add_argument("--step-sec", type=float, default=0.52, help="Seconds between two frames of a clip.")
    ap.add_argument("--min-frames", type=int, default=8, help="The fewest frames a clip may have once thinned.")
    ap.add_argument("--long-side", type=int, default=854, help="The longer side of a written frame.")
    ap.add_argument("--step-tol", type=float, default=0.20,
                    help="The largest difference allowed between the stride's step and --step-sec, as a fraction of "
                         "--step-sec.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace the clips and report an earlier extraction of a video left.")
    args = ap.parse_args()

    # Read the tables, and find the videos, before any video is read.
    rects = load_clip_rects(args.clip_rects)
    ratios = FrameRatios.load(args.frame_ratios)
    table = load_table("atlas120k")
    population = Path(args.population).read_text().split() if args.population else None
    videos = [tuple(v.split("/")) for v in args.videos] if args.videos else ratios.videos()
    bad = ["/".join(v) for v in videos if len(v) != 2]
    if bad:
        raise SystemExit(f"not <procedure>/<video>: {', '.join(bad)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Extract every video, even after one fails, and list the failures at the end.
    failed = []
    for procedure, video in videos:
        print(f"{procedure}/{video}")
        try:
            written = extract_video(Path(args.atlas_root), procedure, video, out, rects, ratios, table,
                                    args.step_sec, args.min_frames, args.long_side, args.step_tol,
                                    overwrite=args.overwrite)
            if population is not None:
                check_population(written, population, procedure, video)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(f"{procedure}/{video}")
    if failed:
        raise SystemExit(f"{len(failed)} of {len(videos)} video(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
