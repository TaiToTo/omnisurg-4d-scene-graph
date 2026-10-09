"""Extract CholecSeg8k clips into the clip directories the pipeline reads.

A clip `VID<nn>_s<stride>_<start>` is `--count` frames of video `nn`,
numbered in CholecSeg8k from `start` in steps of `stride`. A frame
CholecSeg8k annotated takes the image the mask was drawn on, and the mask.
Its `native_frame`, the cholec80 frame it is, is found by matching that image
against the video. A frame CholecSeg8k did not annotate is decoded from the
video. CholecSeg8k numbers 12 of its 17 videos at about 30 frames a second,
so the native frame of such a frame is interpolated between the clip's
annotated frames, or carried on at their rate past the first or the last.
The clip gets `input_images/`, `seg_masks/` and `frame_manifest.json`.

Usage:
    python -m pipeline.extract_cholecseg8k --seg8k-root /path/to/CholecSeg8k --videos-root /path/to/cholec80/videos \\
        --out /path/to/clips --clips VID01_s15_80 [VID25_s15_162 ...] [--count 30] [--overwrite]
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

# How far from its CholecSeg8k number a frame's native frame is searched for. In a video numbered at 30 frames a
# second, a number exceeds its native frame by a sixth of the number, so the window reaches back the number
# divided by `SEARCH_BACK_DIVISOR`, a fifth, and `SEARCH_BACK` frames more.
SEARCH_BACK_DIVISOR = 5
SEARCH_BACK = 150
SEARCH_AHEAD = 60

# The largest mean absolute difference between an annotated image and the video frame it is. The 522 annotated
# frames of the paper's 27 clips match at 0.45 to 6.4.
MATCH_TOL = 10.0

# The rates CholecSeg8k numbers a video at, as native frames per CholecSeg8k number. A video numbered at its own
# 25 frames a second runs at 1; a video numbered at 30 frames a second runs at 25/30.
RATES = (1.0, 25 / 30)
RATE_TOL = 0.02


def parse_clip(name: str) -> tuple[int, int, int]:
    """Return the video number, the stride and the first CholecSeg8k number of a clip's name.

    Raises:
        ValueError: the name is not `VID<nn>_s<stride>_<start>`, with the video number in two digits.
    """
    m = re.fullmatch(r"VID(\d{2})_s(\d+)_(\d+)", name)
    if not m:
        raise ValueError(f"unexpected clip name {name!r}; expected VID<nn>_s<stride>_<start>")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def find_annotation(video_dir: Path, number: int) -> tuple[Path, Path] | None:
    """Return the annotated image and the colour mask of CholecSeg8k number `number`, or None where it has no mask.

    Raises:
        FileNotFoundError: the number has a mask but not the image it was drawn on. Taking the image from the
            video instead would pair a mask with a frame it was not drawn on.
    """
    for chunk in sorted(p for p in video_dir.iterdir() if p.is_dir()):
        mask = chunk / f"frame_{number}_endo_color_mask.png"
        if mask.is_file():
            endo = chunk / f"frame_{number}_endo.png"
            if not endo.is_file():
                raise FileNotFoundError(f"{mask} has no image {endo.name} beside it")
            return endo, mask
    return None


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR, refusing one that cannot be read."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def match_native(cap: cv2.VideoCapture, image: np.ndarray, number: int, n_frames: int) -> int:
    """Return the video frame that `image`, CholecSeg8k number `number`, is.

    The frames of the search window are decoded in order, and the one with the smallest mean absolute difference
    is taken.

    Raises:
        ValueError: a frame is not the image's size, the best match is further than `MATCH_TOL`, or the best
            match is on an edge of the window that is not an end of the video, where the true frame may be
            outside the window.
        RuntimeError: a frame of the window cannot be decoded. The window ends at the video's frame count, so a
            frame that cannot be decoded is a fault, not the end of the video.
    """
    lo = max(0, number - number // SEARCH_BACK_DIVISOR - SEARCH_BACK)
    hi = min(number + SEARCH_AHEAD, n_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    best, best_i = None, None
    for i in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} while matching CholecSeg8k number {number}")
        if frame.shape != image.shape:
            raise ValueError(f"video frame {i} is {frame.shape}, but the annotated image is {image.shape}")
        d = float(np.mean(np.abs(frame.astype(np.int16) - image.astype(np.int16))))
        if best is None or d < best:
            best, best_i = d, i
    if best is None or best > MATCH_TOL:
        raise ValueError(f"CholecSeg8k number {number} matches no video frame in [{lo}, {hi}] "
                         f"(best {best_i}, difference {best:.2f} > {MATCH_TOL})")
    if (best_i == lo and lo > 0) or (best_i == hi and hi < n_frames - 1):
        raise ValueError(f"CholecSeg8k number {number} matched frame {best_i}, on the edge of [{lo}, {hi}]; "
                         "the frame may lie outside the window")
    return best_i


def unannotated_natives(numbers: list[int], annotated: dict[int, int]) -> dict[int, int]:
    """Return the native frame of each CholecSeg8k number in `numbers` that has none in `annotated`.

    Between two annotated frames the native frame is interpolated; past the first or the last it is carried on
    at the rate between the clip's first and last annotated frames. That rate must be one CholecSeg8k numbers
    videos at.

    Raises:
        ValueError: the clip has fewer than two annotated frames, or the rate is neither of `RATES`.
    """
    known = sorted(annotated)
    if len(known) < 2:
        raise ValueError(f"{len(known)} annotated frame(s); the rate needs two")
    span_frames, span_numbers = annotated[known[-1]] - annotated[known[0]], known[-1] - known[0]
    rate = span_frames / span_numbers
    if not any(abs(rate - r) <= RATE_TOL * r for r in RATES):
        raise ValueError(f"the annotated frames run at {rate:.4f} native frames a number, not one of {RATES}")
    out = {}
    for s in numbers:
        if s in annotated:
            continue
        before = [k for k in known if k < s]
        after = [k for k in known if k > s]
        if before and after:
            a, b = before[-1], after[0]
            frames, step = annotated[b] - annotated[a], b - a
        else:
            a = before[-1] if before else after[0]
            frames, step = span_frames, span_numbers
        # (s - a) * frames / step, rounded with halves up, in integers: a float would round some halves down.
        out[s] = annotated[a] + ((s - a) * 2 * frames + step) // (2 * step)
    return out


def extract_clip(seg8k_root: Path, videos_root: Path, out: Path, clip: str, count: int,
                 overwrite: bool = False) -> None:
    """Extract one clip.

    The clip is written under a temporary name and moved into place when it is complete, after every check has
    passed. A clip an earlier run left is removed only then.

    Raises:
        FileNotFoundError: the video or its CholecSeg8k directory is missing, or a mask has no image.
        ValueError: the clip's name is not `VID<nn>_s<stride>_<start>`, the clip already exists and `overwrite`
            is false, an annotated image is not the video's size, matches no video frame, or matches one on the
            edge of its search window, the clip has fewer than two annotated frames, the rate is neither of
            `RATES`, or the native frames do not rise.
        RuntimeError: the video cannot be opened or has no frame count, or an image or a video frame cannot be
            read.
    """
    # Find the video and its annotation, and refuse a clip an earlier run left.
    num, stride, start = parse_clip(clip)
    video_dir = seg8k_root / f"video{num:02d}"
    mp4 = videos_root / f"video{num:02d}.mp4"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"no CholecSeg8k directory {video_dir}")
    if not mp4.is_file():
        raise FileNotFoundError(f"no video {mp4}")
    clip_dir = out / clip
    if clip_dir.exists() and not overwrite:
        raise ValueError(f"{clip_dir} exists; pass --overwrite to replace it")
    numbers = [start + i * stride for i in range(count)]
    pairs = {s: find_annotation(video_dir, s) for s in numbers}

    # A clip a run that stopped left under the temporary name is never complete, so it is removed.
    partial = out / f".{clip}.partial"
    if partial.exists():
        shutil.rmtree(partial)

    cap = cv2.VideoCapture(str(mp4))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {mp4}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            raise RuntimeError(f"{mp4} reports {n_frames} frames")

        # Match each annotated image against the video, carry the rate to the other frames, and refuse native
        # frames that do not rise. No other check refuses a frame decoded out of place.
        annotated = {s: match_native(cap, read_image(p[0]), s, n_frames) for s, p in pairs.items() if p}
        natives = {**annotated, **unannotated_natives(numbers, annotated)}
        order = [natives[s] for s in numbers]
        if any(b <= a for a, b in zip(order, order[1:])):
            raise ValueError(f"{clip}: the native frames do not rise: {order}")

        # Write each frame: the annotated image and its mask, or the video's frame.
        (partial / "input_images").mkdir(parents=True)
        (partial / "seg_masks").mkdir()
        frames = []
        for i, s in enumerate(numbers):
            if pairs[s]:
                endo, mask = pairs[s]
                image = read_image(endo)
                cv2.imwrite(str(partial / "seg_masks" / f"{i:06d}_color_mask.png"), read_image(mask))
                source = f"CholecSeg8k ({endo.parent.name})"
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, natives[s])
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"cannot read frame {natives[s]} of {mp4}")
                source = "Cholec80 video"
            cv2.imwrite(str(partial / "input_images" / f"{i:06d}.png"), image)
            frames.append({"seq_idx": i, "native_frame": natives[s], "seg_frame": s if pairs[s] else None,
                           "timestamp_sec": round(natives[s] / fps, 2), "source": source,
                           "has_seg_mask": bool(pairs[s]), "is_anchor": bool(pairs[s]), "triplets": []})

        # Write the manifest with the workbench's keys, so that a clip extracted here can be compared with one
        # extracted there.
        manifest = {"video_id": f"VID{num:02d}", "video_num": num,
                    "track": {"start_native": start, "stride": stride, "count": count},
                    "n_frames": len(frames), "frames": frames}
        (partial / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        cap.release()

    # Move the complete clip into place, replacing the clip an earlier run left.
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    partial.rename(clip_dir)
    print(f"  {clip}: {len(frames)} frames, {len(annotated)} annotated, native {order[0]}-{order[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg8k-root", required=True, help="The directory that holds CholecSeg8k's video<nn>/.")
    ap.add_argument("--videos-root", required=True, help="The directory that holds cholec80's video<nn>.mp4.")
    ap.add_argument("--out", required=True, help="The directory the clips are written to.")
    ap.add_argument("--clips", nargs="+", required=True, help="Clips to extract, VID<nn>_s<stride>_<start>.")
    ap.add_argument("--count", type=int, default=30, help="The number of frames of a clip.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a clip an earlier run left.")
    args = ap.parse_args()

    # Check every name before any video is read.
    for c in args.clips:
        parse_clip(c)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Extract every clip, even after one fails, and list the failures at the end.
    failed = []
    for clip in args.clips:
        print(clip)
        try:
            extract_clip(Path(args.seg8k_root), Path(args.videos_root), out, clip, args.count, args.overwrite)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(args.clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
