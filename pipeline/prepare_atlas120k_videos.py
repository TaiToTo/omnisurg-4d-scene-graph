"""Build a video root the ATLAS-120k extraction can read: link the release, and convert the videos OpenCV cannot decode.

The extraction reads a video's frame rate and size from its mp4, and checks
the frame ratio against its pixels. One video of the 97, `rarp/NitKIjCcS7U`,
is AV1, which the GPU machine's OpenCV cannot decode. This command makes a
second root. Its `atlas120k/` is a symlink to the release's. Under its
`raw_data/`, every H.264 video is a symlink to the release's file, and every
other video is converted to H.264, keeping its size, frame rate and frame
count. `video_root.json` records what was done to each video. A converted
video that is already there is checked against the source and the record,
not made again.

Usage:
    python -m pipeline.prepare_atlas120k_videos --src /path/to/ATLAS --dst /path/to/ATLAS_h264 [--crf 18]
"""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2

# The codec OpenCV decodes everywhere the pipeline runs. A video in any other codec is converted to it.
READABLE_CODEC = "h264"

# The quality the workbench converted with. The extraction's frame ratio check reads the converted pixels, so a
# conversion at another value is checked against other pixels; the default is kept unless a root is made from
# scratch on purpose.
DEFAULT_CRF = 18

RECORD = "video_root.json"


@dataclass(frozen=True)
class VideoInfo:
    """What `ffprobe` says of a video's first video stream. `frame_rate` is its `r_frame_rate`, as "30000/1001"."""

    codec: str
    width: int
    height: int
    frame_rate: str
    n_frames: int


def require_tools() -> None:
    """Refuse to run without `ffprobe` and `ffmpeg` on the path.

    Raises:
        FileNotFoundError: one of them is missing.
    """
    missing = [t for t in ("ffprobe", "ffmpeg") if shutil.which(t) is None]
    if missing:
        raise FileNotFoundError(f"{' and '.join(missing)} not found on PATH; install FFmpeg")


def probe(path: Path) -> VideoInfo:
    """Return the codec, size, frame rate and frame count of a video's first video stream.

    Raises:
        ValueError: the file has no video stream, or the stream does not say how many frames it has. A frame
            count is what the conversion is checked by, so a video without one cannot be checked.
        subprocess.CalledProcessError: `ffprobe` cannot read the file.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams") or []
    if not streams:
        raise ValueError(f"{path} has no video stream")
    s = streams[0]
    if "nb_frames" not in s:
        raise ValueError(f"{path} does not say how many frames it has, so a conversion of it cannot be checked")
    return VideoInfo(codec=s["codec_name"], width=int(s["width"]), height=int(s["height"]),
                     frame_rate=s["r_frame_rate"], n_frames=int(s["nb_frames"]))


def convert(src: Path, dst: Path, crf: int) -> None:
    """Convert a video to H.264 at the same size, frame rate and frame count, without its audio.

    The file is written under a temporary name and moved to `dst` when FFmpeg has finished, so a run that stops
    leaves no partial file at `dst`.
    """
    part = dst.with_name(dst.stem + ".part" + dst.suffix)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(part)],
        check=True)
    os.replace(part, dst)


def check_converted(src: Path, dst: Path, want: VideoInfo) -> VideoInfo:
    """Refuse a converted video whose size, frame rate or frame count is not the source's, or that OpenCV cannot read.

    The extraction takes the frame size and rate from the mp4: the crop rectangles are in those pixels, the
    stride between frames is computed from the rate, and the frame ratio is checked by frame number. A size, a
    rate or a count that moved would move every frame.

    Raises:
        ValueError: the sizes, frame rates or frame counts differ, or the converted file is not H.264.
        RuntimeError: OpenCV cannot read a frame from the converted file.
    """
    got = probe(dst)
    if got.codec != READABLE_CODEC:
        raise ValueError(f"{dst} is {got.codec}, not {READABLE_CODEC}")
    if (got.width, got.height, got.n_frames) != (want.width, want.height, want.n_frames):
        raise ValueError(f"{dst}: the conversion changed the video from {want.width}x{want.height}, "
                         f"{want.n_frames} frames to {got.width}x{got.height}, {got.n_frames} frames ({src})")
    if Fraction(got.frame_rate) != Fraction(want.frame_rate):
        raise ValueError(f"{dst}: the conversion changed the frame rate from {want.frame_rate} to "
                         f"{got.frame_rate} ({src})")
    cap = cv2.VideoCapture(str(dst))
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"OpenCV cannot read a frame from the converted {dst}")
    return got


def check_recorded_crf(dst_root: Path, procedure: str, video: str, crf: int) -> None:
    """Refuse a converted video that is already there unless `video_root.json` records it at `crf`.

    The quality a conversion was made at is not read back from the file, so the record is the only account of
    it, and a run at another `--crf` would otherwise record a quality the file does not have.

    Raises:
        ValueError: the root has no record of the video's conversion, or the record's crf is another value.
    """
    record_path = dst_root / RECORD
    rows = json.loads(record_path.read_text(encoding="utf-8"))["videos"] if record_path.is_file() else []
    recorded = [r for r in rows if (r["procedure"], r["video"], r["action"]) == (procedure, video, "transcode")]
    if not recorded:
        raise ValueError(f"{procedure}/{video} is already converted, but {record_path} does not record at what "
                         f"crf; remove the conversion to make it again")
    if recorded[0]["crf"] != crf:
        raise ValueError(f"{procedure}/{video} is already converted at crf {recorded[0]['crf']}, not {crf}; "
                         f"remove the conversion to make it again")


def link(target: Path, path: Path) -> None:
    """Make `path` a symlink to `target`, or refuse a `path` that is already something else.

    Raises:
        FileExistsError: `path` exists and is not a symlink to `target`.
    """
    target = target.resolve()
    if path.is_symlink() and path.resolve() == target:
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} exists and is not a symlink to {target}")
    path.symlink_to(target)


def prepare(src_root: Path, dst_root: Path, crf: int = DEFAULT_CRF) -> list[dict]:
    """Build the root and return one record per video, as `video_root.json` holds them.

    Raises:
        FileNotFoundError: `src_root` has no `atlas120k/` or no `raw_data/` with an mp4.
        FileExistsError: a path in `dst_root` is already something other than what the command would make.
        ValueError, RuntimeError: `probe`, `check_recorded_crf` or `check_converted` refuses a video.
    """
    require_tools()
    if not (src_root / "atlas120k").is_dir():
        raise FileNotFoundError(f"no atlas120k/ under {src_root}")
    mp4s = sorted((src_root / "raw_data").glob("*/*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"no raw_data/<procedure>/<video>.mp4 under {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)
    link(src_root / "atlas120k", dst_root / "atlas120k")

    rows = []
    for src in mp4s:
        procedure, video = src.parent.name, src.stem
        dst = dst_root / "raw_data" / procedure / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        info = probe(src)

        # A readable video is linked. Any other is converted once, and checked every time.
        if info.codec == READABLE_CODEC:
            link(src, dst)
            rows.append(dict(procedure=procedure, video=video, action="symlink", **asdict(info)))
            continue
        if dst.is_symlink():
            raise FileExistsError(f"{dst} is a symlink, where a conversion of {src} should be")
        if dst.exists():
            check_recorded_crf(dst_root, procedure, video, crf)
        else:
            print(f"converting {procedure}/{video} ({info.codec} -> {READABLE_CODEC})", flush=True)
            convert(src, dst, crf)
        got = check_converted(src, dst, info)
        rows.append(dict(procedure=procedure, video=video, action="transcode", src_codec=info.codec, crf=crf,
                         **asdict(got)))
        print(f"  {procedure}/{video}: {got.n_frames} frames, {got.width}x{got.height}", flush=True)

    n_converted = sum(r["action"] == "transcode" for r in rows)
    record = dict(src=str(src_root), dst=str(dst_root), crf=crf, n_videos=len(rows), n_transcoded=n_converted,
                  videos=rows)
    with open(dst_root / RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"{len(rows)} videos: {len(rows) - n_converted} linked, {n_converted} converted -> {dst_root}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="The release: the directory that holds atlas120k/ and raw_data/.")
    ap.add_argument("--dst", required=True, help="The root to build. The extraction takes it as --atlas-root.")
    ap.add_argument("--crf", type=int, default=DEFAULT_CRF, help="The quality of a conversion; smaller is better.")
    args = ap.parse_args()
    try:
        prepare(Path(args.src), Path(args.dst), args.crf)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
