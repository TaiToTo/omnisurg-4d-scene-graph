"""Read StereoMIS: the calibration, the rectified frames and their times, camera poses, tissue masks and clips.

A sequence's video stacks two views, the left one on top. A frame is named by
its 0-based position in the video, the frame FFmpeg's `select=eq(n,N)` decodes,
and its time is the video's timestamp. The ground truth's row of a frame, and
the file number of its mask or depth map, are offset from that position by
amounts measured once per sequence and kept in the tables below. Masks and depth
maps are in the rectified left view at half resolution. The ground truth holds
camera-to-world poses in m, and `load_gt` returns them in mm. `clips` cuts a
sequence into the clips the camera trajectory result is measured on; the clip
list keeps them as data, and `read_clip_list` reads it. Every function takes
the dataset's root as an argument.
"""

import configparser
import functools
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SEQUENCES = ("P1", "P2_0", "P2_1", "P2_2", "P2_3", "P2_4", "P2_5", "P2_6", "P2_7", "P2_8")

# The size of the stacked video, of one view, and of the masks and depth maps (width, height).
VIDEO_SIZE = (1280, 2048)
VIEW_SIZE = (1280, 1024)
HALF_SIZE = (640, 512)

# A frame's position less the file number of its mask and depth map: the file number is the position less this
# offset, so -1 numbers a file one ahead of its frame, as every sequence numbers its files from 1. Measured on P2 by
# warping the right view onto the left with the depth, and on P1 by matching its shipped frames (`video_frames/`)
# with the decoded ones. P2_3's camera never moves, so its value is its neighbours'.
DEPTH_FILE_OFFSET = {s: -1 for s in SEQUENCES}

# The ground truth's row of a frame, less the frame's position. Moving depth between frames with the poses reads it
# on seven sequences: their best offsets lie from -5 to -2, and -3 is at most 6.25 % worse than each sequence's best.
GT_ROW_OFFSET = {s: -3 for s in SEQUENCES}

# The ground truth's translations are in m; the depth maps are in mm.
GT_TO_MM = 1000.0

# A clip is 112 frames a stride apart, the stride being 0.2 s at the frame rate the video declares, and clips do not
# overlap. They are fixed before anything is measured.
CLIP_FRAMES = 112
CLIP_INTERVAL_S = 0.2
# The grid's first position, fixed apart from the offsets, so that measuring an offset again moves no clip's frames.
# At even positions every frame of P2 has a mask of its own number.
CLIP_GRID_START = 4

# A depth map shows no surface (the view is inside a trocar, or against tissue) when its median is outside
# 30-400 mm or no more than half its pixels have depth. A clip is left out when fewer than half of its maps show one.
DEPTH_OK_MM = (30.0, 400.0)
DEPTH_OK_VALID = 0.5
CLIP_MIN_DEPTH_OK = 0.5

# A step of more than 5 mm or 5 degrees between two rows is a break in the kinematic record, not a motion: the
# camera moves 0.04 mm a frame at the median and 1.4 mm at the 99th percentile.
GT_JUMP_MM = 5.0
GT_JUMP_DEG = 5.0

# `ate_rel` divides by the true trajectory's RMS radius, so a clip whose radius is below 1 mm is left out. A clip
# whose radius is below 10 mm is in the stratum `slow`, every other one in `moving`.
CLIP_MIN_SPAN_MM = 1.0
CLIP_MOVING_SPAN_MM = 10.0


class Calib:
    """Rectify the two views of a sequence with its `StereoCalibration.ini`.

    Attributes:
        size: the size of one view, (width, height), as the calibration gives it.
        K_full: the rectified left camera's intrinsics, at the size of one view.
        K_half: the same intrinsics at half resolution, the masks' and depth maps' pixels.
        baseline_mm: the stereo baseline in mm, the unit of the calibration's translation.
    """

    def __init__(self, root: Path, seq: str):
        """Read the calibration and compute the rectification.

        Raises:
            FileNotFoundError: the sequence has no `StereoCalibration.ini`.
            ValueError: the calibration is not for views of `VIEW_SIZE`.
        """
        cp = configparser.ConfigParser()
        path = Path(root) / seq / "StereoCalibration.ini"
        if not cp.read(path):
            raise FileNotFoundError(path)

        def g(section: str, key: str) -> float:
            return float(cp[section][key])

        def intrinsics(section: str) -> np.ndarray:
            return np.array([[g(section, "fc_x"), 0.0, g(section, "cc_x")],
                             [0.0, g(section, "fc_y"), g(section, "cc_y")],
                             [0.0, 0.0, 1.0]])

        def distortion(section: str) -> np.ndarray:
            return np.array([g(section, f"kc_{i}") for i in range(5)])

        self.size = (int(g("StereoLeft", "res_x")), int(g("StereoLeft", "res_y")))
        # `K_half` is the masks' and depth maps' pixels only when a view is `VIEW_SIZE`, so a caller that reads it
        # without decoding a frame is refused here too.
        if self.size != VIEW_SIZE:
            raise ValueError(f"{seq}: the calibration is for views of {self.size[0]}x{self.size[1]}, "
                             f"not {VIEW_SIZE[0]}x{VIEW_SIZE[1]}")
        R =np.array([g("StereoRight", f"R_{i}") for i in range(9)]).reshape(3, 3)
        # A column vector, which OpenCV 5 needs: its stereoRectify refuses a 1-D translation.
        T = np.array([g("StereoRight", f"T_{i}") for i in range(3)]).reshape(3, 1)
        self._Kl, self._dl = intrinsics("StereoLeft"), distortion("StereoLeft")
        self._Kr, self._dr = intrinsics("StereoRight"), distortion("StereoRight")
        # With alpha 0 the rectified views keep no black border. P1's shipped frames (`video_frames/`) are rectified
        # this way: the views here at position N - 1 differ from the shipped frame N by 1.5 to 2.2 grey levels on
        # average.
        self._R1, self._R2, P1, P2, *_ = cv2.stereoRectify(
            self._Kl, self._dl, self._Kr, self._dr, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.P_left, self.P_right = P1, P2
        self.K_full = P1[:3, :3].copy()
        self.K_half = self.K_full / 2.0
        self.K_half[2, 2] = 1.0
        self.baseline_mm = float(-P2[0, 3] / P2[0, 0])

    @functools.cached_property
    def map_left(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the left view's rectification map."""
        return cv2.initUndistortRectifyMap(self._Kl, self._dl, self._R1, self.P_left, self.size, cv2.CV_32FC1)

    @functools.cached_property
    def map_right(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the right view's rectification map."""
        return cv2.initUndistortRectifyMap(self._Kr, self._dr, self._R2, self.P_right, self.size, cv2.CV_32FC1)

    def rectify(self, img: np.ndarray, side: str) -> np.ndarray:
        """Rectify one view, `side` "l" or "r", at the size of one view."""
        if side not in ("l", "r"):
            raise ValueError(f"side is 'l' or 'r', not {side!r}")
        m = self.map_left if side == "l" else self.map_right
        return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR)


@functools.lru_cache(maxsize=None)
def calib(root: Path, seq: str) -> Calib:
    """Return the sequence's calibration, read once."""
    return Calib(root, seq)


@functools.lru_cache(maxsize=None)
def video_path(root: Path, seq: str) -> Path:
    """Return the sequence's video: `video.mp4` for P1, `IFBS_ENDOSCOPE-partNNNN.mp4` for P2.

    Raises:
        FileNotFoundError: the sequence holds no mp4, or more than one.
    """
    hits = sorted((Path(root) / seq).glob("*.mp4"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{seq}: one mp4 is needed, found {[h.name for h in hits]}")
    return hits[0]


@functools.lru_cache(maxsize=None)
def video_info(root: Path, seq: str) -> dict:
    """Return the video's width, height, frame count and frame rate, as `ffprobe` reads them.

    The frame rate is the stream's `r_frame_rate`.

    Raises:
        ValueError: the stream does not say how many frames it has, or the video is not two views of
            `VIEW_SIZE` stacked.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(video_path(root, seq))],
        capture_output=True, check=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    if "nb_frames" not in s:
        raise ValueError(f"{seq}: the video does not say how many frames it has, so a position cannot be checked")
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    w, h = int(s["width"]), int(s["height"])
    if (w, h) != VIDEO_SIZE:
        raise ValueError(f"{seq}: two stacked views of {VIDEO_SIZE[0]}x{VIDEO_SIZE[1]} are expected, not {w}x{h}")
    return dict(width=w, height=h, n_frames=int(s["nb_frames"]), fps=num / den)


@functools.lru_cache(maxsize=None)
def frame_times(root: Path, seq: str) -> np.ndarray:
    """Return each frame's time in s, indexed by position: the video's timestamps, in the order they are shown.

    The P2 videos declare 59.94 frames a second, but by their timestamps 12 frames span 0.217 s at the median, and
    some recordings pause. So a time is read from the video, never computed from a position and the frame rate.

    Raises:
        ValueError: a frame has no timestamp, two frames share one, or the video holds another number of frames
            than `video_info` gives.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base:packet=pts",
         "-of", "json", str(video_path(root, seq))], capture_output=True, check=True, text=True).stdout
    data = json.loads(out)
    num, den = (int(x) for x in data["streams"][0]["time_base"].split("/"))
    pts = [p.get("pts") for p in data.get("packets", [])]
    if any(p is None for p in pts):
        raise ValueError(f"{seq}: {sum(p is None for p in pts)} frames have no timestamp")
    # Packets come in the order they are decoded; frames are shown in the order of their timestamps.
    pts = sorted(int(p) for p in pts)
    if len(set(pts)) != len(pts):
        raise ValueError(f"{seq}: {len(pts) - len(set(pts))} frames share a timestamp with another")
    n = video_info(root, seq)["n_frames"]
    if len(pts) != n:
        raise ValueError(f"{seq}: the video holds {len(pts)} timestamps for {n} frames")
    # Integer times the time base, so that a time is the double nearest the exact one.
    t = np.array([p * num / den for p in pts])
    t.setflags(write=False)
    return t


def iter_frame_set(root: Path, seq: str, frames):
    """Decode the given frames in one pass, and yield each one's rectified views at half resolution.

    FFmpeg's `select` counts frames from the start of the input, so the position of each frame is known by
    construction; a seek with `-ss` may land one frame off. One call takes every frame wanted, since a pass
    decodes the whole video. The frames are yielded one at a time, since a stacked frame takes 7.9 MB.

    Args:
        root: the dataset's root.
        seq: the sequence.
        frames: the positions wanted, in any order, repeats allowed.

    Yields:
        `(position, left, right)` in increasing position, each view RGB uint8 of `HALF_SIZE`.

    Raises:
        IndexError: a position is outside the video.
        ValueError: the calibration is not for views of `VIEW_SIZE`.
        RuntimeError: the video ends before every frame wanted is decoded.
    """
    want = sorted({int(f) for f in frames})
    if not want:
        return
    info = video_info(root, seq)
    if want[0] < 0 or want[-1] >= info["n_frames"]:
        raise IndexError(f"{seq}: positions {want[0]}-{want[-1]} are outside 0-{info['n_frames'] - 1}")
    c = calib(root, seq)
    expr = "+".join(f"eq(n\\,{f})" for f in want)
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path(root, seq)), "-vf", f"select='{expr}'",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    px = info["width"] * info["height"] * 3
    view_h = VIEW_SIZE[1]
    got = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=px) as p:
        try:
            for f in want:
                raw = p.stdout.read(px)
                if len(raw) < px:
                    break
                fr = np.frombuffer(raw, np.uint8).reshape(info["height"], info["width"], 3)
                pair = [cv2.resize(c.rectify(fr[top:top + view_h], side), HALF_SIZE, interpolation=cv2.INTER_AREA)
                        for side, top in (("l", 0), ("r", view_h))]
                got += 1
                yield f, pair[0], pair[1]
        finally:
            p.stdout.close()
            p.wait()
    if got != len(want):
        raise RuntimeError(f"{seq}: {len(want)} frames wanted, {got} decoded")


@functools.lru_cache(maxsize=None)
def load_gt(root: Path, seq: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the ground truth's camera-to-world rotations and translations in mm, one per row.

    `groundtruth.txt` holds one row `idx tx ty tz qx qy qz qw` per frame of the record. A point `p` in camera
    coordinates is `R @ p + t` in world coordinates. A row is not a frame's position; `GT_ROW_OFFSET` maps one to
    the other.

    Returns:
        (N, 3, 3) rotations and (N, 3) translations in mm.

    Raises:
        ValueError: the file does not have 8 columns, its first column is not the row number, or a value is not
            finite. The rows are indexed by their place in the file, so a gap in the record would shift every
            pose after it.
    """
    a = np.loadtxt(Path(root) / seq / "groundtruth.txt")
    if a.ndim != 2 or a.shape[1] != 8:
        raise ValueError(f"{seq}: groundtruth.txt has shape {a.shape}, 8 columns are expected")
    if not np.array_equal(a[:, 0], np.arange(len(a))):
        raise ValueError(f"{seq}: the first column of groundtruth.txt is not the row number 0, 1, 2, ...")
    if not np.isfinite(a).all():
        raise ValueError(f"{seq}: groundtruth.txt holds {int((~np.isfinite(a)).sum())} values that are not finite")
    # scipy reads a quaternion as (x, y, z, w), the file's order.
    R = Rotation.from_quat(a[:, 4:8]).as_matrix()
    t = a[:, 1:4] * GT_TO_MM
    # The arrays are cached and shared by every caller, so one that wrote into them would move every later pose.
    R.setflags(write=False)
    t.setflags(write=False)
    return R, t


def measured_offset(table: dict, seq: str) -> int:
    """Return the sequence's offset in `GT_ROW_OFFSET` or `DEPTH_FILE_OFFSET`, refusing one never measured.

    Raises:
        KeyError: the sequence has no measured offset in the table.
    """
    what = "the ground truth's rows" if table is GT_ROW_OFFSET else "the file numbers of its masks and depth maps"
    if seq not in table:
        raise KeyError(f"{seq}: the offset between its frames and {what} has not been measured")
    return table[seq]


def gt_pose(root: Path, seq: str, rows) -> np.ndarray:
    """Return the (N, 4, 4) camera-to-world poses of the ground truth's rows, translations in mm.

    Raises:
        ValueError: no row is given.
        IndexError: a row is outside the ground truth. The pose at the end is not used in its place.
    """
    R, t = load_gt(root, seq)
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        raise ValueError(f"{seq}: no row given")
    if rows.min() < 0 or rows.max() >= len(R):
        raise IndexError(f"{seq}: ground truth rows {rows.min()}-{rows.max()} are outside 0-{len(R) - 1}")
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = R[rows]
    T[:, :3, 3] = t[rows]
    return T


@functools.lru_cache(maxsize=None)
def mask_numbers(root: Path, seq: str) -> frozenset[int]:
    """Return the file numbers of the sequence's masks, `masks/NNNNNNl.png`."""
    return frozenset(int(p.name[:6]) for p in (Path(root) / seq / "masks").glob("[0-9]" * 6 + "l.png"))


def nearest_mask(root: Path, seq: str, frame: int, tol: int = 2) -> int | None:
    """Return the file number of the mask of the frame nearest `frame` that has one, within `tol` frames, or None.

    The masks of P1 to P2_5 are on odd file numbers only, and those of P2_6 to P2_8 on every number, so a frame
    may have none of its own. Instruments move little in one or two frames (17 to 33 ms). Of two frames equally
    near, the earlier one is taken.

    Raises:
        KeyError: the sequence has no measured file offset.
    """
    off, have = measured_offset(DEPTH_FILE_OFFSET, seq), mask_numbers(root, seq)
    for d in range(tol + 1):
        for f in ((frame,) if d == 0 else (frame - d, frame + d)):
            if f - off in have:
                return f - off
    return None


def load_mask(root: Path, seq: str, number: int) -> np.ndarray:
    """Return the tissue mask of a file number: True on tissue, False on instruments and outside the view.

    The dataset's mask is 255 on tissue and 0 elsewhere.

    Raises:
        FileNotFoundError: the sequence has no mask of that number.
        ValueError: the mask file cannot be read, or is not of `HALF_SIZE`.
    """
    path = Path(root) / seq / "masks" / ("%06dl.png" % number)
    if not path.exists():
        raise FileNotFoundError(path)
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError(f"{path} exists but cannot be read")
    if a.shape[::-1] != HALF_SIZE:
        raise ValueError(f"{path} is {a.shape[1]}x{a.shape[0]}, the masks are {HALF_SIZE[0]}x{HALF_SIZE[1]}")
    return a > 127


@functools.lru_cache(maxsize=None)
def gt_jump_rows(root: Path, seq: str) -> np.ndarray:
    """Return each row `i` of the ground truth such that the step from row `i` to row `i + 1` is a break.

    A step is a break when it exceeds `GT_JUMP_MM` or `GT_JUMP_DEG`.
    """
    R, t = load_gt(root, seq)
    d = np.linalg.norm(np.diff(t, axis=0), axis=1)
    dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    rows = np.where((d > GT_JUMP_MM) | (ang > GT_JUMP_DEG))[0]
    rows.setflags(write=False)
    return rows


def has_gt_jump(root: Path, seq: str, lo: int, hi: int) -> bool:
    """Return whether the ground truth breaks between the rows of positions `lo` and `hi`, or at either end.

    Raises:
        KeyError: the sequence has no measured ground truth offset.
    """
    off = measured_offset(GT_ROW_OFFSET, seq)
    j = gt_jump_rows(root, seq)
    return bool(np.any((j >= lo + off - 1) & (j <= hi + off)))


def clip_stride(root: Path, seq: str) -> int:
    """Return the step between a clip's frames that comes nearest `CLIP_INTERVAL_S`: 12 at 60 frames a second."""
    return max(1, int(round(video_info(root, seq)["fps"] * CLIP_INTERVAL_S)))


def load_depth_stats(depth_root: Path, seq: str) -> np.ndarray:
    """Return the depth export's `stats.npy`: one row per depth map, its file number, median in mm and valid share.

    Raises:
        ValueError: the file holds no row, fewer than 3 columns, or a file number that is not whole or that
            repeats. With no row, every clip would be left out as showing no surface.
    """
    st = np.load(Path(depth_root) / seq / "stats.npy")
    if st.ndim != 2 or st.shape[0] == 0 or st.shape[1] < 3:
        raise ValueError(f"{seq}: stats.npy has shape {st.shape}, one row of at least 3 columns per map is expected")
    numbers = st[:, 0]
    if not np.array_equal(numbers, np.round(numbers)):
        raise ValueError(f"{seq}: stats.npy holds a file number that is not whole")
    # `clips` keys the maps by position, so a number given twice would keep one of its rows and drop the other.
    if len(np.unique(numbers)) != len(numbers):
        raise ValueError(f"{seq}: stats.npy holds a file number more than once")
    return st


def shows_surface(median_mm: float, valid: float) -> bool:
    """Return whether a depth map shows a surface: its median within `DEPTH_OK_MM`, more than `DEPTH_OK_VALID` valid."""
    return bool(DEPTH_OK_MM[0] <= median_mm <= DEPTH_OK_MM[1] and valid > DEPTH_OK_VALID)


def drop_of(depth_ok: float, gt_jump: bool, span_mm: float) -> str:
    """Return why a clip is left out, the first of `no_surface`, `gt_jump` and `gt_static` that holds, or ""."""
    return ("no_surface" if depth_ok < CLIP_MIN_DEPTH_OK else
            "gt_jump" if gt_jump else
            "gt_static" if span_mm < CLIP_MIN_SPAN_MM else "")


def stratum_of(span_mm: float) -> str:
    """Return a clip's stratum: `moving` from `CLIP_MOVING_SPAN_MM` up, `slow` below."""
    return "moving" if span_mm >= CLIP_MOVING_SPAN_MM else "slow"


def clips(root: Path, depth_root: Path, seq: str) -> tuple[dict, ...]:
    """Cut a sequence into fixed clips that do not overlap, and mark those its inputs cannot be measured on.

    A clip is left out for its inputs only, never for a score, by the first of these that holds:

    - `no_surface`: fewer than `CLIP_MIN_DEPTH_OK` of its depth maps show a surface;
    - `gt_jump`: the ground truth breaks inside it;
    - `gt_static`: the true camera's RMS radius is below `CLIP_MIN_SPAN_MM`.

    The grid starts at `CLIP_GRID_START`. A clip whose rows run outside the ground truth is not returned.

    Args:
        root: the dataset's root.
        depth_root: the depth export's root, which holds `<seq>/stats.npy`.
        seq: the sequence.

    Returns:
        One dict per clip, with the keys of `CLIP_KEYS`: `drop` is the reason, or ""; `frames` are positions;
        each frame's row is its position plus `gt_row_offset`; `times` are in s; `masks` holds the file number
        of each frame's nearest mask, or None.

    Raises:
        KeyError: the sequence has no measured offset, before anything is read.
        ValueError: no depth map lies on a clip's positions, so whether it shows a surface cannot be measured.
    """
    off, file_off = measured_offset(GT_ROW_OFFSET, seq), measured_offset(DEPTH_FILE_OFFSET, seq)
    info = video_info(root, seq)
    stride = clip_stride(root, seq)
    span = CLIP_FRAMES * stride
    st = load_depth_stats(depth_root, seq)
    d_ok = {int(r[0]) + file_off: shows_surface(r[1], r[2]) for r in st}
    _, tw = load_gt(root, seq)
    n_gt = len(tw)
    times = frame_times(root, seq)
    out = []
    for k, start in enumerate(range(CLIP_GRID_START, info["n_frames"] - span + 1, span)):
        frames = list(range(start, start + span, stride))
        rows = [f + off for f in frames]
        if min(rows) < 0 or max(rows) >= n_gt:
            continue
        inside = [v for f, v in d_ok.items() if start <= f < start + span]
        if not inside:
            raise ValueError(f"{seq}: no depth map lies on positions {start}-{start + span - 1}, so whether "
                             f"{seq}__clip_{k:04d} shows a surface cannot be measured")
        ok = float(np.mean(inside))
        c = tw[np.array(rows)]
        sp = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))
        drop = drop_of(ok, has_gt_jump(root, seq, frames[0], frames[-1]), sp)
        out.append(dict(name=f"{seq}__clip_{k:04d}", seq=seq, usable=not drop, drop=drop, stratum=stratum_of(sp),
                        span_mm=round(sp, 3), depth_ok=round(ok, 3), stride=stride, gt_row_offset=off,
                        frames=frames, times=[float(times[f]) for f in frames],
                        masks=[nearest_mask(root, seq, f) for f in frames]))
    return tuple(out)


# The keys of a clip, in the order the clip list writes them.
CLIP_KEYS = ("name", "seq", "usable", "drop", "stratum", "span_mm", "depth_ok", "stride", "gt_row_offset",
             "frames", "times", "masks")


def gt_rows(clip: dict) -> list[int]:
    """Return the ground truth's row of each frame of a clip of the clip list."""
    return [f + clip["gt_row_offset"] for f in clip["frames"]]


def write_clip_list(clip_list, path: Path) -> None:
    """Write clips as a JSON list, one clip a line, with the keys of `CLIP_KEYS` in order."""
    lines = [json.dumps({k: c[k] for k in CLIP_KEYS}) for c in clip_list]
    Path(path).write_text("[\n" + ",\n".join(lines) + "\n]\n", encoding="utf-8")


def read_clip_list(path: Path) -> tuple[dict, ...]:
    """Return the clips of a clip list, in its order.

    Raises:
        ValueError: the file is not a list of clips with the keys of `CLIP_KEYS`, a name is not
            `<sequence>__clip_NNNN` or is given twice, a clip has not `CLIP_FRAMES` frames, a time and a mask
            for each, its frames are not `stride` apart, its times do not increase, or it is usable while it
            records a reason it is left out, or the reverse.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(c, dict) and tuple(c) == CLIP_KEYS for c in data):
        raise ValueError(f"{path} is not a list of clips with the keys {list(CLIP_KEYS)}, in that order")
    seen = set()
    for c in data:
        name, seq = c["name"], c["seq"]
        if seq not in SEQUENCES or not re.fullmatch(rf"{re.escape(seq)}__clip_\d{{4}}", name) or name in seen:
            raise ValueError(f"{path}: {name!r} of {seq!r} is not <sequence>__clip_NNNN of a sequence, or comes twice")
        seen.add(name)
        frames = c["frames"]
        if len(frames) != CLIP_FRAMES or len(c["times"]) != len(frames) or len(c["masks"]) != len(frames):
            raise ValueError(f"{path}: {name} has {len(frames)} frames, {len(c['times'])} times and "
                             f"{len(c['masks'])} masks, where {CLIP_FRAMES} of each are expected")
        if np.any(np.diff(frames) != c["stride"]) or not np.all(np.diff(c["times"]) > 0):
            raise ValueError(f"{path}: {name}'s frames are not {c['stride']} apart, or its times do not increase")
        if c["usable"] is not (c["drop"] == ""):
            raise ValueError(f"{path}: {name} records usable {c['usable']!r} and the reason {c['drop']!r}; a clip is "
                             "usable, true, exactly when it records no reason")
    return tuple(data)
