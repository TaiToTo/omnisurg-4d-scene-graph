"""Read StereoMIS: the stereo calibration, rectified frames, camera poses and instrument masks.

A sequence's video stacks two views, the left one on top. A frame is named by
its 0-based position in the video, the frame FFmpeg's `select=eq(n,N)` decodes.
The ground truth's row of a frame, and the file number of its mask or depth
map, are offset from that position by amounts measured once per sequence and
kept in the tables below. Masks and depth maps are in the rectified left view
at half resolution. The ground truth holds camera-to-world poses in m, and
`load_gt` returns them in mm. `clips` cuts a sequence into the fixed clips the
camera trajectory result is measured on, and marks the clips whose inputs
cannot be measured. Every function takes the dataset's root as an argument.
"""

import configparser
import functools
import json
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

# The file number of a frame's mask and depth map, less the frame's position. Measured by warping the right view
# onto the left with the depth. P2_3's camera never moves, so its value is its neighbours'.
DEPTH_FILE_OFFSET = {"P1": 0, "P2_0": -1, "P2_1": -1, "P2_2": -1, "P2_3": -1,
                     "P2_4": -1, "P2_5": -1, "P2_6": -1, "P2_7": -1, "P2_8": -1}

# The ground truth's row of a frame, less the frame's position. Moving depth between frames with the poses reads it
# on seven sequences: their best offsets lie from -6 to -2, and -4 is at most 6.5 % worse than each sequence's best.
GT_ROW_OFFSET = {s: -4 for s in SEQUENCES}

# The ground truth's translations are in m; the depth maps are in mm.
GT_TO_MM = 1000.0

# A clip is 112 frames 0.2 s apart, 22.4 s, and clips do not overlap. They are fixed before anything is measured.
CLIP_FRAMES = 112
CLIP_INTERVAL_S = 0.2

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
        R = np.array([g("StereoRight", f"R_{i}") for i in range(9)]).reshape(3, 3)
        # A column vector, which OpenCV 5 needs: its stereoRectify refuses a 1-D translation.
        T = np.array([g("StereoRight", f"T_{i}") for i in range(3)]).reshape(3, 1)
        self._Kl, self._dl = intrinsics("StereoLeft"), distortion("StereoLeft")
        self._Kr, self._dr = intrinsics("StereoRight"), distortion("StereoRight")
        # With alpha 0 the rectified views keep no black border. P1's shipped frames (`video_frames/`) are rectified
        # this way: the views here differ from them by 1.5 to 2.2 grey levels on average.
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
        ValueError: the calibration's view is not the video's.
        RuntimeError: the video ends before every frame wanted is decoded.
    """
    want = sorted({int(f) for f in frames})
    if not want:
        return
    info = video_info(root, seq)
    if want[0] < 0 or want[-1] >= info["n_frames"]:
        raise IndexError(f"{seq}: positions {want[0]}-{want[-1]} are outside 0-{info['n_frames'] - 1}")
    c = calib(root, seq)
    if c.size != VIEW_SIZE:
        raise ValueError(f"{seq}: the calibration is for views of {c.size}, the video's are {VIEW_SIZE}")
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
    coordinates is `R @ p + t` in world coordinates. A row is not a frame's position; `gt_row` maps one to the
    other.

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
    return R, a[:, 1:4] * GT_TO_MM


def gt_row(seq: str, frame: int) -> int:
    """Return the ground truth's row of the frame at a position, by the measured offset only.

    Raises:
        KeyError: the sequence has no measured offset.
    """
    if seq not in GT_ROW_OFFSET:
        raise KeyError(f"{seq}: the offset between its frames and the ground truth's rows has not been measured")
    return frame + GT_ROW_OFFSET[seq]


def gt_pose(root: Path, seq: str, frames) -> np.ndarray:
    """Return the (N, 4, 4) camera-to-world poses of the frames at the given positions, translations in mm.

    Raises:
        ValueError: no frame is given.
        IndexError: a frame's row is outside the ground truth. The pose at the end is not used in its place.
    """
    R, t = load_gt(root, seq)
    rows = np.asarray([gt_row(seq, int(f)) for f in frames])
    if rows.size == 0:
        raise ValueError(f"{seq}: no frame given")
    if rows.min() < 0 or rows.max() >= len(R):
        raise IndexError(f"{seq}: ground truth rows {rows.min()}-{rows.max()} are outside 0-{len(R) - 1}")
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = R[rows]
    T[:, :3, 3] = t[rows]
    return T


def load_mask(root: Path, seq: str, frame: int) -> np.ndarray | None:
    """Return the frame's tissue mask (True on tissue, False on instruments and outside the view), or None.

    The dataset's mask is 255 on tissue and 0 elsewhere. A frame without a mask file gets None.

    Raises:
        ValueError: the mask file cannot be read, or is not of `HALF_SIZE`.
    """
    path = Path(root) / seq / "masks" / ("%06dl.png" % (frame - DEPTH_FILE_OFFSET[seq]))
    if not path.exists():
        return None
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError(f"{path} exists but cannot be read")
    if a.shape[::-1] != HALF_SIZE:
        raise ValueError(f"{path} is {a.shape[1]}x{a.shape[0]}, the masks are {HALF_SIZE[0]}x{HALF_SIZE[1]}")
    return a > 127


def load_mask_nearest(root: Path, seq: str, frame: int, tol: int = 2) -> np.ndarray | None:
    """Return the tissue mask of the frame nearest `frame` that has one, within `tol` frames, or None.

    The masks are on every second frame, on every frame for P2_6 to P2_8, and on odd numbers only in some
    sequences, so a clip's frame often has none. Instruments move little in one or two frames (17 to 33 ms).
    Of two frames equally near, the earlier one is taken.
    """
    for d in range(tol + 1):
        for f in ((frame,) if d == 0 else (frame - d, frame + d)):
            m = load_mask(root, seq, f)
            if m is not None:
                return m
    return None


@functools.lru_cache(maxsize=None)
def gt_jump_rows(root: Path, seq: str) -> np.ndarray:
    """Return each row `i` of the ground truth such that the step from row `i` to row `i + 1` is a break.

    A step is a break when it exceeds `GT_JUMP_MM` or `GT_JUMP_DEG`.
    """
    R, t = load_gt(root, seq)
    d = np.linalg.norm(np.diff(t, axis=0), axis=1)
    dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return np.where((d > GT_JUMP_MM) | (ang > GT_JUMP_DEG))[0]


def has_gt_jump(root: Path, seq: str, lo: int, hi: int) -> bool:
    """Return whether the ground truth breaks between the rows of positions `lo` and `hi`, or at either end."""
    off = GT_ROW_OFFSET[seq]
    j = gt_jump_rows(root, seq)
    return bool(np.any((j >= lo + off - 1) & (j <= hi + off)))


def clip_stride(root: Path, seq: str) -> int:
    """Return the step between a clip's frames that comes nearest `CLIP_INTERVAL_S`: 12 at 60 frames a second."""
    return max(1, int(round(video_info(root, seq)["fps"] * CLIP_INTERVAL_S)))


def load_depth_stats(depth_root: Path, seq: str) -> np.ndarray:
    """Return the depth export's `stats.npy`: one row per depth map, its file number, median in mm and valid share.

    Raises:
        ValueError: the file holds no row, or fewer than 3 columns. With no row, every clip would be left out
            as showing no surface.
    """
    st = np.load(Path(depth_root) / seq / "stats.npy")
    if st.ndim != 2 or st.shape[0] == 0 or st.shape[1] < 3:
        raise ValueError(f"{seq}: stats.npy has shape {st.shape}, one row of at least 3 columns per map is expected")
    return st


def clips(root: Path, depth_root: Path, seq: str) -> tuple[dict, ...]:
    """Cut a sequence into fixed clips that do not overlap, and mark those its inputs cannot be measured on.

    A clip is left out for its inputs only, never for a score, by the first of these that holds:

    - `no_surface`: fewer than `CLIP_MIN_DEPTH_OK` of its depth maps show a surface;
    - `gt_jump`: the ground truth breaks inside it;
    - `gt_static`: the true camera's RMS radius is below `CLIP_MIN_SPAN_MM`.

    The ground truth's rows lag the positions, so the grid starts at the first position that has a row. A clip
    whose rows run past the ground truth's end is not returned.

    Args:
        root: the dataset's root.
        depth_root: the depth export's root, which holds `<seq>/stats.npy`.
        seq: the sequence.

    Returns:
        One dict per clip: `name`, `seq`, `frames` (positions), `times` (s), `stride`, `span_mm`, `depth_ok`,
        `usable`, `drop` (the reason, or "") and `stratum`.
    """
    info = video_info(root, seq)
    stride = clip_stride(root, seq)
    span = CLIP_FRAMES * stride
    off = GT_ROW_OFFSET[seq]
    st = load_depth_stats(depth_root, seq)
    d_ok = {int(r[0]) + DEPTH_FILE_OFFSET[seq]:
            bool(DEPTH_OK_MM[0] <= r[1] <= DEPTH_OK_MM[1] and r[2] > DEPTH_OK_VALID)
            for r in st}
    _, tw = load_gt(root, seq)
    n_gt = len(tw)
    out = []
    first = max(0, -off)
    for k, start in enumerate(range(first, info["n_frames"] - span + 1, span)):
        frames = list(range(start, start + span, stride))
        rows = [f + off for f in frames]
        if min(rows) < 0 or max(rows) >= n_gt:
            continue
        inside = [v for f, v in d_ok.items() if start <= f < start + span]
        ok = float(np.mean(inside)) if inside else 0.0
        c = tw[np.array(rows)]
        sp = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))
        drop = ("no_surface" if ok < CLIP_MIN_DEPTH_OK else
                "gt_jump" if has_gt_jump(root, seq, frames[0], frames[-1]) else
                "gt_static" if sp < CLIP_MIN_SPAN_MM else "")
        out.append(dict(name=f"{seq}__clip_{k:04d}", seq=seq, frames=frames,
                        times=[f / info["fps"] for f in frames], stride=stride,
                        span_mm=round(sp, 3), depth_ok=round(ok, 3),
                        usable=not drop, drop=drop,
                        stratum="moving" if sp >= CLIP_MOVING_SPAN_MM else "slow"))
    return tuple(out)
