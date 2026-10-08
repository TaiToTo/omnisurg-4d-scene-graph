"""The ratio between a clip index's frame numbers and the mp4's.

The frame numbers in `clip_index.json` are not the mp4's frame numbers. In 14
of the 97 videos the annotation numbers frames at a lower rate than the mp4,
and `mp4_frame = native_frame * ratio` corrects it; the measured ratios are
2, 3 and 4.

This is easy to miss. The path that reads the bundled `images/frame_NNNNNN.jpg`
uses the number as a key and nothing goes wrong. Only the path that decodes
the mp4 by frame number drifts, and there it drifts silently: the frame that
comes back is a real frame, from another moment.

How the ratio was measured: for each video, the bundled JPEG of one frame of
its first clip was matched against the mp4 by mean absolute pixel difference,
scanning from the start or seeking to each candidate ratio. A match differs
by 0.7 to 1.9 (the two compressions), a miss by 20 to 190, so there is no
ambiguity. That the ratio holds over the whole video was checked at three
points, early, middle and late, for all 14 videos and 17 controls with ratio
1, with no exception.

Where the ratio comes from: the dataset's extraction script
(`download/process_atlas120k.py` in the ATLAS repository) walks the mp4 and
keeps every `max(1, int(fps / 15))`-th frame, numbering the kept frames from
zero whether or not they fall in the surgical section. The division truncates,
so 60.00 fps gives 4, 59.94 and 50.00 give 3, 30.00 gives 2, and 29.97, 25,
23.98 and 15 give 1. The 14 videos above ratio 1 are exactly those at 30.00
fps or more, and the measured ratios agree with the rule for all 97. The
README's "15 fps" is loose: a 29.97 fps video is kept at its native rate.

The table is still a measurement rather than the rule applied, because the
rule's input is not under our control: the mp4 on disk is whatever the
download produced, not necessarily the file the authors sampled, and the fps
OpenCV reports can fall on either side of the truncation for a video near
30 fps. The rule says which videos to suspect and what to expect; the pixels
say what is.

An unmeasured video is refused, not assumed to be 1. The table lists every
video that was measured, ratio 1 included, so a video missing from it has an
unknown ratio and decoding its mp4 by number may read the wrong moment.

The committed measurement is `atlas120k_meta/frame_ratio.json`.

`docs/figures/atlas120k_frame_ratio.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""

import json
import os

import cv2
import numpy as np

# The smallest mean absolute difference measured between a bundled JPEG and a
# frame that is not it. A tolerance at or above it passes a miss as a match.
MISS_FLOOR = 20.0

# How many mp4 frames either side of the mapped one are read, to absorb seek
# inaccuracy.
WINDOW = 4

# The misses behind `MISS_FLOOR` were measured on frame numbers of 26 and up,
# where a wrong ratio lands at least 26 mp4 frames from the match. A wrong
# ratio moves frame n at least n frames away and the window reads `WINDOW` of
# them back, so below 26 + `WINDOW` the check would compare frames closer than
# any miss was measured at, and in a slow scene those can look the same.
MIN_NATIVE_FRAME = 26 + WINDOW


class FrameRatios:
    """The measured ratios, keyed by `(procedure, video)`."""

    def __init__(self, ratios: dict[tuple[str, str], int], match_tol: float):
        """Hold the ratios and the tolerance, refusing a value that cannot be either.

        Raises:
            ValueError: a ratio is not a positive integer, or `match_tol` is
                not a number between 0 and `MISS_FLOOR`.
        """
        for key, ratio in ratios.items():
            # `bool` is excluded by name: `true` is an `int` to Python and
            # would pass as ratio 1.
            if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
                raise ValueError(f"{'/'.join(key)}: ratio {ratio!r} is not a positive integer")
        # Above this mean absolute difference, a bundled JPEG and the mp4
        # frame it maps to are different frames. Matches sit at 0.7 to 1.9.
        if (isinstance(match_tol, bool) or not isinstance(match_tol, (int, float))
                or not 0 < match_tol < MISS_FLOOR):
            raise ValueError(
                f"match_tol {match_tol!r} does not separate a match from a miss: it "
                f"must be a number above 0 and below {MISS_FLOOR}, the smallest "
                "difference a miss was measured at.")
        self._ratios = dict(ratios)
        self.match_tol = float(match_tol)

    @classmethod
    def load(cls, path: str) -> "FrameRatios":
        """Read the measurement file.

        Args:
            path: a JSON object with `match_tol` and `videos`, each video an
                object with `procedure`, `video` and an integer `ratio`.

        Raises:
            FileNotFoundError: the file is missing. Without it every video is
                unmeasured, and that is reported here, not one video at a time.
            ValueError: a video is listed twice, or a value is refused by the
                constructor.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"no frame-ratio measurement at {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ratios: dict[tuple[str, str], int] = {}
        for row in data["videos"]:
            key = (row["procedure"], row["video"])
            if key in ratios:
                raise ValueError(f"{'/'.join(key)} is measured twice")
            ratios[key] = row["ratio"]
        return cls(ratios, data["match_tol"])

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._ratios

    def __len__(self) -> int:
        return len(self._ratios)

    def videos(self) -> list[tuple[str, str]]:
        """Return the measured videos as `(procedure, video)` pairs, sorted."""
        return sorted(self._ratios)

    def ratio(self, procedure: str, video: str) -> int:
        """The video's `mp4_frame / native_frame`.

        Raises:
            KeyError: the video was not measured. Its ratio is unknown, not 1.
        """
        try:
            return self._ratios[(procedure, video)]
        except KeyError:
            raise KeyError(
                f"{procedure}/{video}: the ratio between its clip index's frame "
                "numbers and the mp4's has not been measured. Decoding the mp4 by "
                "number could read a frame from another moment. Measure it and add "
                "it to the frame-ratio table.") from None

    def mp4_index(self, procedure: str, video: str, native_frame) -> int:
        """Turn a clip-index frame number into the mp4's.

        Args:
            native_frame: the number written in `clip_index.json`, which may be
                a string.
        """
        return int(native_frame) * self.ratio(procedure, video)

    def verify_against_bundled(self, video_path: str, bundled_jpg: str, procedure: str,
                               video: str, native_frame) -> float:
        """Check, on the spot, that the mapped mp4 frame is the bundled JPEG.

        This catches a stale table or a replaced video. The best match within
        `WINDOW` frames either side is taken, to absorb seek inaccuracy: the
        question is whether the frame is there, not exactly where.

        Args:
            native_frame: the clip-index number of the bundled JPEG, at least
                `MIN_NATIVE_FRAME`. Near the start of the video every ratio
                maps to nearly the same mp4 frame, so a wrong one would pass.

        Returns:
            The mean absolute difference of the match.

        Raises:
            ValueError: `native_frame` is below `MIN_NATIVE_FRAME`.
            RuntimeError: the JPEG or the video cannot be opened, no frame
                could be read at the mapped position, or no frame in the
                window is within `match_tol` of the JPEG.
        """
        ratio = self.ratio(procedure, video)
        if int(native_frame) < MIN_NATIVE_FRAME:
            raise ValueError(
                f"{procedure}/{video}: clip-index frame {native_frame} is too close to "
                f"the start of the video to tell ratios apart; verify on frame "
                f"{MIN_NATIVE_FRAME} or later.")
        ref = cv2.imread(bundled_jpg, cv2.IMREAD_COLOR)
        if ref is None:
            raise RuntimeError(f"cannot read the bundled JPEG {bundled_jpg}")
        idx = self.mp4_index(procedure, video, native_frame)

        # The smallest difference from the JPEG over the window around `idx`,
        # and how many frames the window held.
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open the video {video_path}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - WINDOW))
            best, read = float("inf"), 0
            for _ in range(2 * WINDOW + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                a = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
                best = min(best, float(np.abs(a.astype(np.int16) - ref.astype(np.int16)).mean()))
        finally:
            cap.release()

        # Nothing read means the position is past the end. That is not the
        # video's fault alone: a ratio too large puts it there too.
        if read == 0:
            raise RuntimeError(
                f"{procedure}/{video}: clip-index frame {native_frame} maps with ratio "
                f"{ratio} to mp4 frame {idx}, but no frame could be read there "
                f"({n} frames in {os.path.basename(video_path)}). The ratio is too "
                "large, or the video is shorter than the one measured.")
        if best > self.match_tol:
            raise RuntimeError(
                f"frame numbers do not correspond: {procedure}/{video} clip-index "
                f"frame {native_frame} maps with ratio {ratio} to "
                f"mp4 frame {idx}, but the mean absolute difference from "
                f"{os.path.basename(bundled_jpg)} is {best:.1f} (tolerance "
                f"{self.match_tol}, {n} frames in the mp4). The table is stale or "
                "the video was replaced.")
        return best
