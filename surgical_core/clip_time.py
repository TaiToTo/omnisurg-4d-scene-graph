"""The real time of every frame of a clip, from its `frame_manifest.json`.

Both sides of the toolkit need it: the tracking metrics, which pair frames by
how far apart in time they are, and the camera-motion analysis. It belongs to
neither, because the rule that turns a manifest into seconds has exceptions
that are easy to get right in one copy and wrong in the other: CholecSeg8k
has clips whose time does not advance with the frame number, and ATLAS-120k
has clips whose frame numbers mean nothing until a measured ratio is applied.
Two copies of that rule would drift, and the drift would show as a plausible
number, not as a crash.
"""
from __future__ import annotations

import importlib
import json
import os

import numpy as np

# The module that holds the measured ratio of mp4 frame numbers to clip-index
# frame numbers for each ATLAS-120k video. It is imported by name, only when a
# manifest does not record its own ratio: a manifest that does never needs
# the table, and a clip that needs a table that is not there is refused
# rather than passed at a default.
MEASURED_RATIOS = "surgical_core.atlas120k.frame_ratio"


def frame_times(root: str, clip: str) -> np.ndarray:
    """Return the real time of every frame, in seconds, as an (N,) array.

    The times are relative to the clip; only their differences carry meaning.

    A CholecSeg8k clip is not an evenly spaced series. Of the 27 clips of the
    pilot population, 11 have times that do not advance with the frame number.
    The `native_frame` of `VID26_s15_1855_crop`, for example, runs

        1855 1868 1880 ... 1980 | 2020 2035 2050 2065 2080 | 2055 2068 2080 ...

    Steps of 12 and 13 (numbers converted to 25 fps) mix with steps of 15 (not
    converted); there are jumps of 40, 68 and 80 and reversals of -25 and -53;
    and some clips list the same `native_frame` twice (2080 above). That is
    what joining CholecSeg8k's annotation chunks without reordering them
    gives. So no single frame rate describes a clip, and a caller must take
    time differences between the values returned here, pair by pair, never
    from frame indices. Nothing here sorts or repairs the times either: a
    caller that wants the pairs at a given lag selects them by these values.

    ATLAS-120k's `native_frame` is the clip index's number, and in some videos
    the annotation numbers frames at a lower rate than the mp4: the measured
    ratio is 2, 3 or 4. The manifest records it as `frame_ratio`, and
    forgetting to multiply shrinks that clip's time by the ratio. In the
    paper's population of 315 clips, 40 have a ratio other than 1; one 11.5 s
    clip came out as 3.8 s before the ratio was applied.

    Args:
        root: Directory that holds the clips.
        clip: Name of the clip under `root`.

    Returns:
        Seconds, shape (N,): for CholecSeg8k the manifest's `timestamp_sec`,
        for ATLAS-120k `native_frame * frame_ratio / fps_native` (its frames
        are an even stride, with no reversals). When every frame has a
        `timestamp_sec` it is taken as it is, and neither `frame_ratio` nor
        `gt_step_sec_actual` is looked at: an extractor that writes
        timestamps is trusted to have applied the ratio itself.

    Raises:
        RuntimeError: The manifest gives no way to make seconds (there is no
            default), the ratio cannot be established, or the step after
            multiplying does not match the manifest's `gt_step_sec_actual`
            (a forgotten or wrong ratio).
    """
    path = os.path.join(root, clip, "frame_manifest.json")
    with open(path) as f:
        man = json.load(f)
    frames = man.get("frames", [])

    ts = [fr.get("timestamp_sec") for fr in frames]
    if len(ts) >= 2 and all(t is not None for t in ts):
        return np.asarray(ts, dtype=float)

    nf = [fr.get("native_frame") for fr in frames]
    fps = man.get("fps_native")
    if fps and len(nf) >= 2 and all(v is not None for v in nf):
        ratio = _ratio_of(clip, man)
        t = np.asarray(nf, dtype=float) * ratio / float(fps)
        # Only a population that records `gt_step_sec_actual` can be checked.
        # ATLAS-120k frames are an even stride, so the median step must match.
        want = man.get("gt_step_sec_actual")
        if want and len(t) > 1:
            got = float(np.median(np.diff(np.sort(t))))
            if abs(got - float(want)) > 1e-3:
                raise RuntimeError(
                    f"{clip}: the frame step is {got:.4f} s but the manifest says "
                    f"{float(want):.4f} s (frame_ratio={ratio}). A clip whose "
                    "seconds cannot be made is not mixed in.")
        return t

    raise RuntimeError(
        f"{clip}: cannot make frame times from {path}: neither timestamp_sec nor "
        "native_frame with fps_native. A clip whose seconds cannot be made is "
        "not mixed in.")


def _ratio_of(clip: str, man: dict) -> float:
    """Return the ratio to multiply `native_frame` by, from the manifest or the table.

    A manifest without `frame_ratio` is one of two things: a clip that has no
    ratio at all (CholecSeg8k, or a manifest that names no video), or an
    ATLAS-120k clip extracted before the ratio was recorded. The first kind
    passes at 1. The second kind is checked against the table of measured
    ratios, and refused when the table says the ratio is not 1: a default of 1
    would let that clip in with its time shrunk to a half, a third or a
    quarter, and nothing downstream would notice.

    Args:
        clip: Name of the clip, for the message when it is refused.
        man: The parsed `frame_manifest.json`.

    Returns:
        The ratio.

    Raises:
        RuntimeError: The manifest names a video and records no ratio, and
            either the table says the ratio is not 1 or there is no table to
            ask.
    """
    if man.get("frame_ratio") is not None:
        return float(man["frame_ratio"])

    procedure, youtube_id = man.get("procedure"), man.get("youtube_id")
    if not (procedure and youtube_id):
        return 1.0      # no video named: not ATLAS-120k, and nothing to multiply by

    try:
        table = importlib.import_module(MEASURED_RATIOS)
    except ModuleNotFoundError as e:
        # Only the table itself, or a package above it, being absent means
        # "no table". A dependency missing inside the table (`e.name` is then
        # that dependency) is a broken install, and saying "no table" about it
        # would send the reader to the wrong place.
        if not MEASURED_RATIOS.startswith(e.name):
            raise
        raise RuntimeError(
            f"{clip}: the manifest has no `frame_ratio`, and the table of measured "
            f"ratios ({MEASURED_RATIOS}) is not available, so the ratio is "
            "unknown. Re-extract the clip, or write the ratio into the manifest.") from e
    measured = table.frame_ratio(procedure, youtube_id)
    if measured != 1:
        raise RuntimeError(
            f"{clip}: the manifest has no `frame_ratio`, but the measured ratio of "
            f"this video is {measured}. A default of 1 would shrink its seconds to "
            f"1/{measured}. The clip was extracted before the ratio was recorded: "
            "re-extract it, or write the ratio into the manifest.")
    return 1.0
