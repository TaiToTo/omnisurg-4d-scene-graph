"""The crop rectangle of each clip, as a person confirmed it.

Why a rectangle per clip and not per video: the automatic recipe estimates one
rectangle per video from a sample of its frames, and when all 494 clips of the
97 videos were checked by eye, that rectangle was wrong for 305 of them, and
in seven videos the rectangle changes within one mp4 because the recording
conditions switch mid-video. Neither fits a per-video rectangle, so each clip
was judged and the file is that judgement.

The file is what the judging tool writes, one entry per clip:

    {"procedure": ..., "video": ..., "clip": "clip_0001",
     "rect": [x, y, w, h], "verdict": "ok" | "ng" | "skip", ...}

`ok` means the recipe's rectangle was accepted, `ng` that a person redrew it;
either way `rect` is the rectangle to use, so the two are read alike. A clip
marked `skip` (not to be used) or not yet judged is not returned, and the
caller falls back to its per-video rectangle.

Rectangles are in the source video's pixels (`src_size`). On loading they are
rounded to integers, clamped into the frame and refused if that leaves them
degenerate.

The committed judgement is `atlas120k_meta/crop_rects.json`.

`docs/figures/atlas120k_clip_rects.png` shows this on a drawn frame.
"""

import json
import os

# A side shorter than this is a slip of the mouse, not a crop. The judging
# tool enforces the same floor.
MIN_SIDE = 40

Rect = tuple[int, int, int, int]


def _clean(rect: list, src_size: list | None, where: str) -> Rect:
    """Round a rectangle to integers and clamp it into the frame.

    Raises:
        ValueError: a side is shorter than `MIN_SIDE`. A rectangle drawn wrong
            is not used as it is.
    """
    x, y, w, h = (int(round(float(v))) for v in rect)
    if src_size:
        sw, sh = int(src_size[0]), int(src_size[1])
        x, y = max(0, min(x, sw - 1)), max(0, min(y, sh - 1))
        w, h = min(w, sw - x), min(h, sh - y)
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ValueError(f"degenerate rectangle for {where}: {(x, y, w, h)}")
    return (x, y, w, h)


def load_clip_rects(path: str) -> dict[tuple[str, str, str], Rect]:
    """Read the judgement file into `(procedure, video, clip) -> rect`.

    If a clip appears more than once, the later entry wins: the tool appends,
    so a correction comes after what it corrects.

    Args:
        path: the judgement JSON, an array of entries.

    Returns:
        The confirmed rectangles. `skip` and unjudged clips are absent.

    Raises:
        FileNotFoundError: the file is missing. An empty table would be
            indistinguishable from "nothing confirmed", and the caller would
            silently crop every clip with the per-video rectangle.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no confirmed crop rectangles at {path}")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: dict[tuple[str, str, str], Rect] = {}
    for r in rows:
        if r.get("verdict") not in ("ok", "ng"):
            continue
        key = (r["procedure"], r["video"], r["clip"])
        out[key] = _clean(r["rect"], r.get("src_size"), "/".join(key))
    return out


def rect_for(table: dict, procedure: str, video: str, clip: str,
             default: Rect) -> tuple[Rect, str]:
    """The rectangle to crop a clip with, and where it came from.

    Args:
        table: what `load_clip_rects` returned. Empty means always `default`.
        procedure: the procedure directory name.
        video: the video id.
        clip: the clip name in the clip index, such as `clip_0001`.
        default: the per-video rectangle to fall back on.

    Returns:
        `(rect, "confirmed" | "video")`.
    """
    r = table.get((procedure, video, clip))
    return (r, "confirmed") if r else (default, "video")
