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
rounded to integers, cut to the part that lies inside the frame and refused if
that leaves them degenerate. An entry without `src_size` is refused: a
rectangle that cannot be checked against its frame may reach past the left
or top edge, and a negative origin in a numpy slice wraps around to the other
side of the image instead of failing.

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
    """Round a rectangle to integers and keep the part inside the frame.

    Raises:
        ValueError: `src_size` is missing, or a side of what is left is
            shorter than `MIN_SIDE`. A rectangle drawn wrong is not used as
            it is.
    """
    if not src_size:
        raise ValueError(f"{where}: no src_size, so the rectangle cannot be checked against its frame")
    sw, sh = int(src_size[0]), int(src_size[1])
    x, y, w, h = (int(round(float(v))) for v in rect)
    # The intersection with the frame: an edge past the frame is moved to it,
    # and the opposite edge stays where it was drawn.
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(sw, x + w), min(sh, y + h)
    w, h = x1 - x0, y1 - y0
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ValueError(f"degenerate rectangle for {where}: {(x0, y0, w, h)}")
    return (x0, y0, w, h)


def load_clip_rects(path: str) -> dict[tuple[str, str, str], Rect]:
    """Read the judgement file into `(procedure, video, clip) -> rect`.

    If a clip appears more than once, the later entry wins: the tool appends,
    so a correction comes after what it corrects. A later `skip` or
    unjudged entry withdraws the rectangle an earlier entry gave.

    Args:
        path: the judgement JSON, an array of entries.

    Returns:
        The confirmed rectangles. `skip` and unjudged clips are absent.

    Raises:
        FileNotFoundError: the file is missing. An empty table would be
            indistinguishable from "nothing confirmed", and the caller would
            silently crop every clip with the per-video rectangle.
        ValueError: an `ok` or `ng` entry has no `src_size`, or its rectangle
            is degenerate once cut to the frame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no confirmed crop rectangles at {path}")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: dict[tuple[str, str, str], Rect] = {}
    for r in rows:
        key = (r["procedure"], r["video"], r["clip"])
        if r.get("verdict") not in ("ok", "ng"):
            out.pop(key, None)
            continue
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
