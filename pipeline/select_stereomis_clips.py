"""Write the StereoMIS clip list: every clip of every sequence, with its frames, times and masks, usable or not.

`surgical_core.stereomis.clips` cuts each sequence and marks the clips its
inputs cannot be measured on; `stereomis_meta/README.md` gives the rule. The
step reads the video, the ground truth and the masks of each sequence, and
the depth export's `stats.npy`, which nothing else reads. It writes the list
only when every sequence was cut, and reads it back through
`stereomis.read_clip_list` before replacing `--out`, so a list that the
measurements would refuse is not written.

Usage:
    python -m pipeline.select_stereomis_clips --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \\
        --out stereomis_meta/clips.json
"""

import argparse
from collections import Counter
from pathlib import Path

import surgical_core.stereomis as stereomis


def main(argv: list[str] | None = None) -> None:
    """Cut every sequence into clips, check the list, and write it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--depth-root", required=True, help="The depth export, which holds <sequence>/stats.npy.")
    ap.add_argument("--out", required=True, help="The clip list to write, as JSON.")
    args = ap.parse_args(argv)

    # Cut every sequence, and write the list beside `--out`.
    root, depth_root, out = Path(args.root), Path(args.depth_root), Path(args.out)
    clip_list = [c for seq in stereomis.SEQUENCES for c in stereomis.clips(root, depth_root, seq)]
    partial = out.with_name(out.name + ".partial")
    stereomis.write_clip_list(clip_list, partial)

    # Read it back as the measurements will, and move it into place.
    try:
        stereomis.read_clip_list(partial)
    except ValueError as e:
        partial.unlink()
        raise SystemExit(f"ValueError: {e}") from e
    partial.replace(out)

    for seq in stereomis.SEQUENCES:
        here = [c for c in clip_list if c["seq"] == seq]
        reasons = Counter(c["drop"] for c in here if not c["usable"])
        print(f"{seq:5s} {len(here):3d} clips, {sum(c['usable'] for c in here):3d} usable"
              + "".join(f", {n} {r}" for r, n in sorted(reasons.items())))
    print(f"{sum(c['usable'] for c in clip_list)} usable of {len(clip_list)} clips -> {out}")


if __name__ == "__main__":
    main()
