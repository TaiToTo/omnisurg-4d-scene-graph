"""Enumerate one evaluation window per CholecSeg8k clip, by a fixed rule.

Each CholecSeg8k clip (a folder of 80 consecutive annotated frames) gets one
window: the seed is the clip's middle annotated frame (the earlier of the
two middle frames), and the window is `samples` frames `stride` CholecSeg8k
numbers apart with the seed at `seed_index`, as in the AE-CAI windows. A
window is excluded when a frame falls before the video's first frame or
past its last, or when fewer than `min_annotated_frames` of its frames
carry annotation. The script also maps each AE-CAI window to the clips it
covers and to the enumerated window it shares most annotated frames with.

Usage:
    python3 -m aecai_rev.windows --videos-root /path/to/cholec80/videos
"""
import argparse
import os
import re

import cv2
import pandas as pd

from aecai_rev.config import load, write_meta

# Native frames per CholecSeg8k number. CholecSeg8k numbers these five
# videos at their own 25 frames a second and the other twelve at about 30;
# each value was read from the AE-CAI windows' manifests (native frame over
# CholecSeg8k number of the annotated frames).
RATE_25 = {1, 9, 12, 20, 35}


def rate(video):
    """Return the native frames per CholecSeg8k number of a video."""
    return 1.0 if video in RATE_25 else 25 / 30


def clips_of(seg8k_root):
    """Return every CholecSeg8k clip as (video, first number, frame count).

    Raises:
        ValueError: A clip folder is not named `video<nn>_<start>`.
    """
    out = []
    videos = [d for d in sorted(os.listdir(seg8k_root)) if re.fullmatch(r"video\d{2}", d)]
    for vdir in videos:
        for cdir in sorted(os.listdir(os.path.join(seg8k_root, vdir))):
            if cdir.startswith("."):
                continue
            m = re.fullmatch(r"video(\d{2})_(\d+)", cdir)
            if not m:
                raise ValueError(f"unexpected CholecSeg8k folder {cdir!r}")
            frames = [f for f in os.listdir(os.path.join(seg8k_root, vdir, cdir))
                      if f.endswith("_endo_color_mask.png")]
            out.append((int(m.group(1)), int(m.group(2)), len(frames)))
    return out


def video_lengths(videos_root, videos):
    """Return each video's frame count, read from the Cholec80 video."""
    out = {}
    for v in videos:
        cap = cv2.VideoCapture(os.path.join(videos_root, f"video{v:02d}.mp4"))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video{v:02d}.mp4 under {videos_root}")
        out[v] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return out


def annotated_numbers(clips):
    """Return the set of annotated CholecSeg8k numbers of each video."""
    out = {}
    for v, start, n in clips:
        out.setdefault(v, set()).update(range(start, start + n))
    return out


def window_numbers(seed, w):
    """Return a window's CholecSeg8k numbers, the seed at `seed_index`."""
    return [seed + w["stride"] * (k - w["seed_index"]) for k in range(w["samples"])]


def enumerate_windows(clips, lengths, w):
    """Define one window per clip and decide whether it is kept.

    Returns:
        (kept, excluded) lists of dict rows.
    """
    ann = annotated_numbers(clips)
    kept, excluded = [], []
    for v, start, n in clips:
        seed = start + (n - 1) // 2
        nums = window_numbers(seed, w)
        n_ann = sum(1 for f in nums if f in ann[v])
        last_native = int(nums[-1] * rate(v))
        row = dict(video_id=f"VID{v:02d}", clip_id=f"video{v:02d}_{start:05d}",
                   seed_frame=seed, first_frame=nums[0], last_frame=nums[-1],
                   n_annotated=n_ann, frame_list=" ".join(map(str, nums)),
                   clip=f"VID{v:02d}_s{w['stride']}_{nums[0]}_crop")
        reasons = []
        if nums[0] < 0:
            reasons.append("starts before the first frame of the video")
        if last_native >= lengths[v]:
            reasons.append("ends past the last frame of the video")
        if n_ann < w["min_annotated_frames"]:
            reasons.append(f"fewer than {w['min_annotated_frames']} annotated frames")
        if reasons:
            excluded.append(dict(row, reason="; ".join(reasons)))
        else:
            kept.append(row)
    for i, row in enumerate(kept):
        row["window_id"] = f"W{i:03d}"
    return kept, excluded


def legacy_map(legacy, kept, clips, w):
    """Map each AE-CAI window to the clips it covers and to enumerated windows.

    An AE-CAI window `VID<nn>_s15_<start>_crop` holds the numbers `start`,
    `start + stride`, ...; its seed is at `seed_index`. Its frames and an
    enumerated window's frames lie on different phases of the stride, so
    they share clips, not frames. The map lists the enumerated windows
    seeded in a clip the AE-CAI window scores, and names the one whose seed
    is nearest to the AE-CAI seed.
    """
    ann = annotated_numbers(clips)
    by_clip = {row["clip_id"]: row for row in kept}
    rows = []
    for name in legacy:
        m = re.fullmatch(r"VID(\d{2})_s(\d+)_(\d+)_crop", name)
        v, stride, start = int(m.group(1)), int(m.group(2)), int(m.group(3))
        nums = [start + stride * k for k in range(w["samples"])]
        seed = nums[w["seed_index"]]
        scored = {f for f in nums if f in ann[v]}
        covered = sorted({f"video{v:02d}_{s:05d}" for vv, s, n in clips
                          if vv == v and any(s <= f < s + n for f in scored)})
        same = [by_clip[c] for c in covered if c in by_clip]
        near = min(same, key=lambda r: abs(r["seed_frame"] - seed)) if same else None
        rows.append(dict(legacy_clip=name, legacy_seed_frame=seed,
                         legacy_n_annotated=len(scored), clips_covered=" ".join(covered),
                         windows_in_covered_clips=" ".join(r["window_id"] for r in same),
                         nearest_window_id=near["window_id"] if near else None,
                         nearest_seed_frame=near["seed_frame"] if near else None,
                         seed_distance=abs(near["seed_frame"] - seed) if near else None))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos-root", required=True, help="cholec80 videos/ directory")
    args = ap.parse_args()

    # Enumerate: one window per CholecSeg8k clip, kept or excluded.
    cfg = load()
    w = cfg["windows"]
    clips = clips_of(cfg["paths"]["cholecseg8k"])
    lengths = video_lengths(args.videos_root, sorted({v for v, _, _ in clips}))
    kept, excluded = enumerate_windows(clips, lengths, w)

    # Write: the windows, the exclusions and the map from the AE-CAI windows.
    out = os.path.join(cfg["paths"]["results_root"], "00_windows")
    os.makedirs(out, exist_ok=True)
    cols = ["window_id", "video_id", "clip_id", "seed_frame", "frame_list", "n_annotated",
            "first_frame", "last_frame", "clip"]
    pd.DataFrame(kept)[cols].to_csv(os.path.join(out, "windows.csv"), index=False)
    pd.DataFrame(excluded, columns=cols[1:] + ["reason"]).to_csv(
        os.path.join(out, "excluded.csv"), index=False)
    pd.DataFrame(legacy_map(w["legacy27"], kept, clips, w)).to_csv(
        os.path.join(out, "legacy27_map.csv"), index=False)
    write_meta(out, cfg, what="window enumeration", n_clips=len(clips), n_kept=len(kept),
               n_excluded=len(excluded), video_lengths=lengths,
               rate_25_videos=sorted(RATE_25))
    print(f"{len(clips)} clips: {len(kept)} windows kept, {len(excluded)} excluded -> {out}")


if __name__ == "__main__":
    main()
