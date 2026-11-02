"""Score the depth of DA3, Pi3X and MapAnything on StereoMIS against the stereo depth, and print the AbsRel rows.

Each model's depth is resized to the stereo depth's 640 x 512 (nearest), and each frame is scored on the pixels
where the stereo depth and every model have a depth, instrument pixels included. Each answer is scaled by the
ratio of the medians over those pixels before AbsRel is taken. `flat`, one depth for the whole frame, is the
baseline. A clip's AbsRel is the mean over its frames. The rows are means over the clips kept, with intervals
over sequences, and paired differences with marks (`stereomis_summary`).

Usage:
    python paper/ipcai2027/geometry/stereomis_depth.py --clips stereomis_meta/clips.json --clip-root /path/to/clips \\
        --stereo-depth /path/to/stereo_depth --exclude P2_6__clip_0007 P2_7__clip_0001 --out depth.json [--jobs 12]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import surgical_core.stereomis as stereomis
from stereomis_summary import diff_row, interval, kept_clips, mean_row, sequence_of, write_json

# Each model's bundle in a clip.
BUNDLES = {"da3": "exports/mini_npz/results.npz", "pi3x": "exports/mini_npz/results__pi3x.npz",
           "mapanything": "exports/mini_npz/results__mapanything.npz"}
ANSWERS = (*BUNDLES, "flat")
# The fewest pixels a frame needs on the common set to be scored.
MIN_PIXELS = 2000
# The paired differences the paper reports, `a - b`; less is better.
PAIRS = (("pi3x", "flat"), ("da3", "flat"), ("mapanything", "flat"), ("pi3x", "da3"))


def absrel(pred: np.ndarray, ref: np.ndarray) -> float:
    """Return AbsRel of `pred` against `ref` after scaling `pred` by the ratio of the medians."""
    p = pred * (np.median(ref) / np.median(pred))
    return float(np.mean(np.abs(p - ref) / ref))


def measure_clip(job: tuple) -> dict:
    """Return a clip's AbsRel per answer, its frames scored and the share of a frame the common set covers.

    Raises:
        ValueError: a model's bundle holds another number of frames than the stereo depth, or no frame is scored.
    """
    name, clip_root, stereo_depth = job
    with np.load(stereo_depth / f"{name}.npz") as z:
        ref = z["depth_dmm"].astype(np.float32) / 10.0
    n, h, w = ref.shape

    # Read each model's depth at the stereo depth's size; `flat` is one depth everywhere.
    depth = {}
    for model, bundle in BUNDLES.items():
        with np.load(clip_root / name / bundle) as z:
            d = z["depth"].astype(np.float32)
        if len(d) != n:
            raise ValueError(f"{name} {model}: {len(d)} depth frames, the stereo depth has {n}")
        depth[model] = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in d])
    depth["flat"] = np.ones_like(ref)

    # Score each frame on the pixels every model and the stereo depth hold.
    scores, cover = {a: [] for a in ANSWERS}, []
    for i in range(n):
        ok = ref[i] > 0
        for model in BUNDLES:
            ok &= np.isfinite(depth[model][i]) & (depth[model][i] > 0)
        cover.append(float(ok.mean()))
        if ok.sum() >= MIN_PIXELS:
            for a in ANSWERS:
                scores[a].append(absrel(depth[a][i][ok], ref[i][ok]))
    if not scores["flat"]:
        raise ValueError(f"{name}: no frame holds {MIN_PIXELS} pixels on the common set")
    return dict(clip=name, n_frames=n, n_scored=len(scores["flat"]), cover=float(np.median(cover)),
                absrel={a: float(np.mean(v)) for a, v in scores.items()})


def summarize(records: list[dict]) -> dict:
    """Return each answer's mean with its interval, and the paired differences with marks."""
    clips = [r["clip"] for r in records]
    value = {a: [r["absrel"][a] for r in records] for a in ANSWERS}
    return dict(n_clips=len(clips), n_sequences=len({sequence_of(c) for c in clips}),
                frames_scored=sum(r["n_scored"] for r in records), frames=sum(r["n_frames"] for r in records),
                cover_median=round(float(np.median([r["cover"] for r in records])), 3),
                absrel={a: mean_row(value[a], clips) for a in ANSWERS},
                differences={f"{a} - {b}": diff_row(value[a], value[b], clips, -1) for a, b in PAIRS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    ap.add_argument("--clip-root", required=True, help="The clips the depth stages and MapAnything wrote into.")
    ap.add_argument("--stereo-depth", required=True, help="The directory `stereo_depth.py` wrote.")
    ap.add_argument("--exclude", nargs="*", default=[], help="The clips `excluded_clips.py` prints.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    ap.add_argument("--jobs", type=int, default=1, help="Clips measured at once.")
    args = ap.parse_args()

    # Measure every usable clip but the excluded ones.
    names = [c["name"] for c in stereomis.usable_clips(stereomis.read_clip_list(Path(args.clips)))]
    try:
        names = kept_clips(names, args.exclude)
    except ValueError as e:
        raise SystemExit(str(e))
    jobs = [(n, Path(args.clip_root), Path(args.stereo_depth)) for n in names]
    with ProcessPoolExecutor(args.jobs) as ex:
        records = list(ex.map(measure_clip, jobs))

    # Summarize, print the rows and write them.
    result = dict(excluded=sorted(args.exclude), min_pixels=MIN_PIXELS, **summarize(records))
    print(f"StereoMIS AbsRel against stereo depth: {result['n_clips']} clips, {result['n_sequences']} sequences, "
          f"{result['frames_scored']} of {result['frames']} frames, common set {result['cover_median']:.2f} of a "
          "frame (median)")
    for a, row in result["absrel"].items():
        print(f"  {a:12s} {interval(row, 'mean')}")
    for pair, row in result["differences"].items():
        print(f"  {pair:20s} {interval(row, 'delta')}")
    write_json(args.out, result)


if __name__ == "__main__":
    main()
