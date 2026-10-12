"""Score the front/behind relation on D4D: for each pair of regions, whether an answer names the nearer one.

For each side of the population, the tool cuts the frame nearest to the scan into regions (`--regions`) and
answers, for each pair of regions, which one is nearer (`--answer`). The truth is each region's median depth
in the scan, seen from the endoscope's pose. A region is scored when it holds `MIN_NODE_PIX` pixels free of
instruments, of which the scan reaches `MIN_GT_PIX`. A pair whose truths or answers tie is left out. The tool
writes each side's pairs and the share answered right, with bootstrap intervals over clips, sessions and
specimens, and `front_behind_code_sha`. It refuses a side it cannot score, depth made for another frame, and
seed regions whose record names another depth.

Usage:
    python -m evalkit.tools.d4d_front_behind --root /path/to/D4D/extracted --population d4d_meta/population.json \\
        --answer mono --regions ours --depth-dir /path/to/d4d_depth --seed-dir /path/to/d4d_seed --out mono.json \\
        [--answer-pixels all] [--jobs 8]
"""

import argparse
import hashlib
import json
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import surgical_core.d4d as d4d

ANSWERS = ("area", "gt_half", "screen", "stereo", "mono")
REGIONS = ("voronoi", "kmeans", "ours")
ANSWER_PIXELS = ("all", "gt")
UNITS = ("clip", "session", "specimen")

# The scan's pixels a region needs to be scored. The scan is sparser than the pixels, so a median of a few
# points would decide a region.
MIN_GT_PIX = 200
# The pixels free of instruments a region needs to be scored, at the endoscope's resolution.
MIN_NODE_PIX = 400
# The number of regions of the `voronoi` and `kmeans` cuts, and the seed of their random draws.
VORONOI_K = 10
VORONOI_SEED = 0
# The seed of the random half of the scan that `gt_half` answers from.
GT_HALF_SEED = 1
N_BOOT = 2000
BOOT_SEED = 0
# The seed records' fields that differ from side to side, left out when the settings are collected.
PER_SIDE_FIELDS = ("depth_file", "depth_sha256", "frame_index", "n_regions")


def front_behind_code_sha() -> str:
    """Return the sha256 hex digest of this module and `surgical_core.d4d`, each name and content length-prefixed."""
    h = hashlib.sha256()
    for path in (Path(__file__), Path(d4d.__file__)):
        for part in (path.name.encode("utf-8"), path.read_bytes()):
            h.update(struct.pack(">Q", len(part)))
            h.update(part)
    return h.hexdigest()


def voronoi_labels(h: int, w: int, k: int = VORONOI_K, seed: int = VORONOI_SEED) -> np.ndarray:
    """Return the same `k` Voronoi cells of the view for every frame, whatever the frame shows."""
    rng = np.random.RandomState(seed)
    cx = rng.uniform(0.05, 0.95, k) * w
    cy = rng.uniform(0.05, 0.95, k) * h
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2
    return np.argmin(d, axis=0).astype(np.int32)


def kmeans_labels(image: np.ndarray, k: int = VORONOI_K, seed: int = VORONOI_SEED) -> np.ndarray:
    """Return `k` regions of a BGR frame by k-means on colour and position, so their areas follow the frame."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # Weigh position so that a step of one region's side, sqrt(hw / k), counts as 20 levels of colour.
    scale = 20.0 / np.sqrt(h * w / k)
    feat = np.concatenate([lab.reshape(-1, 3), (xx.reshape(-1, 1) * scale).astype(np.float32),
                           (yy.reshape(-1, 1) * scale).astype(np.float32)], axis=1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(seed)
    _, labels, _ = cv2.kmeans(feat, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(h, w).astype(np.int32)


def node_stats(labels: np.ndarray, valid: np.ndarray, gt_depth: np.ndarray, answer_depth: np.ndarray | None,
               answer_pixels: str = "all") -> dict:
    """Return each scored region's area, median scan depth and, with an answer's depth, its median.

    Args:
        labels: (H, W) region ids; -1 is no region.
        valid: (H, W) true on a pixel free of instruments.
        gt_depth: (H, W) the scan's depth, NaN where the scan does not reach.
        answer_depth: (H, W) the answer's depth, or None for an answer by area.
        answer_pixels: `all` takes the answer's median over the region's pixels; `gt` over those the scan
            reaches. A region whose answer has fewer than a quarter of `MIN_NODE_PIX` usable pixels is not scored.
    """
    out = {}
    for lid in np.unique(labels):
        if lid < 0:
            continue
        m = (labels == lid) & valid
        if m.sum() < MIN_NODE_PIX:
            continue
        gsel = m & np.isfinite(gt_depth)
        g = gt_depth[gsel]
        if len(g) < MIN_GT_PIX:
            continue
        rec = dict(area=int(m.sum()), gt=float(np.median(g)), n_gt=int(len(g)))
        if answer_depth is not None:
            a = answer_depth[gsel if answer_pixels == "gt" else m]
            a = a[np.isfinite(a) & (a > 0)]
            if len(a) < MIN_NODE_PIX // 4:
                continue
            rec["ans"] = float(np.median(a))
        out[int(lid)] = rec
    return out


def score_pairs(nodes: dict, answer: str) -> dict:
    """Answer which region of each pair is nearer, and score the answers against the scan.

    `area` names the larger region nearer; every other answer names the region of smaller median depth. A pair
    whose truths or answers tie is left out, so that a tie is not decided by the regions' ids.

    Returns:
        `n_pairs`, `n_correct`, `acc`, `gap_median_mm`, `n_nodes`, and per pair its ids (`pairs`), its gap in
        scan depth in mm (`gap_mm`) and whether it was answered right (`ok`).
    """
    ids = sorted(nodes)
    gaps, ok, pairs = [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = nodes[ids[i]], nodes[ids[j]]
            if a["gt"] == b["gt"]:
                continue
            truth = a["gt"] < b["gt"]
            if answer == "area":
                if a["area"] == b["area"]:
                    continue
                pred = a["area"] > b["area"]
            else:
                if a["ans"] == b["ans"]:
                    continue
                pred = a["ans"] < b["ans"]
            pairs.append([int(ids[i]), int(ids[j])])
            gaps.append(round(abs(a["gt"] - b["gt"]) * 1000.0, 3))
            ok.append(int(pred == truth))
    n = len(ok)
    return dict(n_pairs=n, n_correct=int(sum(ok)), acc=(sum(ok) / n if n else None),
                gap_median_mm=float(np.median(gaps)) if gaps else None, n_nodes=len(ids), pairs=pairs,
                gap_mm=gaps, ok=ok)


def side_depth(depth_dir: Path, key: str, side: str, index: int):
    """Return the side's file of `pipeline.d4d_depth`, opened.

    Raises:
        FileNotFoundError: the file is missing.
        ValueError: the file was made for another frame than `index`.
    """
    path = depth_dir / f"{d4d.side_stem(key, side)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no depth {path}; run pipeline.d4d_depth first")
    z = np.load(path, allow_pickle=False)
    if int(z["frame_index"]) != index:
        raise ValueError(f"{key} {side}: the depth was made for frame {int(z['frame_index'])}, not {index}")
    return z


def check_seeds(seed_dir: Path, depth_dir: Path, key: str, side: str) -> None:
    """Refuse the side's seed regions unless they and their record exist and the record names the side's depth.

    Raises:
        FileNotFoundError: the regions or their record are missing.
        ValueError: the record's `depth_sha256` is not the hash of the side's depth file.
    """
    stem = d4d.side_stem(key, side)
    for suffix in (".npy", ".json"):
        if not (seed_dir / f"{stem}{suffix}").is_file():
            raise FileNotFoundError(f"no {seed_dir / (stem + suffix)}; run pipeline.d4d_seed first")
    record = json.loads((seed_dir / f"{stem}.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((depth_dir / f"{stem}.npz").read_bytes()).hexdigest()
    if record.get("depth_sha256") != digest:
        raise ValueError(f"{key} {side}: the seed regions were made from another depth than {stem}.npz; make them "
                         "again")


def answer_map(clip, index: int, answer: str, gt_points: np.ndarray, pose: np.ndarray, params: dict, z=None):
    """Return the depth an answer reads, (H, W) at the endoscope's resolution; None for `area`.

    - `area`: no depth; the larger region is nearer. A floor that reads no geometry.
    - `screen`: the distance from the principal point, plus one. A floor that reads the image position alone.
    - `gt_half`: the scan's depth from a random half of its points. A ceiling.
    - `stereo`: the dataset's stereo depth of the frame. A reference.
    - `mono`: the depth of `pipeline.d4d_depth`, resized with the nearest neighbour. Only its order is read,
      so its scale is left as it is.

    Raises:
        ValueError: the answer is unknown, or the stereo depth of the frame is missing or named otherwise.
    """
    h, w = int(params["height"]), int(params["width"])
    if answer == "area":
        return None
    if answer == "screen":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return np.sqrt((xx - params["cx"]) ** 2 + (yy - params["cy"]) ** 2) + 1.0
    if answer == "gt_half":
        half = np.random.RandomState(GT_HALF_SEED).rand(len(gt_points)) < 0.5
        return d4d.scan_depth(gt_points[half], pose, params)
    if answer == "stereo":
        paths = clip.stereo_depth_paths
        frame = Path(clip.left_img_paths[index]).stem
        if index >= len(paths) or Path(paths[index]).stem != frame:
            raise ValueError(f"{d4d.clip_key(clip)}: the stereo depth at position {index} is not frame {frame}")
        d = np.load(paths[index]).astype(np.float32)
    elif answer == "mono":
        d = cv2.resize(z["depth"].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"unknown answer {answer!r}; one of {ANSWERS}")
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def region_labels(clip, index: int, regions: str, h: int, w: int, seed_path: Path | None = None, z=None):
    """Return the frame's regions, (H, W) int32 at the endoscope's resolution.

    - `voronoi`: `voronoi_labels`, the same cells on every frame.
    - `kmeans`: `kmeans_labels` of the frame.
    - `ours`: the seed regions of `pipeline.d4d_seed`, resized with the nearest neighbour.

    Raises:
        FileNotFoundError: the frame cannot be read.
        ValueError: the regions are unknown, or the seed regions are not the shape of the depth they were cut on.
    """
    if regions == "voronoi":
        return voronoi_labels(h, w)
    if regions == "kmeans":
        image = cv2.imread(str(clip.left_img_paths[index]))
        if image is None:
            raise FileNotFoundError(f"cannot read the frame {clip.left_img_paths[index]}")
        return kmeans_labels(image)
    if regions == "ours":
        labels = np.load(seed_path)
        if labels.shape != z["depth"].shape:
            raise ValueError(f"{seed_path.name}: regions {labels.shape}, depth {z['depth'].shape}")
        return cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    raise ValueError(f"unknown regions {regions!r}; one of {REGIONS}")


def score_side(task: tuple) -> dict:
    """Return one side's record, from `(clip, side, answer, regions, answer_pixels, depth_dir, seed_dir)`.

    Raises:
        ValueError: the side has no scan, pose or calibration, or a check of the functions it calls fails.
    """
    clip, side, answer, regions, answer_pixels, depth_dir, seed_dir = task
    key, params = d4d.clip_key(clip), clip.endoscope_params
    scan, pose = clip.pointclouds.get(side), clip.poses.get(side)
    if not scan or pose is None or params is None:
        raise ValueError(f"{key} {side}: no scan, pose or calibration to score")

    # Render the scan from the endoscope's pose, and find the frame and its instruments.
    gt_points = d4d.read_scan(scan)
    gt_depth = d4d.scan_depth(gt_points, pose, params)
    index, gap = d4d.nearest_frame(d4d.frame_times(clip), d4d.name_time(scan))
    h, w = int(params["height"]), int(params["width"])
    masks = d4d.frame_files(clip, "left_images_rect_masks", ".png")
    valid = ~d4d.instrument_mask(masks[index], (w, h))

    # Read the side's depth when the answer or the regions need it, then the answer and the regions.
    z = side_depth(depth_dir, key, side, index) if answer == "mono" or regions == "ours" else None
    seed_path = None
    if regions == "ours":
        check_seeds(seed_dir, depth_dir, key, side)
        seed_path = seed_dir / f"{d4d.side_stem(key, side)}.npy"
    ans = answer_map(clip, index, answer, gt_points, pose, params, z=z)
    labels = region_labels(clip, index, regions, h, w, seed_path, z=z)

    # Score every pair of the scored regions.
    rec = score_pairs(node_stats(labels, valid, gt_depth, ans, answer_pixels), answer)
    rec.update(key=key, side=side, answer=answer, regions=regions, frame_index=index, frame_minus_gt_s=gap,
               depth_window=int(z["window"]) if z is not None else None,
               depth_process_res=int(z["process_res"]) if z is not None else None,
               specimen=key.split("/")[0], session="/".join(key.split("/")[:2]))
    return rec


def bootstrap(recs: list, unit: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict | None:
    """Return the share answered right and its 95 % interval, resampling `unit`s; None with no pair scored.

    The share pools the pairs of the units drawn.
    """
    groups: dict = {}
    for r in recs:
        if r["n_pairs"]:
            groups.setdefault(r["key"] if unit == "clip" else r[unit], []).append(r)
    keys = sorted(groups)
    if not keys:
        return None
    rng = np.random.RandomState(seed)

    def acc_of(sel):
        c = sum(r["n_correct"] for k in sel for r in groups[k])
        n = sum(r["n_pairs"] for k in sel for r in groups[k])
        return c / n if n else np.nan

    boots = [acc_of([keys[i] for i in rng.randint(0, len(keys), len(keys))]) for _ in range(n_boot)]
    return dict(unit=unit, n_units=len(keys), acc=float(acc_of(keys)), lo=float(np.nanpercentile(boots, 2.5)),
                hi=float(np.nanpercentile(boots, 97.5)))


def seed_configs(recs: list, seed_dir: Path) -> list:
    """Return the settings of the seed regions the sides were scored on, each with its number of sides."""
    seen: dict = {}
    for r in recs:
        record = json.loads((seed_dir / f"{d4d.side_stem(r['key'], r['side'])}.json").read_text(encoding="utf-8"))
        settings = tuple(sorted((k, v) for k, v in record.items() if k not in PER_SIDE_FIELDS))
        seen[settings] = seen.get(settings, 0) + 1
    return [dict(list(k) + [("n_sides", n)]) for k, n in sorted(seen.items(), key=str)]


def population_tasks(clips: list, population: dict, answer: str, regions: str, answer_pixels: str,
                     depth_dir: Path | None, seed_dir: Path | None) -> list[tuple]:
    """Return the task of each side of the population, in the loader's order of clips, `start` before `end`.

    Raises:
        ValueError: a side of the population names a clip the dataset does not hold.
    """
    keep = {(s["key"], s["side"]) for s in population["sides"]}
    tasks = [(c, side, answer, regions, answer_pixels, depth_dir, seed_dir)
             for c in clips for side in d4d.SIDES if (d4d.clip_key(c), side) in keep]
    if len(tasks) != len(keep):
        found = {(d4d.clip_key(t[0]), t[1]) for t in tasks}
        raise ValueError(f"{len(keep) - len(tasks)} side(s) of the population name a clip the dataset does not "
                         f"hold, the first being {sorted(keep - found)[0]}")
    return tasks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="the dataset's root, which holds the specimen_* directories")
    ap.add_argument("--population", required=True, help="the population file, d4d_meta/population.json")
    ap.add_argument("--answer", required=True, choices=ANSWERS, help="what answers which region is nearer")
    ap.add_argument("--regions", required=True, choices=REGIONS, help="how the frame is cut into regions")
    ap.add_argument("--answer-pixels", default="all", choices=ANSWER_PIXELS,
                    help="the pixels of a region the answer's median is taken over: all, or those the scan reaches")
    ap.add_argument("--depth-dir", help="what pipeline.d4d_depth wrote; --answer mono and --regions ours need it")
    ap.add_argument("--seed-dir", help="what pipeline.d4d_seed wrote; --regions ours needs it")
    ap.add_argument("--out", required=True, help="the JSON file to write")
    ap.add_argument("--jobs", type=int, default=8, help="the sides scored at once")
    args = ap.parse_args(argv)
    if (args.answer == "mono" or args.regions == "ours") and not args.depth_dir:
        ap.error("--answer mono and --regions ours need --depth-dir")
    if args.regions == "ours" and not args.seed_dir:
        ap.error("--regions ours needs --seed-dir")
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    seed_dir = Path(args.seed_dir) if args.seed_dir else None

    # Score every side of the population.
    population = json.loads(Path(args.population).read_text(encoding="utf-8"))
    tasks = population_tasks(d4d.clips(args.root), population, args.answer, args.regions, args.answer_pixels,
                             depth_dir, seed_dir)
    print(f"{len(tasks)} sides, answer={args.answer}, regions={args.regions}, answer_pixels={args.answer_pixels}")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        recs = list(ex.map(score_side, tasks))

    # Write the records with the totals and the intervals.
    total_n = sum(r["n_pairs"] for r in recs)
    total_c = sum(r["n_correct"] for r in recs)
    summary = {u: bootstrap(recs, u) for u in UNITS}
    out = dict(answer=args.answer, regions=args.regions, answer_pixels=args.answer_pixels,
               population=args.population, front_behind_code_sha=front_behind_code_sha(),
               seed_config=seed_configs(recs, seed_dir) if args.regions == "ours" else None,
               depth_window=sorted({r["depth_window"] for r in recs} - {None}),
               depth_process_res=sorted({r["depth_process_res"] for r in recs} - {None}),
               n_frames=len(recs), n_pairs=total_n, acc=(total_c / total_n if total_n else None),
               min_gt_pix=MIN_GT_PIX, min_node_pix=MIN_NODE_PIX, voronoi_k=VORONOI_K, bootstrap=summary,
               per_frame=recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(recs)} sides, {total_n} pairs, acc {out['acc']}")
    for unit, s in summary.items():
        if s:
            print(f"  {unit:9s} n={s['n_units']:4d}  acc={s['acc']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
