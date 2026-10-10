"""Run the DA3 stage, then the Pi3X stage, on every StereoMIS clip, one process per GPU, and check every clip.

The clips are those `surgical_core.stereomis.clips` marks usable, which
`pipeline.extract_stereomis` writes under `--input-dir`. Each stage runs on
the clips that lack its output, dealt out to the GPUs as
`pipeline.depth_population` deals them: DA3 with its point clouds, and Pi3X
with `--max-points 60000`, which thins its point clouds and leaves its poses
alone. Before anything runs, the driver refuses a clip that is not there and
a clip with either stage's output made at another setting. After each stage,
every clip must hold the stage's bundle with one frame per image, at the
driver's setting; the command exits non-zero otherwise, and names the clips.

Usage:
    python -m pipeline.depth_stereomis --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \\
        --input-dir /path/to/clips [--gpus 0 1 2 3]
"""

import argparse
import signal
import sys
import zipfile
from pathlib import Path

import numpy as np

import pipeline.depth_population as population
import surgical_core.stereomis as stereomis
from recon3d_wrapper.pi3x import DEFAULT_PIXEL_LIMIT, Pi3X

# The DA3 stage writes its point clouds, as the workbench's run did; only the viewer reads them.
DA3_COMMAND = population.DEPTH_COMMAND
DA3_ARGS = ["--device", "cuda"]
# The most points a Pi3X point cloud keeps. Unthinned, the Pi3X clouds take about 430 MB a clip.
MAX_POINTS = 60000
PI3X_COMMAND = [sys.executable, "-u", "-m", "pipeline.pi3x"]
PI3X_ARGS = ["--device", "cuda", "--max-points", str(MAX_POINTS)]
PI3X_BUNDLE = Path("exports/mini_npz/results__pi3x.npz")
# What the Pi3X stage records at the setting the driver runs it at: its defaults, besides `--max-points`.
PI3X_SETTING = {"model": Pi3X.model_id, "pixel_limit": DEFAULT_PIXEL_LIMIT, "conf_thre": 0.0, "vo": False}


def usable_clips(root: Path, depth_root: Path) -> list[str]:
    """Return the names of the clips `stereomis.clips` marks usable, sequence by sequence."""
    return [c["name"] for seq in stereomis.SEQUENCES for c in stereomis.clips(root, depth_root, seq) if c["usable"]]


def pi3x_record(manifest: dict) -> dict | None:
    """Return the clip's record of the Pi3X stage, or None when the stage has not finished on it."""
    return manifest.get("geometry_sources", {}).get("pi3x")


def pi3x_departures(manifest: dict) -> list[str]:
    """Return how the clip's Pi3X output departs from the driver's setting: keys that differ, and unthinned clouds."""
    record = pi3x_record(manifest)
    out = [f"{k} {record.get(k, '(none)')!r}" for k, v in PI3X_SETTING.items() if record.get(k, object()) != v]
    most = max((fr.get("geometry_sources", {}).get("pi3x", {}).get("n_vertices", 0)
                for fr in manifest.get("frames", [])), default=0)
    if most > MAX_POINTS:
        out.append(f"clouds of up to {most} points")
    return out


def check_pi3x_setting(manifests: dict[str, dict]) -> None:
    """Refuse a clip whose Pi3X output was made at another setting than the driver runs the stage at.

    Raises:
        ValueError: such a clip. The message names each, with how it departs.
    """
    other = [f"{c} ({', '.join(d)})" for c, m in manifests.items() if pi3x_record(m) is not None
             for d in [pi3x_departures(m)] if d]
    if other:
        raise ValueError(f"{len(other)} clip(s) with Pi3X output made at another setting than {PI3X_SETTING} with "
                         f"at most {MAX_POINTS} points a cloud: {other}")


def check_pi3x(root: Path, clips: list[str]) -> None:
    """Refuse a population in which a clip lacks its Pi3X output, or holds it at another setting or frame count.

    Raises:
        ValueError: a clip has no record or no bundle, `pi3x_departures` lists a departure, its bundle cannot be
            read or holds other keys than the stage writes, or its poses are not one per image.
    """
    manifests = population.read_manifests(root, clips)
    missing, other, unreadable, mismatched = [], [], [], []
    for clip in clips:
        bundle = root / clip / PI3X_BUNDLE
        if pi3x_record(manifests[clip]) is None or not bundle.is_file():
            missing.append(clip)
            continue
        departures = pi3x_departures(manifests[clip])
        if departures:
            other.append(f"{clip} ({', '.join(departures)})")
            continue
        try:
            with np.load(bundle) as z:
                keys, n_poses = set(z.files), int(z["extrinsics"].shape[0]) if "extrinsics" in z else 0
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            unreadable.append(f"{clip} ({type(e).__name__}: {e})")
            continue
        n_images = len(list((root / clip / "input_images").glob("*.png")))
        if keys != population.BUNDLE_KEYS:
            other.append(f"{clip} (keys {sorted(keys)})")
        elif n_poses != n_images:
            mismatched.append(f"{clip} ({n_poses} poses for {n_images} images)")
    problems = [f"{len(v)} {what}: {v}" for what, v in (
        ("clip(s) without Pi3X output", missing), ("with Pi3X output of another setting or version", other),
        ("with a Pi3X bundle that cannot be read", unreadable), ("with another number of poses than images",
                                                                  mismatched)) if v]
    if problems:
        raise ValueError("; ".join(problems))


def run_stereomis(root: Path, clips: list[str], gpus: list[int]) -> None:
    """Run the DA3 stage, then the Pi3X stage, on the clips that lack each one's output, and check every clip.

    Raises:
        FileNotFoundError: a clip is not under `root`.
        ValueError: before anything runs, a manifest cannot be read, a clip's output of either stage was made at
            another setting, or `shards` refuses the GPUs.
        RuntimeError: a process exited non-zero, or the check after a stage refuses the population. The Pi3X
            stage does not start after the DA3 stage fails.
    """
    # Refuse, before any process starts: a clip that is not there or whose manifest cannot be read, and a clip
    # with either stage's output made at another setting.
    manifests = population.read_manifests(root, clips)
    population.check_setting(manifests)
    check_pi3x_setting(manifests)
    print(f"{len(clips)} clips, GPUs {gpus}")

    # Run the DA3 stage on the clips without `depth_info`, and check every clip's depth.
    todo = [c for c in clips if "depth_info" not in manifests[c]]
    print(f"DA3: {len(todo)} clips without depth")
    problems = population.run_shards(DA3_COMMAND, DA3_ARGS, root, population.shards(todo, gpus), "depth")
    try:
        population.check_depth(root, clips)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RuntimeError("DA3: " + "; ".join(problems))

    # Run the Pi3X stage on the clips without its record, and check every clip's Pi3X output.
    manifests = population.read_manifests(root, clips)
    todo = [c for c in clips if pi3x_record(manifests[c]) is None]
    print(f"Pi3X: {len(todo)} clips without Pi3X output")
    problems = population.run_shards(PI3X_COMMAND, PI3X_ARGS, root, population.shards(todo, gpus), "pi3x")
    try:
        check_pi3x(root, clips)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RuntimeError("Pi3X: " + "; ".join(problems))


def main() -> None:
    for s in population.STOP_SIGNALS:
        signal.signal(s, population.stop)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--depth-root", required=True, help="The depth export, which holds <sequence>/stats.npy.")
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0], help="The GPUs to run on, one process each.")
    args = ap.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    try:
        run_stereomis(root, usable_clips(Path(args.root), Path(args.depth_root)), args.gpus)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
