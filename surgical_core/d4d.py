"""Read D4D through its upstream loader: clips, times, instrument masks and scans.

`clips` imports `d4d.loader` from a clone of its repository at `LOADER_COMMIT`, which the README puts on
`PYTHONPATH`. It refuses a missing `d4d`, and one whose files differ from that commit's. A clip's frames,
stereo depth maps and instrument masks are files named by their time, `<sec>_<nsec>`, and so is a scan.
`frame_files` returns a clip's files of one kind and refuses them unless their names are the frames' names.
`instrument_mask` resizes a mask stored at another size with the nearest neighbour, then dilates it.
`pixel_hits` counts a scan's points that fall in each pixel of the endoscope's view.
"""

import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np

SIDES = ("start", "end")

LOADER_REPOSITORY = "https://github.com/reubendocea/d4d"
LOADER_COMMIT = "efe12d265a289b0b102307e64f9910a6985835a3"
# Git's blob hashes of the loader's files at `LOADER_COMMIT`. A copy of the commit without its `.git`, such as
# an archive, is checked as well as a clone.
LOADER_BLOBS = {"__init__.py": "fbb17c4413ef6bdaca5555d70ce482239e0e7e3b",
                "loader.py": "1ad63d5d34bc4b5a7f97fc0d14c562ac09df9d7d"}

# The side of the square that dilates an instrument mask, in pixels. A mask leaves a few pixels of the
# instrument outside its edge.
MASK_DILATE = 9


def blob_hash(path: Path) -> str:
    """Return git's blob hash of the file at `path`, the hash `git hash-object` prints."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_loader(package: Path) -> None:
    """Refuse a `d4d` package whose files differ from those of `LOADER_COMMIT`.

    Raises:
        RuntimeError: a file of `LOADER_BLOBS` is missing from `package`, or holds other bytes.
    """
    for name, want in LOADER_BLOBS.items():
        path = package / name
        got = blob_hash(path) if path.is_file() else "no file"
        if got != want:
            raise RuntimeError(f"{path}: blob {got}, where commit {LOADER_COMMIT[:8]} of {LOADER_REPOSITORY} has "
                               f"{want}; check out that commit")


def loader():
    """Return the loader's `D4D` class, once its files are checked against `LOADER_COMMIT`.

    The package's files are checked before it is imported, so no other code of it runs.

    Raises:
        ImportError: no `d4d` package is on the path.
        RuntimeError: a file of the package differs from that file at `LOADER_COMMIT`.
    """
    spec = importlib.util.find_spec("d4d")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"no `d4d` package: clone {LOADER_REPOSITORY} at commit {LOADER_COMMIT[:8]} and put the "
                          "clone on PYTHONPATH, as the README says")
    check_loader(Path(next(iter(spec.submodule_search_locations))))
    from d4d.loader import D4D
    return D4D


def clips(root: str) -> list:
    """Return every clip of the dataset at `root`, in the loader's order: specimen, session, then clip.

    The loader leaves out the specimens whose directory name holds `ambiguous`.

    Raises:
        FileNotFoundError: `root` does not exist.
        ValueError: `root` holds no clip, as the directory above the dataset's root does.
    """
    found = [clip for specimen in loader()(root) for session in specimen for clip in session]
    if not found:
        raise ValueError(f"{root} holds no clip; give the directory that holds the `specimen_*` directories")
    return found


def clip_key(clip) -> str:
    """Return a clip's key, `<specimen>/<session>/<clip>`."""
    return f"{clip.session.specimen.name}/{clip.session.name}/{clip.name}"


def name_time(path) -> float:
    """Return the time, in seconds, that a file named `<sec>_<nsec>.<ext>` is named by."""
    sec, nsec = Path(path).name.rsplit(".", 1)[0].split("_")
    return float(sec) + float(nsec) * 1e-9


def frame_times(clip) -> np.ndarray:
    """Return the times of a clip's rectified left frames, in seconds."""
    return np.asarray([name_time(p) for p in clip.left_img_paths], dtype=np.float64)


def nearest_frame(times: np.ndarray, t: float) -> tuple[int, float]:
    """Return the index of the frame nearest to time `t`, and that frame's time less `t`, in seconds."""
    i = int(np.argmin(np.abs(times - t)))
    return i, float(times[i] - t)


def frame_files(clip, folder: str, suffix: str) -> list[Path]:
    """Return a clip's files of one kind, one per frame, in the frames' order.

    Raises:
        ValueError: the clip has no frame, or the files' names are not the frames' names. A file missing in
            the middle would pair every later frame with its neighbour's file.
    """
    frames = [Path(p).stem for p in clip.left_img_paths]
    files = sorted((Path(clip.path) / folder).glob(f"*{suffix}"))
    if not frames:
        raise ValueError(f"{clip_key(clip)}: no rectified left frame")
    if [f.stem for f in files] != frames:
        raise ValueError(f"{clip_key(clip)}: {len(files)} files in {folder}/ for {len(frames)} frames, or files "
                         "named by other times than the frames")
    return files


def instrument_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Return the dilated instrument mask of one frame, true on an instrument, at `size` (width, height).

    Raises:
        ValueError: the file cannot be read as an image.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"{path}: not an image")
    if raw.shape != (size[1], size[0]):
        raw = cv2.resize(raw, size, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((MASK_DILATE, MASK_DILATE), np.uint8)
    return cv2.dilate((raw > 0).astype(np.uint8), kernel) > 0


def read_scan(path: str) -> np.ndarray:
    """Return a scan's finite points, (N, 3), in the world frame, in metres.

    Raises:
        ValueError: the scan holds no finite point.
    """
    import trimesh

    scan = trimesh.load(path)
    if isinstance(scan, trimesh.Scene):
        scan = next(g for g in scan.geometry.values() if isinstance(g, trimesh.PointCloud))
    points = np.asarray(scan.vertices, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"{path}: no finite point")
    return points


def pixel_hits(points: np.ndarray, pose: np.ndarray, params: dict) -> np.ndarray:
    """Return how many of `points` fall in each pixel of the endoscope's view, as an (H, W) array.

    Args:
        points: (N, 3) points in the world frame.
        pose: the camera-to-world 4x4 pose of the endoscope.
        params: the loader's `endoscope_params`: `fx`, `fy`, `cx`, `cy`, `width`, `height`.
    """
    # Move the points into the camera's frame and project those in front of it.
    h, w = int(params["height"]), int(params["width"])
    cam = (np.linalg.inv(pose) @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]
    z = cam[:, 2]
    front = z > 1e-6
    safe = np.where(front, z, 1.0)
    u = params["fx"] * cam[:, 0] / safe + params["cx"]
    v = params["fy"] * cam[:, 1] / safe + params["cy"]
    # Count the points that land inside the view, per pixel.
    inside = front & (u >= 0) & (u < params["width"]) & (v >= 0) & (v < params["height"])
    hits = np.zeros(h * w, np.int32)
    np.add.at(hits, np.floor(v[inside]).astype(np.int64) * w + np.floor(u[inside]).astype(np.int64), 1)
    return hits.reshape(h, w)
