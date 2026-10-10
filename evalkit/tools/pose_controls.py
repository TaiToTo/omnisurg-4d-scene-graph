"""Make the camera trajectories the StereoMIS result is compared with: two floors and a ceiling.

- `static_floor`: a camera that never moves. Its `ate_rel` is exactly 1.0.
- `line_floor`: a camera that moves in a straight line at the true
  trajectory's mean speed. It looks plausible and carries no information.
- `StereoVO`: stereo visual odometry, from disparity, sparse tracks and PnP.
  It shows what an estimate with true scale reaches.

No control is cut from the ground truth, because the similarity fit of `ate`
would fit the truth to itself. The ceiling reads the instrument masks, which
a method may not read.
"""

from pathlib import Path

import cv2
import numpy as np

import surgical_core.stereomis as stereomis
from trajectory_eval.pose_metrics import centers_from_cam_to_world

# The depth range a disparity is kept in, in mm. The clips see tissue from 30 to 400 mm.
DEPTH_RANGE_MM = (20.0, 500.0)


def static_floor(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `n` camera centres at the origin and `n` identity rotations."""
    return np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))


def line_floor(gt_c: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a camera that moves along z at the true trajectory's mean speed, and identity rotations.

    The speed is the true path's length over the clip's duration. `ate` does not depend on it, since the
    similarity fit absorbs speed and direction; the speed keeps `rpe` on the truth's scale.

    Raises:
        ValueError: the times are not one per centre, or the clip has no duration.
    """
    times = np.asarray(times, dtype=float)
    if times.shape != (len(gt_c),):
        raise ValueError(f"{times.shape[0] if times.ndim == 1 else times.shape} times for {len(gt_c)} centres")
    dur = float(times[-1] - times[0])
    if not dur > 0:
        raise ValueError(f"the clip lasts {dur} s, so it has no speed")
    path = float(np.linalg.norm(np.diff(gt_c, axis=0), axis=1).sum())
    d = (times - times[0])[:, None] * (path / dur) * np.array([0.0, 0.0, 1.0])
    return d, np.tile(np.eye(3), (len(gt_c), 1, 1))


def sgbm() -> cv2.StereoSGBM:
    """Return the matcher the ceiling takes disparity from.

    At half resolution the clips' depth of 30 to 400 mm is a disparity of 6 to 83 pixels; the number of
    disparities is the next multiple of 16 above, 96.
    """
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=96, blockSize=5,
        P1=8 * 3 * 25, P2=32 * 3 * 25, uniquenessRatio=8,
        speckleWindowSize=120, speckleRange=2, disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def sgbm_depth(matcher, left: np.ndarray, right: np.ndarray, bf: float,
               lo: float = DEPTH_RANGE_MM[0], hi: float = DEPTH_RANGE_MM[1]) -> np.ndarray:
    """Return the depth in mm of a rectified grey pair, 0 where it is not known.

    The matcher gives disparity in sixteenths of a pixel. A disparity of one pixel or less, and a depth
    outside `lo` to `hi`, are not known.

    Args:
        matcher: what `sgbm` returns.
        left: the left view, grey.
        right: the right view, grey.
        bf: the focal length in pixels times the baseline in mm.
        lo: the nearest depth kept, in mm.
        hi: the farthest depth kept, in mm.
    """
    d = matcher.compute(left, right).astype(np.float32) / 16.0
    z = np.where(d > 1.0, bf / np.maximum(d, 1e-3), 0.0)
    return np.where((z >= lo) & (z <= hi), z, 0.0)


class StereoVO:
    """Chain the camera's poses over a clip from stereo depth, sparse tracks and PnP.

    Each step takes depth from the earlier frame only, and the later frame's tracked pixels: PnP needs 3D
    points in one frame and their 2D sightings in the other, so a hole in one frame's depth costs less than
    in a fit of points to points. A step that cannot be solved repeats the earlier pose, and is counted.
    """

    # The fewest tracks a step is solved from, before and after PnP's inliers.
    MIN_TRACKS = 40

    def __init__(self, K: np.ndarray, baseline_mm: float):
        """Make the odometry for a camera of intrinsics `K`, at the views' resolution, and a baseline in mm."""
        self.K = K
        self.bf = float(K[0, 0] * baseline_mm)
        self.m = sgbm()
        self.reset()

    def reset(self) -> None:
        """Start a new trajectory at the identity."""
        self.poses = [np.eye(4)]
        self.prev = None
        self.n_fail = 0

    def _prep(self, left: np.ndarray, right: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Return the left view in grey, and its depth with the pixels outside `mask` unknown."""
        g = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        z = sgbm_depth(self.m, g, cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), self.bf)
        if mask is not None:
            z = np.where(mask, z, 0.0)
        return g, z

    def add(self, left: np.ndarray, right: np.ndarray, mask: np.ndarray | None) -> None:
        """Add a frame, an RGB pair and its tissue mask or None, and extend the trajectory to it."""
        g, z = self._prep(left, right, mask)
        if self.prev is None:
            self.prev = (g, z)
            return
        pg, pz = self.prev

        # Track corners where the earlier frame has depth, forward and back.
        pts = cv2.goodFeaturesToTrack(pg, maxCorners=1200, qualityLevel=0.01, minDistance=7,
                                      mask=(pz > 0).astype(np.uint8))
        ok = False
        if pts is not None and len(pts) >= self.MIN_TRACKS:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, pg, nxt, None, winSize=(21, 21), maxLevel=3)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
            p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)

            # Lift the tracks that have depth to 3D in the earlier camera, and solve the later camera by PnP.
            if len(p0) >= self.MIN_TRACKS:
                u, v = np.round(p0[:, 0]).astype(int), np.round(p0[:, 1]).astype(int)
                zz = pz[v, u]
                f = zz > 0
                if f.sum() >= self.MIN_TRACKS:
                    obj = np.stack([(p0[f, 0] - self.K[0, 2]) / self.K[0, 0] * zz[f],
                                    (p0[f, 1] - self.K[1, 2]) / self.K[1, 1] * zz[f],
                                    zz[f]], 1)
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj.astype(np.float64), p1[f].astype(np.float64), self.K, None,
                        reprojectionError=2.0, iterationsCount=300, confidence=0.999,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    ok = bool(ok) and inl is not None and len(inl) >= self.MIN_TRACKS

        # PnP gives the earlier camera's frame in the later camera's; the pose is camera to world.
        if ok:
            Rm, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rm, tvec.ravel()
            self.poses.append(self.poses[-1] @ np.linalg.inv(T))
        else:
            self.n_fail += 1
            self.poses.append(self.poses[-1].copy())
        self.prev = (g, z)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (N, 3) camera centres and (N, 3, 3) camera-to-world rotations."""
        T = np.stack(self.poses)
        return centers_from_cam_to_world(T), T[:, :3, :3]


def run_sequence(root: Path, depth_root: Path, seq: str) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Run the ceiling on every usable clip of a sequence, decoding the sequence once.

    Each frame takes the tissue mask nearest it, as `stereomis.load_mask_nearest` finds it.

    Args:
        root: the StereoMIS directory, which holds one directory per sequence.
        depth_root: the depth export, which holds `<sequence>/stats.npy`.
        seq: the sequence.

    Returns:
        For each clip's name, its camera centres, its camera-to-world rotations and how many steps failed.

    Raises:
        ValueError: two clips share a frame, a clip holds a frame twice, or two clips interleave.
    """
    # Find the clips, and map each frame to its clip.
    clips = [c for c in stereomis.clips(root, depth_root, seq) if c["usable"]]
    if not clips:
        return {}
    owner = {f: c["name"] for c in clips for f in c["frames"]}

    # Refuse clips one decode cannot feed in turn. A shared frame would go to one clip only, and an interleaved
    # clip would restart its trajectory and lose the frames before.
    order = [owner[f] for f in sorted(owner)]
    n_runs = 1 + sum(a != b for a, b in zip(order, order[1:]))
    if len(owner) != sum(len(c["frames"]) for c in clips) or n_runs != len(clips):
        raise ValueError(f"{seq}: the clips share a frame, repeat one, or interleave, so one decode cannot "
                         "feed them in turn")

    # Decode the sequence once, and chain each clip's poses from its first frame.
    c = stereomis.calib(root, seq)
    vo, cur, out = StereoVO(c.K_half, c.baseline_mm), None, {}
    for f, left, right in stereomis.iter_frame_set(root, seq, sorted(owner)):
        if owner[f] != cur:
            if cur is not None:
                out[cur] = vo.result() + (vo.n_fail,)
            vo.reset()
            cur = owner[f]
        vo.add(left, right, stereomis.load_mask_nearest(root, seq, f))
    out[cur] = vo.result() + (vo.n_fail,)
    return out
