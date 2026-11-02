"""Resample videos with replacement for a statistic that is a ratio of summed counts, and give its 95 % interval.

`paired_stats.boot_ci` bootstraps a mean of values. A ratio of two counts summed over the videos drawn, or the
difference of two such ratios, is not a mean, so it is drawn here. Each draw takes as many videos as there are,
with replacement, from `numpy.random.default_rng(seed)`, and sums each column of their rows.
"""
from __future__ import annotations

import numpy as np


def resampled_sums(per_video: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """Return the column sums of each of `n_boot` draws of the videos' rows, as an (n_boot, k) array.

    Args:
        per_video: (V, k) counts of each video, one row per video in a fixed order.
        n_boot: The number of draws.
        seed: The seed of the generator.

    Raises:
        ValueError: `per_video` is not 2-D or holds fewer than two videos. With one video every draw is the same
            one, and the point that comes out is not an interval.
    """
    per_video = np.asarray(per_video, dtype=np.float64)
    if per_video.ndim != 2 or len(per_video) < 2:
        raise ValueError(f"a video bootstrap needs a (V, k) array of two or more videos, got shape {per_video.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_video), size=(n_boot, len(per_video)))
    return per_video[idx].sum(axis=1)


def interval(draws: np.ndarray) -> list[float]:
    """Return the 2.5th and 97.5th percentiles of a statistic over the draws.

    Raises:
        ValueError: a draw's statistic is not finite: the draw held no event to divide by, and leaving it out
            would narrow the interval without saying so.
    """
    draws = np.asarray(draws, dtype=np.float64)
    bad = int((~np.isfinite(draws)).sum())
    if bad:
        raise ValueError(f"{bad} of {draws.size} draws have no finite statistic")
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
