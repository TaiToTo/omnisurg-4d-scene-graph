"""Window-level confidence intervals and paired tests for every table.

`summarize` gives each condition's mean over windows with a bootstrap 95 %
interval. `paired_tests` compares modalities on the same windows: the
median and mean of the differences, their bootstrap intervals, a two-sided
Wilcoxon signed-rank p-value and its Holm correction over the pairs of one
metric. The bootstrap is the repository's `evalkit.tools.paired_stats.boot_ci`
(10,000 percentile resamples, seeded from the data). The unit is the window,
as the revision asks; because windows of one video are not independent, the
video-level interval (`video_of`) is reported beside it.

Usage:
    from aecai_rev.stats import summarize, paired_tests
"""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats as sps

from evalkit.tools.paired_stats import N_BOOT, SEED, boot_ci, video_of


def _rng(x, tag):
    h = hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float64).tobytes(), digest_size=8)
    h.update(tag.encode())
    return np.random.default_rng([SEED, int.from_bytes(h.digest(), "big")])


def boot_ci_median(d):
    """Return the bootstrap 95 % percentile interval of the median of `d`."""
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        raise ValueError("a bootstrap over fewer than two windows has no interval")
    idx = _rng(d, "median").integers(0, len(d), size=(N_BOOT, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def holm(p):
    """Return Holm-adjusted p-values, None kept where a test had no value.

    Args:
        p: A sequence of p-values or None.

    Returns:
        A list of adjusted p-values in the input order.
    """
    idx = [i for i, v in enumerate(p) if v is not None and not np.isnan(v)]
    m = len(idx)
    out = [None] * len(p)
    order = sorted(idx, key=lambda i: p[i])
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def _ci(values, clips):
    lo, hi = boot_ci(values)
    groups = np.array([video_of(c) for c in clips])
    vlo, vhi = boot_ci(values, groups) if len(np.unique(groups)) >= 2 else (None, None)
    return lo, hi, vlo, vhi


def summarize(long, by):
    """Average a window-level long table per condition, with intervals.

    Args:
        long: Columns `clip`, `metric`, `value` and the columns in `by`.
        by: The columns that name a condition (with `metric`).

    Returns:
        One row per condition and metric: n windows, mean, sd, ci_low,
        ci_high (window bootstrap) and ci_low_video, ci_high_video.
    """
    rows = []
    for key, g in long.dropna(subset=["value"]).groupby(by, sort=False):
        g = g.sort_values("clip")
        v = g["value"].to_numpy(float)
        lo, hi, vlo, vhi = _ci(v, g["clip"].tolist()) if len(v) >= 2 else (None,) * 4
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), n=len(v), mean=float(v.mean()),
                         sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                         ci_low=lo, ci_high=hi, ci_low_video=vlo, ci_high_video=vhi))
    return pd.DataFrame(rows)


def paired_tests(long, by, pairs, side="modality"):
    """Compare the levels of one column on the same windows.

    Args:
        long: Window-level long table (`clip`, `metric`, `value`, ...).
        by: The other columns that name a condition (with `metric`); a test
            is run within each of their combinations.
        pairs: (a, b) levels of `side`; the difference is a - b.
        side: The column whose levels are compared.

    Returns:
        One row per combination and pair: n, median_diff with ci_low and
        ci_high (bootstrap of the median), mean_diff with mean_ci_low and
        mean_ci_high (window bootstrap) and mean_ci_low_video and
        mean_ci_high_video, p_raw (Wilcoxon) and p_holm over the pairs of
        the combination.
    """
    rows = []
    for key, g in long.dropna(subset=["value"]).groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        wide = g.pivot_table(index="clip", columns=side, values="value", aggfunc="first")
        block = []
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            both = wide[[a, b]].dropna()
            d = (both[a] - both[b]).to_numpy(float)
            clips = both.index.tolist()
            row = dict(zip(by, key), pair=f"{a} - {b}", n=len(d))
            if len(d) >= 2:
                row["median_diff"] = float(np.median(d))
                row["ci_low"], row["ci_high"] = boot_ci_median(d)
                row["mean_diff"] = float(d.mean())
                (row["mean_ci_low"], row["mean_ci_high"],
                 row["mean_ci_low_video"], row["mean_ci_high_video"]) = _ci(d, clips)
                row["p_raw"] = (float(sps.wilcoxon(d, zero_method="wilcox").pvalue)
                                if np.any(d != 0) else None)
            block.append(row)
        for row, p in zip(block, holm([r.get("p_raw") for r in block])):
            row["p_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def versions():
    """Return the library versions the statistics depend on."""
    import scipy
    return dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)
