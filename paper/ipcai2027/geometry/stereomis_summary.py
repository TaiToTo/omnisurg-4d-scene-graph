"""Summarize per-clip StereoMIS values: the clips kept, each answer's mean, and paired differences with marks.

`kept_clips` leaves out the clips a command names with `--exclude`. It refuses a name the clips do not hold, so
that a mistyped name cannot leave a clip in. Every interval is `paired_stats.boot_ci` over the sequences, since
the clips of one sequence are not independent. The mark is `paired_stats.mark_of`, with each sequence counted as
a video.
"""

import json
from pathlib import Path

import numpy as np

from evalkit.tools.paired_stats import boot_ci, mark_of, round_keeping_sign


def kept_clips(clips: list[str], exclude: list[str]) -> list[str]:
    """Return `clips` without the excluded ones, in their order.

    Raises:
        ValueError: `exclude` names a clip that `clips` does not hold, or names one twice.
    """
    unknown = sorted(set(exclude) - set(clips))
    if unknown or len(set(exclude)) != len(exclude):
        raise ValueError(f"--exclude names clips the population does not hold, or one twice: {sorted(exclude)}")
    return [c for c in clips if c not in set(exclude)]


def sequence_of(clip: str) -> str:
    """Return the sequence a clip was cut from: `P2_6` for `P2_6__clip_0007`."""
    return clip.split("__")[0]


def mean_row(values, clips: list[str]) -> dict:
    """Return the mean over clips and its interval over sequences."""
    v = np.asarray(values, dtype=float)
    lo, hi = boot_ci(v, np.array([sequence_of(c) for c in clips]))
    return dict(mean=round(float(v.mean()), 4), ci95=[round(lo, 4), round(hi, 4)])


def diff_row(a, b, clips: list[str], sign: int) -> dict:
    """Return the paired difference `a - b` over clips, its interval over sequences and its mark.

    Args:
        a: one value per clip.
        b: one value per clip, the base.
        clips: the clips, in the order of `a` and `b`.
        sign: +1 where more is better, -1 where less is better.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    seqs = np.array([sequence_of(c) for c in clips])
    lo, hi = boot_ci(d, seqs)
    # The mark is read from the interval as written, so an end near zero keeps its sign.
    ci = [round_keeping_sign(lo), round_keeping_sign(hi)]
    n_seq = len(set(seqs))
    return dict(delta=round(float(d.mean()), 4), ci95=ci, mark=mark_of(ci, sign, n_videos=n_seq),
                n_clips=len(d), n_sequences=n_seq)


def interval(row: dict, value: str) -> str:
    """Write a row as `value [lo, hi] mark`, with signs on a difference."""
    spec = "+.3f" if value == "delta" else ".3f"
    lo, hi = row["ci95"]
    return f"{row[value]:{spec}} [{lo:{spec}}, {hi:{spec}}] {row.get('mark', '')}".rstrip()


def write_json(path: str, result: dict) -> None:
    """Write the result as JSON, creating the directory."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {out}")
