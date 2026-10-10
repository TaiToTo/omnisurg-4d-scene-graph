"""Run the tagged `track_sam3.py` on one window and keep its seed map.

The script runs `track_sam3.main` unchanged, with the tagged source on
`sys.path`. It only wraps `seed_label_map`, so that the SAM region map of
the seed frame is written to the cache path `track_sam3` itself reads
(`<out_dir>/<clip>/<mode>sam_group_pps<N>/group_<frame>.npy`). With
`--seed_only` it stops once the seed map is written. Run it with `-P`, so
that this directory does not shadow the tagged modules.

Usage:
    PYTHONPATH=<src>:<src>/sam3_wrapper:<src>/depth_sam_tracking_experiment \
    python3 -P aecai_rev/track_job.py [--seed_only] <track_sam3.py arguments>
"""
import os
import random
import sys

import numpy as np
import torch


class _SeedOnlyDone(Exception):
    """Raised in place of the tracker once the seed map is written."""


def main():
    """Run one tracking job with the seed map kept."""
    seed_only = "--seed_only" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--seed_only"]
    sys.argv = ["track_sam3.py", *argv]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    import track_sam3 as ts

    original = ts.seed_label_map

    def saving_seed_label_map(clip, loader, frame_abs, depth01, args):
        label = original(clip, loader, frame_abs, depth01, args)
        d = os.path.join(args.out_dir, args.clip,
                         f"{args.sam_input}sam_group_pps{args.points_per_side}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"group_{frame_abs:04d}.npy")
        if not os.path.exists(path):
            np.save(path, label)
        return label

    ts.seed_label_map = saving_seed_label_map
    if seed_only:
        def stop(*_a, **_k):
            raise _SeedOnlyDone
        ts.Sam3VideoInstanceSession = stop
        try:
            ts.main()
        except _SeedOnlyDone:
            return
        raise RuntimeError("track_sam3.main returned before building the tracker")
    ts.main()


if __name__ == "__main__":
    main()
