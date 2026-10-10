"""Check the seed maps written for the frozen runs against their tracks.

The frozen AE-CAI runs kept no seed map, so `run_track --seeds` wrote them
again in this environment. A seed map equals the one a frozen run used when
its regions of at least `seed_min_area` pixels, numbered from 1 as the
tracker numbers them, are the identities on the run's seed frame. The script
writes one row per window, modality and points_per_side.

Usage:
    python3 -m aecai_rev.check_seeds
"""
import argparse
import os

import numpy as np
import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.windows_io import window_clips


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    cfg = load()
    if cfg["windows"]["population"] != "legacy27":
        raise SystemExit("frozen runs exist for the legacy27 population only")
    seed = cfg["windows"]["seed_index"]
    work = cfg["paths"]["work_root"]
    rows = []
    for pps, src in sorted(cfg["track"]["sources"].items()):
        if src.startswith(work):
            continue
        for clip in window_clips(cfg):
            for mode in cfg["track"]["modalities"]:
                path = os.path.join(work, "seeds", f"pps{pps}", clip, f"{mode}sam_group_pps{pps}",
                                    f"group_{seed:04d}.npy")
                if not os.path.exists(path):
                    rows.append(dict(pps=pps, clip=clip, modality=mode, status="missing"))
                    continue
                m = np.load(path)
                ids = {int(r) + 1 for r in np.unique(m)
                       if r >= 0 and int((m == r).sum()) >= cfg["track"]["seed_min_area"]}
                lab = np.load(os.path.join(src, clip, f"track_rgb_w1_{mode}", f"label_{seed:04d}.npy"))
                got = {int(v) for v in np.unique(lab) if v >= 0}
                status = "equal" if ids == got else ("tracked subset" if got <= ids else "differs")
                rows.append(dict(pps=pps, clip=clip, modality=mode, status=status,
                                 seed_regions=len(ids), tracked_on_seed_frame=len(got)))
    df = pd.DataFrame(rows)
    out = os.path.join(cfg["paths"]["results_root"], "00_check")
    df.to_csv(os.path.join(out, "seed_maps_check.csv"), index=False)
    write_meta(os.path.join(out, "seed_maps_meta"), cfg, what="seed maps of the frozen runs")
    print(df.groupby(["pps", "status"]).size().to_string())


if __name__ == "__main__":
    main()
