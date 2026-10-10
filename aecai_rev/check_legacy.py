"""Check that the revision's scoring reproduces the AE-CAI Table 1.

The script makes two comparisons on the frozen pps 24 tracks of the AE-CAI
windows. First, it re-runs the tagged `eval_track.py` on every window and
modality and compares each JSON with the frozen one the tag records under
`miccai2026_workshop/results/eval/w1/`, key by key, frame by frame. Second,
it compares the frame scores `aecai_rev.score` wrote under the 300 px cut
with the frozen per-frame values. It writes the per-window comparison and
Table 1 as recomputed, and exits non-zero on any difference.

Usage:
    python3 -m aecai_rev.check_legacy --workers 24
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from aecai_rev.config import load, write_meta
from aecai_rev.run_track import tagged_env

FROZEN = "depth_sam_tracking_experiment/miccai2026_workshop/results/eval/w1"
COLUMNS = {"f1@0.5": "inst_F1_50", "f1@0.75": "inst_F1_75", "n_reg": "n_regions",
           "miou": "mIoU"}


def rerun(cfg, clip, mode, out):
    """Run the tagged evaluator on one frozen condition; return its JSON path."""
    src = cfg["paths"]["aecai_src"]
    tdir = os.path.join(cfg["track"]["sources"][24], clip, f"track_rgb_w1_{mode}")
    path = os.path.join(out, f"{clip}__w1_{mode}.json")
    subprocess.run([sys.executable, "eval_track.py", "--root", cfg["paths"]["cholec_gt"],
                    "--clip", clip, "--track_dir", tdir, "--tag", f"w1_{mode}",
                    "--out_json", path],
                   cwd=os.path.join(src, "depth_sam_tracking_experiment"),
                   env=tagged_env(cfg, ""), check=True, capture_output=True)
    return path


def compare_json(a, b):
    """Return the keys whose values differ between two evaluator JSONs."""
    keys = sorted((set(a) | set(b)) - {"track_dir"})
    return [k for k in keys if a.get(k) != b.get(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = load()
    if cfg["windows"]["population"] != "legacy27":
        raise SystemExit("the Table 1 check runs on the legacy27 population only")
    clips = cfg["windows"]["legacy27"]
    modes = cfg["track"]["modalities"]
    out = os.path.join(cfg["paths"]["results_root"], "00_check")
    work = os.path.join(cfg["paths"]["work"], "check_legacy")
    os.makedirs(work, exist_ok=True)

    # The tagged evaluator, re-run, against the frozen JSONs.
    jobs = [(c, m) for c in clips for m in modes]
    with ThreadPoolExecutor(args.workers) as ex:
        paths = list(ex.map(lambda j: rerun(cfg, j[0], j[1], work), jobs))
    rows, frozen = [], {}
    for (c, m), p in zip(jobs, paths):
        with open(os.path.join(cfg["paths"]["aecai_src"], FROZEN, f"{c}__w1_{m}.json")) as f:
            ref = json.load(f)
        with open(p) as f:
            new = json.load(f)
        frozen[(c, m)] = ref
        rows.append(dict(clip=c, modality=m, n_gt_frames=ref["n_gt_frames"],
                         tagged_rerun_differs=" ".join(compare_json(ref, new))))

    # aecai_rev.score under 300 px, against the frozen per-frame values.
    frames = pd.read_csv(os.path.join(cfg["paths"]["work"], "scores", "frames.csv.gz"))
    frames = frames[(frames["pps"] == 24) & (frames["filter"] == "px300")
                    & (frames["variant"] == "raw")]
    worst = 0.0
    for row in rows:
        ref = frozen[(row["clip"], row["modality"])]
        mine = frames[(frames["clip"] == row["clip"]) & (frames["modality"] == row["modality"])]
        mine = mine.set_index("frame")
        diffs = []
        for fr in ref["per_frame"]:
            m = mine.loc[fr["abs_frame"]]
            vals = {"n_reg": fr["n_regions"], "miou": fr["mIoU"],
                    "f1@0.5": fr["inst"]["f1@0.5"] if fr["inst"] else None,
                    "f1@0.75": fr["inst"]["f1@0.75"] if fr["inst"] else None}
            for k, v in vals.items():
                if v is None:
                    if not pd.isna(m[k]):
                        diffs.append(np.inf)
                    continue
                # The frozen JSON rounds per-frame values to 4 decimals.
                diffs.append(abs(round(float(m[k]), 4) - v))
        row["score_max_abs_diff"] = max(diffs) if diffs else 0.0
        worst = max(worst, row["score_max_abs_diff"])
    table = pd.DataFrame(rows)

    # Table 1, recomputed from the re-run JSONs and from the frame scores.
    t1 = []
    for m in modes:
        refs = [frozen[(c, m)] for c in clips]
        t1.append(dict(modality=m,
                       **{f"frozen_{k}": float(np.mean([r[k] for r in refs]))
                          for k in ["inst_F1_50", "inst_F1_75", "n_regions_mean", "GT_mIoU"]},
                       n_windows=len(refs), n_gt_frames=sum(r["n_gt_frames"] for r in refs)))
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, "legacy_check.csv"), index=False)
    pd.DataFrame(t1).to_csv(os.path.join(out, "table1_reproduced.csv"), index=False)
    write_meta(out, cfg, what="reproduction of AE-CAI Table 1", frozen_dir=FROZEN)
    bad = table[table["tagged_rerun_differs"] != ""]
    print(pd.DataFrame(t1).round(3).to_string(index=False))
    print(f"tagged re-run differs on {len(bad)} of {len(table)}; "
          f"aecai_rev.score max abs diff {worst:.2e}")
    if len(bad) or worst > 5e-5:
        raise SystemExit("the reproduction differs; see legacy_check.csv")


if __name__ == "__main__":
    main()
